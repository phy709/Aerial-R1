from typing import Literal
from collections import OrderedDict
from pycocotools import mask as _mask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from third_parts.mmdet.models.utils.point_sample import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor

class Sa2VAModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # ext
                 # preprocessor=None,
                 # bs
                 training_bs:int=0,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        # FIX: Untie weights for Qwen model
        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
            print("Untying embed_tokens and lm_head weights for Qwen model.")
            self.mllm.model.config.tie_word_embeddings = False
            lm_head = self.mllm.model.get_output_embeddings()
            if lm_head is not None:
                input_embeddings = self.mllm.model.get_input_embeddings()
                lm_head.weight = nn.Parameter(input_embeddings.weight.clone())

        in_dim = self.mllm.get_embedding_size()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

            # FIX: Force update lm_head weight after loading state_dict
            if self.arch_type == 'qwen':
                print("Force updating lm_head weight from pretrained state_dict.")
                lm_head_key = 'mllm.model.lm_head.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)
                    print(f"Successfully updated lm_head weight from key: {lm_head_key}")
                else:
                    print(f"Warning: lm_head weight key '{lm_head_key}' not found in pretrained_state_dict.")

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.bs = training_bs

        if self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()

        # Print gradient status of all weights in self.mllm.model.base_model.model
        print("\n" + "="*80)
        print("GRADIENT STATUS OF MLLM.MODEL WEIGHTS")
        print("="*80)
        
        try:
            base_model = self.mllm.model
            total_params = 0
            trainable_params = 0
            
            for name, param in base_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    grad_status = "✓ TRAINABLE"
                else:
                    grad_status = "✗ FROZEN"
                
                print(f"{name:<60} | {grad_status} | Shape: {tuple(param.shape)} | Params: {param.numel():,}")
            
            print("-" * 80)
            print(f"SUMMARY:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
            print("=" * 80)
            
        except Exception as e:
            print(f"Failed to access self.mllm.model: {e}")
            print("Available attributes in self.mllm.model:")
            print([attr for attr in dir(self.mllm.model) if not attr.startswith('_')])


    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] # required to make add_special_tokens to be False to avoid <bos> or <eos>

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return
        
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_sam2 = self.grounding_encoder.state_dict(*args, prefix=prefix + 'grounding_encoder.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(
            {k: v
             for k, v in state_dict_sam2.items() if k.startswith('grounding_encoder.sam2_model.sam_mask_decoder')})
        to_return.update(state_dict_text)
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video = self.generate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)


        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        return loss_dict


    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def generate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        assert len(pred_embeddings_list) == len(frames_per_batch)
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        raise NotImplementedError("Sa2VA does not support preparing for generation, please use predict_video instead.")
    
    # --- 请将这段代码插入到 Sa2VAModel 类中 ---
    

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

# =================================================================================
# [New Added] Sa2VAGRPOModel for Crop-Consistency Training
# =================================================================================

class Sa2VAGRPOModel(Sa2VAModel):
    """
    Implementation of Sa2VA with S-GRPO (Spatial-Scale Group Relative Policy Optimization).
    Focuses on reinforcing small object segmentation by sampling multiple mask candidates
    and weighting loss based on IoU and object scale.
    """
    def __init__(self, group_size=4, small_obj_bonus=3.0, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.small_obj_bonus = small_obj_bonus
        print(f"Initialized Sa2VAGRPOModel with Group Size: {group_size}, Small Obj Bonus: {small_obj_bonus}")

    def compute_grpo_reward(self, pred_masks, gt_masks):
        """
        组合策略：指数级奖励 (Strategy A) + 硬阈值门槛 (Strategy B)
        """
        # 1. 预处理
        pred_probs = torch.sigmoid(pred_masks)
        pred_binary = (pred_probs > 0.5).float()
        gt_masks = (gt_masks > 0.1).float()

        # 2. 计算 IoU
        intersection = (pred_binary * gt_masks).sum(dim=(-1, -2))
        union = (pred_binary + gt_masks).clamp(0, 1).sum(dim=(-1, -2))
        ious = (intersection + 1e-6) / (union + 1e-6)
        
        # 3. 计算面积占比
        obj_areas = gt_masks.sum(dim=(-1, -2))
        total_area = gt_masks.shape[-1] * gt_masks.shape[-2]
        area_ratios = obj_areas / (total_area + 1e-6)
        
        # ==========================================
        # 策略 A：指数级奖励 (Exponential Reward)
        # ==========================================
        # 面积越小，分母越小，倍率越高。
        # 1% 面积 -> ~1倍; 0.1% 面积 -> ~10倍
        # 加上 clamp 防止倍率过大导致梯度爆炸 (限制在 20 倍以内)
        exp_weights = 0.01 / (area_ratios + 1e-4)
        exp_weights = torch.clamp(exp_weights, min=1.0, max=20.0)

        # ==========================================
        # 策略 B：硬阈值门槛 (Hard Thresholding)
        # ==========================================
        # 只有当 IoU > 0.3 (且是小目标) 时，才激活暴击倍率
        # 这里的 0.3 是一个经验值，您可以根据目前模型的平均 IoU 调整
        iou_threshold = 0.3
        is_small_obj = area_ratios < 0.01
        
        # 核心逻辑：满足条件才暴击
        # 条件：是小目标 AND 预测得比较准 (IoU > 0.3)
        trigger_bonus = is_small_obj & (ious > iou_threshold)
        
        # 最终组合权重
        # 如果触发暴击 -> 使用指数权重
        # 如果没触发   -> 权重为 1.0 (即只拿普通 IoU 奖励)
        final_weights = torch.where(trigger_bonus, exp_weights, 1.0)
        
        rewards = ious * final_weights
        
        return rewards

    def forward(self, data, data_samples=None, mode='loss'):
        if mode != 'loss':
            return super().forward(data, data_samples, mode)

        # -----------------------------------------------------------
        # 1. Group Duplication (Batch Expansion)
        # -----------------------------------------------------------
        # 定义一个辅助函数来处理 Tensor 和 List 的复制
        def expand_data(key, value, group_size):
            if isinstance(value, torch.Tensor):
                return value.repeat_interleave(group_size, dim=0)
            elif isinstance(value, list):
                # List 复制逻辑: [A, B] -> [A, A, ..., B, B, ...]
                return [item for item in value for _ in range(group_size)]
            else:
                return value

        # 逐个处理关键字段
        # Input IDs 和 Attention Mask 通常是 Tensor
        data['input_ids'] = expand_data('input_ids', data['input_ids'], self.group_size)
        data['attention_mask'] = expand_data('attention_mask', data['attention_mask'], self.group_size)
        
        # Labels 也是 Tensor
        if 'labels' in data:
            data['labels'] = expand_data('labels', data['labels'], self.group_size)

        # 修复报错点：pixel_values 可能是 List[Tensor]
        if 'pixel_values' in data:
            data['pixel_values'] = expand_data('pixel_values', data['pixel_values'], self.group_size)
            
        # 处理 image_flags (可能是 List 或 Tensor)
        if 'image_flags' in data:
            data['image_flags'] = expand_data('image_flags', data['image_flags'], self.group_size)

        # 处理 SAM2 需要的 g_pixel_values (可能是 List 或 Tensor)
        if 'g_pixel_values' in data:
             data['g_pixel_values'] = expand_data('g_pixel_values', data['g_pixel_values'], self.group_size)

        # 处理 Masks (Ground Truth) - 之前手动处理的部分保持不变或也用此逻辑
        # 注意：sa2va 的 forward 通常会 pop 出 masks，所以这里要小心处理
        gt_masks_list = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        
        # 扩展 frames_per_batch (List[int])
        if frames_per_batch is not None:
            frames_per_batch = [f for f in frames_per_batch for _ in range(self.group_size)]
        
        # 扩展 GT Masks (List[Tensor])
        new_gt_masks = []
        if gt_masks_list is not None:
            for mask in gt_masks_list:
                for _ in range(self.group_size):
                    # 确保深拷贝，防止引用问题（视情况而定，通常 Tensor 引用即可）
                    new_gt_masks.append(mask)
        gt_masks = new_gt_masks

        # -----------------------------------------------------------
        # 2. Forward Pass (MLLM)
        # -----------------------------------------------------------
        # We rely on Dropout in the MLLM (train mode) to create variance 
        # in the [SEG] token embeddings across the group.
        output = self.mllm(data, data_samples, mode)
        
        # -----------------------------------------------------------
        # 3. Extract [SEG] Tokens & Embeddings (Copied from Sa2VA Logic)
        # -----------------------------------------------------------
        input_ids = data['input_ids'] # Updated expanded input_ids
        seg_token_mask = input_ids == self.seg_token_idx
        
        hidden_states = output.hidden_states
        # Project hidden states to SAM2 embedding space
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        
        pred_embeddings = hidden_states[seg_token_mask]
        
        # Handle batch splitting for video/frames
        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]
        
        # Expand embeddings for video frames
        pred_embeddings_list_video = self.generate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        
        # Process GT masks to match video format
        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        
        # Check obj number
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )

        # -----------------------------------------------------------
        # 4. SAM2 Decoding
        # -----------------------------------------------------------
        # Preprocess images for SAM2
        g_pixel_values = data['g_pixel_values'] # Already expanded
        g_pixel_values_stack = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        
        # Prepare SAM2 inputs
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_stack, expand_size=num_objs)
        
        # PREDICT MASKS
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
        
        # Align GT Masks
        gt_masks_proc = [F.interpolate(gm.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gm in gt_masks_video]
        gt_masks_proc = torch.cat(gt_masks_proc, dim=0)
        pred_masks = pred_masks.flatten(0, 1) # [Total_Objects, H, W]

        # -----------------------------------------------------------
        # 5. GRPO Advantage Calculation
        # -----------------------------------------------------------
        # Calculate Rewards per object/sample
        with torch.no_grad():
            raw_rewards = self.compute_grpo_reward(pred_masks, gt_masks_proc) # [Total_Objects]
            
            # Reshape to [Batch, Group] to calculate relative advantage
            # Note: Total_Objects might vary if mulitple objs per image. 
            # Assuming 1 obj per image for simplicity or averaging across objects in image.
            # Ideally, we calculate advantage per 'Group' (per source image).
            
            # Simplified: Global Normalization or Per-Batch-Group Normalization
            # Reshape based on group_size. 
            # If multiple objects per image, they are flattened. 
            # We assume objects from same image are grouped.
            
            # Let's reshape to [Num_Unique_Images, Group_Size * Num_Objs_Per_Img] roughly
            # Or simpler: reshape to [ -1, group_size ] if 1 object per image.
            
            # Robust Advantage:
            # Group by source sample index. Since we repeated interleave:
            # Index: 0, 1, ..., G-1 are from Sample 0.
            # reshape raw_rewards to [-1, self.group_size]
            
            # Note: This reshaping assumes fixed num objects per image. 
            # If variable, we must track indices. For scale, let's assume standard reshaping.
            rewards_view = raw_rewards.view(-1, self.group_size)
            
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            std_rewards = rewards_view.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards_view - mean_rewards) / std_rewards
            advantages = advantages.flatten() # [Total_Objects]

        # -----------------------------------------------------------
        # 6. Loss Calculation (Advantage Weighted)
        # -----------------------------------------------------------
        # Standard Dice/Mask Loss
        sam_loss_mask = self.loss_mask(pred_masks, gt_masks_proc) # Returns vector or scalar? usually scalar in Sa2VA config.
        # We need element-wise loss to apply weights!
        # Sa2VA config usually uses reduction='mean'. We need to hack this or use manual calculation.
        
        # Re-calculating loss manually to support weighting
        # Binary Cross Entropy
        loss_ce_element = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_proc.float(), reduction='none').mean(dim=(-1, -2))
        
        # Dice Loss Element-wise
        pred_sigmoid = torch.sigmoid(pred_masks)
        numerator = 2 * (pred_sigmoid * gt_masks_proc).sum(dim=(-1, -2))
        denominator = pred_sigmoid.sum(dim=(-1, -2)) + gt_masks_proc.sum(dim=(-1, -2))
        loss_dice_element = 1 - (numerator + 1.0) / (denominator + 1.0)
        
        total_loss_element = 2.0 * loss_ce_element + 0.5 * loss_dice_element
        
        # GRPO Weighting:
        # If Advantage > 0, we want to Minimize Loss MORE (Weight > 1) ??
        # No, Standard RL: Loss = - Advantage * LogProb.
        # Here Loss ~= -LogProb. So Loss_RL = Advantage * Loss.
        # But we must handling negative advantage.
        # If A < 0 (Bad sample), we want to Increase Loss? No, we want to suppress gradient.
        # Standard PPO Clip logic is better, but simple weighting:
        # We want to encourage samples with high A.
        # Let's use: loss = loss * exp(beta * advantage) ? 
        # Or simply: loss = loss * (advantages - advantages.min() + 1e-6) # Normalized positive weights
        
        # Simple/Effective Strategy: 
        # Only learn from the "Good" half of the group?
        # Or standard Policy Gradient on the [SEG] token LLM loss, plus weighted Mask loss.
        
        # Let's apply Advantage to the LLM generation loss (output.loss)
        # This steers the [SEG] token selection.
        # Since output.loss is scalar mean, we can't weight it easily without access to per-token loss.
        # But we can add an auxiliary loss.
        
        # FINAL IMPLEMENTATION CHOICE: Mask Loss Weighted by Positive Advantage
        # We only reinforce samples that are "better than average".
        valid_mask = advantages > 0
        if valid_mask.sum() > 0:
            weighted_loss = (total_loss_element * valid_mask.float() * advantages.abs()).sum() / (valid_mask.sum() + 1e-6)
        else:
            weighted_loss = total_loss_element.mean() # Fallback

        # Combine with LLM loss (Regularization)
        final_loss = output.loss + weighted_loss

        loss_dict = {
            'loss_mask': weighted_loss,
            'loss_dice': torch.tensor(0.0).to(weighted_loss.device), # Placeholders
            'llm_loss': output.loss,
            'reward_mean': raw_rewards.mean()
        }
        return loss_dict


class Sa2VAGRPO2Model(Sa2VAModel):
    """
    GRPO Wrapper for Sa2VA.
    Implements Strategy 2: Global-Local Consistency (Teacher-Student).
    """
    def __init__(self, reward_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # default weights: global_iou=1.0, consistency=1.0
        self.reward_weights = reward_weights or {'global_iou': 1.0, 'consistency': 1.0}

    def _get_crop_box(self, mask, pad_ratio=0.5):
        """Calculate crop box from GT mask."""
        if mask.sum() < 1: return None
        
        rows, cols = torch.where(mask > 0)
        y1, x1 = rows.min(), cols.min()
        y2, x2 = rows.max(), cols.max()
        
        h, w = y2 - y1, x2 - x1
        pad_h, pad_w = int(h * pad_ratio), int(w * pad_ratio)
        
        H, W = mask.shape
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(W, int(x2 + pad_w))
        y2 = min(H, int(y2 + pad_h))
        
        if x2 <= x1 or y2 <= y1: return None
        return (int(x1), int(y1), int(x2), int(y2))

    def _project_local_to_global(self, local_pred, crop_box, global_shape=(1024, 1024)):
        """Project local crop prediction back to global coordinates."""
        x1, y1, x2, y2 = crop_box
        crop_h, crop_w = y2 - y1, x2 - x1
        
        # Resize local prediction (1024x1024) down to actual crop size
        if local_pred.dim() == 2:
            local_pred = local_pred.unsqueeze(0).unsqueeze(0)
            
        local_small = F.interpolate(
            local_pred.float(), 
            size=(crop_h, crop_w), 
            mode='nearest'
        )
        
        global_canvas = torch.zeros(global_shape, device=local_pred.device)
        global_canvas[y1:y2, x1:x2] = local_small.squeeze()
        return global_canvas

    def _compute_iou(self, pred, target):
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        return (inter + 1e-6) / (union + 1e-6)

    def predict_mask_only(self, pixel_values, input_ids, attention_mask):
        """
        Run inference to get masks (Teacher Mode).
        Reproduces the forward pass but returns masks instead of loss.
        """
        # 1. MLLM Forward
        # Construct a dummy data dict as expected by self.mllm
        data = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        # We need to handle the visual processing manually or via mllm helper
        # Since mllm() expects specific keys, let's try to simulate the flow
        
        # MLLM forward to get hidden states
        # Note: We assume mllm can handle the input structure. 
        # Ideally, we follow the same path as forward()
        output = self.mllm(data, mode='loss') # mode='loss' is fine, we just need hidden states
        hidden_states = output.hidden_states
        
        # 2. Text Embeddings
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        
        # Extract SEG token embedding
        seg_token_mask = input_ids == self.seg_token_idx
        _zero = hidden_states.mean() * 0.0
        
        # If no SEG token found (fallback), take first 5 tokens
        if seg_token_mask.sum() > 0:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero
            
        # 3. SAM2 Forward
        # Preprocess image
        bs = pixel_values.shape[0]
        # SAM2 expects list of tensors for preprocessing if not batched correctly, 
        # but here we have a batch tensor [B, 3, H, W]
        # self.grounding_encoder.preprocess_image expects single image usually?
        # Let's check original forward: it uses a list comprehension
        g_pixel_values_processed = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in pixel_values
        ])
        
        # Get SAM2 image embeddings
        # Assuming 1 object per image for the crop (Teacher focuses on the view)
        # We broadcast the text embedding to match batch size
        num_objs = 1 
        num_frames = 1
        
        # Adjust embedding shape [N_total, C] -> [N_total, 1, C]
        language_embeddings = pred_embeddings[:, None, :]
        
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_processed, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, 
            language_embeddings, 
            nf_nobj=(num_frames, num_objs)
        )
        # pred_masks: [B*N, 1, H, W] -> flatten -> [B, H, W]
        pred_masks = pred_masks.squeeze(1)
        return pred_masks

    def compute_grpo_reward(self, 
                          student_pred_mask: torch.Tensor, 
                          gt_mask: torch.Tensor, 
                          pixel_values: torch.Tensor,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          task_type: str = 'global'):
        """
        Compute Crop-Consistency Reward.
        """
        # 1. Base Reward: Global IoU
        reward_global = self._compute_iou(student_pred_mask, gt_mask)
        
        # Filter: Only apply consistency for 'global' tasks with valid GT
        if task_type != 'global' or gt_mask.sum() < 32: # Using 32px threshold as safe guard
            return reward_global

        # 2. Get Crop Box
        crop_box = self._get_crop_box(gt_mask)
        if crop_box is None:
            return reward_global
        x1, y1, x2, y2 = crop_box
        
        # 3. Teacher Inference (Local Crop)
        with torch.no_grad():
            # Crop and Resize Image to 1024x1024
            # pixel_values: [C, H, W] -> assume normalized
            crop_img = pixel_values[:, y1:y2, x1:x2]
            teacher_input_img = F.interpolate(
                crop_img.unsqueeze(0), 
                size=(1024, 1024), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # Remove batch dim for stacking later if batching needed
            
            # Predict
            # Note: We pass the single cropped image. 
            # predict_mask_only expects batch dimension
            teacher_pred_local = self.predict_mask_only(
                teacher_input_img.unsqueeze(0), 
                input_ids.unsqueeze(0), 
                attention_mask.unsqueeze(0)
            )[0] # Take 0-th element
            
        # 4. Project Teacher Prediction
        teacher_projected = self._project_local_to_global(teacher_pred_local, crop_box)
        
        # 5. Consistency Reward
        student_crop_view = student_pred_mask[y1:y2, x1:x2]
        teacher_crop_view = teacher_projected[y1:y2, x1:x2]
        
        consistency_score = self._compute_iou(student_crop_view, teacher_crop_view)
        
        # 6. Total Reward
        total_reward = reward_global + self.reward_weights['consistency'] * consistency_score
        return total_reward
    
class Sa2VARGRPOAntiHallucinationModel(Sa2VAModel):
    """
    R-GRPO Anti-Hallucination (Correct Version)

    - High-res samples:
        * go through MLLM + SAM2
        * rewarded by IoU
    - Low-res samples:
        * do NOT go through MLLM
        * use degraded images + SAM2 only
        * penalized by confidence
    """

    def __init__(
        self,
        group_size=8,
        lambda_low=0.5,
        downsample_ratio=0.25,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert group_size % 2 == 0, "group_size must be even"

        self.group_size = group_size
        self.lambda_low = lambda_low
        self.downsample_ratio = downsample_ratio

    # ------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------
    def make_low_res(self, img):
        """
        img: [C, H, W] or [1, C, H, W]
        return: same shape but low-resolution then upsampled
        """
        squeeze = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze = True

        if img.dtype == torch.uint8:
            img = img.float().div(255.0)

        _, _, h, w = img.shape
        low_h = int(h * self.downsample_ratio)

        # downsample
        low = F.interpolate(
            img,
            size=(low_h, low_h),
            mode="bilinear",
            align_corners=False,
        )

        # upsample back (⚠️ keep SAME size for SAM2)
        low = F.interpolate(
            low,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        if squeeze:
            low = low.squeeze(0)

        return low

    def compute_rgpro_reward(self, pred_masks, gt_masks, is_low_res):
        """
        pred_masks: [N, H, W] (logits)
        gt_masks:   [N, H, W]
        is_low_res: [N] bool
        """
        pred_probs = torch.sigmoid(pred_masks)
        pred_binary = (pred_probs > 0.5).float()
        gt = (gt_masks > 0.5).float()

        inter = (pred_binary * gt).sum(dim=(-1, -2))
        union = (pred_binary + gt).clamp(0, 1).sum(dim=(-1, -2))
        iou = (inter + 1e-6) / (union + 1e-6)

        confidence = pred_probs.mean(dim=(-1, -2))

        rewards = torch.zeros_like(iou)
        rewards[~is_low_res] = iou[~is_low_res]
        rewards[is_low_res] = -self.lambda_low * confidence[is_low_res]

        return rewards

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, data, data_samples=None, mode="loss"):
        if mode != "loss":
            return super().forward(data, data_samples, mode)

        device = data["pixel_values"][0].device
        half = self.group_size // 2

        # =========================================================
        # 1. High-res branch (MLLM + SAM2)
        # =========================================================
        high_data = {}

        # images (HIGH ONLY)
        high_data["pixel_values"] = [
            img for img in data["pixel_values"] for _ in range(half)
        ]
        high_data["g_pixel_values"] = [
            img for img in data["g_pixel_values"] for _ in range(half)
        ]

        def expand(x):
            return x.repeat_interleave(half, dim=0)

        # ---- text inputs (⚠️ MUST be complete) ----
        high_data["input_ids"] = expand(data["input_ids"])
        high_data["attention_mask"] = expand(data["attention_mask"])

        if "position_ids" in data:
            high_data["position_ids"] = expand(data["position_ids"])

        if "labels" in data:
            high_data["labels"] = expand(data["labels"])

        # GT
        gt_masks = data["masks"]
        frames_per_batch = data["frames_per_batch"]

        gt_masks_high = [m for m in gt_masks for _ in range(half)]
        frames_high = [f for f in frames_per_batch for _ in range(half)]

        # ---- MLLM forward (HIGH ONLY) ----
        output = self.mllm(high_data, data_samples, mode)

        # extract SEG embeddings
        input_ids = high_data["input_ids"]
        seg_mask = input_ids == self.seg_token_idx

        hidden = self.text_hidden_fcs(output.hidden_states[-1])
        seg_embeds = hidden[seg_mask]

        seg_counts = seg_mask.int().sum(-1)
        seg_list = torch.split(seg_embeds, seg_counts.tolist(), dim=0)
        seg_list = [x for x in seg_list if len(x) > 0]

        pred_embeds_video = self.generate_video_pred_embeddings(
            seg_list, frames_high
        )
        gt_masks_video = self.process_video_gt_masks(
            gt_masks_high, frames_high
        )

        pred_embeds_video, gt_masks_video = self.check_obj_number(
            pred_embeds_video, gt_masks_video
        )

        # ---- SAM2 decode (HIGH) ----
        g_pixels = torch.stack([
            self.grounding_encoder.preprocess_image(p)
            for p in high_data["g_pixel_values"]
        ])

        num_objs = pred_embeds_video[0].shape[0]
        num_frames = len(pred_embeds_video)

        lang_emb = torch.cat(pred_embeds_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(
            g_pixels, expand_size=num_objs
        )

        pred_masks_high = self.grounding_encoder.inject_language_embd(
            sam_states,
            lang_emb,
            nf_nobj=(num_frames, num_objs),
        ).flatten(0, 1)

        gt_masks_proc = torch.cat([
            F.interpolate(
                m.unsqueeze(0),
                size=pred_masks_high.shape[-2:],
                mode="nearest",
            ).squeeze(0)
            for m in gt_masks_video
        ], dim=0)

        # =========================================================
        # 2. Low-res branch (NO MLLM, SAM2 only)
        # =========================================================
        with torch.no_grad():
            low_g_pixels = []
            for img in data["g_pixel_values"]:
                low = self.make_low_res(img)
                for _ in range(half):
                    low_g_pixels.append(low)

            low_g_pixels = torch.stack([
                self.grounding_encoder.preprocess_image(p)
                for p in low_g_pixels
            ])

            sam_states_low = self.grounding_encoder.get_sam2_embeddings(
                low_g_pixels, expand_size=num_objs
            )

            pred_masks_low = self.grounding_encoder.inject_language_embd(
                sam_states_low,
                lang_emb,
                nf_nobj=(num_frames, num_objs),
            ).flatten(0, 1)

        # =========================================================
        # 3. Reward & Advantage
        # =========================================================
        pred_masks_all = torch.cat(
            [pred_masks_high, pred_masks_low], dim=0
        )
        gt_masks_all = gt_masks_proc.repeat(2, 1, 1)

        is_low_res = torch.zeros(
            pred_masks_all.shape[0],
            device=device,
            dtype=torch.bool,
        )
        is_low_res.view(-1, self.group_size)[:, half:] = True

        with torch.no_grad():
            rewards = self.compute_rgpro_reward(
                pred_masks_all, gt_masks_all, is_low_res
            )

            rewards_g = rewards.view(-1, self.group_size)
            mean = rewards_g.mean(dim=1, keepdim=True)
            std = rewards_g.std(dim=1, keepdim=True) + 1e-6
            advantages = ((rewards_g - mean) / std).flatten()

        # =========================================================
        # 4. Loss
        # =========================================================
        loss_ce = F.binary_cross_entropy_with_logits(
            pred_masks_all, gt_masks_all.float(), reduction="none"
        ).mean(dim=(-1, -2))

        pred_sigmoid = torch.sigmoid(pred_masks_all)
        inter = (pred_sigmoid * gt_masks_all).sum(dim=(-1, -2))
        denom = pred_sigmoid.sum(dim=(-1, -2)) + gt_masks_all.sum(dim=(-1, -2))
        loss_dice = 1 - (2 * inter + 1) / (denom + 1)

        loss_elem = 2.0 * loss_ce + 0.5 * loss_dice

        pos = advantages > 0
        rl_loss = (
            (loss_elem[pos] * advantages[pos]).mean()
            if pos.any()
            else loss_elem.mean()
        )

        total_loss = output.loss + rl_loss

        return {
            "loss": total_loss,
            "llm_loss": output.loss,
            "loss_mask": rl_loss,
            "reward_mean": rewards.mean(),
        }
    
class Sa2VAGRPO_Final(Sa2VAGRPOModel):
    """
    [ICML Strategy] One-Pass GRPO: Consistency + Anti-Hallucination
    
    Modified for "Positive-Only" Datasets via 'Contrastive Resolution Group Sampling'.
    
    Mechanism:
    1.  Splits the group (size G) into two halves:
        -   First Half: High-Res Original Images + Real GTs -> Learn Precision & Recall.
        -   Second Half: Low-Res Degraded Images + Empty GTs -> Learn Refusal (Anti-Hallucination).
    2.  Hybrid Reward Function:
        -   Positive Samples: Reward = IoU + Beta * Consistency
        -   Negative Samples (Low-Res): Reward = +Silence_Reward (if empty) OR -Penalty (if hallucinated).
    """
    def __init__(self, 
                 beta=0.01,
                 group_size=8, 
                 consistency_beta=0.5, 
                 downsample_ratio=0.125,
                 hallucination_penalty=2.0, 
                 silence_reward=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.group_size = group_size
        self.consistency_beta = consistency_beta
        self.downsample_ratio = downsample_ratio
        
        # Hyperparameters for Anti-Hallucination
        self.hallucination_penalty = hallucination_penalty
        self.silence_reward = silence_reward
        
        print(f"Initialized Sa2VAGRPO_Final (Self-Negative Mode):")
        print(f" - Group Size: {group_size}")
        print(f" - Consistency Beta: {consistency_beta}")
        print(f" - Downsample Ratio: {downsample_ratio}")
        print(f" - Hallucination Penalty: -{hallucination_penalty}")
        print(f" - Silence Reward: +{silence_reward}")

    def make_low_res(self, img):
        """
        Visual Degradation Attack:
        Downsample image to lose details, then Upsample back to original size.
        This forces the model to rely on visual evidence rather than text priors.
        """
        # img shape: [C, H, W] or [N, C, H, W]
        is_batch = img.dim() == 4
        if not is_batch:
            img = img.unsqueeze(0)
            
        # Normalize if needed (assuming input is float 0-1 or uint8 0-255)
        if img.dtype == torch.uint8:
            img = img.float().div(255.0)
            
        N, C, H, W = img.shape
        # Ensure minimum size of 8x8 to avoid errors
        low_h = max(8, int(H * self.downsample_ratio))
        low_w = max(8, int(W * self.downsample_ratio))

        # 1. Downsample (Lose visual features)
        low = F.interpolate(
            img,
            size=(low_h, low_w),
            mode="bilinear",
            align_corners=False,
        )

        # 2. Upsample back (Keep tensor shape compatible for batching)
        # Using 'nearest' to maintain the "pixelated/blurry" look
        restored = F.interpolate(
            low,
            size=(H, W),
            mode="nearest", 
        )
        
        if not is_batch:
            restored = restored.squeeze(0)
            
        return restored

    def compute_grpo_reward(self, pred_masks, gt_masks):
        """
        Hybrid Reward Logic.
        Dynamically switches between Consistency Reward and Anti-Hallucination Penalty.
        """
        device = pred_masks.device
        
        # 1. Pre-process Predictions and GT
        pred_probs = torch.sigmoid(pred_masks)
        pred_binary = (pred_probs > 0.5).float()
        gt_binary = (gt_masks > 0.5).float()

        # 2. Identify Negative Samples
        # Since we force Low-Res samples to have Empty GTs, sum() will be 0.
        # Threshold 32 is a safety margin for noise.
        is_negative = gt_binary.sum(dim=(-1, -2)) < 32

        # -------------------------------------------------------
        # Branch A: Negative Samples (Low-Res / Anti-Hallucination)
        # -------------------------------------------------------
        # Check if model predicted "too much" (Hallucination)
        pred_pixel_sum = pred_binary.sum(dim=(-1, -2))
        # Tolerance: Allow < 50 pixels of noise
        is_clean_prediction = pred_pixel_sum < 50 
        
        # Reward Logic:
        # - If clean (refused to segment): Give Silence Reward (+1.0)
        # - If dirty (hallucinated): Give Penalty (-2.0)
        rewards_neg = torch.where(
            is_clean_prediction, 
            torch.tensor(self.silence_reward).to(device), 
            torch.tensor(-self.hallucination_penalty).to(device)
        )

        # -------------------------------------------------------
        # Branch B: Positive Samples (High-Res / Precision)
        # -------------------------------------------------------
        # 2.1 Basic IoU
        intersection = (pred_binary * gt_binary).sum(dim=(-1, -2))
        union = (pred_binary + gt_binary).clamp(0, 1).sum(dim=(-1, -2))
        iou_scores = (intersection + 1e-6) / (union + 1e-6)

        # 2.2 Group Consistency
        # Calculate consistency against the Group Mean
        # Reshape to [Batch_Real, Group, H, W]
        B_real = pred_probs.shape[0] // self.group_size
        preds_view = pred_probs.view(B_real, self.group_size, pred_probs.shape[-2], pred_probs.shape[-1])
        
        # Mean mask of the group
        group_mean = preds_view.mean(dim=1, keepdim=True)
        group_mean_binary = (group_mean > 0.5).float()
        
        # Expand mean back to individual shape for comparison
        group_mean_target = group_mean_binary.expand_as(preds_view).reshape(-1, pred_probs.shape[-2], pred_probs.shape[-1])
        
        # IoU with Group Mean
        inter_consist = (pred_binary * group_mean_target).sum(dim=(-1, -2))
        union_consist = (pred_binary + group_mean_target).clamp(0, 1).sum(dim=(-1, -2))
        consistency_scores = (inter_consist + 1e-6) / (union_consist + 1e-6)

        # Combined Positive Reward
        rewards_pos = iou_scores + self.consistency_beta * consistency_scores

        # -------------------------------------------------------
        # 3. Final Switch
        # -------------------------------------------------------
        # If is_negative (Low-Res) -> Use rewards_neg
        # If not is_negative (High-Res) -> Use rewards_pos
        final_rewards = torch.where(is_negative, rewards_neg, rewards_pos)

        return final_rewards

    def forward(self, data, data_samples=None, mode='loss'):
        if mode != 'loss':
            return super().forward(data, data_samples, mode)

        # ===========================================================
        # 1. Custom Group Expansion (High-Res vs Low-Res Mixing)
        # ===========================================================
        half_group = self.group_size // 2
        
        # --- Helper to expand text inputs (Simple Repetition) ---
        def expand_text(val):
            if isinstance(val, torch.Tensor): 
                return val.repeat_interleave(self.group_size, dim=0)
            elif isinstance(val, list):
                return [item for item in val for _ in range(self.group_size)]
            return val

        data['input_ids'] = expand_text(data['input_ids'])
        data['attention_mask'] = expand_text(data['attention_mask'])
        if 'labels' in data: data['labels'] = expand_text(data['labels'])
        if 'image_flags' in data: data['image_flags'] = expand_text(data['image_flags'])
        
        # --- Expand Visuals (Half High-Res, Half Low-Res) ---
        # We focus on g_pixel_values (SAM2 inputs) as they determine segmentation boundaries.
        orig_g_pixels = data['g_pixel_values'] # List[Tensor] or Tensor
        new_g_pixels = []
        
        batch_len = len(orig_g_pixels)
        for i in range(batch_len):
            raw_img = orig_g_pixels[i]
            # Create Low-Res Version
            low_img = self.make_low_res(raw_img)
            
            # First Half: High-Res (Original)
            for _ in range(half_group):
                new_g_pixels.append(raw_img)
            # Second Half: Low-Res (Degraded)
            for _ in range(self.group_size - half_group):
                new_g_pixels.append(low_img)
        
        # Stack back to Tensor
        if len(new_g_pixels) > 0 and isinstance(new_g_pixels[0], torch.Tensor):
            data['g_pixel_values'] = torch.stack(new_g_pixels)
        else:
            data['g_pixel_values'] = new_g_pixels

        # Expand pixel_values for MLLM (Standard Expansion)
        # Note: We keep MLLM inputs standard to ensure it generates [SEG] tokens correctly.
        # The visual rejection logic is enforced via SAM2's inability to see details.
        if 'pixel_values' in data:
            data['pixel_values'] = expand_text(data['pixel_values'])

        # --- Expand Masks & CREATE EMPTY GTs ---
        gt_masks_list = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        
        if frames_per_batch:
            frames_per_batch = [f for f in frames_per_batch for _ in range(self.group_size)]
        
        new_gt_masks = []
        if gt_masks_list:
            for mask in gt_masks_list:
                # Create Empty Mask (Black)
                empty_mask = torch.zeros_like(mask)
                
                # First Half: Real GT (For High-Res)
                for _ in range(half_group):
                    new_gt_masks.append(mask)
                # Second Half: Empty GT (For Low-Res) -> Forces Negative Sample Logic
                for _ in range(self.group_size - half_group):
                    new_gt_masks.append(empty_mask)
        
        gt_masks = new_gt_masks

        # ===========================================================
        # 2. Standard Forward Pipeline (Inference)
        # ===========================================================
        # (This part is copied from Sa2VAGRPOModel to ensure correct data flow)
        output = self.mllm(data, data_samples, mode)

        input_ids = data['input_ids']
        seg_token_mask = input_ids == self.seg_token_idx
        
        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        
        # Extract embeddings
        _zero = hidden_states.mean() * 0.0
        # If seg token exists, use it; else fallback (though SFT model should produce it)
        if seg_token_mask.any():
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
             # Fallback: take first 5 tokens flattened if no [SEG] found
             # This is just to prevent crash, usually means bad generation
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]
        
        pred_embeddings_list_video = self.generate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)
        
        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )

        # Prepare SAM2 inputs
        g_pixel_values = data['g_pixel_values']
        # Handle list vs tensor compatibility
        if isinstance(g_pixel_values, list):
             g_pixel_values_stack = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
            ])
        else:
             g_pixel_values_stack = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
            ])

        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_stack, expand_size=num_objs)
        
        # Predict Masks
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
        
        # Align GT Masks size
        gt_masks_proc = [F.interpolate(gm.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gm in gt_masks_video]
        gt_masks_proc = torch.cat(gt_masks_proc, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        # ===========================================================
        # 3. Compute GRPO Reward & Loss (With Hybrid Logic)
        # ===========================================================
        with torch.no_grad():
            # Pass to hybrid reward function
            # The function sees Empty GTs in second half and applies penalties
            raw_rewards = self.compute_grpo_reward(pred_masks, gt_masks_proc)
            
            # Normalize Advantage
            rewards_view = raw_rewards.view(-1, self.group_size)
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            std_rewards = rewards_view.std(dim=1, keepdim=True) + 1e-4
            advantages = (rewards_view - mean_rewards) / std_rewards
            advantages = advantages.flatten()

        # Calculate Element-wise Loss
        loss_ce_element = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_proc.float(), reduction='none').mean(dim=(-1, -2))
        
        pred_sigmoid = torch.sigmoid(pred_masks)
        numerator = 2 * (pred_sigmoid * gt_masks_proc).sum(dim=(-1, -2))
        denominator = pred_sigmoid.sum(dim=(-1, -2)) + gt_masks_proc.sum(dim=(-1, -2))
        loss_dice_element = 1 - (numerator + 1.0) / (denominator + 1.0)
        
        total_loss_element = 2.0 * loss_ce_element + 0.5 * loss_dice_element
        
        # Advantage Weighting (GRPO Policy Gradient)
        # Only learn from samples that are better than average (Positive Advantage)
        valid_mask = advantages > 0
        if valid_mask.sum() > 0:
            weighted_loss = (total_loss_element * valid_mask.float() * advantages.abs()).sum() / (valid_mask.sum() + 1e-6)
        else:
            # Fallback if all advantages are negative (rare)
            weighted_loss = total_loss_element.mean()

        final_loss = output.loss + weighted_loss

        loss_dict = {
            'loss_mask': weighted_loss,
            'llm_loss': output.loss,
            'reward_mean': raw_rewards.mean(),
            'reward_std': raw_rewards.std()
        }
        return loss_dict

class Sa2VAGRPO2sModel(Sa2VAGRPOModel):
    """
    True GRPO implementation of Crop-Consistency.
    Combines:
    1. Group Sampling (from Sa2VAGRPOModel) -> Generates multiple student variants
    2. Teacher Consistency (from old Scheme 2) -> Provides high-res supervision
    """
    def __init__(self, reward_weights=None, **kwargs):
        # 继承 Scheme 1 的 group_size 等初始化逻辑
        super().__init__(**kwargs)
        # default weights: global_iou=1.0, consistency=1.0
        self.reward_weights = reward_weights or {'global_iou': 1.0, 'consistency': 1.0}
        print(f"Initialized Consistency GRPO with weights: {self.reward_weights}")

    def _get_crop_box(self, mask, pad_ratio=0.5):
        """Helper: Calculate crop box from GT mask (Same as before)"""
        if mask.sum() < 1: return None
        rows, cols = torch.where(mask > 0)
        y1, x1 = rows.min(), cols.min()
        y2, x2 = rows.max(), cols.max()
        h, w = y2 - y1, x2 - x1
        pad_h, pad_w = int(h * pad_ratio), int(w * pad_ratio)
        H, W = mask.shape
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(W, int(x2 + pad_w))
        y2 = min(H, int(y2 + pad_h))
        if x2 <= x1 or y2 <= y1: return None
        return (int(x1), int(y1), int(x2), int(y2))

    def _project_local_to_global(self, local_pred, crop_box, global_shape=(1024, 1024)):
        """Project local crop prediction back to global coordinates."""
        x1, y1, x2, y2 = crop_box
        crop_h, crop_w = y2 - y1, x2 - x1
        
        # Resize local prediction (1024x1024) down to actual crop size
        if local_pred.dim() == 2:
            local_pred = local_pred.unsqueeze(0).unsqueeze(0)
            
        local_small = F.interpolate(
            local_pred.float(), 
            size=(crop_h, crop_w), 
            mode='nearest'
        )
        
        global_canvas = torch.zeros(global_shape, device=local_pred.device)
        
        # [CRITICAL FIX] 修复 1 像素宽/高导致的 Crash
        # 错误写法: global_canvas[y1:y2, x1:x2] = local_small.squeeze()
        # 原因: 如果 crop_w=1, squeeze() 会把宽维度也压扁，导致 [H, 1] vs [H] 维度不匹配
        
        # 正确写法: 显式取 [0, 0]，保留 H 和 W 维度，即使它们是 1
        global_canvas[y1:y2, x1:x2] = local_small[0, 0]
        
        return global_canvas

    def _compute_iou_batch(self, pred, target):
        """Compute IoU for a batch of masks [N, H, W]"""
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        inter = (pred * target).sum(dim=(-1, -2))
        union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - inter
        return (inter + 1e-6) / (union + 1e-6)

    def predict_mask_only_batch(self, pixel_values_448_batch, pixel_values_1024_batch, input_ids_batch, attention_mask_batch, position_ids_batch=None, labels_batch=None):
        """
        Batched Teacher Inference with Dual-Resolution Support.
        """
        # [Step 1] Resize Input IDs for MLLM (256 tokens for 448x448 image)
        if labels_batch is not None:
            input_ids_batch, labels_batch = self._resize_input_ids_and_labels(
                input_ids_batch, labels_batch, target_len=256
            )
        
        # Re-generate attention mask
        attention_mask_batch = torch.ones_like(input_ids_batch)

        # [Step 2] MLLM Forward
        pixel_values_input = [img for img in pixel_values_448_batch]
        bs = len(pixel_values_input) # <--- Capture Batch Size

        data = {
            'pixel_values': pixel_values_input,
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'position_ids': None
        }
        
        if labels_batch is not None:
            data['labels'] = labels_batch

        with torch.no_grad():
            output = self.mllm(data, mode='loss')
            hidden_states = output.hidden_states
            hidden_states = self.text_hidden_fcs(hidden_states[-1])
            
            # Extract SEG tokens
            seg_token_mask = input_ids_batch == self.seg_token_idx
            pred_embeddings = []
            for i in range(bs):
                mask_i = seg_token_mask[i]
                if mask_i.sum() > 0:
                    pred_embeddings.append(hidden_states[i][mask_i].mean(0))
                else:
                    pred_embeddings.append(hidden_states[i][:5].mean(0))
            pred_embeddings = torch.stack(pred_embeddings)

            # [Step 3] SAM2 Forward
            g_pixel_values_processed = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in pixel_values_1024_batch
            ])
            
            language_embeddings = pred_embeddings[:, None, :] 
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_processed, expand_size=1)
            
            # [CRITICAL FIX] Use actual batch size 'bs' instead of hardcoded 1
            pred_masks = self.grounding_encoder.inject_language_embd(
                sam_states, 
                language_embeddings, 
                nf_nobj=(bs, 1)  # <--- CHANGED FROM (1, 1) TO (bs, 1)
            )
            return pred_masks.squeeze(1)

    def compute_grpo_reward_consistency(self, student_pred_masks, gt_masks, data_dict):
        """
        [Final Version]
        1. Dual Resolution Input: 448 (MLLM) + 1024 (SAM2) -> Fixes Shape Mismatch
        2. Upsampling Strategy: Student (256) -> Teacher (1024) -> Fixes IoU Mismatch
        """
        device = student_pred_masks.device
        
        # 获取 Student 输出尺寸 (256x256)
        student_h, student_w = student_pred_masks.shape[-2:]
        
        # 1. 准备数据维度
        total_images = data_dict['input_ids'].shape[0]        
        total_objects = student_pred_masks.shape[0]           
        if total_images == 0: return torch.zeros_like(student_pred_masks[:, 0, 0]) # Fallback
        num_objects_per_image = total_objects // total_images

        # 2. 计算基础 Global IoU Reward (256x256 下计算，保持高效)
        reward_global = self._compute_iou_batch(student_pred_masks, gt_masks)
        
        # 3. 筛选 Teacher 任务
        teacher_tasks = []
        for img_batch_idx in range(0, total_images, self.group_size):
            start_obj_idx = img_batch_idx * num_objects_per_image
            for k in range(num_objects_per_image):
                obj_idx = start_obj_idx + k
                gt = gt_masks[obj_idx]
                crop_box = self._get_crop_box(gt)
                if crop_box is not None:
                    teacher_tasks.append({
                        'unique_img_idx': img_batch_idx,
                        'unique_obj_idx': obj_idx,
                        'crop_box': crop_box
                    })

        if not teacher_tasks:
            return self.reward_weights['global_iou'] * reward_global

        # 4. 准备双流输入 (Dual Input)
        teacher_pv_448 = []  
        teacher_pv_1024 = [] 
        teacher_input_ids = []
        teacher_labels = [] 
        has_labels = 'labels' in data_dict

        for task in teacher_tasks:
            img_idx = task['unique_img_idx']
            x1, y1, x2, y2 = task['crop_box']
            orig_img = data_dict['pixel_values'][img_idx]
            if orig_img.dim() == 4: orig_img = orig_img[0]
            
            # Crop & Resize
            if orig_img.dim() != 3: # Error handling
                crop_img = torch.zeros((3, 1024, 1024), device=device, dtype=orig_img.dtype)
            else:
                crop_img = orig_img[:, y1:y2, x1:x2]

            # Stream A: 448x448 for InternVL
            img_448 = F.interpolate(crop_img.unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0)
            # Stream B: 1024x1024 for SAM2
            img_1024 = F.interpolate(crop_img.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0)
            
            teacher_pv_448.append(img_448)
            teacher_pv_1024.append(img_1024)
            teacher_input_ids.append(data_dict['input_ids'][img_idx])
            if has_labels: teacher_labels.append(data_dict['labels'][img_idx])

        # 5. Teacher 推理
        teacher_crop_preds = self.predict_mask_only_batch(
            torch.stack(teacher_pv_448), 
            torch.stack(teacher_pv_1024),
            torch.stack(teacher_input_ids), 
            None, 
            position_ids_batch=None,
            labels_batch=torch.stack(teacher_labels) if has_labels else None
        )
        
        # 6. 计算一致性奖励 (使用上采样)
        reward_consistency = torch.zeros_like(reward_global)
        
        for i, task in enumerate(teacher_tasks):
            crop_pred = teacher_crop_preds[i] # [1024, 1024]
            box = task['crop_box']
            unique_obj_idx = task['unique_obj_idx']
            
            # 将 Teacher 投影回全局画布 -> [1024, 1024]
            teacher_global_pred = self._project_local_to_global(crop_pred, box)
            target_teacher = teacher_global_pred.unsqueeze(0).unsqueeze(0) # [1, 1, 1024, 1024]
            
            for g in range(self.group_size):
                target_idx = unique_obj_idx + (g * num_objects_per_image)
                if target_idx < total_objects:
                    # 获取 Student 预测 -> [256, 256]
                    student_pred = student_pred_masks[target_idx]
                    
                    # [关键步骤] 上采样 Student 到 1024 以匹配 Teacher
                    student_pred_up = F.interpolate(
                        student_pred.unsqueeze(0).unsqueeze(0), 
                        size=(1024, 1024), 
                        mode='bilinear', 
                        align_corners=False
                    ) # [1, 1, 1024, 1024]

                    # 计算高分辨率下的 IoU
                    inter = (student_pred_up > 0.5) * (target_teacher > 0.5)
                    union = (student_pred_up > 0.5) + (target_teacher > 0.5)
                    iou = (inter.sum() + 1e-6) / (union.clamp(0, 1).sum() + 1e-6)
                    
                    reward_consistency[target_idx] = iou

        total_reward = (self.reward_weights['global_iou'] * reward_global + 
                        self.reward_weights['consistency'] * reward_consistency)
        
        return total_reward

    def forward(self, data, data_samples=None, mode='loss'):
        """
        Overwrite forward to inject pixel_values into reward calculation.
        Most logic is identical to Sa2VAGRPOModel (Scheme 1).
        """
        if mode != 'loss':
            return super(Sa2VAGRPOModel, self).forward(data, data_samples, mode) # Call Grandparent

        # --- 1. Group Duplication (Same as Scheme 1) ---
        def expand_data(value, group_size):
            if isinstance(value, torch.Tensor):
                return value.repeat_interleave(group_size, dim=0)
            elif isinstance(value, list):
                return [item for item in value for _ in range(group_size)]
            return value

        # Expand inputs
        data['input_ids'] = expand_data(data['input_ids'], self.group_size)
        data['attention_mask'] = expand_data(data['attention_mask'], self.group_size)
        if 'labels' in data: data['labels'] = expand_data(data['labels'], self.group_size)
        if 'pixel_values' in data: data['pixel_values'] = expand_data(data['pixel_values'], self.group_size)
        if 'image_flags' in data: data['image_flags'] = expand_data(data['image_flags'], self.group_size)
        if 'g_pixel_values' in data: data['g_pixel_values'] = expand_data(data['g_pixel_values'], self.group_size)

        gt_masks_list = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        if frames_per_batch: frames_per_batch = [f for f in frames_per_batch for _ in range(self.group_size)]
        
        new_gt_masks = []
        if gt_masks_list:
            for mask in gt_masks_list:
                for _ in range(self.group_size): new_gt_masks.append(mask)
        gt_masks = new_gt_masks

        # --- 2. MLLM Forward (Student) ---
        output = self.mllm(data, data_samples, mode)

        # ... (Include standard Sa2VA extracting embeddings logic here) ...
        # ... (For brevity, I assume this logic is inherited or copied from Scheme 1) ...
        # ... (Let's assume we get 'pred_masks' and 'gt_masks_proc' ready) ...
        
        # [COPY-PASTE Logic from Sa2VAGRPOModel to get pred_masks]
        # START COPY
        input_ids = data['input_ids']
        seg_token_mask = input_ids == self.seg_token_idx
        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        pred_embeddings = hidden_states[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]
        pred_embeddings_list_video = self.generate_video_pred_embeddings(pred_embeddings_list, frames_per_batch)
        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(pred_embeddings_list_video, gt_masks_video)
        
        g_pixel_values_stack = torch.stack([self.grounding_encoder.preprocess_image(pixel) for pixel in data['g_pixel_values']])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values_stack, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
        
        gt_masks_proc = [F.interpolate(gm.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gm in gt_masks_video]
        gt_masks_proc = torch.cat(gt_masks_proc, dim=0)
        pred_masks = pred_masks.flatten(0, 1)
        # END COPY

        # --- 3. GRPO Consistency Reward Calculation ---
        with torch.no_grad():
            # Pass full data dict to access pixel_values for Teacher
            raw_rewards = self.compute_grpo_reward_consistency(pred_masks, gt_masks_proc, data)
            
            # Standard GRPO Advantage Normalization
            rewards_view = raw_rewards.view(-1, self.group_size)
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            std_rewards = rewards_view.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards_view - mean_rewards) / std_rewards
            advantages = advantages.flatten()

        # --- 4. Weighted Loss ---
        loss_ce_element = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_proc.float(), reduction='none').mean(dim=(-1, -2))
        pred_sigmoid = torch.sigmoid(pred_masks)
        numerator = 2 * (pred_sigmoid * gt_masks_proc).sum(dim=(-1, -2))
        denominator = pred_sigmoid.sum(dim=(-1, -2)) + gt_masks_proc.sum(dim=(-1, -2))
        loss_dice_element = 1 - (numerator + 1.0) / (denominator + 1.0)
        
        total_loss_element = 2.0 * loss_ce_element + 0.5 * loss_dice_element
        
        valid_mask = advantages > 0
        if valid_mask.sum() > 0:
            weighted_loss = (total_loss_element * valid_mask.float() * advantages.abs()).sum() / (valid_mask.sum() + 1e-6)
        else:
            weighted_loss = total_loss_element.mean()

        final_loss = output.loss + weighted_loss

        loss_dict = {
            'loss_mask': weighted_loss,
            'llm_loss': output.loss,
            'reward_mean': raw_rewards.mean()
        }
        return loss_dict

    def _resize_input_ids_and_labels(self, input_ids, labels, target_len=256):
        """
        Helper: Resize visual token placeholders in input_ids to match Teacher's image size.
        """
        device = input_ids.device
        new_input_ids_list = []
        new_labels_list = []
        
        # 自动获取当前模型的 img_token_id (通常是 <IMG_CONTEXT>)
        # 如果模型没有暴露，通常可以通过 tokenizer 获取，或者像下面这样统计
        # 这里假设 self.mllm.model.img_context_token_id 存在 (InternVL 标准属性)
        if hasattr(self.mllm.model, 'img_context_token_id'):
             img_token_id = self.mllm.model.img_context_token_id
        else:
             # Fallback: 统计 input_ids[0] 中出现次数最多的 token
             unique, counts = torch.unique(input_ids[0], return_counts=True)
             img_token_id = unique[counts.argmax()]

        for i in range(len(input_ids)):
            seq = input_ids[i]
            lab = labels[i]
            
            # 找到所有图片 token 的掩码
            is_img = (seq == img_token_id)
            
            if not is_img.any():
                new_input_ids_list.append(seq)
                new_labels_list.append(lab)
                continue
                
            # 找到图片 token 的起始和结束位置
            nonzero = torch.nonzero(is_img).flatten()
            start = nonzero[0]
            end = nonzero[-1]
            
            # 保留前缀和后缀文本
            prefix_id = seq[:start]
            suffix_id = seq[end+1:]
            prefix_lab = lab[:start]
            suffix_lab = lab[end+1:]
            
            # 构建新的图片占位符 (长度为 target_len=256)
            img_block_id = torch.full((target_len,), img_token_id, device=device, dtype=seq.dtype)
            img_block_lab = torch.full((target_len,), -100, device=device, dtype=lab.dtype) # -100 忽略 loss
            
            # 拼接
            new_seq = torch.cat([prefix_id, img_block_id, suffix_id])
            new_lab = torch.cat([prefix_lab, img_block_lab, suffix_lab])
            
            new_input_ids_list.append(new_seq)
            new_labels_list.append(new_lab)
            
        return torch.stack(new_input_ids_list), torch.stack(new_labels_list)