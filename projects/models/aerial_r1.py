
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
import torchvision.transforms.functional as TF

#######################################################################
#            Base Model: Sa2VA (SFT Baseline)                         #
#######################################################################

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


#######################################################################
#     Ablation Baseline: SFT + Low-Res Augmentation                   #
#######################################################################

class Sa2VAWithLowResAug(Sa2VAModel):
    def __init__(self, 
                 aug_prob=0.5,             
                 downsample_ratio=0.125,   
                 degradation_type='pixelation',
                 **kwargs):
        super().__init__(**kwargs)
        self.aug_prob = aug_prob
        self.downsample_ratio = downsample_ratio
        self.degradation_type = degradation_type
        
        print(f"[Ablation Baseline] Sa2VA SFT + Low-Res Augmentation Initialized:")
        print(f" - Augmentation Probability: {aug_prob}")
        print(f" - Degradation Type: {degradation_type}")
        print(f" - Downsample Ratio: {downsample_ratio}")

    def make_low_res(self, img):
        import torchvision.transforms.functional as TF
        is_batch = img.dim() == 4
        if not is_batch:
            img = img.unsqueeze(0)
            
        orig_dtype = img.dtype
        if img.dtype == torch.uint8:
            img = img.float().div(255.0)
            
        N, C, H, W = img.shape
        
        if self.degradation_type == 'pixelation':
            low_h = max(8, int(H * self.downsample_ratio))
            low_w = max(8, int(W * self.downsample_ratio))
            low = F.interpolate(img, size=(low_h, low_w), mode="bilinear", align_corners=False)
            restored = F.interpolate(low, size=(H, W), mode="nearest")
        elif self.degradation_type == 'blur':
            restored = TF.gaussian_blur(img, kernel_size=[15, 15], sigma=[5.0, 5.0])
        elif self.degradation_type == 'noise':
            noise_std = 0.3 
            noise = torch.randn_like(img) * noise_std
            restored = torch.clamp(img + noise, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown degradation_type: {self.degradation_type}")

        if orig_dtype == torch.uint8:
            restored = restored.mul(255.0).byte()
            
        if not is_batch:
            restored = restored.squeeze(0)
            
        return restored

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss' and self.training:
            g_pixel_values = data.get('g_pixel_values', None)
            gt_masks_list = data.get('masks', None)

            if g_pixel_values is not None and gt_masks_list is not None:
                new_g_pixels = []
                new_gt_masks = []
                
                for i in range(len(g_pixel_values)):
                    img = g_pixel_values[i]
                    mask = gt_masks_list[i]
                    
                    if torch.rand(1).item() < self.aug_prob:
                        img = self.make_low_res(img)
                        mask = torch.zeros_like(mask) 
                        
                    new_g_pixels.append(img)
                    new_gt_masks.append(mask)

                data['g_pixel_values'] = new_g_pixels
                data['masks'] = new_gt_masks

        return super().forward(data, data_samples, mode)
        
#######################################################################
#            Our Proposed Method: Aerial-R1 (H-GAWR)                  #
#######################################################################

class AerialR1Policy(Sa2VAModel):
    """
    [IEEE TPAMI 2026] Aerial-R1: Reinforcing Aerial Reasoning and Segmentation.
    """
    def __init__(self, 
                 group_size=8, 
                 consistency_beta=0.5, 
                 downsample_ratio=0.125,
                 hallucination_penalty=2.0, 
                 silence_reward=1.0,
                 use_low_res_branch=True,
                 degradation_type='pixelation',  
                 temperature=0.1,                
                 **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.consistency_beta = consistency_beta
        self.downsample_ratio = downsample_ratio
        self.use_low_res_branch = use_low_res_branch
        
        self.hallucination_penalty = hallucination_penalty
        self.silence_reward = silence_reward
        self.degradation_type = degradation_type
        self.temperature = temperature
        
        for module in self.text_hidden_fcs.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.05 
        
        print(f"[Aerial-R1] H-GRPO Policy Initialized:")
        print(f" - Group Size (G): {group_size}")
        print(f" - Consistency Beta: {consistency_beta}")
        print(f" - Downsample Ratio: {downsample_ratio}")
        print(f" - Hallucination Penalty: {hallucination_penalty}")
        print(f" - Silence Reward: {silence_reward}")
        print(f" - Degradation Type: {degradation_type}")
        print(f" - Temperature: {temperature}")

    def make_low_res(self, img):
        """
        Generates visually degraded views to simulate ambiguous aerial conditions.
        Supports 'pixelation', 'blur', and 'noise' for ablation studies.
        """
        is_batch = img.dim() == 4
        if not is_batch:
            img = img.unsqueeze(0)
            
        orig_dtype = img.dtype
        if img.dtype == torch.uint8:
            img = img.float().div(255.0)
            
        N, C, H, W = img.shape
        
        if self.degradation_type == 'pixelation':
            low_h = max(8, int(H * self.downsample_ratio))
            low_w = max(8, int(W * self.downsample_ratio))
            low = F.interpolate(img, size=(low_h, low_w), mode="bilinear", align_corners=False)
            restored = F.interpolate(low, size=(H, W), mode="nearest")

        elif self.degradation_type == 'blur':
            restored = TF.gaussian_blur(img, kernel_size=[15, 15], sigma=[5.0, 5.0])

        elif self.degradation_type == 'noise':
            noise_std = 0.3 
            noise = torch.randn_like(img) * noise_std
            restored = torch.clamp(img + noise, 0.0, 1.0)
            
        else:
            raise ValueError(f"Unknown degradation_type: {self.degradation_type}")

        if orig_dtype == torch.uint8:
            restored = restored.mul(255.0).byte()
            
        if not is_batch:
            restored = restored.squeeze(0)
            
        return restored

    def compute_grpo_reward(self, pred_masks, gt_masks):
        """"""
        device = pred_masks.device
        
        pred_probs = torch.sigmoid(pred_masks)
        pred_binary = (pred_probs > 0.5).float()
        gt_binary = (gt_masks > 0.5).float()

        is_low_res = gt_binary.sum(dim=(-1, -2)) < 32 

        if not getattr(self, 'use_low_res_branch', True):
            is_low_res = torch.zeros_like(is_low_res, dtype=torch.bool)

        # --- Refusal Branch (Low-Res) ---
        pred_pixel_sum = pred_binary.sum(dim=(-1, -2))
        is_successful_refusal = pred_pixel_sum < 50 
        
        rewards_refusal = torch.where(
            is_successful_refusal, 
            torch.tensor(self.silence_reward).to(device), 
            torch.tensor(-self.hallucination_penalty).to(device)
        )

        # --- Precision Branch (High-Res) ---
        intersection = (pred_binary * gt_binary).sum(dim=(-1, -2))
        union = (pred_binary + gt_binary).clamp(0, 1).sum(dim=(-1, -2))
        iou_scores = (intersection + 1e-6) / (union + 1e-6)

        B_real = pred_probs.shape[0] // self.group_size
        preds_view = pred_probs.view(B_real, self.group_size, pred_probs.shape[-2], pred_probs.shape[-1])
        
        group_mean = preds_view.mean(dim=1, keepdim=True)
        group_mean_binary = (group_mean > 0.5).float()
        group_mean_target = group_mean_binary.expand_as(preds_view).reshape(-1, pred_probs.shape[-2], pred_probs.shape[-1])
        
        inter_consist = (pred_binary * group_mean_target).sum(dim=(-1, -2))
        union_consist = (pred_binary + group_mean_target).clamp(0, 1).sum(dim=(-1, -2))
        consistency_scores = (inter_consist + 1e-6) / (union_consist + 1e-6)

        rewards_precision = iou_scores + self.consistency_beta * consistency_scores

        final_rewards = torch.where(is_low_res, rewards_refusal, rewards_precision)
        

        with torch.no_grad():
            has_high_res = (~is_low_res).any()
            has_low_res = is_low_res.any()
            
            detail_metrics = {
                'metric_iou': iou_scores[~is_low_res].mean() if has_high_res else torch.tensor(0.0, device=device),
                'metric_consistency': consistency_scores[~is_low_res].mean() if has_high_res else torch.tensor(0.0, device=device),
                'reward_precision_branch': rewards_precision[~is_low_res].mean() if has_high_res else torch.tensor(0.0, device=device),
                
                'reward_refusal_branch': rewards_refusal[is_low_res].mean() if has_low_res else torch.tensor(0.0, device=device),
                'refusal_success_rate': is_successful_refusal[is_low_res].float().mean() if has_low_res else torch.tensor(0.0, device=device),
            }

        return final_rewards, detail_metrics

    def forward(self, data, data_samples=None, mode='loss'):
        if mode != 'loss':
            return super().forward(data, data_samples, mode)

        # 1. Hybrid-View Group Construction
        half_group = self.group_size // 2 if getattr(self, 'use_low_res_branch', True) else self.group_size
        
        def expand_text(val):
            if isinstance(val, torch.Tensor): 
                return val.repeat_interleave(self.group_size, dim=0)
            elif isinstance(val, list):
                return [item for item in val for _ in range(self.group_size)]
            return val

        keys_to_expand = [
            'input_ids', 'attention_mask', 'labels', 'image_flags', 
            'pixel_values', 'position_ids', 'vp_overall_mask', 'prompt_masks'
        ]
        for k in keys_to_expand:
            if k in data and data[k] is not None:
                data[k] = expand_text(data[k])

        orig_g_pixels = data['g_pixel_values']
        new_g_pixels = []
        
        batch_len = len(orig_g_pixels)
        for i in range(batch_len):
            raw_img = orig_g_pixels[i]
            
            if getattr(self, 'use_low_res_branch', True):
                low_img = self.make_low_res(raw_img) 
            
            for _ in range(half_group):
                new_g_pixels.append(raw_img)
            for _ in range(self.group_size - half_group):
                new_g_pixels.append(low_img)
        
        if len(new_g_pixels) > 0 and isinstance(new_g_pixels[0], torch.Tensor):
            data['g_pixel_values'] = torch.stack(new_g_pixels)
        else:
            data['g_pixel_values'] = new_g_pixels

        gt_masks_list = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        
        if frames_per_batch:
            frames_per_batch = [f for f in frames_per_batch for _ in range(self.group_size)]
        
        new_gt_masks = []
        if gt_masks_list:
            for mask in gt_masks_list:
                if getattr(self, 'use_low_res_branch', True):
                    empty_mask = torch.zeros_like(mask)
                for _ in range(half_group):
                    new_gt_masks.append(mask)
                for _ in range(self.group_size - half_group):
                    new_gt_masks.append(empty_mask)
        gt_masks = new_gt_masks

        # 2. MLLM + SAM2 Inference
        output = self.mllm(data, data_samples, mode)

        input_ids = data['input_ids']
        seg_token_mask = input_ids == self.seg_token_idx
        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        
        if seg_token_mask.any():
            pred_embeddings = hidden_states[seg_token_mask]
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1)

        # ===========================================================
        # Continuous Temperature Sampling
        # ===========================================================
        if self.training and self.temperature > 0:
            noise = torch.randn_like(pred_embeddings)
            feature_norm = torch.norm(pred_embeddings, p=2, dim=-1, keepdim=True)
            pred_embeddings = pred_embeddings + noise * (feature_norm * self.temperature)

        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]
        
        pred_embeddings_list_video = self.generate_video_pred_embeddings(pred_embeddings_list, frames_per_batch)
        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(pred_embeddings_list_video, gt_masks_video)

        g_pixel_values = data['g_pixel_values']
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
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
        
        gt_masks_proc = [F.interpolate(gm.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gm in gt_masks_video]
        gt_masks_proc = torch.cat(gt_masks_proc, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        # 3. GRPO Update
        with torch.no_grad():
            raw_rewards, detail_metrics = self.compute_grpo_reward(pred_masks, gt_masks_proc)
            
            rewards_view = raw_rewards.view(-1, self.group_size)
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            std_rewards = rewards_view.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards_view - mean_rewards) / std_rewards
            advantages = advantages.flatten()

        loss_ce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_proc.float(), reduction='none').mean(dim=(-1, -2))
        pred_sigmoid = torch.sigmoid(pred_masks)
        inter = 2 * (pred_sigmoid * gt_masks_proc).sum(dim=(-1, -2))
        union = pred_sigmoid.sum(dim=(-1, -2)) + gt_masks_proc.sum(dim=(-1, -2))
        loss_dice = 1 - (inter + 1.0) / (union + 1.0)
        
        total_loss_element = 2.0 * loss_ce + 0.5 * loss_dice
        
        valid_mask = advantages > 0
        if valid_mask.sum() > 0:
            weighted_loss = (total_loss_element * valid_mask.float() * advantages.abs()).sum() / (valid_mask.sum() + 1e-6)
        else:
            weighted_loss = total_loss_element.mean()

        final_loss = output.loss + weighted_loss

        loss_dict = {
            'loss_mask': weighted_loss,
            'llm_loss': output.loss,
            'reward_mean': raw_rewards.mean(),
        }
        
        loss_dict.update(detail_metrics)
        
        return loss_dict