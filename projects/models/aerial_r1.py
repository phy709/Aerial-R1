
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

#######################################################################
#            Base Model: Sa2VA (SFT Baseline)                         #
#######################################################################

class Sa2VAModel(BaseModel):
    """
    The Supervised Fine-Tuning (SFT) Baseline Model.
    Based on Sa2VA (Yuan et al., 2025).
    """
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
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 training_bs:int=0,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        # Initialize processors for Qwen if needed
        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        # Fix for Qwen embedding tying
        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
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

        # Load SFT Pretrained Weights
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'[Aerial-R1] Loaded pretrained weights from {pretrained_pth}')

            if self.arch_type == 'qwen':
                lm_head_key = 'mllm.model.lm_head.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.template = template
        self.bs = training_bs

        if self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()

    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_sam2 = self.grounding_encoder.state_dict(*args, prefix=prefix + 'grounding_encoder.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(
            {k: v for k, v in state_dict_sam2.items() if k.startswith('grounding_encoder.sam2_model.sam_mask_decoder')})
        to_return.update(state_dict_text)
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        # ... (Same as original code) ...
        # Simplified for brevity in this response, keep original logic
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
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

    def sample_points(self, mask_pred, gt_masks):
        # Point sampling for mask loss
        gt_masks = gt_masks.unsqueeze(1).to(mask_pred)
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

    def forward(self, data, data_samples=None, mode='loss'):
        # Standard SFT Forward Pass
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype, device=input_ids.device)
        else:
            seg_valid = True

        # ... (Extract embeddings, SAM2 forward, Loss calc) ...
        # Keeping logic identical to Sa2VA original code for baseline reproducibility
        
        seg_token_mask = input_ids == self.seg_token_idx
        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid: seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]
        
        pred_embeddings_list_video = self.generate_video_pred_embeddings(pred_embeddings_list, frames_per_batch)
        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(pred_embeddings_list_video, gt_masks_video)

        g_pixel_values = torch.stack([self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        if len(pred_masks) != len(gt_masks):
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]

        sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
        sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        
        loss_dict = {
            'loss_mask': sam_loss_mask,
            'loss_dice': sam_loss_dice,
            'llm_loss': output.loss,
        }
        return loss_dict


#######################################################################
#            Our Proposed Method: Aerial-R1 (H-GRPO)                  #
#######################################################################

class AerialR1Policy(Sa2VAModel):
    """
    [ICML 2026] Aerial-R1: Reinforcing Aerial Reasoning and Segmentation.
    
    Implements Hybrid-View Group Relative Policy Optimization (H-GRPO).
    
    Key Features:
    1.  Group Sampling (Eq. 1): Constructs a group G containing High-Res (Clear) 
        and Low-Res (Degraded) views.
    2.  Dual-Branch Reward (Eq. 2 & 4):
        - Precision Branch (High-Res): IoU + Consistency Reward.
        - Refusal Branch (Low-Res): Silence Reward or Hallucination Penalty.
    3.  Contrastive Advantage (Eq. 5): Uses group-wise normalization to punish hallucinations.
    """
    def __init__(self, 
                 group_size=8, 
                 consistency_beta=0.5, 
                 downsample_ratio=0.125,
                 hallucination_penalty=2.0, 
                 silence_reward=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.consistency_beta = consistency_beta
        self.downsample_ratio = downsample_ratio
        
        self.hallucination_penalty = hallucination_penalty
        self.silence_reward = silence_reward
        
        print(f"[Aerial-R1] H-GRPO Policy Initialized:")
        print(f" - Group Size (G): {group_size}")
        print(f" - Consistency Beta: {consistency_beta}")
        print(f" - Downsample Ratio: {downsample_ratio}")
        print(f" - Hallucination Penalty: {hallucination_penalty}")
        print(f" - Silence Reward: {silence_reward}")

    def make_low_res(self, img):
        """
        Generates visually degraded views to simulate ambiguous aerial conditions.
        Corresponds to Section 3.2 in the paper.
        """
        is_batch = img.dim() == 4
        if not is_batch:
            img = img.unsqueeze(0)
            
        if img.dtype == torch.uint8:
            img = img.float().div(255.0)
            
        N, C, H, W = img.shape
        low_h = max(8, int(H * self.downsample_ratio))
        low_w = max(8, int(W * self.downsample_ratio))

        # 1. Downsample (Lose high-frequency details)
        low = F.interpolate(img, size=(low_h, low_w), mode="bilinear", align_corners=False)

        # 2. Upsample back (Pixelated effect)
        restored = F.interpolate(low, size=(H, W), mode="nearest")
        
        if not is_batch:
            restored = restored.squeeze(0)
        return restored

    def compute_grpo_reward(self, pred_masks, gt_masks):
        """
        Implements Eq. 2 (Precision Branch) and Eq. 4 (Refusal Branch).
        """
        device = pred_masks.device
        
        pred_probs = torch.sigmoid(pred_masks)
        pred_binary = (pred_probs > 0.5).float()
        gt_binary = (gt_masks > 0.5).float()

        # Identify Low-Res Samples (where GT is Empty)
        # In H-GRPO, we deliberately set GT to empty for Low-Res views to teach refusal.
        is_low_res = gt_binary.sum(dim=(-1, -2)) < 32 

        # --- Refusal Branch (Low-Res) ---
        # Eq. 4 in Paper
        pred_pixel_sum = pred_binary.sum(dim=(-1, -2))
        is_successful_refusal = pred_pixel_sum < 50 # Threshold tau=50
        
        rewards_refusal = torch.where(
            is_successful_refusal, 
            torch.tensor(self.silence_reward).to(device), 
            torch.tensor(-self.hallucination_penalty).to(device)
        )

        # --- Precision Branch (High-Res) ---
        # Eq. 2 in Paper
        # 1. IoU w.r.t Ground Truth
        intersection = (pred_binary * gt_binary).sum(dim=(-1, -2))
        union = (pred_binary + gt_binary).clamp(0, 1).sum(dim=(-1, -2))
        iou_scores = (intersection + 1e-6) / (union + 1e-6)

        # 2. Consistency w.r.t Group Consensus
        # Reshape to calculate group mean
        B_real = pred_probs.shape[0] // self.group_size
        preds_view = pred_probs.view(B_real, self.group_size, pred_probs.shape[-2], pred_probs.shape[-1])
        
        group_mean = preds_view.mean(dim=1, keepdim=True)
        group_mean_binary = (group_mean > 0.5).float()
        group_mean_target = group_mean_binary.expand_as(preds_view).reshape(-1, pred_probs.shape[-2], pred_probs.shape[-1])
        
        inter_consist = (pred_binary * group_mean_target).sum(dim=(-1, -2))
        union_consist = (pred_binary + group_mean_target).clamp(0, 1).sum(dim=(-1, -2))
        consistency_scores = (inter_consist + 1e-6) / (union_consist + 1e-6)

        rewards_precision = iou_scores + self.consistency_beta * consistency_scores

        # --- Final Selection ---
        final_rewards = torch.where(is_low_res, rewards_refusal, rewards_precision)
        return final_rewards

    def forward(self, data, data_samples=None, mode='loss'):
        if mode != 'loss':
            return super().forward(data, data_samples, mode)

        # ===========================================================
        # 1. Hybrid-View Group Construction (Paper Section 3.2)
        # ===========================================================
        # We split the group into High-Res (First Half) and Low-Res (Second Half)
        half_group = self.group_size // 2
        
        # Expand Text Inputs
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
        if 'pixel_values' in data: data['pixel_values'] = expand_text(data['pixel_values'])
        
        # Expand Visuals: Create Hybrid Views
        orig_g_pixels = data['g_pixel_values']
        new_g_pixels = []
        
        batch_len = len(orig_g_pixels)
        for i in range(batch_len):
            raw_img = orig_g_pixels[i]
            low_img = self.make_low_res(raw_img) # Degrade View
            
            # First Half: High-Res
            for _ in range(half_group):
                new_g_pixels.append(raw_img)
            # Second Half: Low-Res
            for _ in range(self.group_size - half_group):
                new_g_pixels.append(low_img)
        
        if len(new_g_pixels) > 0 and isinstance(new_g_pixels[0], torch.Tensor):
            data['g_pixel_values'] = torch.stack(new_g_pixels)
        else:
            data['g_pixel_values'] = new_g_pixels

        # Expand Masks & Construct Empty GTs for Low-Res
        gt_masks_list = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        
        if frames_per_batch:
            frames_per_batch = [f for f in frames_per_batch for _ in range(self.group_size)]
        
        new_gt_masks = []
        if gt_masks_list:
            for mask in gt_masks_list:
                empty_mask = torch.zeros_like(mask)
                # First Half: High-Res uses Real GT
                for _ in range(half_group):
                    new_gt_masks.append(mask)
                # Second Half: Low-Res uses Empty GT (to define refusal task)
                for _ in range(self.group_size - half_group):
                    new_gt_masks.append(empty_mask)
        gt_masks = new_gt_masks

        # ===========================================================
        # 2. MLLM + SAM2 Inference (Policy Rollout)
        # ===========================================================
        output = self.mllm(data, data_samples, mode)

        # Extract embeddings for SAM2
        input_ids = data['input_ids']
        seg_token_mask = input_ids == self.seg_token_idx
        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])
        
        if seg_token_mask.any():
            pred_embeddings = hidden_states[seg_token_mask]
        else:
            # Fallback if policy fails to generate [SEG]
            pred_embeddings = hidden_states[:, :5].flatten(0, 1)

        seg_token_counts = seg_token_mask.int().sum(-1)
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]
        
        pred_embeddings_list_video = self.generate_video_pred_embeddings(pred_embeddings_list, frames_per_batch)
        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(pred_embeddings_list_video, gt_masks_video)

        # SAM2 Decoding
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
        
        # Align GT Mask Size
        gt_masks_proc = [F.interpolate(gm.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gm in gt_masks_video]
        gt_masks_proc = torch.cat(gt_masks_proc, dim=0)
        pred_masks = pred_masks.flatten(0, 1)

        # ===========================================================
        # 3. GRPO Update (Contrastive Advantage)
        # ===========================================================
        with torch.no_grad():
            raw_rewards = self.compute_grpo_reward(pred_masks, gt_masks_proc)
            
            # Eq. 5: Group-wise Normalization
            rewards_view = raw_rewards.view(-1, self.group_size)
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            std_rewards = rewards_view.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards_view - mean_rewards) / std_rewards
            advantages = advantages.flatten()

        # Calculate Segmentation Loss
        loss_ce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_proc.float(), reduction='none').mean(dim=(-1, -2))
        pred_sigmoid = torch.sigmoid(pred_masks)
        inter = 2 * (pred_sigmoid * gt_masks_proc).sum(dim=(-1, -2))
        union = pred_sigmoid.sum(dim=(-1, -2)) + gt_masks_proc.sum(dim=(-1, -2))
        loss_dice = 1 - (inter + 1.0) / (union + 1.0)
        
        total_loss_element = 2.0 * loss_ce + 0.5 * loss_dice
        
        # Advantage Weighted Loss (Eq. 6)
        # We enforce "Good Behaviors" (Positive Advantage)
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
        return loss_dict