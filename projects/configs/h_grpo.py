from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.models import (
    AerialR1Policy,
    SAM2TrainRunner,
    DirectResize,
    InternVLMLLM
)
from projects.datasets import (
    sa2va_collect_fn,
    Sa2VAFinetuneDataset,
    ConcatDatasetSa2VA
)

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = "/models/Sa2VA-InternVL3-2B"
pretrained_pth = "./work_dirs/sft/sft.pth"

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 1
accumulative_counts = 32 # 4 gpus
dataloader_num_workers = 16
max_epochs = 1
optim_type = AdamW
lr = 1e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.05

# Save
save_steps = 100
save_total_limit = 5

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=AerialR1Policy,
    group_size=8,
    consistency_beta=2.0,
    downsample_ratio=0.125,
    hallucination_penalty=2.0,
    silence_reward=1.0,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVLMLLM,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=["embed_tokens", "lm_head"]
        ),
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5)
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = '/data/'

sa2va_default_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
)

sa2va_data_finetune_configs = [
    dict(
        type=Sa2VAFinetuneDataset,
        name='VSAI-Ref-Train',
        data_root=DATA_ROOT + 'VSAI_ref/train/',
        data_prefix=dict(img_path='images/'),
        ann_file='mixed_train.json',
        arch_type='intern_vl',
        serialize_data=False,
        repeats=10,
        **sa2va_default_dataset_configs,
    )
]

train_dataset = dict(
    type=ConcatDatasetSa2VA, datasets=[
        *sa2va_data_finetune_configs,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=sa2va_collect_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=5000)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)