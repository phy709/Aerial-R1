from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer, Qwen2_5_VLProcessor, Qwen3VLProcessor

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.sa2va.models import Sa2VAModel, SAM2TrainRunner, DirectResize, InternVLMLLM
from projects.sa2va.datasets import (
    sa2va_collect_fn, Sa2VA01RefSeg, LLaVADataset, 
    Sa2VA03RefVOS, Sa2VA04VideoQA, Sa2VA05GCGDataset, Sa2VA06VPDataset,
    Sa2VAFinetuneDataset
)

from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA

# 删掉之前所有乱七八糟的补丁代码
from projects.sa2va.metrics import SimpleRefSegMetric, ForceFixValStepHook




#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path = "models/Sa2VA-InternVL3-2B"
pretrained_pth = "models/Sa2VA-InternVL3-2B.pth" # You need to change this path

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 64 # on 2 gpus
dataloader_num_workers = 16
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
lr = 1e-4
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 2000
save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)

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
    type=Sa2VAModel,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVLMLLM, # for QwenVL-based models, replace with Qwen2_5_VL or Qwen3VL
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

DATA_ROOT = '../data/'

# this is for datasets with masks
sa2va_default_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
)

# this is for datasets without masks
sa2va_qa_default_dataset_configs=dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    prompt_template=prompt_template,
    max_length=max_length,
)

######################### ImageRefSeg ##################################

RES_ROOT = DATA_ROOT + 'VSAI/train/'

sa2va_data_finetune_configs = [
    dict(
        type=Sa2VAFinetuneDataset,
        name='FinetuneDataset',
        data_root=RES_ROOT,
        data_prefix=dict(img_path='images/'),
        ann_file='mixed_train3.json',
        arch_type='intern_vl',
        serialize_data=False,
        repeats=50,
        **sa2va_default_dataset_configs,
    )
]

"""
For QwenVL-based models, use the following dataset configs instead:

sa2va_data_finetune_configs = [
    dict(
        type=Sa2VAFinetuneDataset,
        name='FinetuneDataset',
        data_root=RES_ROOT,
        data_prefix=dict(img_path='images/'),
        ann_file='annotations.json',
        serialize_data=False,
        repeats=100,
        arch_type='qwen',
        preprocessor=dict(
            type=Qwen2_5_VLProcessor.from_pretrained, # or Qwen3VLProcessor for Qwen3VL-based models
            pretrained_model_name_or_path=path,
            trust_remote_code=True,
        ),
        **sa2va_default_dataset_configs,
    )
]
"""

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

# --- 在 train_dataloader 定义之后添加 ---

# 假设你的验证集标注文件叫 mixed_val.json (请根据实际修改)
VAL_ANN_FILE = 'mixed_val.json' 
# 如果你想快速测试，可以用 train 的标注文件，并设置 indices 切片
# VAL_ANN_FILE = 'mixed_train_neg.json' 

sa2va_val_dataset = dict(
    type=Sa2VAFinetuneDataset,
    name='ValDataset',
    data_root=DATA_ROOT + 'VSAI/val/',
    data_prefix=dict(img_path='images/'),
    ann_file=VAL_ANN_FILE,
    arch_type='intern_vl',
    serialize_data=False,
    test_mode=True,  # 关键：开启测试模式，关闭随机增强
    # 如果验证集太大，可以只测前 200 个样本来监控指标
    indices=200,     
    **sa2va_default_dataset_configs,
)

val_dataloader = dict(
    batch_size=1, # 验证通常用 batch_size=1 避免 padding 带来的 mask 对齐问题
    num_workers=2,
    dataset=sa2va_val_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type=sa2va_collect_fn)
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type=SimpleRefSegMetric,  # 直接引用上面定义的类
)

test_evaluator = val_evaluator

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
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

# train, val, test setting
# --- 修改 train_cfg 部分 ---
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=500) # 每 500 iter 验证一次

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=ForceFixValStepHook),
]

# configure default hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=True,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
        # 新增：保存最佳模型配置
        save_best='mIoU',  # 对应 SimpleRefSegMetric 返回的 key
        rule='greater'
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
