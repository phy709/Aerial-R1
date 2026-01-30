import random
from typing import Literal, Optional, List
import torch

import numpy as np
from pycocotools import mask as mask_utils

from .common import SEG_QUESTIONS, ANSWER_LIST
from .base import Sa2VABaseDataset

from third_parts.mmdet.datasets.refcoco import RefCocoDataset

import mmengine

class Sa2VAFinetuneDataset(RefCocoDataset, Sa2VABaseDataset):

    def __init__(self,
                 data_root,
                 ann_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='images/'),
                 tokenizer=None,
                 max_length=2048,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 repeats:int = 1,
                 name: str = 'FinetuneDataset',
                 **kwargs):
        
        # Initialize RefCocoDataset
        RefCocoDataset.__init__(self,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file='',
            **kwargs,
        )

        # Initialize Sa2VABaseDataset with common functionality
        Sa2VABaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name
        )
        
        # Dataset-specific configurations
        self.begin_str = f'<image>\n'
        self.image_folder = data_root
        self.single_image_mode = single_image_mode

    def load_data_list(self) -> List[dict]:
        self.annotations = mmengine.load(self.ann_file, file_format='json')
        data_list = []
        img_prefix = self.data_prefix.get('img_path', '')
        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path

        for item in self.annotations:
            # 兼容 image 和 img_path 字段
            path_key = 'img_path' if 'img_path' in item else 'image'
            image_path = item[path_key]
            
            if not image_path.startswith('/') and img_prefix:
                full_path = join_path(img_prefix, image_path)
            else:
                full_path = image_path

            data_info = {
                'img_path': full_path,
                'mask': item['mask'],
                'text': item['text'],
                # 读取关键的新字段
                'task_type': item.get('task_type', 'refseg'),
                'mask_type': item.get('mask_type', 'poly')
            }
            data_list.append(data_info)

        return data_list

    @property
    def modality_length(self):
        return [self._get_modality_length_default() for _ in range(len(self))]

    def _parse_annotations(self, ann_info):
        """
        适配混合数据集 (RefSeg + Global Seg) 的新解析函数
        已添加：空掩码与异常数值的兜底检查
        """
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        # === 1. 读取新字段 ===
        mask_data = ann_info['mask']
        text_prompt = ann_info['text']
        
        task_type = ann_info.get('task_type', 'refseg')
        mask_type = ann_info.get('mask_type', 'poly')

        binary_mask = np.zeros((height, width), dtype=np.uint8)

        # === 2. 解析 Mask (根据 mask_type) ===
        if len(mask_data) > 0:  
            try:
                if mask_type == 'rle':
                    # Global Seg: RLE 格式
                    if isinstance(mask_data.get('counts'), str):
                        mask_data['counts'] = mask_data['counts'].encode('utf-8')
                    m = mask_utils.decode(mask_data)
                    if len(m.shape) == 3:
                        m = m.max(axis=2)
                    binary_mask = m.astype(np.uint8)
                else:
                    # RefSeg: Polygon 格式
                    poly_list = mask_data
                    # 处理嵌套层级
                    if isinstance(poly_list[0], list) and isinstance(poly_list[0][0], list):
                        rles = mask_utils.frPyObjects(poly_list[0], height, width)
                    else:
                        rles = mask_utils.frPyObjects(poly_list, height, width)
                    
                    m = mask_utils.decode(rles)
                    if len(m.shape) == 3:
                        m = m.max(axis=2)
                    binary_mask = m.astype(np.uint8)
            except Exception as e:
                print(f"Error decoding mask for {image_path}: {e}")
                return None

        # ============================================================
        # 🔴 新增：运行时兜底检查 (Sentry Checks)
        # ============================================================
        
        # 检查 1: 防止由 NaN 或无穷大导致的崩溃
        if not np.isfinite(binary_mask).all():
            print(f"[Error] NaN/Inf detected in mask: {image_path}")
            return None


        # === 3. 构造对话 ===
        conversation = []
        
        if task_type == 'global':
            question = self.begin_str + "Please segment all vehicles in this image."
        else:
            # RefSeg 固定格式
            question = self.begin_str + f"Please segment the specific vehicle: {text_prompt}."

        conversation.append({'from': 'human', 'value': question})
        conversation.append({'from': 'gpt', 'value': "Sure, [SEG]."})

        # === 4. 封装返回 ===
        masks = torch.from_numpy(binary_mask).unsqueeze(0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path,
            'task_type': task_type 
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self._read_image(image_file)
            if image is None:
                return None
            
            # Process image using base class method
            image_data = self._process_single_image(image, self.single_image_mode)
            out_data_dict.update(image_data)
            
            # Create image token string and get input/labels
            image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
        else:
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], None)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
        return out_data_dict

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        """Get total length considering repeats."""
        return int(self.real_len() * self.repeats)

    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
        # Handle repeats using index mapping for equal distribution
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]
        
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            # Broken images may cause the returned data to be None
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data
        
        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get valid data after {self._max_refetch + 1} attempts")
