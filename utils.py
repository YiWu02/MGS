import re
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import logging
from dataloader import AGIQADataset_1k, AGIQADataset_3k, AGIQADataset_2023
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, Resize
import re

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def split_1k(prompt):
    """
    关键词堆砌式prompt: the main objects, the second objects, places, styles
    直接按逗号分割，保留短语
    """
    # 使用正则表达式按逗号分割、去除空格
    chunks = re.split(r',\s*', prompt)
    return [c.strip() for c in chunks if c and c.strip()]


def split_3k(prompt):
    """
    混合式prompt: subject, 0-2 details, 0-1 style
    使用介词和标点符号分割prompt
    """
    split_pattern = (
        r'(?=(?:\b(?:in|of|on|at|from|by|with|for|across|inside|near|between|as|to)\b|,|:))'
        # r'(?=(?:\b(?:in|of|on|at|from|by|with|for|across|inside|near|between|as|to)\b))'
        # r'(?=(?:,|:))'
    )
    # 分割文本
    tokens = re.split(split_pattern, prompt, flags=re.IGNORECASE)
    # 清理每个token，去除前后空格和标点符号
    cleaned_tokens = []
    for token in tokens:
        if token and token.strip():
            # 去除前后标点符号和空格
            cleaned = re.sub(r'^[\s,:]+|[\s,:]+$', '', token.strip())
            cleaned = token.strip()
            if cleaned:
                cleaned_tokens.append(cleaned)
    return cleaned_tokens
    

def split_2023(prompt):
    """
    完整句子式prompt
    使用介词和标点符号分割prompt
    """
    split_pattern = (
        r'(?=(?:\b(?:in|of|on|at|from|with|for|across|through|against|over|above|behind|next to|under|into|onto|without)\b|,|\.))'
        # r'(?=(?:\b(?:in|of|on|at|from|with|for|across|through|against|over|above|behind|next to|under|into|onto|without)\b))'
        # r'(?=(?:,|\.))'
    )
    # 分割文本
    tokens = re.split(split_pattern, prompt, flags=re.IGNORECASE)
    # 清理每个token，去除前后空格和标点符号
    cleaned_tokens = []
    for token in tokens:
        if token and token.strip():
            # 去除前后标点符号和空格
            cleaned = re.sub(r'^[\s,.]+|[\s,.]+$', '', token.strip())
            cleaned = token.strip()
            if cleaned:
                cleaned_tokens.append(cleaned)
    return cleaned_tokens


def get_prompt_chunks(dataset_name, prompt):
    """按数据集名称路由到对应的提示分割函数"""
    if dataset_name == 'AGIQA-1K':
        return split_1k(prompt)
    if dataset_name == 'AGIQA-3K':
        return split_3k(prompt)
    if dataset_name == 'AIGCIQA2023':
        return split_2023(prompt)


class AdaptiveResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return Resize(self.size, self.interpolation)(img)
        else:
            return img


# RGB转换函数
def _convert_image_to_rgb(image):
    return image.convert("RGB")


# 数据增强（用于验证和测试）
def preprocess2():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


# 数据增强（用于训练）
def preprocess3():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(512),
        #RandomHorizontalFlip(),  
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_dataloaders_1k(csv_file, bs, data_set, preprocess, test, num_workers=16, pin_memory=True):
    # 创建数据集
    dataset = AGIQADataset_1k(csv_file, data_set, preprocess, test)

    if test:
        shuffle = False
    else:
        shuffle = True

    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def get_dataloaders_3k(csv_file, bs, data_set, preprocess, test, num_workers=16, pin_memory=True):
    # 创建数据集
    dataset = AGIQADataset_3k(csv_file, data_set, preprocess, test)

    if test:
        shuffle = False
    else:
        shuffle = True

    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def get_dataloaders_2023(csv_file, bs, data_set, preprocess, test, num_workers=16, pin_memory=True):
    # 创建数据集
    dataset = AGIQADataset_2023(csv_file, data_set, preprocess, test)

    if test:
        shuffle = False
    else:
        shuffle = True

    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    
    # 清除现有的handlers以避免重复
    if logger.handlers:
        logger.handlers.clear()
    
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
