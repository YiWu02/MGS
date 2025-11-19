# 配置文件
import os
from utils import get_dataloaders_1k, get_dataloaders_3k, get_dataloaders_2023

# 基础配置
base_config = {
    'seed': 42,
    'batch_size': 12,
    'num_folds': 10,
    'weight_decay': 5e-4,
    'grad_clip': 1.0,
    'lr': 5e-6,
    'scheduler_eta_min': 5e-7,
    'num_workers': 16,
    'pin_memory': True
}

# 数据集配置
dataset_configs = {   
    'AGIQA-1K': {
        'columns': ['Image', 'Prompt', 'MOS'],
        'score_types': ['MOS'],  # 质量
        'epochs': 20,
        'image_dir': './data/AGIQA-1K/images',
        'dataloader_func_name': 'get_dataloaders_1k',
        'description': 'AGIQA-1K数据集：评估质量(MOS)维度'
    },
    'AGIQA-3K': {
        'columns': ['name', 'prompt', 'adj1', 'adj2', 'style', 'mos_quality', 'std_quality', 'mos_align', 'std_align'],
        'score_types': ['mos_quality', 'mos_align'],  # 质量和一致性
        'epochs': 15,
        'image_dir': './data/AGIQA-3K/images',
        'dataloader_func_name': 'get_dataloaders_3k',
        'description': 'AGIQA-3K数据集：同时评估质量(mos_quality)和一致性(mos_align)两个维度'
    }, 
    'AIGCIQA2023': {
        'columns': ['model', 'image_name', 'mosz1', 'mosz2', 'mosz3', 'prompt'],
        'score_types': ['mosz1', 'mosz2', 'mosz3'],  # 质量、真实性、一致性
        'epochs': 15,
        'image_dir': './data/AIGCIQA2023/images',
        'dataloader_func_name': 'get_dataloaders_2023',
        'description': 'AIGCIQA2023数据集：同时评估质量(mosz1)、真实性(mosz2)、一致性(mosz3)三个维度'
    } 
}

# 目录配置
dir_config = {
    'checkpoints_base': 'checkpoints',
    'runs_base': 'runs',
    'test_results_base': 'test_results',
    'database_base': './Database'
}

# 日志配置
log_config = {
    'train_log_suffix': 'train_results',
    'test_log_suffix': 'test_results',
    'detailed_results_suffix': 'detailed_test_results'
}

def get_dataloader_func(func_name):
    """根据函数名获取数据加载器函数"""  
    func_map = {
        'get_dataloaders_1k': get_dataloaders_1k,
        'get_dataloaders_3k': get_dataloaders_3k,
        'get_dataloaders_2023': get_dataloaders_2023
    }
    
    return func_map.get(func_name)

def create_directories():
    """创建必要的目录"""
    for dataset_name in dataset_configs.keys():
        # 创建checkpoints目录（每个数据集一个目录）
        checkpoint_dir = os.path.join(dir_config['checkpoints_base'], dataset_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
        # 创建runs目录（每个数据集一个目录，包含fold子目录）
        for fold in range(1, base_config['num_folds'] + 1):
            runs_dir = os.path.join(dir_config['runs_base'], dataset_name, str(fold))
            os.makedirs(runs_dir, exist_ok=True)
    
    # 创建结果目录
    os.makedirs(dir_config['test_results_base'], exist_ok=True) 