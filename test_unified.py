import os
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from model import AGIQAModel
from utils import set_seed, preprocess2, log_and_print, get_logger
from config import base_config, dataset_configs, get_dataloader_func, dir_config, log_config, create_directories
import pandas as pd
import json
from datetime import datetime
import numpy as np

# 设置随机种子
set_seed(base_config['seed'])

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single_fold(dataset_name, fold):
    """测试单个fold"""
    print(f"\n开始测试 {dataset_name} - Fold {fold}")
    
    # 目录
    checkpoint_dir = os.path.join(dir_config['checkpoints_base'], dataset_name)
    log_dir = os.path.join(dir_config['checkpoints_base'], dataset_name)
    results_dir = os.path.join(dir_config['test_results_base'], dataset_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置日志
    logger = get_logger(os.path.join(log_dir, f"{log_config['test_log_suffix']}.log"), 'log')
    
    # 加载模型
    model = AGIQAModel().to(device)
    model_path = os.path.join(checkpoint_dir, f'best_model_fold{fold}.pt')
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    test_csv = os.path.join(dir_config['database_base'], dataset_name, str(fold), 'test.csv')
    
    # 根据数据集选择数据加载器
    dataloader_func = get_dataloader_func(dataset_configs[dataset_name]['dataloader_func_name'])
    image_dir = dataset_configs[dataset_name]['image_dir']
    
    test_loader = dataloader_func(
        test_csv,
        base_config['batch_size'],
        image_dir,
        preprocess2(),
        True,
        num_workers=base_config['num_workers'],
        pin_memory=base_config['pin_memory']
    )
    
    # 测试
    all_gt_scores = {}
    all_pred_scores = {}
    
    with torch.no_grad():
        for sample_batched in tqdm(test_loader, desc='Testing'):
            x = sample_batched['I'].to(device).float()
            patches = sample_batched['patches'].to(device).float()
            prompt = sample_batched['prompt']

            with torch.amp.autocast('cuda', enabled=True):
                # 获取所有预测分数
                predictions = model(x, patches, prompt, dataset_name)
                
                # 收集预测和真实标签
                if dataset_name == 'AIGCIQA2023':
                    for score_type in ['mosz1', 'mosz2', 'mosz3']:
                        if score_type not in all_gt_scores:
                            all_gt_scores[score_type] = []
                            all_pred_scores[score_type] = []
                        
                        pred = predictions[score_type].cpu().numpy()
                        gt = sample_batched[score_type].cpu().numpy()
                        
                        all_gt_scores[score_type].extend(gt.tolist())
                        all_pred_scores[score_type].extend(pred.tolist())
                        
                elif dataset_name == 'AGIQA-1K':
                    all_gt_scores['MOS'] = all_gt_scores.get('MOS', [])
                    all_pred_scores['MOS'] = all_pred_scores.get('MOS', [])
                    
                    pred = predictions['MOS'].cpu().numpy()
                    gt = sample_batched['MOS'].cpu().numpy()
                    
                    all_gt_scores['MOS'].extend(gt.tolist())
                    all_pred_scores['MOS'].extend(pred.tolist())
                    
                elif dataset_name == 'AGIQA-3K':
                    for score_type in ['mos_quality', 'mos_align']:
                        if score_type not in all_gt_scores:
                            all_gt_scores[score_type] = []
                            all_pred_scores[score_type] = []
                        
                        pred = predictions[score_type].cpu().numpy()
                        gt = sample_batched[score_type].cpu().numpy()
                        
                        all_gt_scores[score_type].extend(gt.tolist())
                        all_pred_scores[score_type].extend(pred.tolist())

    # 计算指标
    results = {}
    for score_type in all_gt_scores.keys():
        srcc = spearmanr(all_gt_scores[score_type], all_pred_scores[score_type])[0]
        plcc = pearsonr(all_gt_scores[score_type], all_pred_scores[score_type])[0]
        avg_score = (srcc + plcc) / 2
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((np.array(all_gt_scores[score_type]) - np.array(all_pred_scores[score_type])) ** 2))
        
        results[score_type] = {
            'avg': avg_score,
            'srcc': srcc,
            'plcc': plcc,
            'rmse': rmse,
            'predictions': all_pred_scores[score_type],
            'ground_truth': all_gt_scores[score_type]
        }
    
    
    # 日志记录
    log_msg = f'Fold {fold} - {dataset_name}:'
    for score_type, metrics in results.items():
        log_msg += f' | {score_type}: Avg={metrics["avg"]:.4f}, SRCC={metrics["srcc"]:.4f}, PLCC={metrics["plcc"]:.4f}, RMSE={metrics["rmse"]:.4f}'
    log_and_print(logger, log_msg)
    
    # 保存详细结果
    for score_type, metrics in results.items():
        detailed_results = pd.DataFrame({
            'ground_truth': metrics['ground_truth'],
            'prediction': metrics['predictions']
        })
        detailed_file = os.path.join(results_dir, f'detailed_test_results_{score_type}_fold_{fold}.csv')
        detailed_results.to_csv(detailed_file, index=False)
    
    return results


def main():
    """主测试函数"""
    print("开始统一测试流程...")

    # 创建必要的目录（基于配置）
    create_directories()
    
    # 存储所有结果
    all_results = {}
    
    # 遍历所有数据集
    for dataset_name, cfg in dataset_configs.items():
        print(f"\n{'='*50}")
        print(f"开始测试数据集: {dataset_name}")
        print(f"{'='*50}")
        
        all_results[dataset_name] = {}
        
        # 遍历所有fold
        for fold in range(1, base_config['num_folds'] + 1):
            try:
                result = test_single_fold(dataset_name, fold)
                all_results[dataset_name][fold] = result
                if result:
                    first_score_type = list(result.keys())[0] if result else None
                    best_score = result[first_score_type]['avg'] if first_score_type else 0.0
                    print(f"Fold {fold} 完成: {first_score_type}={best_score:.4f}")
            except Exception as e:
                print(f"Fold {fold} 测试失败: {str(e)}")
                all_results[dataset_name][fold] = None
    
    # 保存总体结果
    save_test_results(all_results)
    print("\n所有测试完成！")


def save_test_results(all_results):
    """保存测试结果"""
   
    # 创建结果目录
    results_dir = dir_config['test_results_base']
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'test_results_{timestamp}.json')
    
    # 转换numpy类型为Python原生类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    # 移除预测结果以减小文件大小
    def clean_results(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if k not in ['predictions', 'ground_truth']:
                    cleaned[k] = clean_results(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_results(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(clean_results(all_results))
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 生成汇总报告
    generate_test_summary_report(all_results, results_dir, timestamp)
    
    print(f"测试结果已保存到: {results_file}")


def generate_test_summary_report(all_results, results_dir, timestamp):
    """生成测试汇总报告"""
    report_file = os.path.join(results_dir, f'test_summary_report_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("测试结果汇总报告\n")
        f.write("="*50 + "\n\n")
        
        for dataset_name, dataset_results in all_results.items():
            f.write(f"数据集: {dataset_name}\n")
            f.write("-"*30 + "\n")
            
            # 计算平均值
            valid_results = [r for r in dataset_results.values() if r is not None]
            if valid_results:
                f.write(f"    成功fold数: {len(valid_results)}/{len(dataset_results)}\n")
                
                # 计算每个分数类型的平均结果
                score_types = set()
                for result in valid_results:
                    for key in result.keys():
                        score_types.add(key)
                
                for score_type in score_types:
                    avgs = [r[score_type]['avg'] for r in valid_results if score_type in r]
                    srccs = [r[score_type]['srcc'] for r in valid_results if score_type in r]
                    plccs = [r[score_type]['plcc'] for r in valid_results if score_type in r]
                    rmses = [r[score_type]['rmse'] for r in valid_results if score_type in r]
                    
                    if avgs:
                        avg_score = sum(avgs) / len(avgs)
                        avg_srcc = sum(srccs) / len(srccs)
                        avg_plcc = sum(plccs) / len(plccs)
                        avg_rmse = sum(rmses) / len(rmses)
                        
                        std_score = np.std(avgs) if len(avgs) > 1 else 0.0
                        std_srcc = np.std(srccs) if len(srccs) > 1 else 0.0
                        std_plcc = np.std(plccs) if len(plccs) > 1 else 0.0
                        std_rmse = np.std(rmses) if len(rmses) > 1 else 0.0
                        
                        f.write(f"    {score_type}: Avg={avg_score:.4f}±{std_score:.4f}, SRCC={avg_srcc:.4f}±{std_srcc:.4f}, PLCC={avg_plcc:.4f}±{std_plcc:.4f}, RMSE={avg_rmse:.4f}±{std_rmse:.4f}\n")
            else:
                f.write(f"    无有效结果\n")
            
            f.write("\n")
    
    print(f"测试汇总报告已保存到: {report_file}")


if __name__ == "__main__":
    main()