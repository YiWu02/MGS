import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter
from model import AGIQAModel
from utils import set_seed, preprocess2, preprocess3, log_and_print, get_logger
from config import base_config, dataset_configs, get_dataloader_func, dir_config, log_config, create_directories
import pandas as pd
import json
from datetime import datetime

# 使用全局随机种子
set_seed(base_config['seed'])

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()  # MAE

def train_epoch(model, optimizer, epoch, train_loader, dataset_name):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    loop = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    for step, sample_batched in enumerate(loop):
        x = sample_batched['I'].to(device).float()
        patches = sample_batched['patches'].to(device).float()
        prompt = sample_batched['prompt']

        # 前向传播
        with torch.amp.autocast('cuda', enabled=True):
            # 获取所有预测分数
            predictions = model(x, patches, prompt, dataset_name)
            
            # 计算总损失
            total_batch_loss = 0.0
            
            if dataset_name == 'AIGCIQA2023':
                # AIGCIQA2023: 三个独立分数
                mosz1_gt = sample_batched['mosz1'].to(device).float()
                mosz2_gt = sample_batched['mosz2'].to(device).float()
                mosz3_gt = sample_batched['mosz3'].to(device).float()
                
                loss1 = criterion_mse(predictions['mosz1'], mosz1_gt)
                loss2 = criterion_mse(predictions['mosz2'], mosz2_gt)
                loss3 = criterion_mse(predictions['mosz3'], mosz3_gt)
                
                total_batch_loss = loss1 + loss2 + loss3
                
            elif dataset_name == 'AGIQA-1K':
                mos_gt = sample_batched['MOS'].to(device).float()
                total_batch_loss = criterion_mse(predictions['MOS'], mos_gt)
                
            elif dataset_name == 'AGIQA-3K':
                mos_quality_gt = sample_batched['mos_quality'].to(device).float()
                mos_align_gt = sample_batched['mos_align'].to(device).float()
                
                loss1 = criterion_mse(predictions['mos_quality'], mos_quality_gt)
                loss2 = criterion_mse(predictions['mos_align'], mos_align_gt)
                
                total_batch_loss = loss1 + loss2

        # 反向传播和优化
        optimizer.zero_grad()
        total_batch_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=base_config['grad_clip'])
        optimizer.step()

        # 更新运行损失
        total_loss += total_batch_loss.item()
        loop.set_postfix(loss=total_batch_loss.item())

    return total_loss / len(train_loader)   # 返回epoch平均损失


def evaluate_epoch(model, val_loader, dataset_name):
    """验证一个epoch"""
    model.eval()
    all_gt_scores = {}
    all_pred_scores = {}
    total_loss = 0.0

    with torch.no_grad():
        for sample_batched in tqdm(val_loader, desc='Validation'):
            x = sample_batched['I'].to(device).float()
            patches = sample_batched['patches'].to(device).float()
            prompt = sample_batched['prompt']

            with torch.amp.autocast('cuda', enabled=True):
                # 获取所有预测分数
                predictions = model(x, patches, prompt, dataset_name)
                
                # 计算总损失
                batch_loss = 0.0
                
                if dataset_name == 'AIGCIQA2023':
                    # AIGCIQA2023: 三个独立分数
                    mosz1_gt = sample_batched['mosz1'].to(device).float()
                    mosz2_gt = sample_batched['mosz2'].to(device).float()
                    mosz3_gt = sample_batched['mosz3'].to(device).float()
                    
                    loss1 = criterion_mse(predictions['mosz1'], mosz1_gt)
                    loss2 = criterion_mse(predictions['mosz2'], mosz2_gt)
                    loss3 = criterion_mse(predictions['mosz3'], mosz3_gt)
                    
                    batch_loss = loss1 + loss2 + loss3
                    
                    # 收集预测和真实标签
                    for score_type in ['mosz1', 'mosz2', 'mosz3']:
                        if score_type not in all_gt_scores:
                            all_gt_scores[score_type] = []
                            all_pred_scores[score_type] = []
                        
                        gt_key = score_type
                        pred = predictions[score_type].cpu().numpy()
                        gt = sample_batched[gt_key].cpu().numpy()
                        
                        all_gt_scores[score_type].extend(gt.tolist())
                        all_pred_scores[score_type].extend(pred.tolist())
                        
                elif dataset_name == 'AGIQA-1K':
                    mos_gt = sample_batched['MOS'].to(device).float()
                    batch_loss = criterion_mse(predictions['MOS'], mos_gt)
                    
                    # 收集预测和真实标签
                    all_gt_scores['MOS'] = all_gt_scores.get('MOS', [])
                    all_pred_scores['MOS'] = all_pred_scores.get('MOS', [])
                    
                    pred = predictions['MOS'].cpu().numpy()
                    gt = sample_batched['MOS'].cpu().numpy()
                    
                    all_gt_scores['MOS'].extend(gt.tolist())
                    all_pred_scores['MOS'].extend(pred.tolist())
                    
                elif dataset_name == 'AGIQA-3K':
                    mos_quality_gt = sample_batched['mos_quality'].to(device).float()
                    mos_align_gt = sample_batched['mos_align'].to(device).float()
                    
                    loss1 = criterion_mse(predictions['mos_quality'], mos_quality_gt)
                    loss2 = criterion_mse(predictions['mos_align'], mos_align_gt)
                    
                    batch_loss = loss1 + loss2
                    
                    # 收集预测和真实标签
                    for score_type in ['mos_quality', 'mos_align']:
                        if score_type not in all_gt_scores:
                            all_gt_scores[score_type] = []
                            all_pred_scores[score_type] = []
                        
                        pred = predictions[score_type].cpu().numpy()
                        gt = sample_batched[score_type].cpu().numpy()
                        
                        all_gt_scores[score_type].extend(gt.tolist())
                        all_pred_scores[score_type].extend(pred.tolist())

            total_loss += batch_loss.item()

    # 计算最终指标
    results = {}
    for score_type in all_gt_scores.keys():
        srcc = spearmanr(all_gt_scores[score_type], all_pred_scores[score_type])[0]
        plcc = pearsonr(all_gt_scores[score_type], all_pred_scores[score_type])[0]
        avg_score = (srcc + plcc) / 2
        results[score_type] = {'srcc': srcc, 'plcc': plcc, 'avg': avg_score}
    
    val_loss = total_loss / len(val_loader)
    return val_loss, results


def train_single_fold(dataset_name, fold):
    """训练单个fold"""
    print(f"\n开始训练 {dataset_name} - Fold {fold}")
    
    # 目录
    checkpoint_dir = os.path.join(dir_config['checkpoints_base'], dataset_name)
    runs_dir = os.path.join(dir_config['runs_base'], dataset_name, str(fold))
    log_dir = os.path.join(dir_config['checkpoints_base'], dataset_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志
    logger = get_logger(os.path.join(log_dir, f"{log_config['train_log_suffix']}.log"), 'log')
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=runs_dir)
    
    # 加载模型
    model = AGIQAModel().to(device)
    
    # 设置优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_config['lr'], weight_decay=base_config['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=base_config['scheduler_eta_min'])
    
    # 初始化最佳结果
    best_result = {}
    best_metric_avg = 0.0  # 使用所有PLCC和SRCC的平均值
    best_model_path = os.path.join(checkpoint_dir, f'best_model_fold{fold}.pt')
    
    # 加载数据
    train_csv = os.path.join(dir_config['database_base'], dataset_name, str(fold), 'train.csv')
    val_csv = os.path.join(dir_config['database_base'], dataset_name, str(fold), 'val.csv')
    
    # 根据数据集选择数据加载器
    dataloader_func = get_dataloader_func(dataset_configs[dataset_name]['dataloader_func_name'])
    image_dir = dataset_configs[dataset_name]['image_dir']
    
    train_loader = dataloader_func(
        train_csv,
        base_config['batch_size'],
        image_dir,
        preprocess3(),
        False,
        num_workers=base_config['num_workers'],
        pin_memory=base_config['pin_memory']
    )
    val_loader = dataloader_func(
        val_csv,
        base_config['batch_size'],
        image_dir,
        preprocess2(),
        True,
        num_workers=base_config['num_workers'],
        pin_memory=base_config['pin_memory']
    )
    
    epochs = dataset_configs[dataset_name]['epochs']
    
    # 训练循环
    for epoch in range(1, epochs+1):
        # 训练
        train_loss = train_epoch(model, optimizer, epoch, train_loader, dataset_name)
        
        # 验证
        val_loss, val_results = evaluate_epoch(model, val_loader, dataset_name)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录每个分数类型的指标
        for score_type, metrics in val_results.items():
            writer.add_scalar(f'Metrics/{score_type}_SRCC', metrics['srcc'], epoch)
            writer.add_scalar(f'Metrics/{score_type}_PLCC', metrics['plcc'], epoch)
        
        # 计算所有PLCC和SRCC的平均值
        all_srcc = [metrics['srcc'] for metrics in val_results.values()]
        all_plcc = [metrics['plcc'] for metrics in val_results.values()]
        current_metric_avg = (sum(all_srcc) + sum(all_plcc)) / (len(all_srcc) + len(all_plcc))

        # 记录PLCC和SRCC的平均值到TensorBoard
        writer.add_scalar('Metrics/Overall_Avg', current_metric_avg, epoch)
        
        # 更新最佳模型 - 使用所有PLCC和SRCC的平均值作为判断标准
        if current_metric_avg > best_metric_avg:
            best_metric_avg = current_metric_avg
            best_result = val_results.copy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold': fold,
                'dataset': dataset_name
            }, best_model_path)
        
        scheduler.step()
        
        # 日志记录
        log_msg = f"Fold {fold} Epoch {epoch}: Train Loss={train_loss:.4f}, Metric Avg={current_metric_avg:.4f}"
        for score_type, metrics in val_results.items():
            log_msg += f" | {score_type}: SRCC={metrics['srcc']:.4f}, PLCC={metrics['plcc']:.4f}"
        log_and_print(logger, log_msg)
    
    writer.close()
    
    # 计算最终最佳结果的平均值
    best_all_srcc = [metrics['srcc'] for metrics in best_result.values()]
    best_all_plcc = [metrics['plcc'] for metrics in best_result.values()]
    final_metric_avg = (sum(best_all_srcc) + sum(best_all_plcc)) / (len(best_all_srcc) + len(best_all_plcc))
    
    # 最终日志
    log_msg = f"Best {dataset_name} result (Metric Avg={final_metric_avg:.4f}):"
    for score_type, metrics in best_result.items():
        log_msg += f" | {score_type}: SRCC={metrics['srcc']:.4f}, PLCC={metrics['plcc']:.4f}"
    log_and_print(logger, log_msg)
    
    return best_result   


def save_single_dataset_results(dataset_name, results):
    """保存单个数据集的结果到对应数据集目录"""
    # 使用对应数据集的checkpoints目录
    results_dir = os.path.join(dir_config['checkpoints_base'], dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results_file = os.path.join(results_dir, f'{dataset_name}_results_{timestamp}.json')
    
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
    
    serializable_results = convert_numpy(results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 生成汇总报告
    report_file = os.path.join(results_dir, f'{dataset_name}_summary_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"{dataset_name} 训练结果汇总报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"数据集: {dataset_name}\n")
        f.write("-" * 30 + "\n\n")
        
        # 计算平均值
        valid_results = [r for r in results.values() if r is not None]
        if valid_results:
            # 计算每个分数类型的平均结果
            score_types = set()
            for result in valid_results:
                for key in result.keys():
                    score_types.add(key)
            
            # 收集所有SRCC和PLCC用于计算总平均值
            all_srccs = []
            all_plccs = []
            
            for score_type in sorted(score_types):
                srccs = [r[score_type]['srcc'] for r in valid_results if score_type in r]
                plccs = [r[score_type]['plcc'] for r in valid_results if score_type in r]
                avgs = [r[score_type]['avg'] for r in valid_results if score_type in r]
                
                if srccs and plccs:
                    # 计算当前分数类型的平均值
                    avg_score = sum(avgs) / len(avgs)
                    avg_srcc = sum(srccs) / len(srccs)
                    avg_plcc = sum(plccs) / len(plccs)
                    
                    # 添加到总集合中
                    all_srccs.extend(srccs)
                    all_plccs.extend(plccs)
                    
                    # 计算成功fold数
                    successful_folds = len([r for r in valid_results if score_type in r])
                    total_folds = len(results)
                    
                    f.write(f"分数类型: {score_type}\n")
                    f.write(f"  平均结果: Avg={avg_score:.4f}, SRCC={avg_srcc:.4f}, PLCC={avg_plcc:.4f}\n")
                    f.write(f"  成功fold数: {successful_folds}/{total_folds}\n\n")
            
            # 计算所有SRCC和PLCC的平均值
            if all_srccs and all_plccs:
                overall_srcc_avg = sum(all_srccs) / len(all_srccs)
                overall_plcc_avg = sum(all_plccs) / len(all_plccs)
                overall_metric_avg = (overall_srcc_avg + overall_plcc_avg) / 2
                
                f.write(f"所有SRCC和PLCC的平均值: {overall_metric_avg:.4f}\n")
        else:
            f.write("无有效结果\n")
    
    print(f"{dataset_name} 结果已保存到: {results_file}")
    print(f"{dataset_name} 汇总报告已保存到: {report_file}")
    
    return results_file, report_file


def main():
    """主训练函数"""
    print("开始统一训练流程...")

    # 创建必要的目录（基于配置）
    create_directories()
    
    # 存储所有结果
    all_results = {}
    
    # 遍历所有数据集
    for dataset_name, cfg in dataset_configs.items():
        print(f"\n{'='*50}")
        print(f"开始训练数据集: {dataset_name}")
        print(f"{'='*50}")
        
        dataset_results = {}
        
        # 遍历所有fold
        for fold in range(1, base_config['num_folds'] + 1):
            try:
                result = train_single_fold(dataset_name, fold)
                dataset_results[fold] = result
                print(f"Fold {fold} 完成")
            except Exception as e:
                print(f"Fold {fold} 训练失败: {str(e)}")
                dataset_results[fold] = None
        
        # 保存当前数据集的结果
        results_file, report_file = save_single_dataset_results(dataset_name, dataset_results)
        all_results[dataset_name] = dataset_results
        
        print(f"{dataset_name} 数据集训练完成！")
        print(f"详细结果: {results_file}")
        print(f"汇总报告: {report_file}")
    
    print("\n所有训练完成！")


if __name__ == "__main__":
    main()