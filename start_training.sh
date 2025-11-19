#!/bin/bash

# AGIQA 统一训练启动脚本

echo "=========================================="
echo "AGIQA 统一训练系统"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练"
fi

# 检查数据目录
echo "检查数据目录..."
for dataset in "AGIQA-1K" "AGIQA-3K" "AIGCIQA2023"; do
    if [ -d "./Database/$dataset" ]; then
        echo "✓ 找到数据集: $dataset"
    else
        echo "✗ 未找到数据集: $dataset"
    fi
done

# 创建必要的目录
echo "创建输出目录..."
mkdir -p checkpoints runs test_results

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 开始训练
echo "开始训练..."
python train_unified.py

echo "训练完成！"