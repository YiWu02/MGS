#!/bin/bash

# AGIQA 统一测试启动脚本

echo "=========================================="
echo "AGIQA 统一测试系统"
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
    echo "警告: 未检测到NVIDIA GPU，将使用CPU测试"
fi

# 检查模型文件
echo "检查模型文件..."
for dataset in "AGIQA-1K" "AGIQA-3K" "AIGCIQA2023"; do
    if [ -d "./checkpoints/$dataset" ]; then
        echo "✓ 找到模型目录: $dataset"
        # 检查是否有训练好的模型
        model_count=$(find "./checkpoints/$dataset" -name "best_model_*.pt" | wc -l)
        echo "  找到 $model_count 个模型文件"
    else
        echo "✗ 未找到模型目录: $dataset"
    fi
done

# 检查数据目录
echo "检查测试数据..."
for dataset in "AGIQA-1K" "AGIQA-3K" "AIGCIQA2023"; do
    if [ -d "./Database/$dataset" ]; then
        echo "✓ 找到数据集: $dataset"
    else
        echo "✗ 未找到数据集: $dataset"
    fi
done

# 创建必要的目录
echo "创建输出目录..."
mkdir -p test_results

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 开始测试
echo "开始测试..."
python test_unified.py

echo "测试完成！"
echo "结果保存在 test_results/ 目录中" 