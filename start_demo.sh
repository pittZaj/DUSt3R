#!/bin/bash
# 煤堆体积测算Demo启动脚本

echo "============================================================"
echo "🚀 煤堆体积测算系统 - 启动脚本"
echo "============================================================"

# 激活conda环境
source /root/anaconda3/bin/activate DUSt3R

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 切换到工作目录
cd /mnt/data3/clip/DUSt3R

# 启动服务
echo "正在启动服务..."
python coal_volume_demo.py

echo "============================================================"
