#!/bin/bash

echo "=========================================="
echo "煤堆体积测算Demo - TRELLIS.2版本"
echo "=========================================="
echo ""

# 激活TRELLIS conda环境
echo "激活TRELLIS环境..."
source /root/anaconda3/bin/activate TRELLIS

# 检查环境
echo "检查Python版本..."
python --version

echo "检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "启动Demo..."
echo "访问地址: http://0.0.0.0:7871"
echo ""

# 运行demo
cd /mnt/data3/clip/DUSt3R
python coal_pile_demo_trellis.py
