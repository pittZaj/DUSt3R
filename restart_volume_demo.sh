#!/bin/bash
# 煤堆体积测算系统 - 重启脚本

echo "========================================"
echo "煤堆体积测算系统 - 重启"
echo "========================================"

# 1. 停止现有服务
echo ""
echo "[步骤1] 停止现有服务..."
lsof -ti:7868 | xargs kill -9 2>/dev/null || true
sleep 2
echo "✓ 现有服务已停止"

# 2. 清理GPU内存
echo ""
echo "[步骤2] 清理GPU内存..."
python3 << EOF
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("✓ GPU内存已清理")
else:
    print("⚠️ 未检测到CUDA")
EOF

# 3. 启动服务
echo ""
echo "[步骤3] 启动煤堆体积测算系统..."
cd /mnt/data3/clip/DUSt3R
nohup python coal_volume_demo.py > /tmp/coal_volume_demo.log 2>&1 &
sleep 5

# 4. 检查服务状态
echo ""
echo "[步骤4] 检查服务状态..."
if lsof -ti:7868 > /dev/null 2>&1; then
    echo "✅ 服务启动成功！"
    echo ""
    echo "访问地址:"
    echo "  - 本地: http://localhost:7868"
    echo "  - 外部: http://$(hostname -I | awk '{print $1}'):7868"
    echo ""
    echo "日志文件: /tmp/coal_volume_demo.log"
else
    echo "❌ 服务启动失败"
    echo "请查看日志: tail -f /tmp/coal_volume_demo.log"
    exit 1
fi

echo ""
echo "========================================"
echo "系统重启完成"
echo "========================================"
