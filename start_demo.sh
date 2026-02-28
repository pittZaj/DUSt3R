#!/bin/bash
# 煤堆体积测算系统 - 启动脚本
# 同时启动主系统(7868)和点云精细处理分析系统(7869)

echo "============================================================"
echo "🚀 煤堆体积测算系统 - 启动脚本"
echo "============================================================"

# 激活conda环境
source /root/anaconda3/bin/activate DUSt3R

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 切换到工作目录
cd /mnt/data3/clip/DUSt3R

# 启动主系统（3D重建 + 体积计算，端口7868）
echo "正在启动主系统（端口7868）..."
nohup python coal_volume_demo.py > /tmp/coal_main.log 2>&1 &
MAIN_PID=$!
echo "主系统 PID: $MAIN_PID"

# 等待主系统初始化
sleep 3

# 启动点云精细处理分析系统（端口7869）
echo "正在启动点云精细处理分析系统（端口7869）..."
nohup python coal_pile_ply_analyzer.py > /tmp/coal_ply.log 2>&1 &
PLY_PID=$!
echo "点云分析系统 PID: $PLY_PID"

echo "============================================================"
echo "✅ 两个服务已启动"
echo ""
echo "📱 主系统（3D重建）:        http://localhost:7868"
echo "🔬 点云精细处理分析系统:    http://localhost:7869"
echo ""
echo "📋 日志文件:"
echo "   主系统:   /tmp/coal_main.log"
echo "   分析系统: /tmp/coal_ply.log"
echo ""
echo "🛑 停止服务请运行: ./stop_demo.sh"
echo "============================================================"
