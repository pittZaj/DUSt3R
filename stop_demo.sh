#!/bin/bash
# 煤堆体积测算系统 - 停止脚本

echo "============================================================"
echo "🛑 煤堆体积测算系统 - 停止脚本"
echo "============================================================"

# 停止主系统
PID=$(ps aux | grep "python coal_volume_demo.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "主系统: 未运行"
else
    echo "停止主系统 (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 1
    kill -9 $PID 2>/dev/null
    echo "✅ 主系统已停止"
fi

# 停止点云分析系统
PID2=$(ps aux | grep "python coal_pile_ply_analyzer.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID2" ]; then
    echo "点云分析系统: 未运行"
else
    echo "停止点云分析系统 (PID: $PID2)..."
    kill $PID2 2>/dev/null
    sleep 1
    kill -9 $PID2 2>/dev/null
    echo "✅ 点云分析系统已停止"
fi

echo "============================================================"
