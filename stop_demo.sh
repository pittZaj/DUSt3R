#!/bin/bash
# 煤堆体积测算Demo停止脚本

echo "============================================================"
echo "🛑 煤堆体积测算系统 - 停止脚本"
echo "============================================================"

# 查找进程
PID=$(ps aux | grep "python coal_volume_demo.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ 未找到运行中的服务"
else
    echo "找到进程: $PID"
    echo "正在停止服务..."
    kill $PID
    sleep 2
    
    # 检查是否成功停止
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️ 进程未响应，强制停止..."
        kill -9 $PID
    fi
    
    echo "✅ 服务已停止"
fi

echo "============================================================"
