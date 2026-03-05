# 系统重启指令说明

## 📋 项目概览

本项目包含两个独立的系统：

### 1. 煤堆体积测算系统（主系统）
- **端口**: 7868
- **启动文件**: `coal_volume_demo.py`
- **功能**: 上传多张煤堆照片，一键输出体积、重量、3D点云
- **重启脚本**: `restart_volume_demo.sh`

### 2. 煤堆点云精细处理分析系统（分析系统）
- **端口**: 7869
- **启动文件**: `coal_pile_ply_analyzer.py`
- **功能**: 对已生成的点云文件进行精细化处理和分析
- **重启脚本**: `restart_coal_system.sh`

---

## 🚀 重启指令

### 方法1: 使用重启脚本（推荐）

#### 重启"煤堆体积测算系统"（端口7868）
```bash
cd /mnt/data3/clip/DUSt3R
bash restart_volume_demo.sh
```

#### 重启"煤堆点云精细处理分析系统"（端口7869）
```bash
cd /mnt/data3/clip/DUSt3R
bash restart_coal_system.sh
```

### 方法2: 手动重启命令

#### 重启"煤堆体积测算系统"（端口7868）
```bash
# 1. 停止服务
lsof -ti:7868 | xargs kill -9 2>/dev/null

# 2. 等待2秒
sleep 2

# 3. 启动服务
cd /mnt/data3/clip/DUSt3R
nohup python coal_volume_demo.py > /tmp/coal_volume_demo.log 2>&1 &

# 4. 等待服务启动（约10秒）
sleep 10

# 5. 检查服务状态
lsof -i:7868 | grep LISTEN
```

#### 重启"煤堆点云精细处理分析系统"（端口7869）
```bash
# 1. 停止服务
lsof -ti:7869 | xargs kill -9 2>/dev/null

# 2. 等待2秒
sleep 2

# 3. 启动服务
cd /mnt/data3/clip/DUSt3R
nohup python coal_pile_ply_analyzer.py > /tmp/coal_pile_analyzer.log 2>&1 &

# 4. 等待服务启动（约10秒）
sleep 10

# 5. 检查服务状态
lsof -i:7869 | grep LISTEN
```

---

## 🔍 常用检查命令

### 检查服务运行状态

```bash
# 检查端口7868（煤堆体积测算系统）
lsof -i:7868

# 检查端口7869（煤堆点云精细处理分析系统）
lsof -i:7869

# 检查进程
ps aux | grep coal_volume_demo
ps aux | grep coal_pile_ply_analyzer
```

### 查看日志

```bash
# 煤堆体积测算系统日志
tail -f /tmp/coal_volume_demo.log

# 煤堆点云精细处理分析系统日志
tail -f /tmp/coal_pile_analyzer.log
```

### 停止服务

```bash
# 停止端口7868服务
lsof -ti:7868 | xargs kill -9

# 停止端口7869服务
lsof -ti:7869 | xargs kill -9
```

---

## 🌐 访问地址

### 煤堆体积测算系统（端口7868）
- 本地访问: http://localhost:7868
- 外部访问: http://YOUR_SERVER_IP:7868

### 煤堆点云精细处理分析系统（端口7869）
- 本地访问: http://localhost:7869
- 外部访问: http://YOUR_SERVER_IP:7869

---

## ⚠️ 注意事项

1. **启动时间**: 系统启动需要约10-15秒，请耐心等待
2. **GPU内存**: 如果系统启动失败，可能是GPU内存不足，建议重启服务器
3. **端口冲突**: 确保端口7868和7869没有被其他程序占用
4. **日志查看**: 如果启动失败，请查看日志文件排查问题

---

## 🛠️ 故障排除

### 问题1: 服务启动失败

**解决方案**:
```bash
# 查看日志
tail -100 /tmp/coal_volume_demo.log

# 检查端口占用
lsof -i:7868

# 清理GPU内存
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 问题2: 端口被占用

**解决方案**:
```bash
# 强制停止占用端口的进程
lsof -ti:7868 | xargs kill -9
lsof -ti:7869 | xargs kill -9
```

### 问题3: 无法访问Web界面

**检查清单**:
1. 确认服务已启动: `lsof -i:7868`
2. 检查防火墙设置
3. 确认浏览器访问的IP和端口正确

---

## 📝 快速参考

| 操作 | 命令 |
|------|------|
| 重启主系统（7868） | `bash restart_volume_demo.sh` |
| 重启分析系统（7869） | `bash restart_coal_system.sh` |
| 查看主系统日志 | `tail -f /tmp/coal_volume_demo.log` |
| 查看分析系统日志 | `tail -f /tmp/coal_pile_analyzer.log` |
| 检查主系统状态 | `lsof -i:7868` |
| 检查分析系统状态 | `lsof -i:7869` |
| 停止主系统 | `lsof -ti:7868 \| xargs kill -9` |
| 停止分析系统 | `lsof -ti:7869 \| xargs kill -9` |

---

**文档更新时间**: 2026-03-05
**系统版本**: v4.3.1
