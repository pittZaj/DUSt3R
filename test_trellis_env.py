#!/usr/bin/env python3
"""
测试TRELLIS.2环境和依赖
"""

import sys
sys.path.insert(0, '/mnt/data3/clip/DUSt3R/TRELLIS.2-main')

print("="*60)
print("TRELLIS.2环境测试")
print("="*60)
print()

# 测试基础依赖
print("1. 测试基础依赖...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"   ✗ PyTorch导入失败: {e}")

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except Exception as e:
    print(f"   ✗ NumPy导入失败: {e}")

try:
    import trimesh
    print(f"   ✓ Trimesh {trimesh.__version__}")
except Exception as e:
    print(f"   ✗ Trimesh导入失败: {e}")

try:
    import gradio as gr
    print(f"   ✓ Gradio {gr.__version__}")
except Exception as e:
    print(f"   ✗ Gradio导入失败: {e}")

print()

# 测试CUDA扩展
print("2. 测试CUDA扩展...")
try:
    import flash_attn
    print(f"   ✓ flash-attn")
except Exception as e:
    print(f"   ✗ flash-attn导入失败: {e}")

try:
    import nvdiffrast
    print(f"   ✓ nvdiffrast")
except Exception as e:
    print(f"   ✗ nvdiffrast导入失败: {e}")

try:
    import cumesh
    print(f"   ✓ cumesh")
except Exception as e:
    print(f"   ✗ cumesh导入失败: {e}")

print()

# 测试TRELLIS.2模块
print("3. 测试TRELLIS.2模块...")
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print(f"   ✓ Trellis2ImageTo3DPipeline导入成功")
except Exception as e:
    print(f"   ✗ Trellis2ImageTo3DPipeline导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from trellis2.utils import render_utils
    print(f"   ✓ render_utils导入成功")
except Exception as e:
    print(f"   ✗ render_utils导入失败: {e}")

try:
    from trellis2.renderers import EnvMap
    print(f"   ✓ EnvMap导入成功")
except Exception as e:
    print(f"   ✗ EnvMap导入失败: {e}")

print()

# 测试o-voxel
print("4. 测试o-voxel...")
try:
    import o_voxel
    print(f"   ✓ o_voxel导入成功")
except Exception as e:
    print(f"   ✗ o_voxel导入失败: {e}")
    print(f"   ℹ 这是可选的，不影响基本功能")

print()

# 测试FlexGEMM
print("5. 测试FlexGEMM...")
try:
    import flex_gemm
    print(f"   ✓ flex_gemm导入成功")
except Exception as e:
    print(f"   ✗ flex_gemm导入失败: {e}")
    print(f"   ℹ 这是可选的，不影响基本功能")

print()
print("="*60)
print("测试完成")
print("="*60)
