#!/bin/bash
# TRELLIS.2依赖安装脚本 (修正版 - 适配CUDA 11.8)

echo "=========================================="
echo "安装TRELLIS.2依赖到DUSt3R环境 (修正版)"
echo "=========================================="

# 激活conda环境
source /root/anaconda3/etc/profile.d/conda.sh
conda activate DUSt3R

# 设置CUDA环境变量 (使用PyTorch对应的CUDA 11.8)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "PyTorch CUDA版本: $(python -c 'import torch; print(torch.version.cuda)')"

# 进入TRELLIS.2目录
cd /mnt/data3/clip/DUSt3R/TRELLIS.2-main

# 安装基础依赖
echo "安装基础依赖..."
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers tensorboard pandas lpips zstandard
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install kornia timm

# 尝试安装nvdiffrast (如果失败则跳过)
echo "安装nvdiffrast..."
mkdir -p /tmp/extensions
if [ ! -d "/tmp/extensions/nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi

# 使用CUDA 11.8编译
cd /tmp/extensions/nvdiffrast
pip install . --no-build-isolation || echo "警告: nvdiffrast安装失败，将尝试继续"

cd /mnt/data3/clip/DUSt3R/TRELLIS.2-main

# 安装nvdiffrec
echo "安装nvdiffrec..."
if [ ! -d "/tmp/extensions/nvdiffrec" ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
fi
cd /tmp/extensions/nvdiffrec
pip install . --no-build-isolation || echo "警告: nvdiffrec安装失败，将尝试继续"

cd /mnt/data3/clip/DUSt3R/TRELLIS.2-main

# 安装CuMesh
echo "安装CuMesh..."
if [ ! -d "/tmp/extensions/CuMesh" ]; then
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
fi
cd /tmp/extensions/CuMesh
pip install . --no-build-isolation || echo "警告: CuMesh安装失败，将尝试继续"

cd /mnt/data3/clip/DUSt3R/TRELLIS.2-main

# 安装FlexGEMM
echo "安装FlexGEMM..."
if [ ! -d "/tmp/extensions/FlexGEMM" ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
fi
cd /tmp/extensions/FlexGEMM
pip install . --no-build-isolation || echo "警告: FlexGEMM安装失败，将尝试继续"

cd /mnt/data3/clip/DUSt3R/TRELLIS.2-main

# 安装o-voxel
echo "安装o-voxel..."
rm -rf /tmp/extensions/o-voxel
cp -r o-voxel /tmp/extensions/o-voxel
cd /tmp/extensions/o-voxel
pip install . --no-build-isolation || echo "警告: o-voxel安装失败，将尝试继续"

cd /mnt/data3/clip/DUSt3R

echo "=========================================="
echo "依赖安装完成!"
echo "=========================================="
echo "注意: 如果有警告信息，某些功能可能受限"
echo "可以尝试运行Demo查看是否正常工作"
