#!/bin/bash
# TRELLIS.2依赖安装脚本 (在DUSt3R环境中)

echo "=========================================="
echo "安装TRELLIS.2依赖到DUSt3R环境"
echo "=========================================="

# 激活conda环境
source /root/anaconda3/etc/profile.d/conda.sh
conda activate DUSt3R

# 进入TRELLIS.2目录
cd /mnt/data3/clip/DUSt3R/TRELLIS.2-main

# 安装基础依赖
echo "安装基础依赖..."
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers tensorboard pandas lpips zstandard
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install kornia timm

# 安装nvdiffrast
echo "安装nvdiffrast..."
mkdir -p /tmp/extensions
if [ ! -d "/tmp/extensions/nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi
pip install /tmp/extensions/nvdiffrast --no-build-isolation

# 安装nvdiffrec
echo "安装nvdiffrec..."
if [ ! -d "/tmp/extensions/nvdiffrec" ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
fi
pip install /tmp/extensions/nvdiffrec --no-build-isolation

# 安装CuMesh
echo "安装CuMesh..."
if [ ! -d "/tmp/extensions/CuMesh" ]; then
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
fi
pip install /tmp/extensions/CuMesh --no-build-isolation

# 安装FlexGEMM
echo "安装FlexGEMM..."
if [ ! -d "/tmp/extensions/FlexGEMM" ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
fi
pip install /tmp/extensions/FlexGEMM --no-build-isolation

# 安装o-voxel
echo "安装o-voxel..."
cp -r o-voxel /tmp/extensions/o-voxel
pip install /tmp/extensions/o-voxel --no-build-isolation

echo "=========================================="
echo "依赖安装完成!"
echo "=========================================="
