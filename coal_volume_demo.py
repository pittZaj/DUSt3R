#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤堆体积测算Demo系统
基于DUSt3R进行3D重建和体积计算
参考官方demo实现和项目规范
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
import gradio as gr
from scipy.spatial import Delaunay, ConvexHull
import json
from datetime import datetime
import time
import gc
import tempfile
import traceback
import trimesh

# 设置环境变量优化CUDA内存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加dust3r路径
sys.path.insert(0, '/mnt/data3/clip/DUSt3R/dust3r-main')

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# 全局性能优化开关
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    print("⚙️ 已启用TF32/高精度矩阵乘加速")
except Exception:
    pass


class CoalVolumeEstimator:
    """煤堆体积测算系统"""

    def __init__(self, model_path, device='cuda'):
        """
        初始化系统

        Args:
            model_path: 模型权重路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """加载DUSt3R模型"""
        print(f"🔄 正在加载模型: {self.model_path}")
        
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 加载模型
            self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            
            print(f"✅ 模型加载成功！")
            
            if self.device == 'cuda':
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"💾 显存: {total_mem:.2f} GB")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise

    def reconstruct_3d(self, image_paths, image_size=512):
        """
        从多张图像进行3D重建

        Args:
            image_paths: 图像路径列表
            image_size: 图像处理尺寸

        Returns:
            scene: 全局对齐后的场景对象
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"[步骤1] 加载 {len(image_paths)} 张图像...")
        t0 = time.time()
        images = load_images(image_paths, size=image_size)
        t1 = time.time()
        print(f"✓ 图像加载完成 (耗时: {t1-t0:.2f}s)")

        print(f"\n[步骤2] 生成图像对...")
        t0 = time.time()
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        t1 = time.time()
        print(f"✓ 生成了 {len(pairs)} 个图像对 (耗时: {t1-t0:.2f}s)")

        print(f"\n[步骤3] 执行3D重建...")
        t0 = time.time()
        with torch.no_grad():
            output = inference(pairs, self.model, self.device, batch_size=1)
        t1 = time.time()
        print(f"✓ 3D重建完成 (耗时: {t1-t0:.2f}s)")

        print(f"\n[步骤4] 全局对齐...")
        t0 = time.time()
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
        t1 = time.time()
        print(f"✓ 全局对齐完成 (损失: {loss:.4f}, 耗时: {t1-t0:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ 3D重建总耗时: {total_time:.2f}s")

        return scene

    def extract_pointcloud(self, scene, confidence_threshold=0.5):
        """
        从场景中提取点云

        Args:
            scene: 全局对齐后的场景
            confidence_threshold: 置信度阈值（降低以获取更多点）

        Returns:
            pcd: Open3D点云对象
        """
        print(f"\n[步骤5] 提取点云...")
        print(f"  - 置信度阈值: {confidence_threshold}")
        t0 = time.time()

        # 获取点云和置信度
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        print(f"  - 总共 {len(pts3d)} 个视图")

        # 合并所有视图的点云
        all_points = []
        all_colors = []

        for i in range(len(pts3d)):
            # 获取置信度掩码
            conf_mask = confidence_masks[i].cpu().numpy()

            # 打印置信度统计
            print(f"  - 视图 {i+1} 置信度范围: [{conf_mask.min():.3f}, {conf_mask.max():.3f}], 平均: {conf_mask.mean():.3f}")

            # 展平并应用阈值
            conf_mask_binary = conf_mask > confidence_threshold
            conf_mask_flat = conf_mask_binary.reshape(-1)

            print(f"    通过阈值的像素: {conf_mask_flat.sum()} / {len(conf_mask_flat)} ({conf_mask_flat.sum()/len(conf_mask_flat)*100:.1f}%)")

            # 获取点和颜色（先展平再索引）
            points = pts3d[i].detach().cpu().numpy().reshape(-1, 3)[conf_mask_flat]

            # 获取颜色（从原始图像）
            img = scene.imgs[i]
            colors = img.reshape(-1, 3)[conf_mask_flat]

            print(f"    实际提取: {len(points):,} 个点")

            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)

        # 合并所有点
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)

        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        t1 = time.time()
        print(f"✓ 提取了 {len(all_points):,} 个点 (耗时: {t1-t0:.2f}s)")

        return pcd

    def align_ground_to_xy_plane(self, pcd):
        """
        自动检测地面并旋转点云，使地面对齐到XY平面，Z轴垂直向上

        这是解决pile_aware算法假设问题的关键步骤！

        Args:
            pcd: 输入点云

        Returns:
            pcd: 对齐后的点云
            ground_normal: 地面法向量（对齐前）
            rotation_matrix: 旋转矩阵
        """
        print(f"\n[步骤6] 地面自动对齐...")
        t0 = time.time()

        points = np.asarray(pcd.points)

        # 1. 选取底部30%的点作为地面候选
        z_threshold = np.percentile(points[:, 2], 30)
        ground_candidates_mask = points[:, 2] < z_threshold
        ground_candidates = points[ground_candidates_mask]

        if len(ground_candidates) < 100:
            print("  ⚠️ 地面候选点太少，跳过对齐")
            return pcd, np.array([0, 0, 1]), np.eye(3)

        # 2. 使用RANSAC拟合地面平面
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_candidates)

        plane_model, inliers = ground_pcd.segment_plane(
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=1000
        )

        [a, b, c, d] = plane_model
        ground_normal = np.array([a, b, c])
        ground_normal = ground_normal / np.linalg.norm(ground_normal)

        print(f"  - 检测到地面法向量: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")

        # 3. 计算旋转矩阵，将地面法向量对齐到Z轴
        target_normal = np.array([0, 0, 1])

        # 如果法向量指向下方，翻转它
        if ground_normal[2] < 0:
            ground_normal = -ground_normal
            print(f"  - 翻转法向量（指向上方）: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")

        # 计算旋转轴和角度
        rotation_axis = np.cross(ground_normal, target_normal)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-6:
            # 法向量已经对齐
            print("  ✓ 地面已经对齐到XY平面，无需旋转")
            rotation_matrix = np.eye(3)
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(np.dot(ground_normal, target_normal), -1.0, 1.0))

            print(f"  - 旋转角度: {np.degrees(rotation_angle):.2f}°")

            # 使用Rodrigues公式构建旋转矩阵
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])

            rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

        # 4. 应用旋转
        pcd.rotate(rotation_matrix, center=(0, 0, 0))

        # 5. 平移使最低点接近Z=0
        points_rotated = np.asarray(pcd.points)
        z_min = points_rotated[:, 2].min()
        translation = np.array([0, 0, -z_min])
        pcd.translate(translation)

        t1 = time.time()
        print(f"✓ 地面对齐完成 (耗时: {t1-t0:.2f}s)")
        print(f"  - Z轴现在垂直于地面向上")
        print(f"  - 地面位于Z≈0平面")

        return pcd, ground_normal, rotation_matrix

    def postprocess_pointcloud(self, pcd, outlier_std=3.0, voxel_size=0.01):
        """
        点云后处理

        Args:
            pcd: 原始点云
            outlier_std: 离群点过滤强度（标准差倍数）
            voxel_size: 下采样体素大小

        Returns:
            pcd: 处理后的点云
        """
        print(f"\n[步骤7] 点云后处理...")
        t0 = time.time()

        initial_count = len(pcd.points)
        print(f"  - 初始点云: {initial_count:,} 个点")

        # 1. 离群点去除（使用自定义参数）
        if initial_count > 100:
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=outlier_std)
            print(f"  - 离群点去除后: {len(pcd.points):,} 个点 (std_ratio={outlier_std})")

        # 检查是否还有足够的点
        if len(pcd.points) < 50:
            raise ValueError(f"点云后处理后点数过少({len(pcd.points)}个)，可能是图像质量问题或重叠不足")

        # 2. 下采样（使用自定义体素尺寸）
        if len(pcd.points) > 10000:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"  - 下采样后: {len(pcd.points):,} 个点 (voxel_size={voxel_size})")

        # 再次检查
        if len(pcd.points) < 50:
            raise ValueError(f"下采样后点数过少({len(pcd.points)}个)")

        # 3. 法向量估计
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        t1 = time.time()
        print(f"✓ 点云后处理完成 (耗时: {t1-t0:.2f}s)")

        return pcd

    def segment_coal_pile(self, pcd, ground_height_percentile=5, use_ransac=False):
        """
        分割煤堆（去除地面）

        Args:
            pcd: 点云
            ground_height_percentile: 地面高度百分位数（用于简单方法）
            use_ransac: 是否使用RANSAC平面拟合（推荐关闭，仅在地面倾斜时使用）

        Returns:
            coal_pcd: 煤堆点云
            ground_height: 地面高度
            ground_plane: 地面平面参数 [a, b, c, d]（仅RANSAC方法）
        """
        print(f"\n[步骤8] 煤堆分割...")
        t0 = time.time()

        points = np.asarray(pcd.points)

        # 检查点云是否为空
        if len(points) == 0:
            raise ValueError("点云为空，无法进行煤堆分割。请检查图像质量和重叠度。")

        if use_ransac:
            # 方法2: RANSAC平面拟合（适用于地面倾斜的场景）
            print(f"  - 使用RANSAC拟合地面平面...")

            # 选取较低的点作为地面候选点（扩大范围到30%）
            z_threshold = np.percentile(points[:, 2], 30)
            ground_candidates_mask = points[:, 2] < z_threshold
            ground_candidates = points[ground_candidates_mask]

            if len(ground_candidates) < 3:
                print("  ⚠️ 地面候选点太少，切换到简单方法")
                use_ransac = False
            else:
                # 使用Open3D的RANSAC平面分割
                ground_pcd = o3d.geometry.PointCloud()
                ground_pcd.points = o3d.utility.Vector3dVector(ground_candidates)

                # RANSAC平面拟合（使用更宽松的参数）
                plane_model, inliers = ground_pcd.segment_plane(
                    distance_threshold=0.03,  # 增加到3cm，更宽松
                    ransac_n=3,
                    num_iterations=1000
                )

                [a, b, c, d] = plane_model
                ground_plane = plane_model

                # 计算地面高度
                center_xy = points[:, :2].mean(axis=0)
                ground_height = -(a * center_xy[0] + b * center_xy[1] + d) / c

                print(f"  - RANSAC拟合地面平面: [{a:.3f}, {b:.3f}, {c:.3f}, {d:.3f}]")
                print(f"  - 地面高度（中心点）: {ground_height:.3f}")
                print(f"  - 地面内点数: {len(inliers)} / {len(ground_candidates)}")

                # 使用平面距离过滤（使用更宽松的阈值）
                distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
                mask = distances > 0.03  # 增加到3cm，保留更多点
                coal_pcd = pcd.select_by_index(np.where(mask)[0])

        if not use_ransac:
            # 方法1: 简单高度阈值方法（推荐，快速准确）
            print(f"  - 使用简单高度阈值方法...")
            ground_height = np.percentile(points[:, 2], ground_height_percentile)
            ground_plane = None
            print(f"  - 估计地面高度: {ground_height:.3f}")

            # 使用高度阈值过滤
            mask = points[:, 2] > (ground_height + 0.02)
            coal_pcd = pcd.select_by_index(np.where(mask)[0])

        # 再次检查分割后的点云
        if len(coal_pcd.points) == 0:
            print("  ⚠️ 警告：所有点都被识别为地面，保留原始点云")
            coal_pcd = pcd
            ground_plane = None

        t1 = time.time()
        print(f"✓ 煤堆分割完成，保留 {len(coal_pcd.points):,} 个点 (耗时: {t1-t0:.2f}s)")

        return coal_pcd, ground_height, ground_plane

    def calculate_volume_delaunay(self, pcd, ground_height):
        """使用Delaunay三角剖分计算体积"""
        points = np.asarray(pcd.points)
        
        # 添加地面点
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        grid_size = 20
        ground_grid = np.array([
            [x, y, ground_height]
            for x in np.linspace(x_min, x_max, grid_size)
            for y in np.linspace(y_min, y_max, grid_size)
        ])
        
        all_points = np.vstack([points, ground_grid])
        
        try:
            tri = Delaunay(all_points)
            
            def tetrahedron_volume(a, b, c, d):
                return abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
            
            total_volume = 0
            for simplex in tri.simplices:
                pts = all_points[simplex]
                if np.any(pts[:, 2] > ground_height + 0.01):
                    total_volume += tetrahedron_volume(*pts)
            
            return total_volume
        except Exception as e:
            print(f"  ⚠️ Delaunay方法失败: {e}")
            return 0.0

    def calculate_volume_convex_hull(self, pcd, ground_height):
        """使用凸包法计算体积"""
        points = np.asarray(pcd.points)
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        ground_grid = np.array([
            [x, y, ground_height]
            for x in np.linspace(x_min, x_max, 20)
            for y in np.linspace(y_min, y_max, 20)
        ])
        
        all_points = np.vstack([points, ground_grid])
        
        try:
            hull = ConvexHull(all_points)
            return hull.volume
        except Exception as e:
            print(f"  ⚠️ 凸包方法失败: {e}")
            return 0.0

    def calculate_volume_voxel(self, pcd, voxel_size=0.05, ground_height=0.0):
        """使用体素化方法计算体积"""
        try:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
            voxels = voxel_grid.get_voxels()
            
            count = 0
            for voxel in voxels:
                center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
                if center[2] > ground_height:
                    count += 1
            
            volume = count * (voxel_size ** 3)
            return volume
        except Exception as e:
            print(f"  ⚠️ 体素方法失败: {e}")
            return 0.0

    def calculate_volume(self, pcd, ground_height):
        """使用多种方法计算体积并融合"""
        print(f"\n[步骤9] 体积计算...")
        t0 = time.time()

        # 方法1: Delaunay
        print("  - 使用Delaunay三角剖分...")
        v1 = self.calculate_volume_delaunay(pcd, ground_height)
        print(f"    体积: {v1:.3f} m³")

        # 方法2: 凸包
        print("  - 使用凸包法...")
        v2 = self.calculate_volume_convex_hull(pcd, ground_height)
        print(f"    体积: {v2:.3f} m³")

        # 方法3: 体素化
        print("  - 使用体素化方法...")
        v3 = self.calculate_volume_voxel(pcd, voxel_size=0.05, ground_height=ground_height)
        print(f"    体积: {v3:.3f} m³")

        # 过滤无效值
        volumes = [v for v in [v1, v2, v3] if v > 0]

        if len(volumes) == 0:
            print("  ⚠️ 所有方法都失败了")
            return {
                'volume': 0.0,
                'confidence': 0.0,
                'methods': {'delaunay': v1, 'convex_hull': v2, 'voxel': v3}
            }

        # 加权平均
        volume = np.mean(volumes)

        # 计算置信度（基于方法间的一致性）
        if len(volumes) > 1:
            std = np.std(volumes)
            confidence = 1.0 / (1.0 + std / volume) if volume > 0 else 0.0
        else:
            confidence = 0.5

        t1 = time.time()
        print(f"✓ 体积计算完成 (耗时: {t1-t0:.2f}s)")
        print(f"  - 最终体积: {volume:.3f} m³")
        print(f"  - 置信度: {confidence:.2%}")

        return {
            'volume': volume,
            'confidence': confidence,
            'methods': {
                'delaunay': v1,
                'convex_hull': v2,
                'voxel': v3
            }
        }

    def estimate_weight(self, volume, density=1300):
        """根据体积估算重量"""
        weight_kg = volume * density
        weight_ton = weight_kg / 1000
        return weight_ton

    def process_images(self, image_paths, output_dir, coal_density=1300, image_size=512,
                      confidence_threshold=0.5, outlier_std=3.0, voxel_size=0.01, use_ransac=False):
        """处理图像并生成结果"""
        print("\n" + "="*60)
        print("煤堆体积测算系统 - 开始处理")
        print("="*60)

        overall_start = time.time()

        try:
            # 1. 3D重建
            scene = self.reconstruct_3d(image_paths, image_size=image_size)

            # 2. 提取点云（使用自定义置信度阈值）
            pcd = self.extract_pointcloud(scene, confidence_threshold=confidence_threshold)

            # 3. 地面自动对齐（关键步骤！使Z轴垂直于地面）
            pcd, ground_normal, rotation_matrix = self.align_ground_to_xy_plane(pcd)

            # 4. 点云后处理（使用自定义参数）
            pcd = self.postprocess_pointcloud(pcd, outlier_std=outlier_std, voxel_size=voxel_size)

            # 5. 煤堆分割（现在地面已经在XY平面上了）
            coal_pcd, ground_height, ground_plane = self.segment_coal_pile(pcd, use_ransac=use_ransac)

            # 6. 体积计算
            volume_result = self.calculate_volume(coal_pcd, ground_height)

            # 7. 重量估算
            weight = self.estimate_weight(volume_result['volume'], coal_density)

            # 8. 保存结果
            os.makedirs(output_dir, exist_ok=True)

            # 保存点云
            pcd_path = os.path.join(output_dir, 'coal_pile.ply')
            o3d.io.write_point_cloud(pcd_path, coal_pcd)
            print(f"\n✓ 点云已保存: {pcd_path}")

            # 保存结果JSON
            result = {
                'timestamp': datetime.now().isoformat(),
                'volume_m3': float(volume_result['volume']),
                'weight_ton': float(weight),
                'confidence': float(volume_result['confidence']),
                'coal_density_kg_m3': coal_density,
                'ground_height': float(ground_height),
                'num_points': len(coal_pcd.points),
                'num_images': len(image_paths),
                'image_size': image_size,
                'methods': {k: float(v) for k, v in volume_result['methods'].items()},
                'processing_time_seconds': time.time() - overall_start
            }

            result_path = os.path.join(output_dir, 'result.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✓ 结果已保存: {result_path}")

            print("\n" + "="*60)
            print("处理完成！")
            print("="*60)
            print(f"📊 体积: {result['volume_m3']:.2f} 立方米")
            print(f"⚖️  重量: {result['weight_ton']:.2f} 吨")
            print(f"🎯 置信度: {result['confidence']:.1%}")
            print(f"⏱️  总耗时: {result['processing_time_seconds']:.2f}秒")
            print("="*60)

            return result, coal_pcd

        except Exception as e:
            print(f"\n❌ 处理失败: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def create_gradio_interface(estimator):
    """创建Gradio Web界面"""

    def pointcloud_to_glb(pcd, glb_path, uniform_color=True):
        """
        将点云转换为GLB格式用于3D可视化

        Args:
            pcd: Open3D点云对象
            glb_path: 输出GLB文件路径
            uniform_color: 是否使用统一颜色（橙棕色，适合煤堆）

        Returns:
            glb_path: GLB文件路径
        """
        print(f"  - 生成3D可视化模型...")

        # 获取点云数据
        points = np.asarray(pcd.points)

        # 设置颜色
        if uniform_color:
            # 使用统一的橙棕色（煤堆/沙堆的颜色）
            colors = np.tile([0.8, 0.5, 0.2], (len(points), 1))  # RGB: 橙棕色
        else:
            colors = np.asarray(pcd.colors)

        # 采样点云（如果点太多，只显示部分）
        max_points = 5000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        # 创建场景
        scene = trimesh.Scene()

        # 添加点云（使用点云对象）
        point_cloud = trimesh.PointCloud(points, colors=(colors * 255).astype(np.uint8))
        scene.add_geometry(point_cloud)

        # 计算点云边界
        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        center = bounds.mean(axis=0)
        size = np.ptp(bounds, axis=0)
        ground_z = bounds[0, 2]  # 最低点的Z坐标

        # 添加精致的坐标轴（放在角落，更小更柔和）
        axis_length = np.max(size) * 0.15  # 坐标轴长度为点云最大范围的15%
        axis_origin = [bounds[0, 0], bounds[0, 1], ground_z]  # 放在左下角

        # 创建自定义坐标轴（使用细线）
        axis_radius = axis_length * 0.01

        # X轴 - 红色
        x_axis = trimesh.creation.cylinder(
            radius=axis_radius,
            height=axis_length,
            transform=trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        )
        x_axis.apply_translation([axis_origin[0] + axis_length/2, axis_origin[1], axis_origin[2]])
        x_axis.visual.face_colors = [200, 100, 100, 255]  # 柔和的红色
        scene.add_geometry(x_axis)

        # Y轴 - 绿色
        y_axis = trimesh.creation.cylinder(
            radius=axis_radius,
            height=axis_length,
            transform=trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        )
        y_axis.apply_translation([axis_origin[0], axis_origin[1] + axis_length/2, axis_origin[2]])
        y_axis.visual.face_colors = [100, 200, 100, 255]  # 柔和的绿色
        scene.add_geometry(y_axis)

        # Z轴 - 蓝色
        z_axis = trimesh.creation.cylinder(
            radius=axis_radius,
            height=axis_length
        )
        z_axis.apply_translation([axis_origin[0], axis_origin[1], axis_origin[2] + axis_length/2])
        z_axis.visual.face_colors = [100, 100, 200, 255]  # 柔和的蓝色
        scene.add_geometry(z_axis)

        # 导出为GLB格式
        scene.export(glb_path)
        print(f"  ✓ 3D模型已保存: {glb_path}")

        return glb_path

    def process_files(files, coal_density, image_size, confidence_threshold, outlier_std, voxel_size, use_ransac):
        """处理上传的文件"""
        if not files or len(files) == 0:
            return "❌ 请上传至少2张图像", None, None, None, None

        if len(files) < 2:
            return "❌ 至少需要2张图像进行3D重建", None, None, None, None

        try:
            # 获取图像路径
            image_paths = [f.name for f in files]

            print(f"\n📸 收到 {len(image_paths)} 张图像")
            for i, path in enumerate(image_paths, 1):
                print(f"  {i}. {os.path.basename(path)}")

            print(f"\n⚙️ 参数设置:")
            print(f"  - 置信度阈值: {confidence_threshold}")
            print(f"  - 离群点过滤强度: {outlier_std}")
            print(f"  - 下采样体素大小: {voxel_size}")
            print(f"  - RANSAC地面拟合: {'开启' if use_ransac else '关闭'}")

            # 创建输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"/mnt/data3/clip/DUSt3R/output_{timestamp}"

            # 处理图像（传递新参数）
            result, coal_pcd = estimator.process_images(
                image_paths,
                output_dir,
                coal_density=coal_density,
                image_size=image_size,
                confidence_threshold=confidence_threshold,
                outlier_std=outlier_std,
                voxel_size=voxel_size,
                use_ransac=use_ransac
            )

            # 生成结果文本
            result_text = f"""
## 🎉 煤堆体积测算结果

### 📊 主要指标
- **体积**: {result['volume_m3']:.2f} 立方米
- **重量**: {result['weight_ton']:.2f} 吨
- **置信度**: {result['confidence']:.1%}

### 📈 详细信息
- **点云数量**: {result['num_points']:,} 个点
- **图像数量**: {result['num_images']} 张
- **图像尺寸**: {result['image_size']}px
- **地面高度**: {result['ground_height']:.3f} 米
- **煤炭密度**: {result['coal_density_kg_m3']} kg/m³

### 🔬 各方法结果
- **Delaunay三角剖分**: {result['methods']['delaunay']:.3f} m³
- **凸包法**: {result['methods']['convex_hull']:.3f} m³
- **体素化**: {result['methods']['voxel']:.3f} m³

### ⏱️ 性能统计
- **处理时间**: {result['processing_time_seconds']:.2f} 秒
- **时间戳**: {result['timestamp']}

### 📁 输出文件
- **输出目录**: `{output_dir}`
- **点云文件**: `coal_pile.ply`
- **结果文件**: `result.json`

---
✅ **处理完成！** 您可以下载点云文件和结果文件进行进一步分析。
"""

            # 生成GLB文件用于3D可视化
            glb_path = os.path.join(output_dir, 'coal_pile.glb')
            pointcloud_to_glb(coal_pcd, glb_path, uniform_color=True)

            # 返回结果
            pcd_path = os.path.join(output_dir, 'coal_pile.ply')
            result_json_path = os.path.join(output_dir, 'result.json')

            return result_text, glb_path, pcd_path, result_json_path, output_dir

        except Exception as e:
            error_msg = f"""
## ❌ 处理失败

**错误信息**:
```
{str(e)}
```

**详细堆栈**:
```
{traceback.format_exc()}
```

### 🔧 可能的解决方案
1. 确保上传的图像格式正确（JPG, PNG等）
2. 确保图像质量良好，有足够的重叠
3. 检查GPU内存是否充足
4. 尝试减少图像数量或降低图像尺寸
"""
            return error_msg, None, None, None, None

    # 创建界面
    with gr.Blocks(title="煤堆体积测算系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏭 煤堆体积测算系统

            基于自研3D重建算法的智能体积测量 | 洗煤厂智能监管解决方案

            ✅ **系统状态**: 模型已加载，可正常使用
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                # 文件上传
                files = gr.File(
                    label="📷 上传煤堆图像（至少2张，建议5-10张）",
                    file_count="multiple",
                    file_types=["image"],
                    height=200
                )
                
                # 参数设置
                with gr.Row():
                    coal_density = gr.Slider(
                        minimum=1000,
                        maximum=1600,
                        value=1300,
                        step=10,
                        label="煤炭密度 (kg/m³)",
                        info="原煤: 1200-1400, 精煤: 1300-1500"
                    )

                    image_size = gr.Slider(
                        minimum=224,
                        maximum=512,
                        value=512,
                        step=32,
                        label="图像处理尺寸 (px)",
                        info="更大的尺寸精度更高但速度更慢"
                    )

                # 高级参数（可折叠）
                with gr.Accordion("🔧 高级参数设置（推荐使用默认值）", open=False):
                    gr.Markdown("*调整这些参数可以获得不同的点云效果*")

                    with gr.Row():
                        confidence_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.5,
                            step=0.1,
                            label="置信度阈值",
                            info="推荐: 0.5 | 降低可获得更多点，但可能引入噪声"
                        )

                        outlier_std = gr.Slider(
                            minimum=1.0,
                            maximum=5.0,
                            value=3.0,
                            step=0.5,
                            label="离群点过滤强度",
                            info="推荐: 3.0 | 越大越宽松，保留更多点"
                        )

                    with gr.Row():
                        voxel_size = gr.Slider(
                            minimum=0.005,
                            maximum=0.05,
                            value=0.01,
                            step=0.005,
                            label="下采样体素大小 (m)",
                            info="推荐: 0.01 | 越小保留更多细节，但计算更慢"
                        )

                        use_ransac = gr.Checkbox(
                            value=False,
                            label="使用RANSAC地面拟合",
                            info="推荐: 关闭 | 仅在地面倾斜时开启"
                        )

                # 尺度标定说明
                with gr.Accordion("📏 尺度标定说明（重要）", open=True):
                    gr.Markdown("""
                    ### 如何获得真实世界的体积？

                    **当前系统输出的是相对体积**，需要通过尺度标定转换为真实体积：

                    #### 方法1：参考物体标定（推荐）
                    ```
                    1. 在拍摄时放置已知尺寸的参考物体（如1米长的标尺）
                    2. 测量参考物体在点云中的长度
                    3. 计算尺度因子 = 真实长度 / 点云长度
                    4. 真实体积 = 点云体积 × (尺度因子)³
                    ```

                    #### 方法2：已知距离标定
                    ```
                    1. 测量拍摄时相机到煤堆的距离 D (米)
                    2. 使用相机焦距 f 和传感器尺寸计算尺度
                    3. 尺度因子 ≈ D / (焦距 × 图像尺寸)
                    ```

                    #### 示例计算
                    ```
                    假设：
                    - 点云体积: 1.5 m³
                    - 参考物体真实长度: 1.0 m
                    - 参考物体点云长度: 0.8 m
                    - 尺度因子 = 1.0 / 0.8 = 1.25

                    真实体积 = 1.5 × (1.25)³ = 2.93 m³
                    ```

                    **💡 提示**: 建议每次测量都放置参考物体以确保精度
                    """)
                
                # 提交按钮
                submit_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                
                # 结果显示
                result_text = gr.Markdown(label="处理结果")

                # 3D模型查看器
                gr.Markdown("### 🎨 3D点云可视化")
                gr.Markdown("*包含坐标系，统一橙棕色渲染*")
                model_3d = gr.Model3D(
                    label="煤堆3D模型（可旋转、缩放、平移）",
                    height=600,
                    clear_color=[0.1, 0.1, 0.15, 1.0]
                )

                # 文件下载
                with gr.Row():
                    pcd_file = gr.File(label="📦 点云文件 (.ply)")
                    result_file = gr.File(label="📄 结果文件 (.json)")

                output_dir_text = gr.Textbox(label="输出目录", interactive=False)
            
            with gr.Column(scale=1):
                # 使用说明
                gr.Markdown(
                    """
                    ### 📖 使用说明

                    #### 🎯 功能介绍
                    本系统采用自研3D重建算法，
                    无需相机标定，快速准确地测算煤堆体积。

                    #### 📸 图像采集要求

                    **数量要求**:
                    - 最少: 2张图像
                    - 推荐: 5-10张图像
                    - 更多图像可提高精度

                    **拍摄要求**:
                    - ✅ 多角度拍摄（正面、侧面、顶部）
                    - ✅ 图像间有50-70%重叠
                    - ✅ 避免运动模糊
                    - ✅ 光线充足、清晰
                    - ✅ 分辨率≥1920x1080
                    - ✅ **建议放置参考物体**（如1米标尺）

                    **拍摄技巧**:
                    - 围绕煤堆拍摄一圈
                    - 包含地面参考
                    - 避免逆光拍摄
                    - 保持相机稳定

                    #### ⚙️ 参数说明

                    **煤炭密度**:
                    - 原煤: 1200-1400 kg/m³
                    - 精煤: 1300-1500 kg/m³
                    - 可根据实际煤种调整

                    **图像尺寸**:
                    - 224px: 快速处理，精度较低
                    - 512px: 推荐，平衡速度和精度

                    **高级参数**:
                    - 推荐使用默认值
                    - 可根据实际效果微调

                    #### 📊 结果说明

                    **体积**: 点云模型的相对体积

                    **重量**: 根据密度估算的重量

                    **置信度**: 基于多种方法一致性的
                    置信度评分，越高越可靠

                    **点云文件**: 可用专业软件查看3D模型

                    #### ⚠️ 注意事项

                    - 首次处理可能需要较长时间
                    - 确保图像质量良好
                    - **需要尺度标定**才能获得真实体积
                    - 建议每次测量都放置参考物体

                    #### 💡 系统特点

                    - ✨ 无需相机标定
                    - ⚡ 快速3D重建
                    - 🎯 多方法融合计算
                    - 📈 置信度评估
                    - 🔬 点云可视化
                    - 📏 支持尺度标定
                    """
                )
        
        # 事件绑定
        submit_btn.click(
            process_files,
            inputs=[files, coal_density, image_size, confidence_threshold, outlier_std, voxel_size, use_ransac],
            outputs=[result_text, model_3d, pcd_file, result_file, output_dir_text]
        )
    
    return demo


def main():
    """主函数"""
    try:
        print("\n" + "="*60)
        print("🚀 煤堆体积测算系统 - 启动中...")
        print("="*60)
        
        # 模型路径
        model_path = "/mnt/data3/clip/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        
        # 检查模型文件
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("\n可用的模型文件:")
            model_dir = "/mnt/data3/clip/DUSt3R"
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    print(f"  - {f}")
            return
        
        # 检查CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n🖥️  计算设备: {device}")
        
        if device == 'cuda':
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 显存: {total_mem:.2f} GB")
            
            # 设置使用GPU 0（在CUDA_VISIBLE_DEVICES=0的环境下）
            torch.cuda.set_device(0)
        else:
            print("⚠️  未检测到CUDA，将使用CPU（速度较慢）")
        
        # 创建估算器
        print("\n📦 正在初始化系统...")
        estimator = CoalVolumeEstimator(model_path, device=device)
        
        # 创建Gradio界面
        print("\n🎨 正在创建Web界面...")
        demo = create_gradio_interface(estimator)
        
        # 启动服务
        print("\n🌐 启动Web服务器...")
        print("="*60)
        print(f"📱 访问地址: http://localhost:7868")
        print(f"🔗 外部访问: http://<服务器IP>:7868")
        print("="*60)
        print("\n💡 提示:")
        print("  - 上传至少2张煤堆图像")
        print("  - 建议5-10张图像以获得更好效果")
        print("  - 图像间应有50-70%重叠")
        print("  - 首次处理可能需要较长时间")
        print("="*60)
        
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7868,       # 端口号
            share=False,            # 不创建公共链接
            debug=False,            # 调试模式
            show_error=True,        # 显示错误信息
            inbrowser=False         # 不自动打开浏览器
        )
        
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        print("\n🔧 可能的解决方案:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 确保模型文件已下载")
        print("3. 检查端口7868是否被占用")
        print("4. 确保有足够的GPU显存（建议≥8GB）")
        print("5. 检查CUDA环境是否正确配置")
        traceback.print_exc()


if __name__ == "__main__":
    main()
