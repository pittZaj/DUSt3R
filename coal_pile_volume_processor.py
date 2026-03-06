#!/usr/bin/env python3
"""
煤堆点云体积测量处理模块
实现点云预处理、拼接、分割、细化、曲面重构、边界提取和体积计算功能
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA


class CoalPileVolumeProcessor:
    """煤堆点云体积测量处理器"""

    def __init__(self):
        """初始化处理器"""
        self.point_cloud = None
        self.processed_cloud = None
        self.ground_plane = None
        self.ground_plane_mesh = None  # 地面平面网格（用于可视化）
        self.pile_clouds = []
        self.mesh = None          # 曲面重构结果
        self.processing_log = []

    def log(self, message: str):
        """记录处理日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(log_entry)

    def load_point_cloud(self, file_path: str) -> Dict:
        """加载点云文件"""
        self.log(f"正在加载点云文件: {file_path}")
        self.point_cloud = o3d.io.read_point_cloud(file_path)
        points = np.asarray(self.point_cloud.points)
        info = {
            "点数量": len(points),
            "是否有颜色": self.point_cloud.has_colors(),
            "是否有法向量": self.point_cloud.has_normals(),
            "边界框": {
                "最小值": points.min(axis=0).tolist(),
                "最大值": points.max(axis=0).tolist(),
                "尺寸": (points.max(axis=0) - points.min(axis=0)).tolist()
            }
        }
        self.log(f"点云加载成功: {info['点数量']} 个点")
        return info

    def remove_layered_points(self, xy_threshold: float = 0.001) -> Dict:
        """
        剔除分层叠加的点：只保留最外层（表面）的点

        策略：对于XY位置非常接近的点（<xy_threshold），只保留Z值最大的点（表面点）
        这样可以去除DUSt3R重建时产生的内部重叠点，保留煤堆表面形态

        Args:
            xy_threshold: XY平面距离阈值（米），小于此值的点被认为是同一位置
                         默认0.001m（1mm），只处理真正重叠的点

        Returns:
            剔除结果信息
        """
        self.log("开始剔除分层叠加点（保留最外层表面点）...")
        if self.point_cloud is None:
            raise ValueError("请先加载点云文件")

        points = np.asarray(self.point_cloud.points)
        original_count = len(points)

        # 使用网格化方法：将XY平面划分为小网格，每个网格只保留Z值最大的点
        # 这样可以保留表面形态，同时去除内部重叠点

        # 计算网格大小
        grid_size = xy_threshold
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)

        # 计算每个点所属的网格索引
        grid_x = ((points[:, 0] - x_min) / grid_size).astype(int)
        grid_y = ((points[:, 1] - y_min) / grid_size).astype(int)
        grid_keys = list(zip(grid_x, grid_y))

        # 对每个网格，只保留Z值最大的点
        from collections import defaultdict
        grid_dict = defaultdict(list)
        for idx, key in enumerate(grid_keys):
            grid_dict[key].append(idx)

        keep_indices = []
        for key, indices in grid_dict.items():
            if len(indices) == 1:
                keep_indices.append(indices[0])
            else:
                # 保留Z值最大的点（表面点）
                z_values = points[indices, 2]
                max_z_idx = indices[z_values.argmax()]
                keep_indices.append(max_z_idx)

        # 创建清理后的点云
        cleaned_cloud = self.point_cloud.select_by_index(keep_indices)
        self.point_cloud = cleaned_cloud

        result = {
            "原始点数": original_count,
            "剔除后点数": len(keep_indices),
            "剔除点数": original_count - len(keep_indices),
            "保留比例": f"{len(keep_indices)/original_count*100:.2f}%"
        }
        self.log(f"分层点剔除完成: 移除 {result['剔除点数']} 个内部重叠点，保留表面点")
        return result

    def preprocess_point_cloud(self,
                               voxel_size: float = 0.01,
                               nb_neighbors: int = 20,
                               std_ratio: float = 2.0,
                               remove_layers: bool = True) -> Dict:
        """
        点云预处理：分层点剔除 + 体素下采样 + 统计离群点去除 + 法向量估计

        Args:
            voxel_size: 体素下采样大小（米），建议煤堆使用0.002-0.005
            nb_neighbors: 统计离群点检测的邻居数量
            std_ratio: 标准差比率，越大越宽松（建议煤堆使用3.0-4.0）
            remove_layers: 是否先剔除分层叠加点
        """
        self.log("开始点云预处理...")
        if self.point_cloud is None:
            raise ValueError("请先加载点云文件")

        original_count = len(self.point_cloud.points)

        # 0. 剔除分层叠加点（可选）
        layer_result = None
        if remove_layers:
            # 使用更小的阈值，只处理真正重叠的点
            layer_result = self.remove_layered_points(xy_threshold=0.001)

        # 1. 体素下采样（对于稀疏点云，使用更小的体素）
        self.log(f"执行体素下采样 (voxel_size={voxel_size})...")
        downsampled = self.point_cloud.voxel_down_sample(voxel_size=voxel_size)
        after_downsample = len(downsampled.points)

        # 2. 统计离群点去除（对于煤堆，使用更宽松的标准）
        self.log(f"执行统计离群点去除 (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        cl, ind = downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        self.processed_cloud = downsampled.select_by_index(ind)
        after_outlier_removal = len(self.processed_cloud.points)

        # 3. 估计法向量
        self.log("估计点云法向量...")
        self.processed_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2,
                max_nn=30
            )
        )
        self.processed_cloud.orient_normals_consistent_tangent_plane(k=15)

        result = {
            "原始点数": original_count,
            "分层剔除": layer_result if layer_result else "未执行",
            "下采样后点数": after_downsample,
            "去除离群点后点数": after_outlier_removal,
            "保留比例": f"{after_outlier_removal/original_count*100:.2f}%"
        }
        self.log(f"预处理完成: 保留 {result['保留比例']} 的点")
        return result

    def merge_point_clouds(self, cloud_list: List[o3d.geometry.PointCloud],
                           voxel_size: float = 0.01) -> Dict:
        """
        点云拼接：基于局部特征匹配，将多个点云合并到同一坐标系
        适用于多次扫描同一料堆的场景

        Args:
            cloud_list: 待拼接的点云列表
            voxel_size: 拼接后的下采样体素大小

        Returns:
            拼接结果信息
        """
        self.log(f"开始点云拼接，共 {len(cloud_list)} 个点云...")

        if len(cloud_list) == 0:
            raise ValueError("点云列表为空")

        if len(cloud_list) == 1:
            self.processed_cloud = cloud_list[0]
            self.log("只有一个点云，无需拼接")
            return {"拼接点云数": 1, "总点数": len(self.processed_cloud.points)}

        # 以第一个点云为基准，逐步配准拼接
        merged = cloud_list[0]
        total_before = sum(len(c.points) for c in cloud_list)

        for i, cloud in enumerate(cloud_list[1:], 1):
            self.log(f"  配准第 {i+1} 个点云...")

            # 下采样用于特征提取
            src_down = cloud.voxel_down_sample(voxel_size)
            tgt_down = merged.voxel_down_sample(voxel_size)

            # 估计法向量
            for c in [src_down, tgt_down]:
                c.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 2, max_nn=30))

            # FPFH特征提取
            radius_feature = voxel_size * 5
            src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                src_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                tgt_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

            # RANSAC全局配准
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down, tgt_down, src_fpfh, tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=voxel_size * 2,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )

            # ICP精细配准
            result_icp = o3d.pipelines.registration.registration_icp(
                cloud, merged,
                max_correspondence_distance=voxel_size,
                init=result_ransac.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            cloud.transform(result_icp.transformation)
            merged = merged + cloud
            self.log(f"  第 {i+1} 个点云配准完成，fitness={result_icp.fitness:.3f}")

        # 合并后下采样去重
        merged = merged.voxel_down_sample(voxel_size)
        self.processed_cloud = merged

        result = {
            "拼接点云数": len(cloud_list),
            "拼接前总点数": total_before,
            "拼接后点数": len(merged.points)
        }
        self.log(f"点云拼接完成: {result['拼接后点数']} 个点")
        return result

    def segment_ground_plane(self,
                             distance_threshold: float = 0.02,
                             ransac_n: int = 3,
                             num_iterations: int = 1000,
                             keep_all_points: bool = True,
                             method: str = "max_cross_section") -> Dict:
        """
        识别地面平面

        核心约束：物料堆"顶端尖、地面宽"
        - 地面是横截面面积最大的平面
        - 地面上方包含所有点云

        Args:
            method:
                - "pile_aware"（强烈推荐，NEW！）: 基于物料堆真空特性
                - "csf": CSF布料模拟，适合复杂地形
                - "deterministic": 完全确定性算法，结果100%稳定
                - "max_cross_section": 基于最大横截面面积识别地面
                - "ransac": RANSAC找最大平面（不稳定，不推荐）
                - "convex_hull_base": 3D凸包底面
                - "adaptive": 结合横截面面积和密度双重约束
        """
        self.log(f"开始地面平面识别（方法: {method}）...")
        if self.processed_cloud is None:
            raise ValueError("请先执行预处理")

        points = np.asarray(self.processed_cloud.points)

        # 固定随机种子，确保RANSAC结果可复现
        np.random.seed(42)

        if method == "pile_aware":
            plane_model, inliers = self._fit_ground_pile_aware(
                self.processed_cloud, distance_threshold)
        elif method == "csf":
            plane_model, inliers = self._fit_ground_csf(
                self.processed_cloud, distance_threshold)
        elif method == "deterministic":
            plane_model, inliers = self._fit_ground_deterministic(
                self.processed_cloud, distance_threshold)
        elif method == "max_cross_section":
            plane_model, inliers = self._fit_ground_max_cross_section(
                self.processed_cloud, distance_threshold)
        elif method == "region_growing":
            plane_model, inliers = self._fit_ground_region_growing(
                self.processed_cloud, distance_threshold)
        elif method == "ransac":
            plane_model, inliers = self.processed_cloud.segment_plane(
                distance_threshold, ransac_n, num_iterations)
        elif method == "convex_hull_base":
            plane_model, inliers = self._fit_ground_from_convex_hull(
                self.processed_cloud)
        elif method == "adaptive":
            plane_model, inliers = self._fit_ground_adaptive(
                self.processed_cloud, distance_threshold)
        elif method == "lowest_points":
            plane_model, inliers = self._fit_ground_from_lowest_points(
                self.processed_cloud, distance_threshold)
        elif method == "coordinate_correction":
            plane_model, inliers = self._fit_ground_exclude_sides(
                self.processed_cloud, distance_threshold)
        elif method == "normal_filter":
            plane_model, inliers = self._fit_ground_with_coordinate_correction(
                self.processed_cloud, distance_threshold)
        else:
            raise ValueError(f"不支持的地面识别方法: {method}")

        [a, b, c, d] = plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        self.log(f"初始地面平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # 计算所有点到平面的有向距离
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm

        # 确保法向量方向朝上（大多数点在正侧）
        if (distances > 0).sum() < (distances < 0).sum():
            # 法向量方向反了，翻转
            a, b, c, d = -a, -b, -c, -d
            plane_model = np.array([a, b, c, d])
            distances = -distances
            self.log(f"  翻转法向量方向，确保大多数点在地面上方")

        # 将地面平移到所有点的下方（确保所有点在地面上方）
        min_dist = distances.min()
        if min_dist < 0.005:  # 如果有点低于地面5mm以内
            # 向下平移，使最低点距地面5mm
            shift = min_dist - 0.005  # 负值表示需要向下移
            d_new = d - shift * norm
            plane_model = np.array([a, b, c, d_new])
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d_new) / norm
            self.log(f"  向下平移 {abs(shift)*1000:.1f} mm，确保所有点在地面上方")

        [a, b, c, d] = plane_model
        self.ground_plane = plane_model

        # 计算地面Z高度用于日志
        center_xy = points[:, :2].mean(axis=0)
        if abs(c) > 1e-6:
            ground_z_at_center = -(a * center_xy[0] + b * center_xy[1] + d) / c
        else:
            ground_z_at_center = points[:, 2].min()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        ground_pct = (ground_z_at_center - z_min) / (z_max - z_min) * 100

        self.log(f"最终地面平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        self.log(f"地面位置: Z={ground_z_at_center:.4f}m ({ground_pct:.1f}%高度处)")
        self.log(f"距离范围: [{distances.min()*1000:.1f}, {distances.max()*1000:.1f}] mm")

        # 创建地面平面可视化网格
        self.ground_plane_mesh = self._create_ground_plane_mesh(plane_model)

        if keep_all_points:
            pile_cloud = self.processed_cloud
            pile_cloud.paint_uniform_color([0.8, 0.4, 0.2])
            result = {
                "地面点数": len(inliers),
                "料堆点数": len(self.processed_cloud.points),
                "保留所有点": True,
                "地面平面方程": plane_model.tolist(),
                "识别方法": method,
                "地面高度百分比": f"{ground_pct:.1f}%",
                "地面上方点数": int((distances > 0).sum()),
                "料堆占比": "100.00%（保留所有点）"
            }
            self.pile_clouds = [pile_cloud]
            self.log(f"地面识别完成: 地面在{ground_pct:.1f}%高度处，{result['地面上方点数']}个点在地面上方")
        else:
            ground_cloud = self.processed_cloud.select_by_index(inliers)
            pile_cloud = self.processed_cloud.select_by_index(inliers, invert=True)
            ground_cloud.paint_uniform_color([0.5, 0.5, 0.5])
            pile_cloud.paint_uniform_color([0.8, 0.4, 0.2])
            result = {
                "地面点数": len(inliers),
                "料堆点数": len(pile_cloud.points),
                "保留所有点": False,
                "地面平面方程": plane_model.tolist(),
                "识别方法": method,
                "地面高度百分比": f"{ground_pct:.1f}%",
                "地面上方点数": int((distances > 0).sum()),
                "料堆占比": f"{len(pile_cloud.points)/len(self.processed_cloud.points)*100:.2f}%"
            }
            self.pile_clouds = [pile_cloud]
            self.log(f"地面分割完成: 地面 {result['地面点数']} 点, 料堆 {result['料堆点数']} 点")

        return result

    def _fit_ground_deterministic(self, cloud: o3d.geometry.PointCloud,
                                   distance_threshold: float) -> tuple:
        """
        完全确定性的地面识别算法（推荐）

        核心原理：
        1. 按Z值排序，取最低的5-10%点（确定性操作）
        2. 使用最小二乘法拟合平面（确定性算法，无随机性）
        3. 找到距离平面小于阈值的所有点作为内点

        优势：
        - 完全确定性，每次运行结果完全一致
        - 不依赖随机采样，无RANSAC的不稳定性
        - 更快（无需迭代）
        - 更适合DUSt3R点云特性（地面点少，主要是表面点）

        适用场景：
        - 需要结果稳定可复现的生产环境
        - 地面点较少的点云（如DUSt3R重建结果）
        - 对算法速度有要求的场景
        """
        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        self.log(f"  点云Z范围: [{z_min:.4f}, {z_max:.4f}] m, 高度差: {z_range:.4f} m")

        # Step 1: 按Z值排序，取最低的8%点（确定性）
        z_sorted_indices = np.argsort(points[:, 2])
        bottom_percentile = 0.08  # 固定比例，确保确定性
        n_bottom = max(10, int(len(points) * bottom_percentile))
        bottom_indices = z_sorted_indices[:n_bottom]

        self.log(f"  取Z值最低的{bottom_percentile*100:.0f}%点: {n_bottom}个点")

        # Step 2: 使用最小二乘法拟合平面（确定性算法）
        bottom_points = points[bottom_indices]
        plane_model = self._fit_plane_least_squares(bottom_points)

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # 计算倾角
        z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.clip(np.abs(np.dot(normal, z_axis)), 0, 1))
        angle_deg = np.degrees(angle)

        self.log(f"  最小二乘法拟合平面: 倾角={angle_deg:.2f}°")

        # Step 3: 找到距离平面小于阈值的所有点（确定性）
        norm = np.sqrt(a**2 + b**2 + c**2)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
        inliers = np.where(distances < distance_threshold)[0]

        self.log(f"  地面内点数: {len(inliers)} ({len(inliers)/len(points)*100:.1f}%)")

        return plane_model, inliers.tolist()

    def _fit_plane_least_squares(self, points: np.ndarray) -> np.ndarray:
        """
        使用最小二乘法拟合平面（完全确定性）

        原理：
        1. 计算点云质心
        2. 对中心化后的点云进行SVD分解
        3. 最小奇异值对应的向量即为平面法向量

        Args:
            points: Nx3点云数组

        Returns:
            平面方程 [a, b, c, d]，表示 ax + by + cz + d = 0
        """
        # 计算质心
        centroid = points.mean(axis=0)

        # 中心化
        centered = points - centroid

        # SVD分解（确定性算法）
        _, _, vh = np.linalg.svd(centered)

        # 最小奇异值对应的向量即为平面法向量
        normal = vh[2, :]

        # 确保法向量向上（Z分量为正）
        if normal[2] < 0:
            normal = -normal

        # 归一化
        normal = normal / np.linalg.norm(normal)

        # 计算d: ax + by + cz + d = 0 => d = -(a*x0 + b*y0 + c*z0)
        d = -np.dot(normal, centroid)

        return np.array([normal[0], normal[1], normal[2], d])

    def _fit_ground_pile_aware(self, cloud: o3d.geometry.PointCloud,
                                distance_threshold: float) -> tuple:
        """
        基于物料堆特性的智能地面识别算法 - V4 全自动版

        设计理念：
        - 自动找平面：自动检测地面法向量，无需假设坐标系
        - 自动验证：多重验证机制，确保地面在底部而非顶部
        - 自动拟合：迭代优化，自适应参数，精确贴合点云

        算法流程：
        1. 自动检测地面方向（RANSAC + 方向验证）
        2. 自动定位地面位置（面积分析 + 多重验证）
        3. 自动精确拟合（迭代优化 + 质量评估）

        核心特性：
        - 无需人工调参，适应各种点云
        - 鲁棒性强，处理各种异常情况
        - 提供详细诊断信息，便于问题追踪

        适用场景：
        - DUSt3R生成的点云（地面点稀少）
        - 物料堆、煤堆、矿石堆等
        - 地面可以在任意方向
        """
        from scipy.spatial import ConvexHull

        points = np.asarray(cloud.points)
        total_points = len(points)

        self.log(f"  ═══ 智能地面识别算法 V4 ═══")
        self.log(f"  总点数: {total_points}")

        # ========== 阶段1: 自动检测地面方向 ==========
        self.log(f"\n  【阶段1】自动检测地面方向...")

        # 使用RANSAC检测主平面
        plane_model_ransac, inliers_ransac = cloud.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=1000
        )

        a, b, c, d = plane_model_ransac
        ground_normal = np.array([a, b, c])
        ground_normal = ground_normal / np.linalg.norm(ground_normal)

        self.log(f"  RANSAC检测到平面法向量: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")
        self.log(f"  RANSAC内点数: {len(inliers_ransac)} ({len(inliers_ransac)/total_points*100:.1f}%)")

        # 自动验证法向量方向（确保指向点云主体）
        signed_distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.linalg.norm([a, b, c])
        if np.median(signed_distances) < 0:
            ground_normal = -ground_normal
            a, b, c, d = -a, -b, -c, -d
            self.log(f"  自动翻转法向量（指向点云主体）")

        # ========== 阶段2: 自动定位地面位置 ==========
        self.log(f"\n  【阶段2】自动定位地面位置...")

        # 重新计算有向距离
        signed_distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.linalg.norm([a, b, c])
        dist_min, dist_max = signed_distances.min(), signed_distances.max()
        dist_range = dist_max - dist_min

        self.log(f"  点到平面的距离范围: [{dist_min:.4f}, {dist_max:.4f}] m (范围: {dist_range:.4f} m)")

        # 自适应确定测试点数量（根据点云大小和距离范围）
        num_tests = min(50, max(20, int(dist_range / 0.05)))  # 每5cm一个测试点
        test_distances = np.linspace(dist_min, dist_max, num_tests)

        self.log(f"  沿法向量方向测试{num_tests}个位置...")

        # 建立垂直于法向量的坐标系（用于投影）
        if abs(ground_normal[2]) < 0.9:
            v1 = np.cross(ground_normal, [0, 0, 1])
        else:
            v1 = np.cross(ground_normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(ground_normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        # 分析每个位置的凸包面积和点数
        areas_at_distances = []
        point_counts = []

        for test_dist in test_distances:
            below_plane_mask = signed_distances <= test_dist
            below_points = points[below_plane_mask]
            point_counts.append(len(below_points))

            if len(below_points) > 3:
                try:
                    # 投影到2D平面
                    projected_2d = np.column_stack([
                        np.dot(below_points, v1),
                        np.dot(below_points, v2)
                    ])
                    hull = ConvexHull(projected_2d)
                    areas_at_distances.append(hull.volume)
                except:
                    areas_at_distances.append(0)
            else:
                areas_at_distances.append(0)

        areas_array = np.array(areas_at_distances)
        point_counts_array = np.array(point_counts)

        # 自动验证：找到面积开始显著增加的位置
        valid_mask = areas_array > 0
        if np.sum(valid_mask) < 3:
            self.log(f"  ⚠️ 警告：有效测试点太少，使用RANSAC结果")
            return plane_model_ransac, inliers_ransac

        valid_distances = test_distances[valid_mask]
        valid_areas = areas_array[valid_mask]
        valid_counts = point_counts_array[valid_mask]

        # 计算面积变化率
        area_gradients = np.gradient(valid_areas)
        area_gradients_normalized = area_gradients / (valid_areas.max() + 1e-6)

        # 多重验证策略
        self.log(f"  应用多重验证策略...")

        # 策略1：找到面积梯度最大的位置（面积增长最快）
        max_gradient_idx = np.argmax(area_gradients)

        # 策略2：找到面积达到一定比例的位置（自适应阈值）
        max_area = valid_areas.max()
        # 自适应阈值：根据面积分布自动确定
        area_percentiles = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_candidates = []
        for percentile in area_percentiles:
            target_area = max_area * percentile
            above_threshold = valid_areas >= target_area
            if np.any(above_threshold):
                idx = np.where(above_threshold)[0][0]
                threshold_candidates.append(idx)

        # 策略3：找到点数达到一定比例的位置
        max_count = valid_counts.max()
        target_count = max_count * 0.15  # 15%的点数
        above_count_threshold = valid_counts >= target_count
        if np.any(above_count_threshold):
            count_threshold_idx = np.where(above_count_threshold)[0][0]
        else:
            count_threshold_idx = max_gradient_idx

        # 综合决策：选择最保守的位置（最靠近底部）
        all_candidates = [max_gradient_idx, count_threshold_idx] + threshold_candidates
        optimal_idx = max(all_candidates)  # 选择最大的索引（最靠近底部）
        optimal_distance = valid_distances[optimal_idx]

        self.log(f"  面积范围: [{valid_areas.min():.6f}, {valid_areas.max():.6f}]")
        self.log(f"  最大梯度位置: 距离={valid_distances[max_gradient_idx]:.4f}")
        self.log(f"  点数阈值位置: 距离={valid_distances[count_threshold_idx]:.4f}")
        self.log(f"  综合决策选择: 距离={optimal_distance:.4f}")

        # 自动验证：检查是否从顶部到底部（面积应该增加）
        area_trend = valid_areas[-1] - valid_areas[0]
        if area_trend > 0:
            self.log(f"  ✅ 验证通过：面积从{valid_areas[0]:.6f}增加到{valid_areas[-1]:.6f}（底部面积更大）")
        else:
            self.log(f"  ⚠️ 警告：面积趋势异常，可能需要调整")

        # ========== 阶段3: 自动精确拟合 ==========
        self.log(f"\n  【阶段3】自动精确拟合地面平面...")

        # 自适应确定候选点范围
        # 根据点云密度和距离范围自动调整margin
        adaptive_margin = max(distance_threshold * 3, dist_range * 0.02)  # 至少3倍阈值或2%范围

        # 迭代优化：多次拟合取最优结果
        best_plane = None
        best_score = float('inf')
        best_inliers = []

        # 尝试不同的margin值
        margin_candidates = [adaptive_margin * 0.5, adaptive_margin, adaptive_margin * 1.5]

        for margin in margin_candidates:
            # 选择候选点
            near_optimal_mask = (signed_distances >= optimal_distance - margin) & \
                               (signed_distances <= optimal_distance + margin)
            candidate_points = points[near_optimal_mask]

            if len(candidate_points) < 10:
                continue

            # 拟合平面
            try:
                plane_candidate = self._fit_plane_least_squares(candidate_points)
                a_c, b_c, c_c, d_c = plane_candidate

                # 计算拟合质量评分
                norm_c = np.sqrt(a_c**2 + b_c**2 + c_c**2)
                distances_c = np.abs(a_c * candidate_points[:, 0] +
                                    b_c * candidate_points[:, 1] +
                                    c_c * candidate_points[:, 2] + d_c) / norm_c

                # 评分：平均距离越小越好
                score = distances_c.mean()

                if score < best_score:
                    best_score = score
                    best_plane = plane_candidate
                    # 找到所有内点
                    all_distances = np.abs(a_c * points[:, 0] +
                                          b_c * points[:, 1] +
                                          c_c * points[:, 2] + d_c) / norm_c
                    best_inliers = np.where(all_distances < distance_threshold)[0].tolist()

            except:
                continue

        if best_plane is None:
            self.log(f"  ⚠️ 精确拟合失败，使用简单平移方法")
            # 使用简单平移
            d_new = d - optimal_distance * np.linalg.norm([a, b, c])
            best_plane = np.array([a, b, c, d_new])
            norm = np.sqrt(a**2 + b**2 + c**2)
            distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d_new) / norm
            best_inliers = np.where(distances < distance_threshold)[0].tolist()

        # 最终验证
        a_final, b_final, c_final, d_final = best_plane
        normal_final = np.array([a_final, b_final, c_final])
        normal_final = normal_final / np.linalg.norm(normal_final)

        # 检查法向量一致性
        dot_product = np.dot(ground_normal, normal_final)
        angle_diff = np.degrees(np.arccos(np.clip(np.abs(dot_product), 0, 1)))

        self.log(f"  最优拟合质量评分: {best_score:.6f}")
        self.log(f"  最终法向量: [{normal_final[0]:.3f}, {normal_final[1]:.3f}, {normal_final[2]:.3f}]")
        self.log(f"  法向量角度差异: {angle_diff:.2f}°")
        self.log(f"  地面内点数: {len(best_inliers)} ({len(best_inliers)/total_points*100:.1f}%)")

        if len(best_inliers) > 0:
            ground_points = points[best_inliers]
            norm_final = np.sqrt(a_final**2 + b_final**2 + c_final**2)
            ground_distances = np.abs(a_final * ground_points[:, 0] +
                                     b_final * ground_points[:, 1] +
                                     c_final * ground_points[:, 2] + d_final) / norm_final
            self.log(f"  地面点距离统计: 平均={ground_distances.mean():.4f}m, 最大={ground_distances.max():.4f}m")

        self.log(f"\n  ✅ 智能地面识别完成（全自动：检测+验证+拟合）")

        return best_plane, best_inliers

    def _fit_ground_csf(self, cloud: o3d.geometry.PointCloud,
                        distance_threshold: float) -> tuple:
        """
        使用CSF (Cloth Simulation Filter) 进行地面识别

        CSF是基于布料模拟的地面过滤算法，专门用于LiDAR点云地面分类。

        核心原理：
        1. 模拟一块布在重力作用下覆盖点云表面
        2. 布会自然地贴合地面，而不会陷入物体内部
        3. 通过布的最终位置识别地面点

        优势：
        - 适应复杂地形（倾斜、不规则、起伏地面）
        - 对点云不完整、地面点稀少有更好的容错性
        - 比RANSAC更鲁棒，不会误判侧面为地面
        - 参数可调，适应不同场景

        适用场景：
        - 煤堆、矿石堆等不规则地面
        - 倾斜地面、地形起伏
        - DUSt3R生成的稀疏地面点云

        参数说明：
        - cloth_resolution: 布料分辨率，越小越精细（0.1-1.0）
        - rigidness: 刚度，1=平坦地形，3=复杂地形
        - iterations: 迭代次数，越多越精确（500-1000）
        - class_threshold: 分类阈值，点到布的距离阈值
        """
        try:
            import CSF
        except ImportError:
            self.log("  ⚠️ CSF库未安装，请运行: pip install cloth-simulation-filter")
            self.log("  回退到deterministic方法")
            return self._fit_ground_deterministic(cloud, distance_threshold)

        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        self.log(f"  点云Z范围: [{z_min:.4f}, {z_max:.4f}] m, 高度差: {z_range:.4f} m")
        self.log(f"  使用CSF (Cloth Simulation Filter) 进行地面识别...")

        # 创建CSF对象
        csf = CSF.CSF()

        # 根据点云大小自适应调整参数
        # 计算点云XY范围
        xy_range = max(points[:, 0].max() - points[:, 0].min(),
                      points[:, 1].max() - points[:, 1].min())

        # 自适应布料分辨率（根据点云大小）
        cloth_resolution = max(0.05, min(0.5, xy_range / 10))

        # 设置参数（针对煤堆场景优化）
        csf.params.bSloopSmooth = False  # 不平滑斜坡（保留真实地形）
        csf.params.cloth_resolution = cloth_resolution  # 自适应布料分辨率
        csf.params.rigidness = 3  # 刚度=3，适合复杂地形（煤堆）
        csf.params.time_step = 0.65  # 时间步长
        csf.params.class_threshold = max(0.05, z_range * 0.05)  # 自适应分类阈值
        csf.params.interations = 500  # 迭代次数

        self.log(f"  CSF参数: 布料分辨率={csf.params.cloth_resolution:.3f}m, "
                f"刚度={csf.params.rigidness}, 分类阈值={csf.params.class_threshold:.3f}m, "
                f"迭代={csf.params.interations}次")

        # 设置点云
        csf.setPointCloud(points)

        # 执行地面过滤
        ground = CSF.VecInt()  # 地面点索引
        non_ground = CSF.VecInt()  # 非地面点索引
        csf.do_filtering(ground, non_ground)

        # 转换为numpy数组
        ground_indices = np.array(ground)
        non_ground_indices = np.array(non_ground)

        self.log(f"  CSF识别结果: 地面点={len(ground_indices)}, "
                f"非地面点={len(non_ground_indices)}")

        if len(ground_indices) < 3:
            self.log(f"  ⚠️ 地面点过少({len(ground_indices)})，回退到deterministic方法")
            return self._fit_ground_deterministic(cloud, distance_threshold)

        # 使用地面点拟合平面（最小二乘法）
        ground_points = points[ground_indices]
        plane_model = self._fit_plane_least_squares(ground_points)

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # 计算倾角
        z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.clip(np.abs(np.dot(normal, z_axis)), 0, 1))
        angle_deg = np.degrees(angle)

        self.log(f"  地面平面拟合: 倾角={angle_deg:.2f}°, 地面点数={len(ground_indices)}")

        # 如果倾角过大，说明CSF识别不准确，回退到deterministic方法
        if angle_deg > 15.0:
            self.log(f"  ⚠️ CSF识别的地面倾角过大({angle_deg:.2f}° > 15°)")
            self.log(f"  这可能是因为点云规模较小或地面点稀少")
            self.log(f"  回退到deterministic方法（更适合DUSt3R点云）")
            return self._fit_ground_deterministic(cloud, distance_threshold)

        return plane_model, ground_indices.tolist()

    def _fit_ground_max_cross_section(self, cloud: o3d.geometry.PointCloud,
                                       distance_threshold: float) -> tuple:
        """
        多候选平面 + 水平度优先的地面识别算法

        核心原理：
        - 地面必须是水平的（法向量接近[0,0,1]）
        - 用RANSAC迭代找多个候选平面（最多5个）
        - 综合评分：水平度60% + 内点数30% + 位置10%
        - 选择得分最高的平面作为地面

        为什么水平度优先：
        - 即使侧面的内点数更多，只要地面平面足够水平，也会被选中
        - 避免误判煤堆侧面为地面
        """
        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        # Step 1: 取底部30%的点
        bottom_percentile = 0.30
        z_threshold = z_min + z_range * bottom_percentile
        bottom_mask = points[:, 2] <= z_threshold
        bottom_indices = np.where(bottom_mask)[0]

        self.log(f"  底部{bottom_percentile*100:.0f}%点数: {len(bottom_indices)}")

        if len(bottom_indices) < 10:
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # Step 2: 迭代找多个候选平面
        bottom_cloud = cloud.select_by_index(bottom_indices.tolist())
        bottom_points = np.asarray(bottom_cloud.points)

        candidates = []
        remaining_indices = np.arange(len(bottom_points))

        for iteration in range(5):
            if len(remaining_indices) < 10:
                break

            # 从剩余点中找平面
            temp_cloud = o3d.geometry.PointCloud()
            temp_cloud.points = o3d.utility.Vector3dVector(bottom_points[remaining_indices])

            try:
                plane_model, inliers_temp = temp_cloud.segment_plane(
                    distance_threshold, 3, 2000)

                if len(inliers_temp) < 10:
                    break

                # 映射回bottom_points的索引
                inliers_in_remaining = remaining_indices[inliers_temp]

                # 计算平面特征
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                normal = normal / np.linalg.norm(normal)
                z_axis = np.array([0, 0, 1])
                angle = np.arccos(np.clip(np.abs(np.dot(normal, z_axis)), 0, 1))
                angle_deg = np.degrees(angle)

                # 计算平面平均高度
                plane_points = bottom_points[inliers_in_remaining]
                avg_z = np.mean(plane_points[:, 2])

                candidates.append({
                    'plane_model': plane_model,
                    'inliers': inliers_in_remaining,
                    'angle_deg': angle_deg,
                    'num_inliers': len(inliers_in_remaining),
                    'avg_z': avg_z
                })

                self.log(f"  候选平面{iteration+1}: 内点数={len(inliers_in_remaining)}, 倾角={angle_deg:.1f}°, 平均Z={avg_z:.3f}")

                # 移除这个平面的内点
                remaining_indices = np.setdiff1d(remaining_indices, inliers_in_remaining)

            except Exception as e:
                self.log(f"  候选平面{iteration+1}拟合失败: {e}")
                break

        if not candidates:
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # Step 3: 综合评分选择最佳平面
        best_candidate = None
        best_score = -1

        max_inliers = max(c['num_inliers'] for c in candidates)
        z_range_bottom = z_max - z_min if z_range > 0 else 1.0

        for i, candidate in enumerate(candidates):
            # 水平度得分（0-1，越接近水平越高）
            horizontal_score = max(0, 1 - candidate['angle_deg'] / 15.0)
            horizontal_score = min(1.0, horizontal_score)

            # 内点数得分（0-1，归一化）
            inlier_score = candidate['num_inliers'] / max_inliers if max_inliers > 0 else 0

            # 位置得分（0-1，越低越好）
            position_score = 1 - (candidate['avg_z'] - z_min) / z_range_bottom if z_range_bottom > 0 else 0
            position_score = max(0, min(1.0, position_score))

            # 综合评分：水平度60% + 内点数30% + 位置10%
            total_score = horizontal_score * 0.6 + inlier_score * 0.3 + position_score * 0.1

            self.log(f"  候选{i+1}评分: 水平度={horizontal_score:.3f}, 内点数={inlier_score:.3f}, 位置={position_score:.3f}, 总分={total_score:.3f}")

            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate

        self.log(f"  ✓ 选择最佳平面: 倾角={best_candidate['angle_deg']:.1f}°, 内点数={best_candidate['num_inliers']}, 总分={best_score:.3f}")

        # 映射回全局索引
        inliers_global = [bottom_indices[i] for i in best_candidate['inliers']]
        return best_candidate['plane_model'], inliers_global

    def _fit_ground_from_lowest_points(self, cloud: o3d.geometry.PointCloud,
                                        distance_threshold: float) -> tuple:
        """
        多候选平面 + 水平度优先的地面识别（底部10%点版本）

        策略：
        - 取底部10%点，迭代找多个候选平面
        - 综合评分：水平度60% + 内点数30% + 位置10%
        - 选择得分最高的平面
        """
        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        # 取底部10%的点
        bottom_percentile = 0.10
        z_threshold = z_min + z_range * bottom_percentile
        bottom_mask = points[:, 2] <= z_threshold
        bottom_indices = np.where(bottom_mask)[0]

        self.log(f"  底部{bottom_percentile*100:.0f}%点数: {len(bottom_indices)}")

        if len(bottom_indices) < 10:
            bottom_percentile = 0.20
            z_threshold = z_min + z_range * bottom_percentile
            bottom_mask = points[:, 2] <= z_threshold
            bottom_indices = np.where(bottom_mask)[0]
            self.log(f"  扩大到底部{bottom_percentile*100:.0f}%点数: {len(bottom_indices)}")

        if len(bottom_indices) < 10:
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # 迭代找多个候选平面
        bottom_cloud = cloud.select_by_index(bottom_indices.tolist())
        bottom_points = np.asarray(bottom_cloud.points)

        candidates = []
        remaining_indices = np.arange(len(bottom_points))

        for iteration in range(5):
            if len(remaining_indices) < 10:
                break

            temp_cloud = o3d.geometry.PointCloud()
            temp_cloud.points = o3d.utility.Vector3dVector(bottom_points[remaining_indices])

            try:
                plane_model, inliers_temp = temp_cloud.segment_plane(
                    distance_threshold, 3, 2000)

                if len(inliers_temp) < 10:
                    break

                inliers_in_remaining = remaining_indices[inliers_temp]

                # 计算平面特征
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                normal = normal / np.linalg.norm(normal)
                z_axis = np.array([0, 0, 1])
                angle = np.arccos(np.clip(np.abs(np.dot(normal, z_axis)), 0, 1))
                angle_deg = np.degrees(angle)

                plane_points = bottom_points[inliers_in_remaining]
                avg_z = np.mean(plane_points[:, 2])

                candidates.append({
                    'plane_model': plane_model,
                    'inliers': inliers_in_remaining,
                    'angle_deg': angle_deg,
                    'num_inliers': len(inliers_in_remaining),
                    'avg_z': avg_z
                })

                self.log(f"  候选平面{iteration+1}: 内点数={len(inliers_in_remaining)}, 倾角={angle_deg:.1f}°")

                remaining_indices = np.setdiff1d(remaining_indices, inliers_in_remaining)

            except Exception as e:
                break

        if not candidates:
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # 综合评分选择最佳平面
        best_candidate = None
        best_score = -1

        max_inliers = max(c['num_inliers'] for c in candidates)
        z_range_bottom = z_max - z_min if z_range > 0 else 1.0

        for i, candidate in enumerate(candidates):
            horizontal_score = max(0, 1 - candidate['angle_deg'] / 15.0)
            horizontal_score = min(1.0, horizontal_score)

            inlier_score = candidate['num_inliers'] / max_inliers if max_inliers > 0 else 0

            position_score = 1 - (candidate['avg_z'] - z_min) / z_range_bottom if z_range_bottom > 0 else 0
            position_score = max(0, min(1.0, position_score))

            total_score = horizontal_score * 0.6 + inlier_score * 0.3 + position_score * 0.1

            self.log(f"  候选{i+1}评分: 总分={total_score:.3f}")

            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate

        self.log(f"  ✓ 选择最佳平面: 倾角={best_candidate['angle_deg']:.1f}°, 总分={best_score:.3f}")

        inliers_global = [bottom_indices[i] for i in best_candidate['inliers']]
        return best_candidate['plane_model'], inliers_global
        return plane_model, inliers_global

    def _fit_ground_exclude_sides(self, cloud: o3d.geometry.PointCloud,
                                   distance_threshold: float) -> tuple:
        """
        物料堆地面识别：先排除侧面，再找地面

        核心洞察：
        - 物料堆的侧面通常是最大的平面（面积占60%以上）
        - 侧面是倾斜的（30-60°）
        - 地面是水平的（<10°），但点数很少

        策略：
        1. 使用RANSAC找到最大的平面
        2. 如果是倾斜的（>15°），认为是侧面，排除这些点
        3. 在剩余的点中，重复步骤1-2，直到找到水平平面
        4. 水平平面就是地面
        """
        points = np.asarray(cloud.points)
        self.log(f"  点云Z范围: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] m")
        self.log(f"  总点数: {len(points)}")

        remaining_cloud = cloud
        remaining_indices = np.arange(len(points))
        excluded_planes = []

        # 最多尝试5次，排除倾斜的平面
        for iteration in range(5):
            if len(remaining_indices) < 50:
                self.log(f"  剩余点数过少({len(remaining_indices)})，停止迭代")
                break

            # 在剩余点中找最大的平面
            try:
                plane_model, inliers_local = remaining_cloud.segment_plane(
                    distance_threshold, 3, 1000)
                a, b, c, d = plane_model

                # 计算倾角
                plane_normal = np.array([a, b, c])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                z_axis = np.array([0, 0, 1])
                angle = np.degrees(np.arccos(np.clip(abs(plane_normal @ z_axis), 0, 1)))

                # 映射到全局索引
                inliers_global = remaining_indices[inliers_local]
                inlier_ratio = len(inliers_local) / len(remaining_indices)

                self.log(f"  迭代{iteration+1}: 找到平面 {len(inliers_local)}点 "
                        f"({inlier_ratio*100:.1f}%), 倾角={angle:.1f}°")

                # 如果是水平平面（<15°），认为是地面
                if angle < 15.0:
                    self.log(f"  ✅ 找到水平平面（倾角{angle:.1f}° < 15°），认为是地面")

                    # 验证这个平面是否在底部
                    inlier_points = points[inliers_global]
                    z_median = np.median(inlier_points[:, 2])
                    z_min_all = points[:, 2].min()
                    z_max_all = points[:, 2].max()
                    z_position = (z_median - z_min_all) / (z_max_all - z_min_all)

                    self.log(f"  地面位置: Z中位数={z_median:.4f}, 相对位置={z_position*100:.1f}%")

                    if z_position < 0.3:  # 在底部30%范围内
                        self.log(f"  ✅ 地面在底部，验证通过")
                        return plane_model, inliers_global.tolist()
                    else:
                        self.log(f"  ⚠️ 平面不在底部（{z_position*100:.1f}% > 30%），继续寻找")

                # 如果是倾斜平面（>15°），认为是侧面，排除
                if angle >= 15.0 and inlier_ratio > 0.2:  # 至少占20%
                    self.log(f"  ⚠️ 倾斜平面（倾角{angle:.1f}° > 15°），可能是侧面，排除")
                    excluded_planes.append({
                        'plane': plane_model,
                        'angle': angle,
                        'points': len(inliers_local)
                    })

                    # 排除这些点，继续在剩余点中寻找
                    mask = np.ones(len(remaining_indices), dtype=bool)
                    mask[inliers_local] = False
                    remaining_indices = remaining_indices[mask]

                    remaining_points = points[remaining_indices]
                    remaining_cloud = o3d.geometry.PointCloud()
                    remaining_cloud.points = o3d.utility.Vector3dVector(remaining_points)

                    self.log(f"  剩余点数: {len(remaining_indices)} ({len(remaining_indices)/len(points)*100:.1f}%)")
                else:
                    # 平面太小或倾角不够大，停止迭代
                    self.log(f"  平面太小或倾角不够大，停止迭代")
                    break

            except Exception as e:
                self.log(f"  平面拟合失败: {e}")
                break

        # 如果没有找到水平平面，在剩余点中按Z值取底部点
        self.log(f"  未找到明显的水平平面，在剩余点中按Z值取底部")

        if len(remaining_indices) < 10:
            self.log(f"  ⚠️ 剩余点数过少({len(remaining_indices)})，使用全局底部点")
            # 在全局点云中取Z值最低的5-10%
            z_sorted_global = np.argsort(points[:, 2])
            n_bottom = max(50, int(len(points) * 0.05))
            bottom_indices = z_sorted_global[:n_bottom]

            bottom_cloud = cloud.select_by_index(bottom_indices.tolist())
            try:
                plane_model, inliers_local = bottom_cloud.segment_plane(
                    distance_threshold, 3, 1000)
                a, b, c, d = plane_model
                angle = np.degrees(np.arccos(abs(c)/np.sqrt(a**2+b**2+c**2)))

                inliers_global = bottom_indices[inliers_local]
                self.log(f"  全局底部平面: {len(inliers_global)}点, 倾角={angle:.1f}°")

                return plane_model, inliers_global.tolist()
            except:
                self.log(f"  ⚠️ 底部平面拟合失败，使用最低点拟合")
                # 直接用最低的点拟合平面
                return self._fit_plane_from_lowest_points(cloud, points, 0.05)

        # 在剩余点中，取Z值最低的10%
        remaining_points = points[remaining_indices]
        z_sorted = np.argsort(remaining_points[:, 2])
        n_bottom = max(10, int(len(remaining_points) * 0.1))
        bottom_local = z_sorted[:n_bottom]
        bottom_global = remaining_indices[bottom_local]

        bottom_cloud = cloud.select_by_index(bottom_global.tolist())
        try:
            plane_model, inliers_local = bottom_cloud.segment_plane(
                distance_threshold, 3, 1000)
            a, b, c, d = plane_model
            angle = np.degrees(np.arccos(abs(c)/np.sqrt(a**2+b**2+c**2)))

            inliers_global = bottom_global[inliers_local]
            self.log(f"  底部平面: {len(inliers_global)}点, 倾角={angle:.1f}°")

            return plane_model, inliers_global.tolist()
        except:
            self.log(f"  ⚠️ 底部平面拟合失败，回退到全局RANSAC")
            return cloud.segment_plane(distance_threshold, 3, 1000)

    def _fit_ground_with_coordinate_correction(self, cloud: o3d.geometry.PointCloud,
                                                distance_threshold: float) -> tuple:
        """
        物料堆专用地面识别：法线验证 + 多策略融合

        参考hdl_graph_slam的滤除法，针对物料堆场景优化：
        1. 法线预过滤：只保留法向量向上的点（地面特征）
        2. Z值分层：尝试不同高度范围的点
        3. RANSAC拟合：在过滤后的点上拟合平面
        4. 法线验证：确保平面法线与Z轴夹角<10°（水平度强约束）
        5. 多候选选择：综合水平度、覆盖率、中心性选择最佳平面

        适配物料堆特性：
        - DUSt3R点云主要是表面点，地面点极少
        - 不依赖传感器高度（无标定场景）
        - 利用法线信息提高鲁棒性
        """
        from scipy.spatial import ConvexHull

        points = np.asarray(cloud.points)
        self.log(f"  点云Z范围: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] m")

        # 确保点云有法向量
        if not cloud.has_normals():
            self.log("  计算点云法向量...")
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            cloud.orient_normals_consistent_tangent_plane(30)

        normals = np.asarray(cloud.normals)

        # 计算XY中心和总面积
        xy_center = points[:, :2].mean(axis=0)
        try:
            xy_hull_total = ConvexHull(points[:, :2])
            xy_area_total = xy_hull_total.volume
        except:
            xy_area_total = (points[:,0].max()-points[:,0].min()) * (points[:,1].max()-points[:,1].min())

        xy_span = np.sqrt((points[:,0].max()-points[:,0].min())**2 +
                         (points[:,1].max()-points[:,1].min())**2)

        # 第一步：法线预过滤 - 只保留法向量向上的点（地面特征）
        # 地面点的法向量应该向上（与Z轴夹角<60°）
        z_axis = np.array([0, 0, 1])
        normal_angles = np.degrees(np.arccos(np.clip(np.abs(normals @ z_axis), 0, 1)))
        upward_mask = normal_angles < 60.0  # 法向量与Z轴夹角<60°

        upward_indices = np.where(upward_mask)[0]
        self.log(f"  法线预过滤: {len(upward_indices)}/{len(points)} 点法向量向上（<60°）")

        if len(upward_indices) < 10:
            self.log("  ⚠️ 法向量向上的点太少，跳过法线过滤")
            upward_indices = np.arange(len(points))

        # 第二步：在法线过滤后的点中，按Z值分层尝试
        upward_points = points[upward_indices]
        z_sorted_local = np.argsort(upward_points[:, 2])

        candidates = []

        # 策略1：法线过滤 + Z值分层
        for percentile in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
            n_points = int(len(upward_points) * percentile)
            if n_points < 10:
                continue

            # 取法线向上的点中Z值最低的
            bottom_local_indices = z_sorted_local[:n_points]
            bottom_global_indices = upward_indices[bottom_local_indices]
            bottom_cloud = cloud.select_by_index(bottom_global_indices.tolist())

            try:
                plane_model, inliers_local = bottom_cloud.segment_plane(
                    distance_threshold, 3, 1000)
                a, b, c, d = plane_model

                # 验证平面法线与Z轴夹角（水平度强约束）
                plane_normal = np.array([a, b, c])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                angle = np.degrees(np.arccos(np.clip(abs(plane_normal @ z_axis), 0, 1)))

                # 只接受接近水平的平面（<10°）
                if angle > 10.0:
                    continue

                bottom_pts = points[bottom_global_indices]
                try:
                    bottom_hull = ConvexHull(bottom_pts[:, :2])
                    coverage = bottom_hull.volume / xy_area_total
                except:
                    coverage = 0.0

                bottom_center = bottom_pts[:, :2].mean(axis=0)
                offset_ratio = np.linalg.norm(bottom_center - xy_center) / xy_span

                # 计算综合得分
                horizontal_score = 1.0 - angle / 10.0  # 水平度得分
                coverage_score = min(1.0, coverage / 0.3)  # 覆盖率得分（降低阈值到30%）
                center_score = max(0, 1.0 - offset_ratio / 0.25)  # 中心性得分

                # 权重：水平度50% + 覆盖率30% + 中心性20%
                score = horizontal_score * 0.5 + coverage_score * 0.3 + center_score * 0.2

                candidates.append({
                    'method': f'法线过滤+底部{percentile*100:.0f}%',
                    'plane': plane_model,
                    'inliers': [int(bottom_global_indices[i]) for i in inliers_local],
                    'angle': angle,
                    'coverage': coverage,
                    'offset': offset_ratio,
                    'score': score,
                    'n_points': n_points
                })

                self.log(f"  候选{len(candidates)}: 法线过滤+底部{percentile*100:.0f}%({n_points}点), "
                        f"倾角={angle:.1f}°, 覆盖={coverage*100:.1f}%, 偏移={offset_ratio*100:.1f}%, 得分={score:.3f}")

            except:
                continue

        # 策略2：不使用法线过滤，直接按Z值（作为备选）
        z_sorted_indices = np.argsort(points[:, 2])
        for percentile in [0.05, 0.08, 0.10]:
            n_points = int(len(points) * percentile)
            if n_points < 10:
                continue

            bottom_indices = z_sorted_indices[:n_points]
            bottom_cloud = cloud.select_by_index(bottom_indices.tolist())

            try:
                plane_model, inliers_local = bottom_cloud.segment_plane(
                    distance_threshold, 3, 1000)
                a, b, c, d = plane_model

                plane_normal = np.array([a, b, c])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                angle = np.degrees(np.arccos(np.clip(abs(plane_normal @ z_axis), 0, 1)))

                if angle > 10.0:
                    continue

                bottom_pts = points[bottom_indices]
                try:
                    bottom_hull = ConvexHull(bottom_pts[:, :2])
                    coverage = bottom_hull.volume / xy_area_total
                except:
                    coverage = 0.0

                bottom_center = bottom_pts[:, :2].mean(axis=0)
                offset_ratio = np.linalg.norm(bottom_center - xy_center) / xy_span

                horizontal_score = 1.0 - angle / 10.0
                coverage_score = min(1.0, coverage / 0.3)
                center_score = max(0, 1.0 - offset_ratio / 0.25)
                score = horizontal_score * 0.5 + coverage_score * 0.3 + center_score * 0.2

                candidates.append({
                    'method': f'直接Z值+底部{percentile*100:.0f}%',
                    'plane': plane_model,
                    'inliers': [int(bottom_indices[i]) for i in inliers_local],
                    'angle': angle,
                    'coverage': coverage,
                    'offset': offset_ratio,
                    'score': score,
                    'n_points': n_points
                })

                self.log(f"  候选{len(candidates)}: 直接Z值+底部{percentile*100:.0f}%({n_points}点), "
                        f"倾角={angle:.1f}°, 覆盖={coverage*100:.1f}%, 偏移={offset_ratio*100:.1f}%, 得分={score:.3f}")

            except:
                continue

        # 如果没有找到水平的候选平面
        if not candidates:
            self.log("  ⚠️ 未找到水平平面（倾角<10°），回退到RANSAC")
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # 选择得分最高的候选
        best = max(candidates, key=lambda x: x['score'])

        self.log(f"  ✅ 选择最佳平面: {best['method']}, 倾角={best['angle']:.1f}°, "
                f"覆盖={best['coverage']*100:.1f}%, 偏移={best['offset']*100:.1f}%, 得分={best['score']:.3f}")

        return best['plane'], best['inliers']


    def _fit_ground_from_convex_hull(self, cloud: o3d.geometry.PointCloud) -> tuple:
        """从3D凸包的底面拟合地面"""
        points = np.asarray(cloud.points)
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(points)
            # 找到Z值最小的面作为底面
            min_z_face = None
            min_z = float('inf')
            for simplex in hull.simplices:
                face_points = points[simplex]
                face_z = face_points[:, 2].mean()
                if face_z < min_z:
                    min_z = face_z
                    min_z_face = simplex

            # 用底面的点拟合平面
            face_points = points[min_z_face]
            # 计算平面方程
            v1 = face_points[1] - face_points[0]
            v2 = face_points[2] - face_points[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, face_points[0])
            plane_model = np.array([normal[0], normal[1], normal[2], d])

            # 找到接近这个平面的点作为内点
            norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            distances = np.abs((normal[0] * points[:, 0] + normal[1] * points[:, 1] +
                               normal[2] * points[:, 2] + d) / norm)
            inliers = np.where(distances < 0.02)[0].tolist()

            return plane_model, inliers
        except Exception:
            # 降级到RANSAC
            return cloud.segment_plane(0.02, 3, 1000)

    def _fit_ground_adaptive(self, cloud: o3d.geometry.PointCloud,
                            distance_threshold: float) -> tuple:
        """
        自适应地面拟合：多层次底部点分析 + 水平度优先

        策略：
        1. 分别尝试底部10%、15%、20%、25%、30%的点
        2. 对每个范围用RANSAC迭代找多个候选平面
        3. 综合评分：水平度60% + 内点数30% + 位置10%
        4. 选择得分最高的平面
        """
        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        all_candidates = []

        # 尝试不同的底部百分比
        for percentile in [0.10, 0.15, 0.20, 0.25, 0.30]:
            z_threshold = z_min + z_range * percentile
            bottom_mask = points[:, 2] <= z_threshold
            bottom_indices = np.where(bottom_mask)[0]

            if len(bottom_indices) < 10:
                continue

            bottom_cloud = cloud.select_by_index(bottom_indices.tolist())
            bottom_points = np.asarray(bottom_cloud.points)

            # 对每个百分比范围，找多个候选平面
            remaining_indices = np.arange(len(bottom_points))

            for iteration in range(3):  # 每个范围最多找3个平面
                if len(remaining_indices) < 10:
                    break

                temp_cloud = o3d.geometry.PointCloud()
                temp_cloud.points = o3d.utility.Vector3dVector(bottom_points[remaining_indices])

                try:
                    plane_model, inliers_temp = temp_cloud.segment_plane(
                        distance_threshold, 3, 2000)

                    if len(inliers_temp) < 10:
                        break

                    inliers_in_remaining = remaining_indices[inliers_temp]

                    # 计算平面特征
                    a, b, c, d = plane_model
                    normal = np.array([a, b, c])
                    normal = normal / np.linalg.norm(normal)
                    z_axis = np.array([0, 0, 1])
                    angle = np.arccos(np.clip(np.abs(np.dot(normal, z_axis)), 0, 1))
                    angle_deg = np.degrees(angle)

                    plane_points = bottom_points[inliers_in_remaining]
                    avg_z = np.mean(plane_points[:, 2])

                    all_candidates.append({
                        'plane_model': plane_model,
                        'inliers': [bottom_indices[i] for i in inliers_in_remaining],
                        'angle_deg': angle_deg,
                        'num_inliers': len(inliers_in_remaining),
                        'avg_z': avg_z,
                        'percentile': percentile
                    })

                    remaining_indices = np.setdiff1d(remaining_indices, inliers_in_remaining)

                except Exception as e:
                    break

        if not all_candidates:
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # 综合评分选择最佳平面
        best_candidate = None
        best_score = -1

        max_inliers = max(c['num_inliers'] for c in all_candidates)
        z_range_total = z_max - z_min if z_range > 0 else 1.0

        for i, candidate in enumerate(all_candidates):
            # 水平度得分（0-1）
            horizontal_score = max(0, 1 - candidate['angle_deg'] / 15.0)
            horizontal_score = min(1.0, horizontal_score)

            # 内点数得分（0-1）
            inlier_score = candidate['num_inliers'] / max_inliers if max_inliers > 0 else 0

            # 位置得分（0-1，越低越好）
            position_score = 1 - (candidate['avg_z'] - z_min) / z_range_total if z_range_total > 0 else 0
            position_score = max(0, min(1.0, position_score))

            # 综合评分：水平度60% + 内点数30% + 位置10%
            total_score = horizontal_score * 0.6 + inlier_score * 0.3 + position_score * 0.1

            self.log(f"  底部{candidate['percentile']*100:.0f}%-候选{i+1}: 倾角={candidate['angle_deg']:.1f}°, 内点={candidate['num_inliers']}, 得分={total_score:.3f}")

            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate

        self.log(f"  ✓ 自适应方法选择最佳平面: 倾角={best_candidate['angle_deg']:.1f}°, 总分={best_score:.3f}")
        return best_candidate['plane_model'], best_candidate['inliers']

    def _fit_ground_region_growing(self, cloud: o3d.geometry.PointCloud,
                                    distance_threshold: float) -> tuple:
        """
        基于区域生长的地面分割（专为物料堆设计）

        算法：
        1. 取最低5-10%的点作为种子点
        2. 计算所有点的法向量
        3. 从种子点开始区域生长，法向量夹角<15-20度的点加入地面
        4. 煤堆侧面法向量突变，生长自然停止

        参数：
        - distance_threshold: RANSAC距离阈值（用于最后的平面拟合）

        返回：
        - plane_model: 地面平面方程
        - inliers_global: 地面点的全局索引
        """
        from scipy.spatial import cKDTree

        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        # 第一阶段：选择种子点（最低8%的点）
        seed_percentile = 0.08
        seed_threshold = z_min + z_range * seed_percentile
        seed_mask = points[:, 2] <= seed_threshold
        seed_indices = np.where(seed_mask)[0]

        self.log(f"  种子点数: {len(seed_indices)} (最低{seed_percentile*100:.0f}%)")

        if len(seed_indices) < 10:
            self.log(f"  种子点过少，回退到RANSAC方法")
            return cloud.segment_plane(distance_threshold, 3, 1000)

        # 第二阶段：计算所有点的法向量
        self.log(f"  计算法向量中...")
        if not cloud.has_normals():
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
            )
        normals = np.asarray(cloud.normals)

        # 确保法向量方向一致（朝上）
        for i in range(len(normals)):
            if normals[i, 2] < 0:
                normals[i] = -normals[i]

        # 第三阶段：区域生长
        self.log(f"  区域生长中...")

        # 构建KD树用于邻域搜索
        kdtree = cKDTree(points)

        # 初始化
        ground_set = set(seed_indices.tolist())
        frontier = list(seed_indices)
        visited = set(seed_indices.tolist())

        # 法向量夹角阈值（度）
        angle_threshold = 18.0

        # 区域生长主循环
        iteration = 0
        max_iterations = 100000  # 防止无限循环

        while frontier and iteration < max_iterations:
            current_idx = frontier.pop(0)
            current_normal = normals[current_idx]

            # 查找邻域点（k=20）
            try:
                distances, neighbors = kdtree.query(points[current_idx], k=21)
            except:
                continue

            for neighbor_idx in neighbors[1:]:  # 跳过自己
                if neighbor_idx >= len(points):
                    continue

                if neighbor_idx in visited:
                    continue

                visited.add(neighbor_idx)

                # 计算法向量夹角
                neighbor_normal = normals[neighbor_idx]
                cos_angle = np.dot(current_normal, neighbor_normal)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))

                # 如果夹角小于阈值，加入地面区域
                if angle_deg < angle_threshold:
                    ground_set.add(neighbor_idx)
                    frontier.append(neighbor_idx)

            iteration += 1

            # 每1000次迭代输出一次进度
            if iteration % 1000 == 0:
                self.log(f"    迭代 {iteration}: 地面点数={len(ground_set)}, 待处理={len(frontier)}")

        ground_indices = list(ground_set)
        self.log(f"  生长完成: 地面点数={len(ground_indices)}")
        self.log(f"  地面占比: {len(ground_indices)/len(points)*100:.1f}%")

        # 第四阶段：用RANSAC拟合平面（提高精度）
        if len(ground_indices) < 10:
            self.log(f"  地面点过少，回退到RANSAC方法")
            return cloud.segment_plane(distance_threshold, 3, 1000)

        ground_cloud = cloud.select_by_index(ground_indices)
        try:
            plane_model, inliers_local = ground_cloud.segment_plane(
                distance_threshold, 3, 1000)
            inliers_global = [ground_indices[i] for i in inliers_local]

            self.log(f"  平面拟合完成: 内点数={len(inliers_global)}")
            return plane_model, inliers_global
        except:
            self.log(f"  平面拟合失败，使用所有地面点")
            return cloud.segment_plane(distance_threshold, 3, 1000)

    def _create_ground_plane_mesh(self, plane_model: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        创建地面平面的可视化网格（半透明绿色平面）

        Args:
            plane_model: 平面方程 [a, b, c, d]，表示 ax + by + cz + d = 0

        Returns:
            地面平面网格
        """
        points = np.asarray(self.processed_cloud.points)

        # 获取点云的XY范围，扩展20%作为地面平面的范围
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.2
        x_max += x_range * 0.2
        y_min -= y_range * 0.2
        y_max += y_range * 0.2

        # 创建网格顶点（在XY平面上划分网格，Z值根据平面方程计算）
        a, b, c, d = plane_model
        nx, ny = 20, 20  # 网格分辨率

        vertices = []
        for i in range(nx):
            for j in range(ny):
                x = x_min + (x_max - x_min) * i / (nx - 1)
                y = y_min + (y_max - y_min) * j / (ny - 1)
                # 根据平面方程计算Z值: z = -(ax + by + d) / c
                if abs(c) > 1e-6:
                    z = -(a * x + b * y + d) / c
                else:
                    z = 0
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # 创建三角形面片
        triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                v0 = i * ny + j
                v1 = (i + 1) * ny + j
                v2 = (i + 1) * ny + (j + 1)
                v3 = i * ny + (j + 1)
                triangles.append([v0, v1, v2])
                triangles.append([v0, v2, v3])

        triangles = np.array(triangles)

        # 创建网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        # 设置颜色为半透明绿色
        mesh.paint_uniform_color([0.2, 0.8, 0.2])  # 绿色

        return mesh

    def refine_point_cloud(self,
                           pile_index: int = 0,
                           iterations: int = 3,
                           nb_neighbors: int = 30,
                           std_ratio: float = 1.5,
                           min_retention_ratio: float = 0.7,
                           skip_refine: bool = False) -> Dict:
        """
        细化处理：基于迭代统计滤波的鲁棒性评估

        Args:
            pile_index: 料堆索引
            iterations: 迭代次数
            nb_neighbors: 邻居数量
            std_ratio: 标准差比率（越小越严格）
            min_retention_ratio: 最小保留比例，低于此比例则停止迭代
            skip_refine: 是否跳过细化（保留所有点用于重构）

        Returns:
            细化结果信息
        """
        if pile_index >= len(self.pile_clouds):
            raise ValueError(f"料堆索引 {pile_index} 超出范围")

        cloud = self.pile_clouds[pile_index]
        original_count = len(cloud.points)

        if skip_refine:
            self.log(f"跳过细化处理，保留所有 {original_count} 个点用于曲面重构")
            # 仍然需要估计法向量
            if not cloud.has_normals():
                cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                )
                cloud.orient_normals_consistent_tangent_plane(k=15)

            result = {
                "料堆索引": pile_index,
                "细化前点数": original_count,
                "细化后点数": original_count,
                "去除点数": 0,
                "保留比例": "100.00%",
                "迭代次数": 0,
                "跳过细化": True
            }
            return result

        self.log(f"开始细化处理料堆 #{pile_index}（迭代 {iterations} 次）...")
        current_cloud = cloud

        for i in range(iterations):
            before = len(current_cloud.points)

            # 检查是否已经低于最小保留比例
            current_ratio = before / original_count
            if current_ratio < min_retention_ratio:
                self.log(f"  当前保留比例 {current_ratio*100:.1f}% 低于阈值 {min_retention_ratio*100:.1f}%，停止迭代")
                break

            # 统计离群点去除
            cl, ind = current_cloud.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            current_cloud = current_cloud.select_by_index(ind)
            after = len(current_cloud.points)
            removed = before - after
            self.log(f"  迭代 {i+1}: 去除 {removed} 个离群点，剩余 {after} 个点 ({after/original_count*100:.1f}%)")

            if removed == 0:
                self.log(f"  第 {i+1} 轮无变化，提前结束迭代")
                break

        # 更新料堆点云
        self.pile_clouds[pile_index] = current_cloud

        # 重新估计法向量
        current_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        current_cloud.orient_normals_consistent_tangent_plane(k=15)

        result = {
            "料堆索引": pile_index,
            "细化前点数": original_count,
            "细化后点数": len(current_cloud.points),
            "去除点数": original_count - len(current_cloud.points),
            "保留比例": f"{len(current_cloud.points)/original_count*100:.2f}%",
            "迭代次数": iterations,
            "跳过细化": False
        }
        self.log(f"细化完成: 保留 {result['保留比例']} 的点")
        return result

    def reconstruct_surface(self,
                            pile_index: int = 0,
                            method: str = "screened_poisson",
                            depth: int = 8,
                            alpha: float = None) -> Dict:
        """
        曲面重构：从点云重建三角网格曲面

        Args:
            pile_index: 料堆索引
            method: 重构方法
                - "convex_hull"（稀疏点云推荐）: 凸包法，100%封闭，适合极度稀疏点云
                - "convex_hull_shrink"（稀疏点云推荐）: 收缩包裹法，100%封闭，保留更多细节
                - "screened_poisson": Screened Poisson重建，类似CGAL泊松，最适合中等密度点云
                - "advancing_front": Advancing Front重建，类似CGAL Advancing Front，保留细节
                - "scale_space": Scale Space重建，多尺度融合，鲁棒性好
                - "poisson_enhanced": 增强泊松重构，更高质量，更多细节
                - "poisson": 标准泊松重构，适合中等密度点云
                - "bpa_enhanced": 增强BPA，更多半径层级，更好的孔洞填充
                - "bpa": 标准球旋转算法，适合密集点云
                - "alpha_shape": Alpha Shapes算法，适合不规则形状和稀疏点云
                - "pile_convex": 高度图重构（仅适合近似水平地面）
            depth: 泊松重构深度（默认8，范围6-10）
            alpha: Alpha Shapes参数（自动计算如果为None）

        注意：
        - 所有方法都使用在"地面识别"步骤中已确定的地面平面
        - 自动提取表面点，解决内部重叠点问题
        """
        self.log(f"开始曲面重构（方法: {method}）...")

        if pile_index >= len(self.pile_clouds):
            raise ValueError(f"料堆索引 {pile_index} 超出范围")

        cloud = self.pile_clouds[pile_index]
        points = np.asarray(cloud.points)

        # 确保有法向量，且法向量朝外
        if not cloud.has_normals():
            self.log("估计法向量...")
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=np.mean(cloud.compute_nearest_neighbor_distance()) * 3,
                    max_nn=30
                )
            )
        # 统一法向量朝向（朝向点云外侧）
        cloud.orient_normals_consistent_tangent_plane(k=15)

        if method == "screened_poisson":
            mesh = self._reconstruct_screened_poisson(cloud, depth)
        elif method == "advancing_front":
            mesh = self._reconstruct_advancing_front(cloud)
        elif method == "scale_space":
            mesh = self._reconstruct_scale_space(cloud)
        elif method == "poisson":
            mesh = self._reconstruct_poisson_with_base(cloud, depth)
        elif method == "poisson_enhanced":
            mesh = self._reconstruct_poisson_enhanced(cloud, depth)
        elif method == "bpa":
            # 使用原始BPA方法（用户测试后确认最适合）
            mesh = self._reconstruct_bpa_with_base(cloud)
        elif method == "bpa_cloth_draping":
            # 落布法BPA（实验性，可能过度平滑）
            mesh = self._reconstruct_bpa_cloth_draping(cloud)
        elif method == "bpa_enhanced":
            mesh = self._reconstruct_bpa_enhanced(cloud)
        elif method == "bpa_original":
            # 原始BPA方法（与bpa相同）
            mesh = self._reconstruct_bpa_with_base(cloud)
        elif method == "alpha_shape":
            mesh = self._reconstruct_alpha_shape(cloud, alpha)
        elif method == "pile_convex":
            mesh = self._reconstruct_pile_convex(cloud)
        elif method == "convex_hull":
            mesh = self._reconstruct_convex_hull(cloud)
        elif method == "convex_hull_shrink":
            mesh = self._reconstruct_convex_hull_shrink(cloud)
        else:
            raise ValueError(f"不支持的重构方法: {method}")

        # 后处理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        self.mesh = mesh

        result = {
            "料堆索引": pile_index,
            "重构方法": method,
            "顶点数": len(mesh.vertices),
            "三角面数": len(mesh.triangles),
            "是否水密": mesh.is_watertight()
        }
        self.log(f"曲面重构完成: {result['顶点数']} 顶点, {result['三角面数']} 三角面, 水密: {result['是否水密']}")
        return result

    def _get_ground_z(self, points: np.ndarray) -> float:
        """获取地面高度（基于地面平面方程或点云最低点）"""
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            return float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            return float(np.percentile(points[:, 2], 5))

    def _add_ground_base(self, mesh: o3d.geometry.TriangleMesh,
                         points: np.ndarray,
                         ground_z: float) -> o3d.geometry.TriangleMesh:
        """
        为曲面网格添加地面封底：
        1. 将网格底部顶点投影到地面高度
        2. 用Delaunay三角剖分生成底面
        3. 合并顶面和底面
        """
        from scipy.spatial import Delaunay

        # 找到网格底部边界（Z值接近地面的顶点）
        verts = np.asarray(mesh.vertices)
        z_range = verts[:, 2].max() - ground_z
        bottom_mask = verts[:, 2] < (ground_z + z_range * 0.15)
        bottom_verts = verts[bottom_mask]

        if len(bottom_verts) < 3:
            # 使用点云XY轮廓作为底面
            bottom_verts = points.copy()

        # 将底部顶点投影到地面
        bottom_proj = bottom_verts.copy()
        bottom_proj[:, 2] = ground_z

        # 用凸包生成底面轮廓
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(bottom_proj[:, :2])
            hull_pts = bottom_proj[hull.vertices]
        except Exception:
            hull_pts = bottom_proj

        # 生成底面三角形（扇形剖分）
        base_mesh = o3d.geometry.TriangleMesh()
        n = len(hull_pts)
        center = hull_pts.mean(axis=0)
        center[2] = ground_z

        base_verts = np.vstack([hull_pts, [center]])
        base_tris = []
        for i in range(n):
            base_tris.append([i, (i + 1) % n, n])  # 底面朝下

        base_mesh.vertices = o3d.utility.Vector3dVector(base_verts)
        base_mesh.triangles = o3d.utility.Vector3iVector(base_tris)

        # 合并顶面和底面
        combined = mesh + base_mesh
        return combined

    def _reconstruct_poisson_with_base(self, cloud: o3d.geometry.PointCloud,
                                        depth: int) -> o3d.geometry.TriangleMesh:
        """
        智能泊松重构 + 封闭网格生成（V2优化版）

        核心改进：
        1. 自动参数优化：根据点云特征自适应调整参数
        2. 智能密度裁剪：多阶段裁剪，保留真实表面，去除悬空面
        3. 孔洞填充：自动检测并填充小孔洞
        4. 地面封底：确保生成封闭的三维模型
        5. 表面平滑：保留细节的同时去除噪声

        解决问题：
        - BPA多余平面问题：使用Poisson重构，生成连续表面
        - 曲面不完整问题：智能密度裁剪 + 孔洞填充
        - 缺失纹理问题：保留原始点云颜色信息
        - 非封闭问题：添加地面封底，确保watertight
        """
        self.log(f"执行智能泊松曲面重构 (depth={depth})...")
        points = np.asarray(cloud.points)
        total_points = len(points)

        # 自动确定地面高度
        ground_z = self._get_ground_z(points)
        self.log(f"  地面高度: {ground_z:.4f}")

        # 阶段1：泊松重构
        self.log(f"  [阶段1] 泊松重构...")

        # 自适应参数：根据点云大小调整depth
        if total_points < 1000:
            depth = min(depth, 8)
        elif total_points < 5000:
            depth = min(depth, 9)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        self.log(f"  初始网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        # 阶段2：智能密度裁剪
        self.log(f"  [阶段2] 智能密度裁剪...")

        densities = np.asarray(densities)
        verts = np.asarray(mesh.vertices)

        # 策略1：去除极低密度区域（明显的悬空面）
        density_threshold_low = np.percentile(densities, 5)
        low_density_mask = densities < density_threshold_low

        # 策略2：去除远离点云的顶点
        # 计算每个顶点到最近点云点的距离
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances_to_cloud, _ = tree.query(verts)

        # 自适应距离阈值
        avg_nn_dist = np.mean(cloud.compute_nearest_neighbor_distance())
        distance_threshold = avg_nn_dist * 5  # 5倍平均最近邻距离

        far_from_cloud_mask = distances_to_cloud > distance_threshold

        # 策略3：去除地面以下的顶点
        below_ground_mask = verts[:, 2] < (ground_z - 0.01)

        # 综合裁剪
        vertices_to_remove = low_density_mask | far_from_cloud_mask | below_ground_mask

        removed_count = np.sum(vertices_to_remove)
        self.log(f"  裁剪策略:")
        self.log(f"    - 极低密度: {np.sum(low_density_mask)} 顶点")
        self.log(f"    - 远离点云: {np.sum(far_from_cloud_mask)} 顶点")
        self.log(f"    - 地面以下: {np.sum(below_ground_mask)} 顶点")
        self.log(f"    - 总计移除: {removed_count} 顶点")

        mesh.remove_vertices_by_mask(vertices_to_remove)
        self.log(f"  裁剪后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段3：孔洞填充（可选）
        # 注意：Open3D没有直接的孔洞填充功能，这里我们通过地面封底来确保封闭

        # 阶段4：地面封底（确保封闭）
        self.log(f"  [阶段3] 添加地面封底...")

        verts = np.asarray(mesh.vertices)

        # 找到底部边界点（Z值接近地面的点）
        margin = avg_nn_dist * 3
        bottom_mask = verts[:, 2] < (ground_z + margin)
        bottom_verts = verts[bottom_mask]

        if len(bottom_verts) > 3:
            # 计算底部点的2D凸包
            from scipy.spatial import ConvexHull
            try:
                hull_2d = ConvexHull(bottom_verts[:, :2])
                hull_indices = hull_2d.vertices

                # 创建地面封底网格
                hull_pts = bottom_verts[hull_indices]
                hull_pts[:, 2] = ground_z  # 投影到地面

                # 使用扇形三角化
                n = len(hull_pts)
                center = hull_pts.mean(axis=0)

                # 添加中心点和底面三角形
                base_verts = np.vstack([hull_pts, [center]])
                base_tris = []
                for i in range(n):
                    base_tris.append([i, (i + 1) % n, n])

                # 创建底面网格
                base_mesh = o3d.geometry.TriangleMesh()
                base_mesh.vertices = o3d.utility.Vector3dVector(base_verts)
                base_mesh.triangles = o3d.utility.Vector3iVector(base_tris)

                # 合并网格
                mesh = mesh + base_mesh

                self.log(f"  地面封底完成: 添加 {len(base_tris)} 个三角面")
            except Exception as e:
                self.log(f"  ⚠️ 地面封底失败: {e}")

        # 阶段5：网格优化
        self.log(f"  [阶段4] 网格优化...")

        # 移除重复顶点和退化三角形
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # 计算法向量
        mesh.compute_vertex_normals()

        # 可选：轻微平滑（保留细节）
        # mesh = mesh.filter_smooth_simple(number_of_iterations=1)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        # 统计信息
        verts_final = np.asarray(mesh.vertices)
        self.log(f"  网格范围:")
        self.log(f"    X: [{verts_final[:, 0].min():.4f}, {verts_final[:, 0].max():.4f}]")
        self.log(f"    Y: [{verts_final[:, 1].min():.4f}, {verts_final[:, 1].max():.4f}]")
        self.log(f"    Z: [{verts_final[:, 2].min():.4f}, {verts_final[:, 2].max():.4f}]")

        return mesh

    def _reconstruct_bpa_cloth_draping(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        改进的BPA重建：保留原始BPA的优势，增强孔洞填充

        核心策略：
        1. 使用原始BPA生成初始网格（保留细节和高度）
        2. 检测并填充孔洞
        3. 确保封闭性

        相比落布法：
        - 保留原始BPA的细节和高度
        - 只在必要时填充孔洞
        - 不使用高度图（避免过度平滑）
        """
        self.log("执行改进BPA重建（保留细节+填充孔洞）...")
        points = np.asarray(cloud.points)

        # 获取地面信息
        if self.ground_plane is None:
            ground_z = float(np.percentile(points[:, 2], 5))
            a, b, c, d = 0, 0, 1, -ground_z
        else:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)

        self.log(f"  地面高度: {ground_z:.4f}")

        # 阶段1：使用原始BPA生成初始网格
        self.log(f"  [阶段1] 原始BPA重建（保留细节）...")

        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)

        # 使用密集的半径层级
        radii = [
            min_dist * 0.6,
            min_dist * 0.8,
            min_dist * 1.0,
            avg_dist * 0.2,
            avg_dist * 0.3,
            avg_dist * 0.4,
            avg_dist * 0.5,
            avg_dist * 0.7,
            avg_dist * 1.0,
            avg_dist * 1.3,
            avg_dist * 1.5,
            avg_dist * 2.0,
            avg_dist * 2.5,
            avg_dist * 3.0,
            avg_dist * 4.0,
            avg_dist * 6.0,
            avg_dist * 8.0,
            avg_dist * 12.0,
            avg_dist * 16.0,
            avg_dist * 20.0
        ]

        self.log(f"    使用{len(radii)}个半径层级")

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii)
        )

        self.log(f"    初始BPA网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 连通性过滤
        if len(mesh.triangles) > 0:
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)

            if len(cluster_n_triangles) > 1:
                largest_cluster_idx = cluster_n_triangles.argmax()
                triangles_to_remove = triangle_clusters != largest_cluster_idx
                mesh.remove_triangles_by_mask(triangles_to_remove)
                mesh.remove_unreferenced_vertices()
                self.log(f"    移除小分量后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段2：专业孔洞填充（使用PyMeshLab）
        self.log(f"  [阶段2] 专业孔洞填充...")

        initial_triangles = len(mesh.triangles)

        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()

            # 转换为PyMeshLab格式
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            ms.add_mesh(pymeshlab.Mesh(verts, faces))

            # 检测孔洞数量
            ms.compute_selection_by_non_manifold_edges_per_face()
            self.log(f"    初始网格: {len(verts)} 顶点, {len(faces)} 三角面")

            # 关闭所有孔洞
            ms.meshing_close_holes(maxholesize=1000)
            self.log(f"    孔洞填充完成")

            # 平滑填充的孔洞区域
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, cotangentweight=False)
            self.log(f"    平滑完成")

            # 转换回Open3D格式
            m = ms.current_mesh()
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(m.vertex_matrix())
            mesh.triangles = o3d.utility.Vector3iVector(m.face_matrix())

            self.log(f"    填充后网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        except Exception as e:
            self.log(f"    ⚠️ PyMeshLab孔洞填充失败: {e}")
            self.log(f"    回退到基础清理")

        # 清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        # 阶段3：地面封底与侧面封闭
        self.log(f"  [阶段3] 地面封底与侧面封闭...")

        try:
            # 获取网格的实际边界边
            mesh_verts = np.asarray(mesh.vertices)
            mesh_tris = np.asarray(mesh.triangles)

            # 构建边到三角形的映射
            edge_count = {}
            for tri in mesh_tris:
                edges = [
                    tuple(sorted([tri[0], tri[1]])),
                    tuple(sorted([tri[1], tri[2]])),
                    tuple(sorted([tri[2], tri[0]]))
                ]
                for edge in edges:
                    edge_count[edge] = edge_count.get(edge, 0) + 1

            # 边界边：只被一个三角形使用的边
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

            if len(boundary_edges) == 0:
                self.log(f"    ⚠️ 未检测到边界边，跳过封底")
            else:
                self.log(f"    检测到{len(boundary_edges)}条边界边")

                # 构建边界顶点的有序链
                boundary_verts_ordered = []
                used_edges = set()
                current_edge = boundary_edges[0]
                boundary_verts_ordered.append(current_edge[0])
                boundary_verts_ordered.append(current_edge[1])
                used_edges.add(current_edge)

                while len(used_edges) < len(boundary_edges):
                    last_vert = boundary_verts_ordered[-1]
                    found = False
                    for edge in boundary_edges:
                        if edge in used_edges:
                            continue
                        if last_vert in edge:
                            next_vert = edge[1] if edge[0] == last_vert else edge[0]
                            if next_vert == boundary_verts_ordered[0]:
                                # 闭合了
                                used_edges.add(edge)
                                break
                            boundary_verts_ordered.append(next_vert)
                            used_edges.add(edge)
                            found = True
                            break
                    if not found:
                        break

                # 投影边界顶点到地面
                ground_normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
                boundary_verts_top = mesh_verts[boundary_verts_ordered]
                boundary_verts_bottom = []
                for pt in boundary_verts_top:
                    dist = (a * pt[0] + b * pt[1] + c * pt[2] + d) / (a**2 + b**2 + c**2)
                    projected_pt = pt - dist * ground_normal
                    boundary_verts_bottom.append(projected_pt)
                boundary_verts_bottom = np.array(boundary_verts_bottom)

                # 计算地面中心点
                center = boundary_verts_bottom.mean(axis=0)
                dist = (a * center[0] + b * center[1] + c * center[2] + d) / (a**2 + b**2 + c**2)
                center = center - dist * ground_normal

                # 构建封闭网格
                n_boundary = len(boundary_verts_ordered)
                n_mesh_verts = len(mesh_verts)

                # 新顶点：原网格顶点 + 地面边界顶点 + 地面中心点
                new_verts = np.vstack([
                    mesh_verts,                    # 0 ~ n_mesh_verts-1
                    boundary_verts_bottom,         # n_mesh_verts ~ n_mesh_verts+n_boundary-1
                    [center]                       # n_mesh_verts+n_boundary
                ])

                # 新三角面：原网格三角面 + 侧面 + 底面
                new_tris = list(mesh_tris)

                # 添加侧面（连接顶部边界和底部边界）
                for i in range(n_boundary):
                    top_idx1 = boundary_verts_ordered[i]
                    top_idx2 = boundary_verts_ordered[(i + 1) % n_boundary]
                    bottom_idx1 = n_mesh_verts + i
                    bottom_idx2 = n_mesh_verts + (i + 1) % n_boundary

                    # 两个三角形组成一个矩形侧面
                    new_tris.append([top_idx1, top_idx2, bottom_idx1])
                    new_tris.append([top_idx2, bottom_idx2, bottom_idx1])

                # 添加底面（扇形三角面）
                center_idx = n_mesh_verts + n_boundary
                for i in range(n_boundary):
                    bottom_idx1 = n_mesh_verts + i
                    bottom_idx2 = n_mesh_verts + (i + 1) % n_boundary
                    new_tris.append([bottom_idx1, center_idx, bottom_idx2])

                # 创建新网格
                closed_mesh = o3d.geometry.TriangleMesh()
                closed_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
                closed_mesh.triangles = o3d.utility.Vector3iVector(new_tris)

                mesh = closed_mesh

                self.log(f"    地面封底完成: 添加{n_boundary*3}个三角面（侧面+底面）")
                self.log(f"    边界顶点数: {n_boundary}")

        except Exception as e:
            import traceback
            self.log(f"    ⚠️ 地面封底失败: {e}")
            self.log(f"    {traceback.format_exc()}")

        # 阶段4：最终优化
        self.log(f"  [阶段4] 最终优化...")

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        # Taubin平滑（保留细节）
        mesh = mesh.filter_smooth_taubin(number_of_iterations=5, lambda_filter=0.5, mu=-0.53)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_bpa_with_base(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        智能BPA球旋转算法 + 封闭网格生成（V3优化版）

        核心改进：
        1. 使用地面识别的实际地面平面（不是简单的Z约束）
        2. 智能过滤多余平面（连通性分析）
        3. 曲面补全（多半径BPA）
        4. 地面封底（使用识别的地面平面）

        地面约束：
        - 使用self.ground_plane（地面识别算法的结果）
        - 地面平面方程：ax + by + cz + d = 0
        - 确保地面是识别到的实际平面，而不是水平面
        """
        self.log("执行智能BPA球旋转算法（使用识别的地面平面）...")
        points = np.asarray(cloud.points)
        total_points = len(points)

        # 检查是否有地面平面信息
        if self.ground_plane is None:
            self.log("  ⚠️ 警告：未找到地面平面信息，将使用简化方法")
            # 如果没有地面信息，使用简化的Z约束
            ground_z = float(np.percentile(points[:, 2], 5))
            ground_normal = np.array([0, 0, 1])
            use_ground_plane = False
        else:
            # 使用地面识别的结果
            a, b, c, d = self.ground_plane
            ground_normal = np.array([a, b, c])
            ground_normal = ground_normal / np.linalg.norm(ground_normal)

            # 计算地面参考高度（用于显示）
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)

            use_ground_plane = True
            self.log(f"  使用识别的地面平面: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            self.log(f"  地面法向量: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}]")
            self.log(f"  地面参考高度: {ground_z:.4f}")

        # 阶段1：多半径BPA重构（极致细节增强）
        self.log(f"  [阶段1] 多半径BPA重构（极致细节刻画）...")

        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)
        median_dist = np.median(distances)

        self.log(f"  点云密度分析:")
        self.log(f"    最小距离: {min_dist:.6f}")
        self.log(f"    中位距离: {median_dist:.6f}")
        self.log(f"    平均距离: {avg_dist:.6f}")

        # 使用更密集的半径层级，特别加强超细节层级
        # 策略：在小半径区间使用更密集的采样
        radii = [
            min_dist * 0.6,   # 超超细节（新增）
            min_dist * 0.8,   # 超细节
            min_dist * 1.0,   # 最小距离（新增）
            avg_dist * 0.2,   # 极细节（新增）
            avg_dist * 0.3,   # 极细节
            avg_dist * 0.4,   # 细节（新增）
            avg_dist * 0.5,   # 细节
            avg_dist * 0.7,   # 细节（新增）
            avg_dist * 1.0,   # 标准
            avg_dist * 1.3,   # 中等（新增）
            avg_dist * 1.5,   # 中等
            avg_dist * 2.0,   # 中等间隙
            avg_dist * 2.5,   # 大间隙（新增）
            avg_dist * 3.0,   # 大间隙
            avg_dist * 4.0,   # 大间隙
            avg_dist * 6.0,   # 超大间隙
            avg_dist * 8.0,   # 超大间隙
            avg_dist * 12.0,  # 极大间隙
            avg_dist * 16.0,  # 极大间隙
            avg_dist * 20.0   # 超极大间隙（新增）
        ]
        self.log(f"  使用{len(radii)}个半径层级（极致细节刻画）")
        self.log(f"  半径范围: [{radii[0]:.6f}, {radii[-1]:.6f}]")

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii)
        )

        self.log(f"  初始BPA网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段2：智能过滤多余平面
        self.log(f"  [阶段2] 智能过滤多余平面...")

        # 策略1：连通性分析
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) > 1:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx

            self.log(f"  检测到{len(cluster_n_triangles)}个连通分量")
            self.log(f"  最大分量: {cluster_n_triangles[largest_cluster_idx]} 个三角面")
            self.log(f"  移除{np.sum(triangles_to_remove)}个小分量（多余平面）")

            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()

        # 策略2：几何过滤
        verts = np.asarray(mesh.vertices)

        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances_to_cloud, _ = tree.query(verts)

        distance_threshold = avg_dist * 3
        far_from_cloud_mask = distances_to_cloud > distance_threshold

        if np.any(far_from_cloud_mask):
            self.log(f"  移除{np.sum(far_from_cloud_mask)}个远离点云的顶点")
            mesh.remove_vertices_by_mask(far_from_cloud_mask)

        # 策略3：使用地面平面过滤
        if use_ground_plane:
            verts = np.asarray(mesh.vertices)
            # 计算点到地面平面的有向距离
            signed_distances = (a * verts[:, 0] + b * verts[:, 1] + c * verts[:, 2] + d) / np.linalg.norm([a, b, c])

            # 移除地面以下的顶点（有向距离为负）
            below_ground_mask = signed_distances < -avg_dist

            if np.any(below_ground_mask):
                self.log(f"  移除{np.sum(below_ground_mask)}个地面以下的顶点（使用识别的地面平面）")
                mesh.remove_vertices_by_mask(below_ground_mask)
        else:
            # 简化方法：使用Z约束
            verts = np.asarray(mesh.vertices)
            below_ground_mask = verts[:, 2] < (ground_z - avg_dist)
            if np.any(below_ground_mask):
                self.log(f"  移除{np.sum(below_ground_mask)}个地面以下的顶点")
                mesh.remove_vertices_by_mask(below_ground_mask)

        self.log(f"  过滤后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段2.5：孔洞填充（解决镂空问题）
        self.log(f"  [阶段2.5] 孔洞检测与填充...")

        # 检测边界边（孔洞的边缘）
        mesh.compute_vertex_normals()

        # 统计孔洞信息
        # Open3D没有直接的孔洞检测API，我们通过边界边来估计
        # 一个封闭网格没有边界边，有边界边说明有孔洞

        # 简单的孔洞填充策略：使用网格细分和平滑
        initial_triangles = len(mesh.triangles)

        # 策略1：网格细分（增加三角面密度，有助于后续填充）
        # 注意：细分会增加顶点和三角面数量，改善细节
        if len(mesh.triangles) < 5000:  # 只对小网格进行细分
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            self.log(f"  网格细分: {initial_triangles} → {len(mesh.triangles)} 三角面")

        # 策略2：轻微平滑（填充小孔洞，保留大致形状）
        # 使用Laplacian平滑，迭代次数要少，避免过度平滑
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=2, lambda_filter=0.5)
        self.log(f"  Laplacian平滑完成（填充小孔洞）")

        # 策略3：移除退化三角形（清理细分和平滑产生的问题）
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

        self.log(f"  孔洞填充后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段3：地面封底（使用所有点云点投影到地面）
        self.log(f"  [阶段3] 添加地面封底（使用所有点云点投影）...")

        # 关键改进：使用所有原始点云点投影到地面，而不是只用网格底部点
        # 这样地面范围会覆盖整个点云的投影区域
        if use_ground_plane:
            # 将所有点云点投影到地面平面上
            self.log(f"  将{len(points)}个点云点投影到地面平面...")

            # 建立垂直于地面法向量的坐标系
            if abs(ground_normal[2]) < 0.9:
                v1 = np.cross(ground_normal, [0, 0, 1])
            else:
                v1 = np.cross(ground_normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(ground_normal, v1)
            v2 = v2 / np.linalg.norm(v2)

            # 将所有点云点投影到地面平面上
            projected_points = []
            for pt in points:
                # 点到平面的投影：p' = p - ((p·n + d) / ||n||²) * n
                dist = (a * pt[0] + b * pt[1] + c * pt[2] + d) / (a**2 + b**2 + c**2)
                projected_pt = pt - dist * ground_normal
                projected_points.append(projected_pt)

            projected_points = np.array(projected_points)

            # 在2D平面上计算凸包
            projected_2d = np.column_stack([
                np.dot(projected_points, v1),
                np.dot(projected_points, v2)
            ])

            try:
                from scipy.spatial import ConvexHull
                hull_2d = ConvexHull(projected_2d)
                hull_indices = hull_2d.vertices
                hull_pts = projected_points[hull_indices]

                self.log(f"  地面凸包: {len(hull_pts)} 个顶点")

                # 计算地面中心点（也在地面平面上）
                center = hull_pts.mean(axis=0)
                # 确保中心点也在地面平面上
                dist = (a * center[0] + b * center[1] + c * center[2] + d) / (a**2 + b**2 + c**2)
                center = center - dist * ground_normal

                # 创建底面网格（扇形三角化）
                n = len(hull_pts)
                base_verts = np.vstack([hull_pts, [center]])
                base_tris = []
                for i in range(n):
                    base_tris.append([i, (i + 1) % n, n])

                base_mesh = o3d.geometry.TriangleMesh()
                base_mesh.vertices = o3d.utility.Vector3dVector(base_verts)
                base_mesh.triangles = o3d.utility.Vector3iVector(base_tris)

                # 合并网格
                mesh = mesh + base_mesh

                self.log(f"  地面封底完成: 添加{len(base_tris)}个三角面")
                self.log(f"  地面使用所有点云点的投影（覆盖范围更大）")

            except Exception as e:
                self.log(f"  ⚠️ 地面封底失败: {e}")
                import traceback
                traceback.print_exc()

        else:
            # 简化方法：使用Z约束和XY投影
            self.log(f"  使用简化方法（水平面投影）...")

            # 将所有点投影到Z=ground_z平面
            projected_points = points.copy()
            projected_points[:, 2] = ground_z

            try:
                from scipy.spatial import ConvexHull
                hull_2d = ConvexHull(projected_points[:, :2])
                hull_indices = hull_2d.vertices
                hull_pts = projected_points[hull_indices]

                center = hull_pts.mean(axis=0)
                center[2] = ground_z

                n = len(hull_pts)
                base_verts = np.vstack([hull_pts, [center]])
                base_tris = []
                for i in range(n):
                    base_tris.append([i, (i + 1) % n, n])

                base_mesh = o3d.geometry.TriangleMesh()
                base_mesh.vertices = o3d.utility.Vector3dVector(base_verts)
                base_mesh.triangles = o3d.utility.Vector3iVector(base_tris)

                mesh = mesh + base_mesh

                self.log(f"  地面封底完成: 添加{len(base_tris)}个三角面")
                self.log(f"  地面使用水平面（Z={ground_z:.4f}）")

            except Exception as e:
                self.log(f"  ⚠️ 地面封底失败: {e}")
            try:
                from scipy.spatial import ConvexHull

                # 计算2D凸包（在垂直于地面法向量的平面上）
                if use_ground_plane:
                    # 建立垂直于地面法向量的坐标系
                    if abs(ground_normal[2]) < 0.9:
                        v1 = np.cross(ground_normal, [0, 0, 1])
                    else:
                        v1 = np.cross(ground_normal, [1, 0, 0])
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = np.cross(ground_normal, v1)
                    v2 = v2 / np.linalg.norm(v2)

                    # 投影到2D平面
                    projected_2d = np.column_stack([
                        np.dot(bottom_verts, v1),
                        np.dot(bottom_verts, v2)
                    ])

                    hull_2d = ConvexHull(projected_2d)
                    hull_indices = hull_2d.vertices
                    hull_pts = bottom_verts[hull_indices]

                    # 将凸包点投影到地面平面上
                    # 点到平面的投影：p' = p - ((p·n + d) / ||n||²) * n
                    for i in range(len(hull_pts)):
                        pt = hull_pts[i]
                        dist = (a * pt[0] + b * pt[1] + c * pt[2] + d) / (a**2 + b**2 + c**2)
                        hull_pts[i] = pt - dist * ground_normal

                    # 计算地面中心点（也在地面平面上）
                    center = hull_pts.mean(axis=0)
                    # 确保中心点也在地面平面上
                    dist = (a * center[0] + b * center[1] + c * center[2] + d) / (a**2 + b**2 + c**2)
                    center = center - dist * ground_normal

                else:
                    # 简化方法：XY平面凸包
                    hull_2d = ConvexHull(bottom_verts[:, :2])
                    hull_indices = hull_2d.vertices
                    hull_pts = bottom_verts[hull_indices]
                    hull_pts[:, 2] = ground_z

                    center = hull_pts.mean(axis=0)
                    center[2] = ground_z

                # 创建底面网格（扇形三角化）
                n = len(hull_pts)
                base_verts = np.vstack([hull_pts, [center]])
                base_tris = []
                for i in range(n):
                    base_tris.append([i, (i + 1) % n, n])

                base_mesh = o3d.geometry.TriangleMesh()
                base_mesh.vertices = o3d.utility.Vector3dVector(base_verts)
                base_mesh.triangles = o3d.utility.Vector3iVector(base_tris)

                # 合并网格
                mesh = mesh + base_mesh

                self.log(f"  地面封底完成: 添加{len(base_tris)}个三角面")

                if use_ground_plane:
                    self.log(f"  地面使用识别的平面方程（非水平面）")
                else:
                    self.log(f"  地面使用水平面（Z={ground_z:.4f}）")

            except Exception as e:
                self.log(f"  ⚠️ 地面封底失败: {e}")
                import traceback
                traceback.print_exc()

        # 阶段4：网格优化与细节增强
        self.log(f"  [阶段4] 网格优化与细节增强...")

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # 重新计算法向量（确保法向量一致性）
        mesh.compute_vertex_normals()

        # 可选：非常轻微的Taubin平滑（保留细节的同时去除噪声）
        # Taubin平滑比Laplacian平滑更好地保留细节
        mesh = mesh.filter_smooth_taubin(number_of_iterations=5, lambda_filter=0.5, mu=-0.53)
        self.log(f"  Taubin平滑完成（保留细节，去除噪声）")

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        # 统计信息
        verts_final = np.asarray(mesh.vertices)
        self.log(f"  网格范围:")
        self.log(f"    X: [{verts_final[:, 0].min():.4f}, {verts_final[:, 0].max():.4f}]")
        self.log(f"    Y: [{verts_final[:, 1].min():.4f}, {verts_final[:, 1].max():.4f}]")
        self.log(f"    Z: [{verts_final[:, 2].min():.4f}, {verts_final[:, 2].max():.4f}]")

        # 验证地面平面
        if use_ground_plane:
            # 重新计算最终顶点到地面平面的有向距离
            signed_distances_final = (a * verts_final[:, 0] + b * verts_final[:, 1] +
                                     c * verts_final[:, 2] + d) / np.linalg.norm([a, b, c])

            # 检查地面点是否在识别的平面上
            ground_verts_mask = signed_distances_final < avg_dist * 2
            if np.any(ground_verts_mask):
                ground_verts = verts_final[ground_verts_mask]
                ground_distances = np.abs(a * ground_verts[:, 0] + b * ground_verts[:, 1] +
                                         c * ground_verts[:, 2] + d) / np.linalg.norm([a, b, c])
                self.log(f"  地面点到识别平面的距离: 平均={ground_distances.mean():.6f}m, 最大={ground_distances.max():.6f}m")

        return mesh

    def _reconstruct_pile_convex(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        高度图重构：基于XY网格，每格取Z最大值生成表面
        注意：此方法假设地面近似水平，适合地面水平度>0.9的场景
        """
        self.log("执行高度图曲面重构...")
        points = np.asarray(cloud.points)
        ground_z = self._get_ground_z(points)

        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)

        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        grid_size = max(avg_dist * 2.0, (x_max - x_min) / 50)

        nx = max(int((x_max - x_min) / grid_size) + 1, 5)
        ny = max(int((y_max - y_min) / grid_size) + 1, 5)
        self.log(f"  网格分辨率: {nx}×{ny}，网格大小: {grid_size:.4f} m")

        # 构建高度图（每格取Z最大值）
        height_map = np.full((nx, ny), np.nan)
        for pt in points:
            ix = min(int((pt[0] - x_min) / grid_size), nx - 1)
            iy = min(int((pt[1] - y_min) / grid_size), ny - 1)
            if np.isnan(height_map[ix, iy]) or pt[2] > height_map[ix, iy]:
                height_map[ix, iy] = pt[2]

        # 填充空缺（用邻域均值，保留真实凹陷）
        from scipy.ndimage import generic_filter
        def fill_nan_mean(arr):
            valid = arr[~np.isnan(arr)]
            return np.mean(valid) if len(valid) > 0 else np.nan

        filled_map = height_map.copy()
        for _ in range(10):
            if not np.any(np.isnan(filled_map)):
                break
            new_map = generic_filter(filled_map, fill_nan_mean, size=3,
                                     mode='constant', cval=np.nan)
            mask = np.isnan(filled_map)
            filled_map[mask] = new_map[mask]

        filled_map[np.isnan(filled_map)] = ground_z
        filled_map = np.maximum(filled_map, ground_z)

        # 生成网格
        vertices, triangles = [], []
        def top_idx(ix, iy): return ix * ny + iy
        def bot_idx(ix, iy): return nx * ny + ix * ny + iy

        for ix in range(nx):
            for iy in range(ny):
                x = x_min + ix * grid_size
                y = y_min + iy * grid_size
                vertices.append([x, y, filled_map[ix, iy]])
        for ix in range(nx):
            for iy in range(ny):
                x = x_min + ix * grid_size
                y = y_min + iy * grid_size
                vertices.append([x, y, ground_z])

        for ix in range(nx - 1):
            for iy in range(ny - 1):
                v0, v1 = top_idx(ix, iy), top_idx(ix+1, iy)
                v2, v3 = top_idx(ix+1, iy+1), top_idx(ix, iy+1)
                triangles += [[v0, v1, v2], [v0, v2, v3]]
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                v0, v1 = bot_idx(ix, iy), bot_idx(ix+1, iy)
                v2, v3 = bot_idx(ix+1, iy+1), bot_idx(ix, iy+1)
                triangles += [[v0, v2, v1], [v0, v3, v2]]
        for ix in range(nx - 1):
            triangles += [
                [top_idx(ix,0), bot_idx(ix,0), bot_idx(ix+1,0)],
                [top_idx(ix,0), bot_idx(ix+1,0), top_idx(ix+1,0)],
                [top_idx(ix,ny-1), bot_idx(ix+1,ny-1), bot_idx(ix,ny-1)],
                [top_idx(ix,ny-1), top_idx(ix+1,ny-1), bot_idx(ix+1,ny-1)]
            ]
        for iy in range(ny - 1):
            triangles += [
                [top_idx(0,iy), bot_idx(0,iy+1), bot_idx(0,iy)],
                [top_idx(0,iy), top_idx(0,iy+1), bot_idx(0,iy+1)],
                [top_idx(nx-1,iy), bot_idx(nx-1,iy), bot_idx(nx-1,iy+1)],
                [top_idx(nx-1,iy), bot_idx(nx-1,iy+1), top_idx(nx-1,iy+1)]
            ]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.compute_vertex_normals()
        self.log(f"  高度图重构完成: {len(vertices)} 顶点, {len(triangles)} 三角面")
        return mesh

    def _reconstruct_poisson_enhanced(self, cloud: o3d.geometry.PointCloud, depth: int) -> o3d.geometry.TriangleMesh:
        """
        增强泊松重构：针对稀疏点云优化

        核心改进：
        1. 更激进的参数设置，适合稀疏点云
        2. 多尺度重构与融合
        3. 更智能的密度裁剪
        4. 网格细化与投影回点云
        5. 更好的孔洞填充

        适用场景：从图片重建的稀疏点云
        """
        self.log(f"执行增强泊松曲面重构 (depth={depth}, 针对稀疏点云优化)...")
        points = np.asarray(cloud.points)
        total_points = len(points)

        # 自动确定地面高度
        ground_z = self._get_ground_z(points)
        self.log(f"  地面高度: {ground_z:.4f}")

        # 阶段1：自适应参数优化
        self.log(f"  [阶段1] 自适应参数优化...")

        # 根据点云密度自适应调整depth
        avg_nn_dist = np.mean(cloud.compute_nearest_neighbor_distance())
        self.log(f"  平均最近邻距离: {avg_nn_dist:.6f}")

        # 稀疏点云使用较低的depth，避免过度平滑
        if total_points < 1000:
            depth = min(depth, 7)
            self.log(f"  点云较稀疏({total_points}点)，降低depth到{depth}")
        elif total_points < 5000:
            depth = min(depth, 8)

        # 阶段2：泊松重构（使用更宽松的参数）
        self.log(f"  [阶段2] 泊松重构（scale=1.2，更宽松的边界）...")

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth, width=0, scale=1.2, linear_fit=False
        )

        self.log(f"  初始网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段3：更智能的密度裁剪（保留更多细节）
        self.log(f"  [阶段3] 智能密度裁剪（保留细节）...")

        densities = np.asarray(densities)
        verts = np.asarray(mesh.vertices)

        # 使用更低的密度阈值，保留更多细节
        density_threshold = np.percentile(densities, 3)  # 从5%降到3%
        low_density_mask = densities < density_threshold

        # 计算到点云的距离
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances_to_cloud, _ = tree.query(verts)

        # 使用更宽松的距离阈值
        distance_threshold = avg_nn_dist * 8  # 从5倍增加到8倍
        far_from_cloud_mask = distances_to_cloud > distance_threshold

        # 地面以下的顶点
        below_ground_mask = verts[:, 2] < (ground_z - avg_nn_dist * 2)

        # 综合裁剪（更保守）
        vertices_to_remove = low_density_mask | far_from_cloud_mask | below_ground_mask

        self.log(f"  裁剪策略:")
        self.log(f"    - 极低密度: {np.sum(low_density_mask)} 顶点")
        self.log(f"    - 远离点云: {np.sum(far_from_cloud_mask)} 顶点")
        self.log(f"    - 地面以下: {np.sum(below_ground_mask)} 顶点")
        self.log(f"    - 总计移除: {np.sum(vertices_to_remove)} 顶点")

        mesh.remove_vertices_by_mask(vertices_to_remove)

        self.log(f"  裁剪后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段4：网格细化（增加三角面密度）
        self.log(f"  [阶段4] 网格细化...")

        if len(mesh.triangles) < 10000:  # 只对小网格进行细分
            initial_triangles = len(mesh.triangles)
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            self.log(f"  网格细分: {initial_triangles} → {len(mesh.triangles)} 三角面")

        # 阶段5：投影回点云（增强细节）
        self.log(f"  [阶段5] 投影回点云（增强细节）...")

        verts = np.asarray(mesh.vertices)
        # 对于接近点云的顶点，投影到最近的点云点
        distances_to_cloud, nearest_indices = tree.query(verts)

        # 只投影距离较近的顶点
        projection_threshold = avg_nn_dist * 3
        close_mask = distances_to_cloud < projection_threshold

        if np.any(close_mask):
            # 投影到最近的点云点
            verts[close_mask] = points[nearest_indices[close_mask]]
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            self.log(f"  投影{np.sum(close_mask)}个顶点到点云（增强细节）")

        # 阶段6：孔洞填充
        self.log(f"  [阶段6] 孔洞填充...")

        # 使用Laplacian平滑填充小孔洞
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.3)
        self.log(f"  Laplacian平滑完成")

        # 清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

        # 阶段7：地面封底
        self.log(f"  [阶段7] 地面封底...")
        mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 阶段8：最终优化
        self.log(f"  [阶段8] 最终优化...")

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()

        # 轻微Taubin平滑（保留细节）
        mesh = mesh.filter_smooth_taubin(number_of_iterations=3, lambda_filter=0.5, mu=-0.53)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_bpa_enhanced(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        增强BPA重构：更多半径层级，更好的孔洞填充

        核心改进：
        1. 更密集的半径采样（30个层级）
        2. 更激进的孔洞填充策略
        3. 网格细化与平滑
        4. 更好的地面封底
        """
        self.log("执行增强BPA球旋转算法...")
        points = np.asarray(cloud.points)

        # 检查地面平面
        if self.ground_plane is None:
            ground_z = float(np.percentile(points[:, 2], 5))
            use_ground_plane = False
        else:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
            use_ground_plane = True

        self.log(f"  地面高度: {ground_z:.4f}")

        # 阶段1：超密集多半径BPA重构
        self.log(f"  [阶段1] 超密集多半径BPA重构...")

        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        min_dist = np.min(distances)

        self.log(f"  点云密度: 最小={min_dist:.6f}, 平均={avg_dist:.6f}")

        # 使用30个半径层级，覆盖从超细节到超大间隙
        radii = []
        # 超细节层级（10个）
        for i in range(10):
            radii.append(min_dist * (0.5 + i * 0.1))
        # 细节层级（10个）
        for i in range(10):
            radii.append(avg_dist * (0.2 + i * 0.15))
        # 大间隙层级（10个）
        for i in range(10):
            radii.append(avg_dist * (2.0 + i * 2.0))

        self.log(f"  使用{len(radii)}个半径层级")
        self.log(f"  半径范围: [{radii[0]:.6f}, {radii[-1]:.6f}]")

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii)
        )

        self.log(f"  初始BPA网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段2：连通性过滤
        self.log(f"  [阶段2] 连通性过滤...")

        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) > 1:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            self.log(f"  移除{np.sum(triangles_to_remove)}个小分量")

        # 阶段3：激进的孔洞填充
        self.log(f"  [阶段3] 激进的孔洞填充...")

        initial_triangles = len(mesh.triangles)

        # 多次细分和平滑
        if len(mesh.triangles) < 8000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=2)
            self.log(f"  网格细分: {initial_triangles} → {len(mesh.triangles)} 三角面")

        # 多次Laplacian平滑
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.5)
        self.log(f"  Laplacian平滑完成（填充孔洞）")

        # 清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        # 阶段4：地面封底
        self.log(f"  [阶段4] 地面封底...")
        mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 阶段5：最终优化
        self.log(f"  [阶段5] 最终优化...")

        mesh.compute_vertex_normals()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=5, lambda_filter=0.5, mu=-0.53)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_alpha_shape(self, cloud: o3d.geometry.PointCloud, alpha: float = None) -> o3d.geometry.TriangleMesh:
        """
        Alpha Shapes重构：适合不规则形状和稀疏点云（改进版）

        改进点：
        1. 提取外部表面点，去除内部冗余点
        2. 使用地面识别的地面平面作为底部
        3. 改进封闭性（填充孔洞而不是替换算法）

        Args:
            cloud: 输入点云
            alpha: Alpha参数（自动计算如果为None）
        """
        self.log("执行Alpha Shapes曲面重构（改进版）...")
        points = np.asarray(cloud.points)

        # 阶段1：提取外部表面点（去除内部冗余点）
        self.log(f"  [阶段1] 提取外部表面点...")
        surface_cloud = self._extract_surface_points(cloud)
        surface_points = np.asarray(surface_cloud.points)
        self.log(f"  表面点提取: {len(points)} → {len(surface_points)} 点")

        # 阶段2：自动确定alpha参数
        if alpha is None:
            distances = surface_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            alpha = avg_dist * 2.5
            self.log(f"  自动计算alpha: {alpha:.6f}")
        else:
            self.log(f"  使用指定alpha: {alpha:.6f}")

        # 阶段3：Alpha Shape重构
        self.log(f"  [阶段3] Alpha Shape重构...")
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(surface_cloud, alpha)
            self.log(f"  Alpha Shape网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        except Exception as e:
            self.log(f"  ⚠️ Alpha Shape失败: {e}，回退到BPA")
            return self._reconstruct_bpa_with_base(surface_cloud)

        # 阶段4：清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        # 阶段5：使用地面平面封底
        ground_z = self._get_ground_z(surface_points)
        self.log(f"  [阶段5] 使用地面平面封底 (ground_z={ground_z:.4f})...")
        mesh = self._add_ground_base_enhanced(mesh, surface_points, ground_z)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _add_ground_base_enhanced(self, mesh: o3d.geometry.TriangleMesh,
                                   points: np.ndarray,
                                   ground_z: float = None) -> o3d.geometry.TriangleMesh:
        """
        增强的地面封底：使用地面平面方程投影所有点云点

        核心改进：
        1. 使用self.ground_plane平面方程投影点（不是简单的ground_z）
        2. 创建侧壁连接顶面和底面，确保水密
        3. 使用凸包生成底面
        """
        from scipy.spatial import ConvexHull

        # 获取地面平面参数
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            ground_normal = np.array([a, b, c])
            ground_normal = ground_normal / np.linalg.norm(ground_normal)
            self.log(f"  使用地面平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        else:
            # 回退到简单的水平面
            if ground_z is None:
                ground_z = float(np.percentile(points[:, 2], 5))
            a, b, c, d = 0, 0, 1, -ground_z
            ground_normal = np.array([0, 0, 1])
            self.log(f"  使用水平地面: z = {ground_z:.4f}")

        try:
            # 将所有点云点投影到地面平面
            projected_points = []
            for pt in points:
                # 点到平面的有向距离
                dist = (a * pt[0] + b * pt[1] + c * pt[2] + d) / np.sqrt(a**2 + b**2 + c**2)
                # 投影点 = 原点 - 距离 * 法向量
                projected_pt = pt - dist * ground_normal
                projected_points.append(projected_pt)

            projected_points = np.array(projected_points)

            # 建立2D坐标系（在地面平面上）
            if abs(ground_normal[2]) < 0.9:
                v1 = np.cross(ground_normal, [0, 0, 1])
            else:
                v1 = np.cross(ground_normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(ground_normal, v1)
            v2 = v2 / np.linalg.norm(v2)

            # 将投影点转换到2D坐标系
            projected_2d = np.column_stack([
                np.dot(projected_points, v1),
                np.dot(projected_points, v2)
            ])

            # 计算2D凸包
            hull_2d = ConvexHull(projected_2d)
            hull_indices = hull_2d.vertices
            hull_pts = projected_points[hull_indices]

            # 计算中心点（也在地面平面上）
            center = hull_pts.mean(axis=0)
            # 确保中心点在地面平面上
            dist = (a * center[0] + b * center[1] + c * center[2] + d) / np.sqrt(a**2 + b**2 + c**2)
            center = center - dist * ground_normal

            # 创建底面网格（扇形三角化）
            n = len(hull_pts)
            base_verts = np.vstack([hull_pts, [center]])
            base_tris = []
            for i in range(n):
                base_tris.append([i, (i + 1) % n, n])

            base_mesh = o3d.geometry.TriangleMesh()
            base_mesh.vertices = o3d.utility.Vector3dVector(base_verts)
            base_mesh.triangles = o3d.utility.Vector3iVector(base_tris)

            # 创建侧壁：连接顶面边界和底面边界
            mesh_verts = np.asarray(mesh.vertices)

            # 找到顶面网格的底部边界点（接近地面的点）
            distances_to_plane = np.abs(
                (a * mesh_verts[:, 0] + b * mesh_verts[:, 1] + c * mesh_verts[:, 2] + d) /
                np.sqrt(a**2 + b**2 + c**2)
            )
            z_range = mesh_verts[:, 2].max() - mesh_verts[:, 2].min()
            bottom_threshold = np.percentile(distances_to_plane, 20)  # 底部20%的点

            bottom_mask = distances_to_plane <= bottom_threshold
            bottom_verts_indices = np.where(bottom_mask)[0]

            if len(bottom_verts_indices) > 0:
                # 将底部顶点投影到地面平面
                bottom_verts = mesh_verts[bottom_verts_indices]
                bottom_projected = []
                for pt in bottom_verts:
                    dist = (a * pt[0] + b * pt[1] + c * pt[2] + d) / np.sqrt(a**2 + b**2 + c**2)
                    projected_pt = pt - dist * ground_normal
                    bottom_projected.append(projected_pt)
                bottom_projected = np.array(bottom_projected)

                # 创建侧壁三角形（连接原始底部点和投影点）
                side_verts = []
                side_tris = []
                base_idx = len(mesh_verts)

                for i, orig_idx in enumerate(bottom_verts_indices):
                    side_verts.append(bottom_projected[i])

                # 按角度排序底部点，形成环
                center_2d = bottom_projected.mean(axis=0)
                angles = np.arctan2(
                    np.dot(bottom_projected - center_2d, v2),
                    np.dot(bottom_projected - center_2d, v1)
                )
                sorted_indices = np.argsort(angles)

                # 创建侧壁三角形
                for i in range(len(sorted_indices)):
                    curr_orig = bottom_verts_indices[sorted_indices[i]]
                    next_orig = bottom_verts_indices[sorted_indices[(i + 1) % len(sorted_indices)]]
                    curr_proj = base_idx + sorted_indices[i]
                    next_proj = base_idx + sorted_indices[(i + 1) % len(sorted_indices)]

                    # 两个三角形形成一个四边形侧壁
                    side_tris.append([curr_orig, next_orig, next_proj])
                    side_tris.append([curr_orig, next_proj, curr_proj])

                if len(side_tris) > 0:
                    # 创建侧壁网格
                    side_mesh = o3d.geometry.TriangleMesh()
                    all_verts = np.vstack([mesh_verts, np.array(side_verts)])
                    side_mesh.vertices = o3d.utility.Vector3dVector(all_verts)
                    side_mesh.triangles = o3d.utility.Vector3iVector(
                        np.vstack([np.asarray(mesh.triangles), side_tris])
                    )

                    # 合并所有网格
                    mesh = side_mesh + base_mesh
                    self.log(f"  地面封底完成: 底面{len(base_tris)}个三角面, 侧壁{len(side_tris)}个三角面")
                else:
                    mesh = mesh + base_mesh
                    self.log(f"  地面封底完成: 添加{len(base_tris)}个三角面（无侧壁）")
            else:
                # 没有找到底部点，直接合并底面
                mesh = mesh + base_mesh
                self.log(f"  地面封底完成: 添加{len(base_tris)}个三角面")

        except Exception as e:
            self.log(f"  ⚠️ 地面封底失败: {e}")
            import traceback
            traceback.print_exc()

        return mesh

    def _extract_surface_points(self, cloud: o3d.geometry.PointCloud, radius_multiplier: float = 3.0) -> o3d.geometry.PointCloud:
        """
        提取表面点：解决内部重叠点问题

        策略：
        1. 计算每个点的局部密度
        2. 估计法向量
        3. 只保留表面点（最外圈）

        Args:
            cloud: 输入点云
            radius_multiplier: 搜索半径倍数

        Returns:
            表面点云
        """
        self.log("  [表面点提取] 解决内部重叠点问题...")

        points = np.asarray(cloud.points)
        n_points = len(points)

        # 计算平均最近邻距离
        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        search_radius = avg_dist * radius_multiplier

        self.log(f"    原始点数: {n_points}")
        self.log(f"    搜索半径: {search_radius:.6f}")

        # 确保有法向量
        if not cloud.has_normals():
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=search_radius, max_nn=30
                )
            )
            cloud.orient_normals_consistent_tangent_plane(30)

        normals = np.asarray(cloud.normals)

        # 方法1：基于法向量的表面点检测
        # 表面点的法向量应该指向外部，且周围点的法向量应该一致
        from scipy.spatial import cKDTree
        tree = cKDTree(points)

        surface_mask = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            # 查找邻域点
            indices = tree.query_ball_point(points[i], search_radius)
            if len(indices) < 5:
                continue

            # 计算邻域点的法向量一致性
            neighbor_normals = normals[indices]
            normal_consistency = np.abs(np.dot(neighbor_normals, normals[i])).mean()

            # 如果法向量不一致，可能是内部点
            if normal_consistency < 0.7:
                surface_mask[i] = False

        # 方法2：基于局部密度的表面点检测
        # 表面点的密度应该低于内部点
        local_densities = np.array([len(tree.query_ball_point(pt, search_radius)) for pt in points])
        density_threshold = np.percentile(local_densities, 70)  # 保留密度较低的70%

        low_density_mask = local_densities <= density_threshold

        # 综合两种方法
        final_surface_mask = surface_mask | low_density_mask

        # 提取表面点
        surface_cloud = cloud.select_by_index(np.where(final_surface_mask)[0].tolist())

        n_surface = len(surface_cloud.points)
        self.log(f"    表面点数: {n_surface} ({n_surface/n_points*100:.1f}%)")

        return surface_cloud

    def _reconstruct_screened_poisson(self, cloud: o3d.geometry.PointCloud, depth: int = 8) -> o3d.geometry.TriangleMesh:
        """
        Screened Poisson重建（PyMeshLab实现，类似CGAL泊松）

        核心优势：
        1. Screened Poisson比标准Poisson更适合有噪声的点云
        2. 能够更好地处理稀疏区域
        3. 生成更平滑、更完整的表面
        4. 使用已识别的地面平面

        适用场景：
        - 从图片重建的稀疏点云
        - 有内部重叠点的点云
        - 需要高质量表面的场景
        """
        self.log(f"执行Screened Poisson重建 (depth={depth}, PyMeshLab实现)...")

        try:
            import pymeshlab
        except ImportError:
            self.log("  ⚠️ PyMeshLab未安装，回退到标准泊松重建")
            return self._reconstruct_poisson_with_base(cloud, depth)

        points = np.asarray(cloud.points)

        # 阶段1：表面点提取
        self.log(f"  [阶段1] 表面点提取...")
        surface_cloud = self._extract_surface_points(cloud, radius_multiplier=3.0)
        surface_points = np.asarray(surface_cloud.points)

        # 确保有法向量
        if not surface_cloud.has_normals():
            avg_dist = np.mean(surface_cloud.compute_nearest_neighbor_distance())
            surface_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=avg_dist * 3, max_nn=30
                )
            )
            surface_cloud.orient_normals_consistent_tangent_plane(30)

        # 阶段2：使用PyMeshLab进行Screened Poisson重建
        self.log(f"  [阶段2] Screened Poisson重建...")

        # 创建MeshSet
        ms = pymeshlab.MeshSet()

        # 将点云转换为PyMeshLab格式
        vertex_matrix = surface_points
        normal_matrix = np.asarray(surface_cloud.normals)

        # 创建mesh（只有顶点和法向量）
        m = pymeshlab.Mesh(vertex_matrix=vertex_matrix, v_normals_matrix=normal_matrix)
        ms.add_mesh(m)

        self.log(f"    输入点数: {len(vertex_matrix)}")

        # 执行Screened Poisson重建
        # 参数说明：
        # - depth: 八叉树深度（8-10适合大多数场景）
        # - samplespernode: 每个节点的最小样本数（建议1.5）
        # - pointweight: 点权重（4.0适合有噪声的点云）
        ms.generate_surface_reconstruction_screened_poisson(
            depth=depth,
            samplespernode=1.5,
            pointweight=4.0,
            preclean=True
        )

        self.log(f"    重建网格: {ms.current_mesh().vertex_number()} 顶点, {ms.current_mesh().face_number()} 三角面")

        # 阶段3：密度裁剪（移除低密度区域）
        self.log(f"  [阶段3] 密度裁剪...")

        # 计算顶点到原始点云的距离
        mesh_vertices = ms.current_mesh().vertex_matrix()

        from scipy.spatial import cKDTree
        tree = cKDTree(surface_points)
        distances_to_cloud, _ = tree.query(mesh_vertices)

        avg_nn_dist = np.mean(surface_cloud.compute_nearest_neighbor_distance())
        distance_threshold = avg_nn_dist * 5

        # 标记要保留的顶点
        keep_mask = distances_to_cloud < distance_threshold

        # 转换为Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(ms.current_mesh().face_matrix())

        # 移除远离点云的顶点
        mesh.remove_vertices_by_mask(~keep_mask)

        self.log(f"    裁剪后: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

        # 阶段4：地面封底
        self.log(f"  [阶段4] 地面封底...")
        ground_z = self._get_ground_z(points)
        mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 阶段5：后处理
        self.log(f"  [阶段5] 后处理...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        # 轻微平滑
        mesh = mesh.filter_smooth_taubin(number_of_iterations=3, lambda_filter=0.5, mu=-0.53)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_advancing_front(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Advancing Front重建（PyMeshLab实现，类似CGAL Advancing Front）

        核心优势：
        1. 从点云边界开始，逐步向内推进
        2. 适合处理有边界的点云
        3. 能够保留更多细节
        4. 对稀疏点云鲁棒性好

        适用场景：
        - 有明确边界的点云
        - 需要保留细节的场景
        - 不规则形状的煤堆
        """
        self.log("执行Advancing Front重建 (PyMeshLab实现)...")

        try:
            import pymeshlab
        except ImportError:
            self.log("  ⚠️ PyMeshLab未安装，回退到BPA算法")
            return self._reconstruct_bpa_with_base(cloud)

        points = np.asarray(cloud.points)

        # 阶段1：表面点提取
        self.log(f"  [阶段1] 表面点提取...")
        surface_cloud = self._extract_surface_points(cloud, radius_multiplier=2.5)
        surface_points = np.asarray(surface_cloud.points)

        # 确保有法向量
        if not surface_cloud.has_normals():
            avg_dist = np.mean(surface_cloud.compute_nearest_neighbor_distance())
            surface_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=avg_dist * 3, max_nn=30
                )
            )
            surface_cloud.orient_normals_consistent_tangent_plane(30)

        # 阶段2：使用PyMeshLab进行Advancing Front重建
        self.log(f"  [阶段2] Advancing Front重建...")

        # 创建MeshSet
        ms = pymeshlab.MeshSet()

        # 将点云转换为PyMeshLab格式
        vertex_matrix = surface_points
        normal_matrix = np.asarray(surface_cloud.normals)

        m = pymeshlab.Mesh(vertex_matrix=vertex_matrix, v_normals_matrix=normal_matrix)
        ms.add_mesh(m)

        self.log(f"    输入点数: {len(vertex_matrix)}")

        # 执行Ball Pivoting（PyMeshLab的实现，类似Advancing Front）
        # 计算合适的半径
        distances = surface_cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)

        # 使用多个半径
        radii = [
            avg_dist * 0.5,
            avg_dist * 1.0,
            avg_dist * 1.5,
            avg_dist * 2.0,
            avg_dist * 3.0,
            avg_dist * 5.0
        ]

        self.log(f"    使用{len(radii)}个半径: [{radii[0]:.6f}, {radii[-1]:.6f}]")

        ms.generate_surface_reconstruction_ball_pivoting(
            ballradius=pymeshlab.AbsoluteValue(radii[0]),
            clustering=20.0,
            creasethr=90.0,
            deletefaces=True
        )

        # 如果第一次重建失败或结果太少，尝试更大的半径
        if ms.current_mesh().face_number() < 100:
            self.log(f"    初始重建结果较少，尝试更大半径...")
            for radius in radii[1:]:
                ms.generate_surface_reconstruction_ball_pivoting(
                    ballradius=pymeshlab.AbsoluteValue(radius),
                    clustering=20.0,
                    creasethr=90.0,
                    deletefaces=False  # 不删除已有的面
                )

        self.log(f"    重建网格: {ms.current_mesh().vertex_number()} 顶点, {ms.current_mesh().face_number()} 三角面")

        # 转换为Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(ms.current_mesh().vertex_matrix())
        mesh.triangles = o3d.utility.Vector3iVector(ms.current_mesh().face_matrix())

        # 阶段3：孔洞填充
        self.log(f"  [阶段3] 孔洞填充...")

        # 网格细分
        if len(mesh.triangles) < 5000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            self.log(f"    网格细分完成")

        # 平滑填充孔洞
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)

        # 清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        # 阶段4：地面封底
        self.log(f"  [阶段4] 地面封底...")
        ground_z = self._get_ground_z(points)
        mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 阶段5：最终优化
        self.log(f"  [阶段5] 最终优化...")
        mesh.compute_vertex_normals()
        mesh = mesh.filter_smooth_taubin(number_of_iterations=3, lambda_filter=0.5, mu=-0.53)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_scale_space(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Scale Space重建（基于多尺度分析）

        核心思想：
        1. 在多个尺度上分析点云
        2. 提取稳定的几何特征
        3. 融合多尺度信息

        实现策略：
        由于PyMeshLab没有直接的Scale Space实现，我们使用多尺度Poisson重建的融合

        适用场景：
        - 点云密度不均匀
        - 需要鲁棒性的场景
        - 复杂形状的煤堆
        """
        self.log("执行Scale Space重建 (多尺度融合策略)...")

        points = np.asarray(cloud.points)

        # 阶段1：表面点提取
        self.log(f"  [阶段1] 表面点提取...")
        surface_cloud = self._extract_surface_points(cloud, radius_multiplier=3.0)

        # 阶段2：多尺度重建
        self.log(f"  [阶段2] 多尺度重建...")

        # 在不同depth上进行泊松重建
        depths = [6, 7, 8]
        meshes = []

        for depth in depths:
            self.log(f"    尺度{depth}重建...")
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    surface_cloud, depth=depth, width=0, scale=1.2, linear_fit=False
                )

                # 简单的密度裁剪
                densities = np.asarray(densities)
                verts = np.asarray(mesh.vertices)

                density_threshold = np.percentile(densities, 5)
                keep_mask = densities >= density_threshold

                mesh.remove_vertices_by_mask(~keep_mask)

                meshes.append(mesh)
                self.log(f"      {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
            except Exception as e:
                self.log(f"      ⚠️ 尺度{depth}重建失败: {e}")

        if not meshes:
            self.log("  ⚠️ 所有尺度重建失败，回退到标准泊松")
            return self._reconstruct_poisson_with_base(surface_cloud, 7)

        # 阶段3：选择最佳尺度（选择三角面数适中的）
        self.log(f"  [阶段3] 选择最佳尺度...")

        # 选择三角面数在中位数附近的mesh
        face_counts = [len(m.triangles) for m in meshes]
        median_faces = np.median(face_counts)

        best_idx = np.argmin([abs(fc - median_faces) for fc in face_counts])
        mesh = meshes[best_idx]

        self.log(f"    选择尺度{depths[best_idx]}的结果")

        # 阶段4：地面封底
        self.log(f"  [阶段4] 地面封底...")
        ground_z = self._get_ground_z(points)
        mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 阶段5：后处理
        self.log(f"  [阶段5] 后处理...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        # 平滑
        mesh = mesh.filter_smooth_taubin(number_of_iterations=5, lambda_filter=0.5, mu=-0.53)

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_pile_convex_old(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """旧版堆料凸包重构（已废弃，保留备用）"""
        self.log("执行堆料凸包重构（垂直落布法）...")

        points = np.asarray(cloud.points)

        # 1. 确定地面高度
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            ground_z = float(points[:, 2].min())

        self.log(f"  地面高度: {ground_z:.4f} m")

        # 2. 将XY平面划分为网格，每格取Z最大值（表面点）
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)

        # 网格分辨率：根据点云密度自适应
        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        grid_size = max(avg_dist * 1.5, (x_max - x_min) / 60)

        nx = max(int((x_max - x_min) / grid_size) + 1, 5)
        ny = max(int((y_max - y_min) / grid_size) + 1, 5)
        self.log(f"  网格分辨率: {nx}×{ny}，网格大小: {grid_size:.4f} m")

        # 3. 构建高度图（每个网格取Z最大值）
        height_map = np.full((nx, ny), np.nan)
        for pt in points:
            ix = min(int((pt[0] - x_min) / grid_size), nx - 1)
            iy = min(int((pt[1] - y_min) / grid_size), ny - 1)
            if np.isnan(height_map[ix, iy]) or pt[2] > height_map[ix, iy]:
                height_map[ix, iy] = pt[2]

        # 4. 填充空缺网格（插值）：避免凹陷
        from scipy.ndimage import generic_filter
        def fill_nan(arr):
            # 用邻域最大值填充NaN（保守估计，往大估算）
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                return np.max(valid)
            return np.nan

        # 迭代填充，直到没有NaN
        filled_map = height_map.copy()
        for _ in range(10):
            if not np.any(np.isnan(filled_map)):
                break
            new_map = generic_filter(filled_map, fill_nan, size=3, mode='constant', cval=np.nan)
            mask = np.isnan(filled_map)
            filled_map[mask] = new_map[mask]

        # 仍有NaN的格子设为地面高度
        filled_map[np.isnan(filled_map)] = ground_z

        # 确保所有高度不低于地面
        filled_map = np.maximum(filled_map, ground_z)

        self.log(f"  高度图构建完成，最大高度: {filled_map.max():.4f} m，最小高度: {filled_map.min():.4f} m")

        # 5. 从高度图生成三角网格
        vertices = []
        # 顶面顶点
        for ix in range(nx):
            for iy in range(ny):
                x = x_min + ix * grid_size
                y = y_min + iy * grid_size
                z = filled_map[ix, iy]
                vertices.append([x, y, z])

        # 底面顶点（地面高度）
        bottom_offset = nx * ny
        for ix in range(nx):
            for iy in range(ny):
                x = x_min + ix * grid_size
                y = y_min + iy * grid_size
                vertices.append([x, y, ground_z])

        vertices = np.array(vertices)
        triangles = []

        def top_idx(ix, iy): return ix * ny + iy
        def bot_idx(ix, iy): return bottom_offset + ix * ny + iy

        # 顶面三角形
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                v0 = top_idx(ix, iy)
                v1 = top_idx(ix + 1, iy)
                v2 = top_idx(ix + 1, iy + 1)
                v3 = top_idx(ix, iy + 1)
                triangles.append([v0, v1, v2])
                triangles.append([v0, v2, v3])

        # 底面三角形（法向量朝下）
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                v0 = bot_idx(ix, iy)
                v1 = bot_idx(ix + 1, iy)
                v2 = bot_idx(ix + 1, iy + 1)
                v3 = bot_idx(ix, iy + 1)
                triangles.append([v0, v2, v1])
                triangles.append([v0, v3, v2])

        # 四周侧面
        for ix in range(nx - 1):
            # 前边 (iy=0)
            triangles.append([top_idx(ix, 0), bot_idx(ix, 0), bot_idx(ix+1, 0)])
            triangles.append([top_idx(ix, 0), bot_idx(ix+1, 0), top_idx(ix+1, 0)])
            # 后边 (iy=ny-1)
            triangles.append([top_idx(ix, ny-1), bot_idx(ix+1, ny-1), bot_idx(ix, ny-1)])
            triangles.append([top_idx(ix, ny-1), top_idx(ix+1, ny-1), bot_idx(ix+1, ny-1)])

        for iy in range(ny - 1):
            # 左边 (ix=0)
            triangles.append([top_idx(0, iy), bot_idx(0, iy+1), bot_idx(0, iy)])
            triangles.append([top_idx(0, iy), top_idx(0, iy+1), bot_idx(0, iy+1)])
            # 右边 (ix=nx-1)
            triangles.append([top_idx(nx-1, iy), bot_idx(nx-1, iy), bot_idx(nx-1, iy+1)])
            triangles.append([top_idx(nx-1, iy), bot_idx(nx-1, iy+1), top_idx(nx-1, iy+1)])

        triangles = np.array(triangles)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        self.log(f"  堆料凸包重构完成: {len(vertices)} 顶点, {len(triangles)} 三角面, 水密: {mesh.is_watertight()}")
        return mesh

    def calculate_boundary(self, pile_index: int = 0) -> Dict:
        """
        计算料堆边界：包括2D投影轮廓和3D边界框

        Args:
            pile_index: 料堆索引

        Returns:
            边界信息字典
        """
        self.log(f"计算料堆 #{pile_index} 边界...")

        if pile_index >= len(self.pile_clouds):
            raise ValueError(f"料堆索引 {pile_index} 超出范围")

        cloud = self.pile_clouds[pile_index]
        points = np.asarray(cloud.points)

        # 3D轴对齐边界框
        bbox = cloud.get_axis_aligned_bounding_box()
        # 3D有向边界框（更紧凑）
        obb = cloud.get_oriented_bounding_box()

        # 计算点到地面平面的距离
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            # 计算所有点到地面平面的垂直距离（带符号）
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

            # 堆顶位置：距离地面最远的点（取绝对值）
            max_dist_idx = np.argmax(np.abs(distances))
            peak_position = points[max_dist_idx].tolist()
            pile_height = float(np.abs(distances[max_dist_idx]))

            # 地面高度：中心点在地面平面上的Z坐标
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)

            # 将所有点投影到地面平面上（垂直投影）
            # 投影公式：P' = P - distance * normal
            normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            projected_points = points - distances[:, np.newaxis] * normal

            # 使用投影后的点计算2D边界（XY平面）
            xy_points = projected_points[:, :2]
        else:
            # 没有地面平面时，使用原始XY投影
            xy_points = points[:, :2]
            max_z_idx = points[:, 2].argmax()
            peak_position = points[max_z_idx].tolist()
            ground_z = float(points[:, 2].min())
            pile_height = float(peak_position[2] - ground_z)

        # 2D投影轮廓（基于投影到地面平面的点）
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import pdist, squareform
        try:
            hull_2d = ConvexHull(xy_points)
            hull_area = hull_2d.volume  # 2D凸包面积
            hull_perimeter = hull_2d.area  # 2D凸包周长
            hull_vertices = xy_points[hull_2d.vertices].tolist()

            # 计算凸包顶点
            hull_points = xy_points[hull_2d.vertices]

            # 计算投影边界的长宽（轴对齐边界框，用于参考）
            min_xy = hull_points.min(axis=0)
            max_xy = hull_points.max(axis=0)
            projected_length = float(max_xy[0] - min_xy[0])
            projected_width = float(max_xy[1] - min_xy[1])

            # 计算凸包顶点间的最大距离（新增）
            if len(hull_points) >= 2:
                distances_matrix = squareform(pdist(hull_points))
                max_span = float(distances_matrix.max())
                max_dist_indices = np.unravel_index(distances_matrix.argmax(), distances_matrix.shape)
                span_point1 = hull_points[max_dist_indices[0]].tolist()
                span_point2 = hull_points[max_dist_indices[1]].tolist()
            else:
                max_span = 0.0
                span_point1 = [0.0, 0.0]
                span_point2 = [0.0, 0.0]
        except Exception:
            hull_area = 0.0
            hull_perimeter = 0.0
            hull_vertices = []
            projected_length = 0.0
            projected_width = 0.0
            max_span = 0.0
            span_point1 = [0.0, 0.0]
            span_point2 = [0.0, 0.0]

        result = {
            "料堆索引": pile_index,
            "堆顶位置": peak_position,
            "地面高度": ground_z,
            "料堆高度": pile_height,
            "轴对齐边界框": {
                "最小值": bbox.min_bound.tolist(),
                "最大值": bbox.max_bound.tolist(),
                "尺寸": (bbox.max_bound - bbox.min_bound).tolist()
            },
            "有向边界框": {
                "中心": obb.center.tolist(),
                "尺寸": obb.extent.tolist()
            },
            "底面投影": {
                "凸包面积(m²)": round(hull_area, 4),
                "凸包周长(m)": round(hull_perimeter, 4),
                "轮廓顶点数": len(hull_vertices),
                "投影边界长度": projected_length,
                "投影边界宽度": projected_width,
                "最大跨度": max_span,
                "最大跨度端点1": span_point1,
                "最大跨度端点2": span_point2
            }
        }
        self.log(f"边界计算完成: 高度={pile_height:.3f}m, 底面积={hull_area:.3f}m²")
        return result

    def cluster_piles(self,
                     eps: float = 0.05,
                     min_points: int = 100) -> Dict:
        """使用DBSCAN聚类分割多个料堆"""
        self.log("开始料堆聚类分割...")

        if not self.pile_clouds or len(self.pile_clouds) == 0:
            raise ValueError("请先执行地面分割")

        pile_cloud = self.pile_clouds[0]
        labels = np.array(pile_cloud.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        ))

        max_label = labels.max()
        self.log(f"检测到 {max_label + 1} 个料堆簇")

        try:
            import matplotlib.pyplot as plt
            colors = plt.cm.tab10(np.linspace(0, 1, max(max_label + 1, 1)))[:, :3]
        except ImportError:
            colors = np.random.rand(max(max_label + 1, 1), 3)

        pile_info = []
        self.pile_clouds = []

        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) < min_points:
                continue

            cluster_cloud = pile_cloud.select_by_index(cluster_indices)
            cluster_cloud.paint_uniform_color(colors[i % len(colors)])

            points = np.asarray(cluster_cloud.points)
            bbox = cluster_cloud.get_axis_aligned_bounding_box()

            info = {
                "簇编号": i,
                "点数": len(cluster_indices),
                "中心": points.mean(axis=0).tolist(),
                "边界框": {
                    "最小值": bbox.min_bound.tolist(),
                    "最大值": bbox.max_bound.tolist()
                }
            }
            pile_info.append(info)
            self.pile_clouds.append(cluster_cloud)
            self.log(f"料堆 #{i}: {len(cluster_indices)} 点")

        result = {
            "料堆数量": len(self.pile_clouds),
            "料堆信息": pile_info
        }
        self.log(f"聚类完成: 识别出 {len(self.pile_clouds)} 个料堆")
        return result

    def calculate_pile_volume(self,
                             pile_index: int = 0,
                             method: str = "auto") -> Dict:
        """
        计算料堆体积（支持多方法融合与置信度评估）

        Args:
            pile_index: 料堆索引
            method: 计算方法
                - "auto": 智能自动选择（基于点云特征）
                - "mesh": 网格法
                - "convex_hull": 凸包法
                - "grid": 栅格法
                - "grid_adaptive": 自适应栅格法（新增）
                - "horizontal_section": 水平截面法（新增，博客推荐）
                - "voxel": 体素化方法（新增，博客推荐）
                - "multi": 多方法融合（原有）
                - "multi_enhanced": 增强多方法融合（新增，包含所有方法）

        Returns:
            体积计算结果
        """
        self.log(f"开始计算料堆 #{pile_index} 的体积 (方法: {method})...")

        if pile_index >= len(self.pile_clouds):
            raise ValueError(f"料堆索引 {pile_index} 超出范围")

        pile_cloud = self.pile_clouds[pile_index]
        points = np.asarray(pile_cloud.points)

        if len(points) == 0:
            raise ValueError("料堆点云为空")

        bbox = pile_cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound

        # 计算堆顶位置和高度（与calculate_boundary保持一致）
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            # 计算所有点到地面平面的垂直距离（带符号）
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

            # 堆顶位置：距离地面最远的点（取绝对值）
            max_dist_idx = np.argmax(np.abs(distances))
            peak_position = points[max_dist_idx]
            pile_height = float(np.abs(distances[max_dist_idx]))

            # 地面高度：中心点在地面平面上的Z坐标
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            # 没有地面平面时，使用Z坐标最大的点
            max_z_idx = points[:, 2].argmax()
            peak_position = points[max_z_idx]
            ground_z = float(points[:, 2].min())
            pile_height = float(peak_position[2] - ground_z)

        # 智能自动选择最佳方法
        if method == "auto":
            characteristics = self._analyze_point_cloud_characteristics(pile_cloud)
            method = self._select_best_volume_method(characteristics)
            self.log(f"智能选择: {method}")
            self.log(f"  点云特征: 点数={characteristics['点数']}, 均匀性={characteristics['分布均匀性']:.2f}, 高宽比={characteristics['高宽比']:.2f}")

        # 计算体积
        volumes = {}
        if method == "multi_enhanced":
            # 增强多方法融合（包含所有方法）
            self.log("执行增强多方法融合（包含所有新方法）...")
            if self.mesh is not None and self.mesh.is_watertight():
                volumes["mesh"] = float(self.mesh.get_volume())
            volumes["convex_hull"] = self._calculate_volume_convex_hull(pile_cloud, ground_z)
            volumes["grid"] = self._calculate_volume_grid(pile_cloud, ground_z)
            volumes["grid_adaptive"] = self._calculate_volume_grid_adaptive(pile_cloud, ground_z)
            volumes["horizontal_section"] = self._calculate_volume_horizontal_section(pile_cloud, ground_z)
            volumes["voxel"] = self._calculate_volume_voxel(pile_cloud, ground_z)

            # 使用加权平均（根据方法可靠性）
            weights = {
                "mesh": 1.0,
                "convex_hull": 0.7,
                "grid": 0.8,
                "grid_adaptive": 0.9,
                "horizontal_section": 0.95,
                "voxel": 0.85
            }

            weighted_sum = sum(volumes[k] * weights.get(k, 0.8) for k in volumes.keys())
            total_weight = sum(weights.get(k, 0.8) for k in volumes.keys())
            volume = float(weighted_sum / total_weight)

            confidence = self._calculate_volume_confidence(volumes)
            self.log(f"增强多方法融合: {volumes}")
            self.log(f"  加权平均={volume:.4f} m³, 置信度={confidence:.2f}")

        elif method == "multi":
            # 原有多方法融合
            if self.mesh is not None and self.mesh.is_watertight():
                volumes["mesh"] = float(self.mesh.get_volume())
            volumes["convex_hull"] = self._calculate_volume_convex_hull(pile_cloud, ground_z)
            volumes["grid"] = self._calculate_volume_grid(pile_cloud, ground_z)

            # 使用中位数作为最终结果
            volume = float(np.median(list(volumes.values())))
            confidence = self._calculate_volume_confidence(volumes)
            self.log(f"多方法融合: {volumes}, 中位数={volume:.4f}, 置信度={confidence:.2f}")

        elif method == "mesh" and self.mesh is not None and self.mesh.is_watertight():
            volume = float(self.mesh.get_volume())
            confidence = 0.95
            self.log(f"使用水密网格计算体积: {volume:.4f} m³")

        elif method == "convex_hull":
            volume = self._calculate_volume_convex_hull(pile_cloud, ground_z)
            confidence = 0.75

        elif method == "grid":
            volume = self._calculate_volume_grid(pile_cloud, ground_z)
            confidence = 0.80

        elif method == "grid_adaptive":
            volume = self._calculate_volume_grid_adaptive(pile_cloud, ground_z)
            confidence = 0.85

        elif method == "horizontal_section":
            volume = self._calculate_volume_horizontal_section(pile_cloud, ground_z)
            confidence = 0.90

        elif method == "voxel":
            volume = self._calculate_volume_voxel(pile_cloud, ground_z)
            confidence = 0.82

        else:
            # 降级到凸包法
            self.log(f"⚠️ 未知方法 '{method}'，降级到凸包法")
            volume = self._calculate_volume_convex_hull(pile_cloud, ground_z)
            confidence = 0.70

        result = {
            "料堆索引": pile_index,
            "点数": len(points),
            "堆顶位置": peak_position.tolist(),
            "地面高度": ground_z,
            "料堆高度": pile_height,
            "边界框": {
                "最小值": min_bound.tolist(),
                "最大值": max_bound.tolist(),
                "尺寸": (max_bound - min_bound).tolist()
            },
            "体积(立方米)": float(volume),
            "计算方法": method,
            "置信度": confidence if method != "multi" else confidence,
            "多方法结果": volumes if method == "multi" else None
        }
        self.log(f"体积计算完成: {volume:.4f} 立方米 (高度: {pile_height:.2f}m, 置信度: {confidence:.2f})")
        return result

    def _calculate_volume_confidence(self, volumes: Dict[str, float]) -> float:
        """计算多方法体积的置信度（基于一致性）"""
        if len(volumes) < 2:
            return 0.5

        values = list(volumes.values())
        mean_vol = np.mean(values)
        std_vol = np.std(values)
        cv = std_vol / mean_vol if mean_vol > 0 else 1.0  # 变异系数

        # 变异系数越小，置信度越高
        confidence = max(0.5, 1.0 - cv)
        return confidence

    def _calculate_volume_convex_hull(self, cloud: o3d.geometry.PointCloud, ground_z: float) -> float:
        """
        凸包法计算体积

        原理：
        1. 将料堆点云投影到地面平面上
        2. 合并料堆点和投影点
        3. 计算3D凸包体积
        """
        points = np.asarray(cloud.points)

        if self.ground_plane is not None:
            # 使用地面平面进行正确的投影
            a, b, c, d = self.ground_plane

            # 计算所有点到地面平面的垂直距离
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

            # 将点投影到地面平面上（垂直投影）
            normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            ground_points = points - distances[:, np.newaxis] * normal
        else:
            # 没有地面平面时，使用简单的Z坐标投影
            ground_points = points.copy()
            ground_points[:, 2] = ground_z

        # 合并料堆点和地面投影点
        all_points = np.vstack([points, ground_points])

        temp_cloud = o3d.geometry.PointCloud()
        temp_cloud.points = o3d.utility.Vector3dVector(all_points)
        try:
            hull, _ = temp_cloud.compute_convex_hull()
            volume = float(hull.get_volume())
            self.log(f"凸包法计算体积: {volume:.6f} m³")
            return volume
        except Exception as e:
            self.log(f"凸包计算失败: {str(e)}, 使用网格方法")
            return self._calculate_volume_grid(cloud, ground_z)

    def _calculate_volume_grid(self, cloud: o3d.geometry.PointCloud, ground_z: float, grid_size: float = None) -> float:
        """
        栅格法计算体积（自适应网格大小）

        原理：
        1. 将XY平面划分为网格
        2. 每个网格计算点到地面平面的最大垂直距离
        3. 累加所有网格的体积

        改进：
        - 自适应确定网格大小（基于点云密度）
        - 统一使用地面平面计算高度
        """
        points = np.asarray(cloud.points)

        # 自适应确定网格大小
        if grid_size is None:
            distances_nn = cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances_nn)
            # 网格大小为平均点间距的2-3倍
            grid_size = avg_dist * 2.5

        # 统一使用地面平面计算垂直距离
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
            normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            projected_points = points - distances[:, np.newaxis] * normal
            xy_points = projected_points[:, :2]
        else:
            xy_points = points[:, :2]
            distances = points[:, 2] - ground_z

        x_min, y_min = xy_points.min(axis=0)
        x_max, y_max = xy_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        volume = 0.0
        grid_count = 0
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (
                    (xy_points[:, 0] >= x_bins[i]) & (xy_points[:, 0] < x_bins[i+1]) &
                    (xy_points[:, 1] >= y_bins[j]) & (xy_points[:, 1] < y_bins[j+1])
                )
                if mask.any():
                    # 该网格内点到地面平面的最大垂直距离（只取正值）
                    grid_heights = distances[mask]
                    # 过滤地面以下的点
                    grid_heights = grid_heights[grid_heights > 0]
                    if len(grid_heights) > 0:
                        max_height = grid_heights.max()
                        volume += grid_size * grid_size * max_height
                        grid_count += 1

        self.log(f"栅格法计算体积: {volume:.6f} m³ (网格数: {grid_count}, 自适应网格大小: {grid_size:.4f}m)")
        return volume

    def _calculate_volume_horizontal_section(self, cloud: o3d.geometry.PointCloud, ground_z: float, num_layers: int = 50) -> float:
        """
        水平截面法计算体积（博客推荐方法）

        原理：
        1. 将煤堆按高度分层
        2. 计算每层的2D面积（使用凸包或Alpha Shape）
        3. 使用梯形法则累加体积

        优势：
        - 适合不规则形状的煤堆
        - 对稀疏点云鲁棒
        - 能够捕捉煤堆的形状变化

        Args:
            cloud: 点云
            ground_z: 地面高度
            num_layers: 分层数量（默认50层）
        """
        from scipy.spatial import ConvexHull

        points = np.asarray(cloud.points)

        # 计算垂直距离（统一使用地面平面）
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
            heights = np.abs(distances)
        else:
            heights = points[:, 2] - ground_z

        max_height = heights.max()

        # 自适应确定分层数（基于点云尺寸）
        # 对于小点云，减少分层数
        bbox = cloud.get_axis_aligned_bounding_box()
        bbox_size = bbox.max_bound - bbox.min_bound
        pile_height = max_height

        # 每层厚度约为平均点间距的2-3倍
        distances_nn = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances_nn)
        optimal_layer_thickness = avg_dist * 2.5

        # 计算最优分层数
        optimal_num_layers = max(10, min(50, int(pile_height / optimal_layer_thickness)))
        num_layers = optimal_num_layers

        layer_thickness = max_height / num_layers

        volume = 0.0
        prev_area = 0.0

        # 计算地面面积（作为第0层）
        try:
            hull_2d_ground = ConvexHull(points[:, :2])
            ground_area = hull_2d_ground.volume
        except:
            ground_area = 0.0

        for i in range(num_layers + 1):  # 包括地面层（i=0）
            h = i * layer_thickness

            # 选择该高度以上的点
            mask = heights >= h
            if not mask.any():
                # 没有点了，使用面积0
                area = 0.0
            else:
                layer_points = points[mask]

                # 计算该层的2D面积（投影到XY平面）
                try:
                    if len(layer_points) >= 3:
                        hull_2d = ConvexHull(layer_points[:, :2])
                        area = hull_2d.volume  # 2D凸包的volume就是面积
                    else:
                        area = 0.0
                except:
                    area = 0.0

            # 梯形法则累加体积
            if i > 0:
                volume += (prev_area + area) / 2 * layer_thickness

            prev_area = area

        self.log(f"水平截面法计算体积: {volume:.6f} m³ (自适应分层数: {num_layers}, 层厚: {layer_thickness:.4f}m)")
        return volume

    def _calculate_volume_voxel(self, cloud: o3d.geometry.PointCloud, ground_z: float, voxel_size: float = None) -> float:
        """
        体素化方法计算体积（自适应体素大小）

        原理：
        1. 将3D空间划分为小立方体（体素）
        2. 统计煤堆占据的体素数量
        3. 累加体素体积

        改进：
        - 自适应确定体素大小（基于点云密度）
        - 使用地面平面正确过滤地面以下的体素

        Args:
            cloud: 点云
            ground_z: 地面高度
            voxel_size: 体素大小（None时自动计算）
        """
        points = np.asarray(cloud.points)

        # 自适应确定体素大小
        if voxel_size is None:
            distances_nn = cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances_nn)
            # 体素大小为平均点间距的2.5倍（与网格法一致）
            voxel_size = avg_dist * 2.5

        # 创建体素网格
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size=voxel_size)

        # 获取所有体素
        voxels = voxel_grid.get_voxels()

        # 过滤地面以下的体素（使用地面平面）
        valid_voxel_count = 0
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            norm = np.sqrt(a**2 + b**2 + c**2)
            for voxel in voxels:
                voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
                # 计算体素中心到地面平面的有向距离
                distance = (a * voxel_center[0] + b * voxel_center[1] + c * voxel_center[2] + d) / norm
                # 只统计地面以上的体素
                if distance > 0:
                    valid_voxel_count += 1
        else:
            # 没有地面平面时，使用简单的Z坐标判断
            for voxel in voxels:
                voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
                if voxel_center[2] > ground_z:
                    valid_voxel_count += 1

        volume = valid_voxel_count * (voxel_size ** 3)

        self.log(f"体素化方法计算体积: {volume:.6f} m³ (体素数: {valid_voxel_count}, 自适应体素大小: {voxel_size:.4f}m)")
        return volume

    def _calculate_volume_grid_adaptive(self, cloud: o3d.geometry.PointCloud, ground_z: float) -> float:
        """
        自适应栅格法计算体积（改进版）

        改进：
        1. 根据点云密度自动确定网格大小
        2. 使用90百分位数而非最大值，避免离群点影响
        3. 统一使用地面平面计算高度

        Args:
            cloud: 点云
            ground_z: 地面高度
        """
        points = np.asarray(cloud.points)

        # 自动确定网格大小（基于点云密度）
        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        grid_size = avg_dist * 2.5  # 网格大小为平均点间距的2.5倍

        # 统一使用地面平面计算垂直距离
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
            normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            projected_points = points - distances[:, np.newaxis] * normal
            xy_points = projected_points[:, :2]
        else:
            xy_points = points[:, :2]
            distances = points[:, 2] - ground_z

        x_min, y_min = xy_points.min(axis=0)
        x_max, y_max = xy_points.max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        volume = 0.0
        grid_count = 0

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (
                    (xy_points[:, 0] >= x_bins[i]) & (xy_points[:, 0] < x_bins[i+1]) &
                    (xy_points[:, 1] >= y_bins[j]) & (xy_points[:, 1] < y_bins[j+1])
                )
                if mask.any():
                    grid_distances = distances[mask]
                    # 只考虑地面以上的点
                    grid_distances = grid_distances[grid_distances > 0]
                    if len(grid_distances) > 0:
                        # 使用90百分位数而非最大值，避免离群点影响
                        height = np.percentile(grid_distances, 90)
                        volume += grid_size * grid_size * height
                        grid_count += 1

        self.log(f"自适应栅格法计算体积: {volume:.6f} m³ (网格数: {grid_count}, 自适应网格大小: {grid_size:.4f}m)")
        return volume

    def _analyze_point_cloud_characteristics(self, cloud: o3d.geometry.PointCloud) -> Dict:
        """
        分析点云特征，用于智能方法选择

        Returns:
            点云特征字典
        """
        points = np.asarray(cloud.points)
        n_points = len(points)

        # 计算点云密度
        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)

        # 计算点云形状特征
        bbox = cloud.get_axis_aligned_bounding_box()
        bbox_size = bbox.max_bound - bbox.min_bound
        aspect_ratio = bbox_size[2] / max(bbox_size[0], bbox_size[1])  # 高度/底面

        # 判断点云分布均匀性
        uniformity = 1.0 - (std_dist / avg_dist) if avg_dist > 0 else 0.0

        characteristics = {
            "点数": n_points,
            "平均点间距": avg_dist,
            "点间距标准差": std_dist,
            "分布均匀性": uniformity,
            "高宽比": aspect_ratio,
            "边界框尺寸": bbox_size.tolist()
        }

        return characteristics

    def _select_best_volume_method(self, characteristics: Dict) -> str:
        """
        基于点云特征智能选择最佳体积计算方法

        Args:
            characteristics: 点云特征

        Returns:
            推荐的方法名称
        """
        n_points = characteristics["点数"]
        uniformity = characteristics["分布均匀性"]
        aspect_ratio = characteristics["高宽比"]

        # 决策逻辑
        if n_points < 500:
            # 稀疏点云：使用凸包法
            return "convex_hull"
        elif n_points > 5000 and uniformity > 0.7:
            # 密集且均匀：使用体素化方法
            return "voxel"
        elif aspect_ratio > 0.5:
            # 高耸的煤堆：使用水平截面法
            return "horizontal_section"
        elif uniformity < 0.5:
            # 分布不均匀：使用自适应栅格法
            return "grid_adaptive"
        else:
            # 默认：使用多方法融合
            return "multi_enhanced"

    def save_processed_cloud(self, output_path: str):
        """保存处理后的点云"""
        self.log(f"保存处理后的点云到: {output_path}")
        if not self.pile_clouds:
            raise ValueError("没有可保存的料堆点云")

        combined_cloud = self.pile_clouds[0]
        for pile_cloud in self.pile_clouds[1:]:
            combined_cloud = combined_cloud + pile_cloud

        o3d.io.write_point_cloud(output_path, combined_cloud)
        self.log(f"点云保存成功: {len(combined_cloud.points)} 点")

    def save_mesh(self, output_path: str):
        """保存曲面重构网格"""
        if self.mesh is None:
            raise ValueError("尚未执行曲面重构")
        o3d.io.write_triangle_mesh(output_path, self.mesh)
        self.log(f"网格保存成功: {output_path}")

    def generate_report(self, output_path: str, volume_results: List[Dict]):
        """生成处理报告"""
        self.log(f"生成处理报告: {output_path}")
        report = {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "处理日志": self.processing_log,
            "料堆数量": len(self.pile_clouds),
            "体积结果": volume_results,
            "总体积": sum(r["体积(立方米)"] for r in volume_results)
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.log(f"报告生成完成: 总体积 {report['总体积']:.4f} 立方米")
        return report

    def _reconstruct_convex_hull(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        凸包法曲面重构（适合稀疏点云）

        原理：
        1. 计算包裹所有点的最小凸多面体
        2. 100%保证封闭
        3. 使用与BPA相同的地面处理逻辑

        优势：
        - 简单快速
        - 100%封闭
        - 体积计算准确
        - 适合稀疏点云

        劣势：
        - 丢失凹陷细节
        - 体积会偏大（包含凹陷空间）
        """
        self.log("执行凸包法曲面重构...")
        from scipy.spatial import ConvexHull

        points = np.asarray(cloud.points)

        # 获取地面高度
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            ground_z = float(np.percentile(points[:, 2], 5))

        self.log(f"  地面高度: {ground_z:.4f}")

        # 阶段1: 计算3D凸包
        self.log(f"  [阶段1] 计算3D凸包...")
        self.log(f"    输入点数: {len(points)}")

        try:
            hull = ConvexHull(points)

            # 创建Open3D网格
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)

            self.log(f"    凸包网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
            self.log(f"    凸包是否封闭: {mesh.is_watertight()}")

        except Exception as e:
            self.log(f"  ❌ 凸包计算失败: {e}")
            raise

        # 阶段2: 清理和优化（凸包本身就是封闭的，不需要添加地面）
        self.log(f"  [阶段2] 清理和优化...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        if mesh.is_watertight():
            volume = mesh.get_volume()
            self.log(f"  体积: {volume:.6f} m³")

        self.log(f"曲面重构完成: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面, 水密: {mesh.is_watertight()}")

        return mesh

    def _reconstruct_convex_hull_shrink(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        收缩包裹法曲面重构（适合稀疏点云，保留更多细节）

        原理：
        1. 从放大的凸包开始
        2. 逐步收缩，贴合点云表面
        3. 保留更多细节，同时保证封闭
        4. 使用与BPA相同的地面处理逻辑

        优势：
        - 100%封闭
        - 保留更多细节（比纯凸包好）
        - 体积更准确

        参数：
        - initial_scale: 初始放大倍数（默认1.3）
        - shrink_iterations: 收缩迭代次数（默认30）
        """
        self.log("执行收缩包裹法曲面重构...")
        from scipy.spatial import ConvexHull

        points = np.asarray(cloud.points)

        # 获取地面高度
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            ground_z = float(np.percentile(points[:, 2], 5))

        self.log(f"  地面高度: {ground_z:.4f}")

        # 参数
        initial_scale = 1.3
        shrink_iterations = 30

        # 阶段1: 创建初始凸包（放大）
        self.log(f"  [阶段1] 创建初始凸包（放大{initial_scale}倍）...")

        center = points.mean(axis=0)
        scaled_points = center + (points - center) * initial_scale

        try:
            hull = ConvexHull(scaled_points)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(scaled_points)
            mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)

            self.log(f"    初始网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面")

        except Exception as e:
            self.log(f"  ❌ 凸包计算失败: {e}")
            raise

        # 阶段2: 构建KD树
        self.log(f"  [阶段2] 构建KD树...")
        pcd_tree = o3d.geometry.KDTreeFlann(cloud)

        # 阶段3: 迭代收缩
        self.log(f"  [阶段3] 迭代收缩 ({shrink_iterations}次)...")

        vertices = np.asarray(mesh.vertices)

        for iter in range(shrink_iterations):
            if iter % 10 == 0:
                self.log(f"    迭代 {iter}/{shrink_iterations}...")

            # 计算每个顶点的法向量
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)

            # 向内收缩
            shrink_factor = 0.02  # 每次收缩2%

            for i in range(len(vertices)):
                # 查找最近的点云点
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(vertices[i], 1)
                nearest_point = points[idx[0]]

                # 如果距离点云太远，向内收缩
                if dist[0] > 0.001:  # 1mm阈值
                    # 向点云方向移动
                    direction = nearest_point - vertices[i]
                    direction = direction / (np.linalg.norm(direction) + 1e-8)

                    vertices[i] += direction * shrink_factor

            mesh.vertices = o3d.utility.Vector3dVector(vertices)

        # 阶段4: 平滑
        self.log(f"  [阶段4] 平滑处理...")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)

        # 阶段5: 清理（收缩后的凸包仍然是封闭的，不需要添加地面）
        self.log(f"  [阶段5] 清理...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        self.log(f"  最终网格: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")
        self.log(f"  是否封闭: {mesh.is_watertight()}")

        if mesh.is_watertight():
            volume = mesh.get_volume()
            self.log(f"  体积: {volume:.6f} m³")

        self.log(f"曲面重构完成: {len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面, 水密: {mesh.is_watertight()}")

        return mesh

    def _extract_surface_points(self, cloud: o3d.geometry.PointCloud,
                                radius_multiplier: float = 2.5) -> o3d.geometry.PointCloud:
        """
        提取外部表面点，去除内部冗余点

        使用基于法向量和局部密度的方法识别表面点：
        1. 计算每个点的法向量
        2. 检查点是否在表面（法向量一致性）
        3. 去除内部重叠点

        Args:
            cloud: 输入点云
            radius_multiplier: 搜索半径倍数

        Returns:
            表面点云
        """
        self.log("  提取外部表面点...")
        points = np.asarray(cloud.points)

        # 确保有法向量
        if not cloud.has_normals():
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            cloud.orient_normals_consistent_tangent_plane(k=15)

        normals = np.asarray(cloud.normals)

        # 计算平均点间距
        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        search_radius = avg_dist * radius_multiplier

        # 构建KD树
        pcd_tree = o3d.geometry.KDTreeFlann(cloud)

        # 标记表面点
        is_surface = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            # 搜索邻域
            [k, idx, _] = pcd_tree.search_radius_vector_3d(points[i], search_radius)

            if k < 5:  # 孤立点，保留
                is_surface[i] = True
                continue

            # 计算法向量一致性
            neighbor_normals = normals[idx[1:]]  # 排除自己
            current_normal = normals[i]

            # 计算与邻居法向量的点积
            dot_products = np.abs(np.dot(neighbor_normals, current_normal))
            consistency = np.mean(dot_products)

            # 如果法向量一致性高，说明是表面点
            # 如果一致性低，说明可能是内部点（周围法向量指向不同方向）
            if consistency > 0.7:  # 阈值可调
                is_surface[i] = True
            else:
                # 进一步检查：计算点到邻居的方向与法向量的一致性
                neighbor_points = points[idx[1:]]
                directions = neighbor_points - points[i]
                directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)

                # 表面点的法向量应该与到邻居的方向大致垂直
                perpendicularity = np.abs(np.dot(directions, current_normal))
                avg_perp = np.mean(perpendicularity)

                if avg_perp < 0.5:  # 大致垂直
                    is_surface[i] = True

        # 创建表面点云
        surface_indices = np.where(is_surface)[0]
        surface_cloud = cloud.select_by_index(surface_indices)

        self.log(f"    表面点提取: {len(points)} → {len(surface_indices)} 点 ({len(surface_indices)/len(points)*100:.1f}%)")

        return surface_cloud

    def _repair_mesh_closure(self, mesh: o3d.geometry.TriangleMesh,
                             points: np.ndarray,
                             ground_z: float) -> o3d.geometry.TriangleMesh:
        """
        修复网格封闭性

        策略：
        1. 检测边界边
        2. 填充孔洞
        3. 确保地面封底
        4. 修复非流形边

        Args:
            mesh: 输入网格
            points: 原始点云
            ground_z: 地面高度

        Returns:
            修复后的网格
        """
        self.log("  修复网格封闭性...")

        # 1. 检测并填充小孔洞
        self.log("    填充小孔洞...")
        initial_triangles = len(mesh.triangles)

        # 使用细分和平滑来填充孔洞
        if initial_triangles < 10000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            self.log(f"    网格细分: {initial_triangles} → {len(mesh.triangles)} 三角面")

        # Laplacian平滑可以帮助填充小孔洞
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)

        # 2. 清理非流形结构
        self.log("    清理非流形结构...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # 3. 确保地面封底
        if not mesh.is_watertight():
            self.log("    重新添加地面封底...")
            mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 4. 最终检查
        if mesh.is_watertight():
            self.log("    ✓ 网格已封闭")
        else:
            self.log("    ⚠️ 网格仍未完全封闭，但已尽力修复")

        return mesh

    def _ensure_watertight_mesh(self, mesh: o3d.geometry.TriangleMesh,
                                cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        确保网格封闭（修复方法，不替换算法）

        策略：
        1. 检测并填充孔洞
        2. 改进地面封底
        3. 修复非流形边
        4. 如果仍未封闭，尝试更激进的修复

        Args:
            mesh: 输入网格
            cloud: 原始点云

        Returns:
            修复后的网格（尽可能封闭）
        """
        if mesh.is_watertight():
            self.log("  ✓ 网格已封闭")
            return mesh

        self.log("  ⚠️ 网格未封闭，执行修复...")

        points = np.asarray(cloud.points)
        ground_z = self._get_ground_z(points)

        # 策略1：填充小孔洞
        self.log("    尝试填充孔洞...")
        initial_triangles = len(mesh.triangles)

        # 细分可以帮助填充小孔洞
        if initial_triangles < 10000:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            self.log(f"    网格细分: {initial_triangles} → {len(mesh.triangles)} 三角面")

        # Laplacian平滑
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.5)

        # 清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # 检查是否封闭
        if mesh.is_watertight():
            self.log("    ✓ 孔洞填充成功，网格已封闭")
            return mesh

        # 策略2：重新添加地面封底（更激进）
        self.log("    重新添加地面封底...")
        mesh = self._add_ground_base_enhanced(mesh, points, ground_z)

        # 再次清理
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

        # 最终检查
        if mesh.is_watertight():
            self.log("    ✓ 地面封底成功，网格已封闭")
        else:
            self.log("    ⚠️ 网格仍未完全封闭（保留原算法结果）")

        return mesh
