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
        基于物料堆特性的地面识别算法（推荐用于DUSt3R点云）

        核心洞察（来自用户反馈）：
        1. DUSt3R生成的点云主要是物料堆表面点，地面点极少
        2. 物料堆内部是真空的（没有点）
        3. 如果地面朝上（正确）：点云呈现凹陷（内部真空）
        4. 如果地面朝下（错误）：点云呈现凸起（表面点）

        算法策略：
        1. 分析不同高度层的XY投影面积
        2. 如果面积随高度递减 → 地面在底部（正确）
        3. 如果面积随高度递增 → 地面在顶部（需要翻转）
        4. 基于面积变化趋势判断地面方向

        适用场景：
        - DUSt3R生成的点云（地面点稀少）
        - 物料堆、煤堆、矿石堆等
        - 点云主要是表面点的场景
        """
        from scipy.spatial import ConvexHull

        points = np.asarray(cloud.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_range = z_max - z_min

        self.log(f"  点云Z范围: [{z_min:.4f}, {z_max:.4f}] m, 高度差: {z_range:.4f} m")
        self.log(f"  使用物料堆感知算法（基于真空特性）...")

        # Step 1: 分析不同高度层的XY投影面积
        self.log(f"  分析不同高度层的XY投影面积（判断地面方向）...")

        areas = []
        percentiles = [10, 30, 50, 70, 90]

        for percentile in percentiles:
            z_threshold = z_min + z_range * percentile / 100
            layer_points = points[points[:, 2] <= z_threshold]

            if len(layer_points) > 3:
                try:
                    hull = ConvexHull(layer_points[:, :2])
                    area = hull.volume  # 2D凸包的面积
                    areas.append(area)
                    self.log(f"    底部{percentile}%高度: {len(layer_points)}点, XY面积={area:.6f}")
                except:
                    areas.append(0)
            else:
                areas.append(0)

        # Step 2: 判断面积变化趋势
        if len(areas) >= 3:
            # 计算面积变化趋势（线性回归斜率）
            x = np.array(percentiles[:len(areas)])
            y = np.array(areas)

            # 简单线性拟合
            slope = np.polyfit(x, y, 1)[0]

            self.log(f"  面积变化趋势: 斜率={slope:.8f}")

            if slope > 0:
                self.log(f"  ✅ 面积随高度递增 → 地面在底部（正确方向）")
                ground_at_bottom = True
            else:
                self.log(f"  ⚠️ 面积随高度递减 → 地面在顶部（需要翻转）")
                self.log(f"  这种情况很少见，可能是点云坐标系问题")
                ground_at_bottom = False
        else:
            self.log(f"  ⚠️ 无法判断面积趋势，假设地面在底部")
            ground_at_bottom = True

        # Step 3: 基于判断结果选择地面点
        if ground_at_bottom:
            # 地面在底部，取Z值最低的点
            z_sorted_indices = np.argsort(points[:, 2])
            bottom_percentile = 0.08
            n_bottom = max(10, int(len(points) * bottom_percentile))
            bottom_indices = z_sorted_indices[:n_bottom]

            self.log(f"  取Z值最低的{bottom_percentile*100:.0f}%点: {n_bottom}个点")
        else:
            # 地面在顶部（罕见情况），取Z值最高的点
            z_sorted_indices = np.argsort(points[:, 2])
            top_percentile = 0.08
            n_top = max(10, int(len(points) * top_percentile))
            bottom_indices = z_sorted_indices[-n_top:]

            self.log(f"  取Z值最高的{top_percentile*100:.0f}%点: {n_top}个点")

        # Step 4: 使用最小二乘法拟合平面
        bottom_points = points[bottom_indices]
        plane_model = self._fit_plane_least_squares(bottom_points)

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # 计算倾角
        z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.clip(np.abs(np.dot(normal, z_axis)), 0, 1))
        angle_deg = np.degrees(angle)

        self.log(f"  地面平面拟合: 倾角={angle_deg:.2f}°")

        # Step 5: 找到距离平面小于阈值的所有点
        norm = np.sqrt(a**2 + b**2 + c**2)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
        inliers = np.where(distances < distance_threshold)[0]

        self.log(f"  地面内点数: {len(inliers)} ({len(inliers)/len(points)*100:.1f}%)")

        return plane_model, inliers.tolist()

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
                            method: str = "poisson",
                            depth: int = 7) -> Dict:
        """
        曲面重构：从点云重建三角网格曲面

        Args:
            pile_index: 料堆索引
            method: 重构方法
                - "poisson"（推荐）: 泊松重构 + 地面封底，解决凹陷问题
                - "bpa": 球旋转算法 + 地面封底，解决空缺问题
                - "pile_convex": 高度图重构（仅适合近似水平地面）
            depth: 泊松重构深度（默认7）
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

        if method == "poisson":
            mesh = self._reconstruct_poisson_with_base(cloud, depth)
        elif method == "bpa":
            mesh = self._reconstruct_bpa_with_base(cloud)
        elif method == "pile_convex":
            mesh = self._reconstruct_pile_convex(cloud)
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
        泊松重构 + 地面封底：
        - 解决凹陷问题：使用更激进的密度裁剪（去除低密度悬空面）
        - 解决底部开口：添加地面封底
        """
        self.log(f"执行泊松曲面重构 (depth={depth})...")
        points = np.asarray(cloud.points)
        ground_z = self._get_ground_z(points)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        # 裁剪低密度面（去除悬空面和大凹陷）
        densities = np.asarray(densities)
        # 使用25%分位数作为阈值（比之前的10%更激进，去除更多悬空面）
        density_threshold = np.percentile(densities, 25)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        self.log(f"  泊松重构完成: {len(mesh.triangles)} 个三角面")

        # 裁剪低于地面的部分
        verts = np.asarray(mesh.vertices)
        below_ground = verts[:, 2] < (ground_z - 0.005)
        if below_ground.any():
            mesh.remove_vertices_by_mask(below_ground)
            self.log(f"  裁剪地面以下顶点后: {len(mesh.triangles)} 个三角面")

        # 添加地面封底
        mesh = self._add_ground_base(mesh, points, ground_z)
        self.log(f"  添加地面封底后: {len(mesh.triangles)} 个三角面")
        return mesh

    def _reconstruct_bpa_with_base(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        BPA球旋转算法 + 地面封底：
        - 解决空缺问题：使用多个半径层级覆盖不同尺度的空缺
        - 解决底部开口：添加地面封底
        """
        self.log("执行球旋转算法（BPA）曲面重构...")
        points = np.asarray(cloud.points)
        ground_z = self._get_ground_z(points)

        distances = cloud.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)

        # 使用更多半径层级，覆盖不同尺度的空缺
        radii = [avg_dist * 0.5, avg_dist, avg_dist * 2,
                 avg_dist * 4, avg_dist * 8, avg_dist * 16]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii)
        )
        self.log(f"  BPA重构完成: {len(mesh.triangles)} 个三角面")

        # 添加地面封底
        mesh = self._add_ground_base(mesh, points, ground_z)
        self.log(f"  添加地面封底后: {len(mesh.triangles)} 个三角面")
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

        # 2D投影轮廓（XY平面）
        from scipy.spatial import ConvexHull
        xy_points = points[:, :2]
        try:
            hull_2d = ConvexHull(xy_points)
            hull_area = hull_2d.volume  # 2D凸包面积
            hull_perimeter = hull_2d.area  # 2D凸包周长
            hull_vertices = xy_points[hull_2d.vertices].tolist()
        except Exception:
            hull_area = 0.0
            hull_perimeter = 0.0
            hull_vertices = []

        # 堆顶位置（Z最大点）
        max_z_idx = points[:, 2].argmax()
        peak_position = points[max_z_idx].tolist()

        # 地面高度
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            ground_z = float(points[:, 2].min())

        pile_height = float(peak_position[2] - ground_z)

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
                "轮廓顶点数": len(hull_vertices)
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
            method: 计算方法 ("auto" 自动选择 | "mesh" 网格法 | "convex_hull" 凸包法 | "grid" 栅格法 | "multi" 多方法融合)

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

        max_z_idx = points[:, 2].argmax()
        peak_position = points[max_z_idx]

        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            center_xy = points[:, :2].mean(axis=0)
            ground_z = float(-(a * center_xy[0] + b * center_xy[1] + d) / c)
        else:
            ground_z = float(points[:, 2].min())

        pile_height = float(peak_position[2] - ground_z)

        # 自动选择最佳方法
        if method == "auto":
            if self.mesh is not None and self.mesh.is_watertight() and len(points) > 500:
                method = "mesh"
                self.log("自动选择: 网格法（水密网格可用）")
            elif len(points) > 1000:
                method = "multi"
                self.log("自动选择: 多方法融合（点云充足）")
            else:
                method = "convex_hull"
                self.log("自动选择: 凸包法（点云稀疏）")

        # 计算体积
        volumes = {}
        if method == "multi":
            # 多方法融合
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
        else:
            # 降级到凸包法
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
        """凸包法计算体积"""
        points = np.asarray(cloud.points)
        ground_points = points.copy()
        ground_points[:, 2] = ground_z
        all_points = np.vstack([points, ground_points])

        temp_cloud = o3d.geometry.PointCloud()
        temp_cloud.points = o3d.utility.Vector3dVector(all_points)
        try:
            hull, _ = temp_cloud.compute_convex_hull()
            return float(hull.get_volume())
        except Exception as e:
            self.log(f"凸包计算失败: {str(e)}, 使用网格方法")
            return self._calculate_volume_grid(cloud, ground_z)

    def _calculate_volume_grid(self, cloud: o3d.geometry.PointCloud, ground_z: float, grid_size: float = 0.05) -> float:
        """栅格法计算体积"""
        points = np.asarray(cloud.points)
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        volume = 0.0
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (
                    (points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i+1]) &
                    (points[:, 1] >= y_bins[j]) & (points[:, 1] < y_bins[j+1])
                )
                if mask.any():
                    max_z = points[mask, 2].max()
                    height = max(0, max_z - ground_z)
                    volume += grid_size * grid_size * height
        return volume

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
