#!/usr/bin/env python3
"""
煤堆点云体积测量处理模块
实现点云预处理、分割、边界提取和体积计算功能
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime


class CoalPileVolumeProcessor:
    """煤堆点云体积测量处理器"""

    def __init__(self):
        """初始化处理器"""
        self.point_cloud = None
        self.processed_cloud = None
        self.ground_plane = None
        self.pile_clouds = []
        self.processing_log = []

    def log(self, message: str):
        """记录处理日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(log_entry)

    def load_point_cloud(self, file_path: str) -> Dict:
        """
        加载点云文件

        Args:
            file_path: 点云文件路径

        Returns:
            包含点云信息的字典
        """
        self.log(f"正在加载点云文件: {file_path}")

        try:
            self.point_cloud = o3d.io.read_point_cloud(file_path)

            points = np.asarray(self.point_cloud.points)
            colors = np.asarray(self.point_cloud.colors) if self.point_cloud.has_colors() else None

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

        except Exception as e:
            self.log(f"加载点云失败: {str(e)}")
            raise

    def preprocess_point_cloud(self,
                               voxel_size: float = 0.01,
                               nb_neighbors: int = 20,
                               std_ratio: float = 2.0) -> Dict:
        """
        点云预处理：下采样、去噪、离群点去除

        Args:
            voxel_size: 体素下采样大小
            nb_neighbors: 统计离群点去除的邻居数量
            std_ratio: 标准差比率阈值

        Returns:
            处理结果信息
        """
        self.log("开始点云预处理...")

        if self.point_cloud is None:
            raise ValueError("请先加载点云文件")

        original_count = len(self.point_cloud.points)

        # 1. 体素下采样
        self.log(f"执行体素下采样 (voxel_size={voxel_size})...")
        downsampled = self.point_cloud.voxel_down_sample(voxel_size=voxel_size)
        after_downsample = len(downsampled.points)
        self.log(f"下采样完成: {original_count} -> {after_downsample} 点")

        # 2. 统计离群点去除
        self.log(f"执行统计离群点去除 (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        cl, ind = downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        self.processed_cloud = downsampled.select_by_index(ind)
        after_outlier_removal = len(self.processed_cloud.points)
        self.log(f"离群点去除完成: {after_downsample} -> {after_outlier_removal} 点")

        # 3. 估计法向量（用于后续处理）
        self.log("估计点云法向量...")
        self.processed_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2,
                max_nn=30
            )
        )
        self.log("法向量估计完成")

        result = {
            "原始点数": original_count,
            "下采样后点数": after_downsample,
            "去除离群点后点数": after_outlier_removal,
            "保留比例": f"{after_outlier_removal/original_count*100:.2f}%"
        }

        self.log(f"预处理完成: 保留 {result['保留比例']} 的点")
        return result

    def segment_ground_plane(self,
                            distance_threshold: float = 0.02,
                            ransac_n: int = 3,
                            num_iterations: int = 1000) -> Dict:
        """
        使用RANSAC分割地面平面

        Args:
            distance_threshold: 点到平面的距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: RANSAC迭代次数

        Returns:
            地面分割结果信息
        """
        self.log("开始地面平面分割...")

        if self.processed_cloud is None:
            raise ValueError("请先执行预处理")

        # 使用RANSAC拟合平面
        plane_model, inliers = self.processed_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        [a, b, c, d] = plane_model
        self.ground_plane = plane_model
        self.log(f"地面平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # 分离地面和料堆
        ground_cloud = self.processed_cloud.select_by_index(inliers)
        pile_cloud = self.processed_cloud.select_by_index(inliers, invert=True)

        # 为地面和料堆着色（便于可视化）
        ground_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
        pile_cloud.paint_uniform_color([0.8, 0.4, 0.2])    # 棕色

        result = {
            "地面点数": len(inliers),
            "料堆点数": len(pile_cloud.points),
            "地面平面方程": plane_model.tolist(),
            "料堆占比": f"{len(pile_cloud.points)/len(self.processed_cloud.points)*100:.2f}%"
        }

        # 保存分割后的料堆点云
        self.pile_clouds = [pile_cloud]

        self.log(f"地面分割完成: 地面 {result['地面点数']} 点, 料堆 {result['料堆点数']} 点")
        return result

    def cluster_piles(self,
                     eps: float = 0.05,
                     min_points: int = 100) -> Dict:
        """
        使用DBSCAN聚类分割多个料堆

        Args:
            eps: DBSCAN邻域半径
            min_points: 最小点数

        Returns:
            聚类结果信息
        """
        self.log("开始料堆聚类分割...")

        if not self.pile_clouds or len(self.pile_clouds) == 0:
            raise ValueError("请先执行地面分割")

        pile_cloud = self.pile_clouds[0]

        # DBSCAN聚类
        labels = np.array(pile_cloud.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        ))

        max_label = labels.max()
        self.log(f"检测到 {max_label + 1} 个料堆簇")

        # 为每个簇分配不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, max_label + 1))[:, :3]

        pile_info = []
        self.pile_clouds = []

        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) < min_points:
                continue

            cluster_cloud = pile_cloud.select_by_index(cluster_indices)
            cluster_cloud.paint_uniform_color(colors[i])

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
                             method: str = "convex_hull") -> Dict:
        """
        计算料堆体积

        Args:
            pile_index: 料堆索引
            method: 计算方法 ("convex_hull", "alpha_shape", "grid")

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

        # 计算料堆的基本信息
        bbox = pile_cloud.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound

        # 计算堆顶位置（Z坐标最大的点）
        max_z_idx = points[:, 2].argmax()
        peak_position = points[max_z_idx]

        # 计算地面高度（使用地面平面方程或最小Z值）
        if self.ground_plane is not None:
            a, b, c, d = self.ground_plane
            # 计算料堆中心点在地面平面上的投影高度
            center_xy = points[:, :2].mean(axis=0)
            ground_z = -(a * center_xy[0] + b * center_xy[1] + d) / c
        else:
            ground_z = points[:, 2].min()

        pile_height = peak_position[2] - ground_z

        # 根据方法计算体积
        if method == "convex_hull":
            volume = self._calculate_volume_convex_hull(pile_cloud, ground_z)
        elif method == "alpha_shape":
            volume = self._calculate_volume_alpha_shape(pile_cloud, ground_z)
        elif method == "grid":
            volume = self._calculate_volume_grid(pile_cloud, ground_z)
        else:
            raise ValueError(f"不支持的体积计算方法: {method}")

        result = {
            "料堆索引": pile_index,
            "点数": len(points),
            "堆顶位置": peak_position.tolist(),
            "地面高度": float(ground_z),
            "料堆高度": float(pile_height),
            "边界框": {
                "最小值": min_bound.tolist(),
                "最大值": max_bound.tolist(),
                "尺寸": (max_bound - min_bound).tolist()
            },
            "体积(立方米)": float(volume),
            "计算方法": method
        }

        self.log(f"体积计算完成: {volume:.4f} 立方米 (高度: {pile_height:.2f}m)")
        return result

    def _calculate_volume_convex_hull(self, cloud: o3d.geometry.PointCloud, ground_z: float) -> float:
        """使用凸包方法计算体积"""
        points = np.asarray(cloud.points)

        # 添加地面投影点
        ground_points = points.copy()
        ground_points[:, 2] = ground_z

        # 合并料堆点和地面投影点
        all_points = np.vstack([points, ground_points])

        # 创建点云并计算凸包
        temp_cloud = o3d.geometry.PointCloud()
        temp_cloud.points = o3d.utility.Vector3dVector(all_points)

        try:
            hull, _ = temp_cloud.compute_convex_hull()
            volume = hull.get_volume()
            return volume
        except Exception as e:
            self.log(f"凸包计算失败: {str(e)}, 使用网格方法")
            return self._calculate_volume_grid(cloud, ground_z)

    def _calculate_volume_alpha_shape(self, cloud: o3d.geometry.PointCloud, ground_z: float, alpha: float = 0.1) -> float:
        """使用Alpha Shape方法计算体积"""
        # Alpha Shape需要更复杂的实现，这里使用简化版本
        # 实际应用中可以使用scipy.spatial.Delaunay
        return self._calculate_volume_grid(cloud, ground_z)

    def _calculate_volume_grid(self, cloud: o3d.geometry.PointCloud, ground_z: float, grid_size: float = 0.05) -> float:
        """使用网格方法计算体积"""
        points = np.asarray(cloud.points)

        # 创建XY平面的网格
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)

        x_bins = np.arange(x_min, x_max + grid_size, grid_size)
        y_bins = np.arange(y_min, y_max + grid_size, grid_size)

        # 计算每个网格单元的最大高度
        volume = 0.0
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                # 找到该网格单元内的点
                mask = (
                    (points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i+1]) &
                    (points[:, 1] >= y_bins[j]) & (points[:, 1] < y_bins[j+1])
                )

                if mask.any():
                    max_z = points[mask, 2].max()
                    height = max(0, max_z - ground_z)
                    cell_volume = grid_size * grid_size * height
                    volume += cell_volume

        return volume

    def save_processed_cloud(self, output_path: str, include_ground: bool = False):
        """
        保存处理后的点云

        Args:
            output_path: 输出文件路径
            include_ground: 是否包含地面点云
        """
        self.log(f"保存处理后的点云到: {output_path}")

        if not self.pile_clouds:
            raise ValueError("没有可保存的料堆点云")

        # 合并所有料堆点云
        combined_cloud = self.pile_clouds[0]
        for pile_cloud in self.pile_clouds[1:]:
            combined_cloud += pile_cloud

        o3d.io.write_point_cloud(output_path, combined_cloud)
        self.log(f"点云保存成功: {len(combined_cloud.points)} 点")

    def generate_report(self, output_path: str, volume_results: List[Dict]):
        """
        生成处理报告

        Args:
            output_path: 报告输出路径
            volume_results: 体积计算结果列表
        """
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


# 导入matplotlib用于颜色映射
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("警告: matplotlib未安装，聚类可视化功能将受限")


def process_coal_pile_pipeline(
    input_file: str,
    output_dir: str,
    voxel_size: float = 0.01,
    ground_threshold: float = 0.02,
    cluster_eps: float = 0.05,
    volume_method: str = "convex_hull"
) -> Dict:
    """
    完整的煤堆点云处理流程

    Args:
        input_file: 输入点云文件路径
        output_dir: 输出目录
        voxel_size: 下采样体素大小
        ground_threshold: 地面分割阈值
        cluster_eps: 聚类邻域半径
        volume_method: 体积计算方法

    Returns:
        处理结果字典
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化处理器
    processor = CoalPileVolumeProcessor()

    # 1. 加载点云
    cloud_info = processor.load_point_cloud(input_file)

    # 2. 预处理
    preprocess_result = processor.preprocess_point_cloud(voxel_size=voxel_size)

    # 3. 地面分割
    ground_result = processor.segment_ground_plane(distance_threshold=ground_threshold)

    # 4. 料堆聚类（如果有多个料堆）
    try:
        cluster_result = processor.cluster_piles(eps=cluster_eps)
    except Exception as e:
        processor.log(f"聚类失败，假设只有一个料堆: {str(e)}")
        cluster_result = {"料堆数量": 1}

    # 5. 计算每个料堆的体积
    volume_results = []
    for i in range(len(processor.pile_clouds)):
        volume_result = processor.calculate_pile_volume(i, method=volume_method)
        volume_results.append(volume_result)

    # 6. 保存处理后的点云
    processed_cloud_path = output_path / "processed_pile.ply"
    processor.save_processed_cloud(str(processed_cloud_path))

    # 7. 生成报告
    report_path = output_path / "volume_report.json"
    report = processor.generate_report(str(report_path), volume_results)

    return {
        "点云信息": cloud_info,
        "预处理结果": preprocess_result,
        "地面分割结果": ground_result,
        "聚类结果": cluster_result,
        "体积结果": volume_results,
        "总体积": report["总体积"],
        "输出目录": str(output_path)
    }


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"

        result = process_coal_pile_pipeline(input_file, output_dir)
        print("\n" + "="*50)
        print("处理完成!")
        print(f"总体积: {result['总体积']:.4f} 立方米")
        print(f"输出目录: {result['输出目录']}")
        print("="*50)
    else:
        print("用法: python coal_pile_volume_processor.py <input_file> [output_dir]")
