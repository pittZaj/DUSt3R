#!/usr/bin/env python3
"""
诊断地面封底问题
"""

import sys
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from coal_pile_volume_processor import CoalPileVolumeProcessor


def diagnose_ground_base():
    """诊断地面封底问题"""

    test_file = "/mnt/data3/clip/DUSt3R/test/coal_pile (4).ply"

    print("=" * 80)
    print("地面封底问题诊断")
    print("=" * 80)

    processor = CoalPileVolumeProcessor()

    # 准备
    print("\n[1] 准备...")
    processor.load_point_cloud(test_file)
    processor.preprocess_point_cloud(voxel_size=0.01)
    processor.segment_ground_plane(method="deterministic")
    processor.pile_clouds = [processor.processed_cloud]

    # 测试alpha_shape
    print("\n[2] 测试alpha_shape...")
    result = processor.reconstruct_surface(pile_index=0, method="alpha_shape")

    mesh = processor.mesh

    print(f"\n结果:")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  三角面数: {len(mesh.triangles)}")
    print(f"  是否封闭: {mesh.is_watertight()}")

    # 检查网格边界
    print(f"\n[3] 检查网格边界...")
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    print(f"  非流形边数: {len(edges)}")

    # 检查边界边
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.compute_vertex_normals()

    # 获取边界边
    triangles = np.asarray(mesh.triangles)
    edges_dict = {}

    for tri in triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            edges_dict[edge] = edges_dict.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edges_dict.items() if count == 1]
    print(f"  边界边数: {len(boundary_edges)}")

    if len(boundary_edges) > 0:
        print(f"\n  边界边示例（前10个）:")
        for i, edge in enumerate(boundary_edges[:10]):
            v1, v2 = edge
            p1 = np.asarray(mesh.vertices)[v1]
            p2 = np.asarray(mesh.vertices)[v2]
            print(f"    边{i}: v{v1}({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}) - "
                  f"v{v2}({p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f})")

    # 检查地面平面
    print(f"\n[4] 检查地面平面...")
    if processor.ground_plane is not None:
        a, b, c, d = processor.ground_plane
        print(f"  地面平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # 检查顶点到地面的距离
        vertices = np.asarray(mesh.vertices)
        distances = np.abs(a * vertices[:, 0] + b * vertices[:, 1] + c * vertices[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

        print(f"  顶点到地面距离:")
        print(f"    最小: {distances.min():.6f} m")
        print(f"    最大: {distances.max():.6f} m")
        print(f"    平均: {distances.mean():.6f} m")

        # 找到接近地面的顶点
        ground_threshold = 0.005  # 5mm
        near_ground = distances < ground_threshold
        print(f"  接近地面的顶点数（<5mm）: {near_ground.sum()}")

    print(f"\n{'='*80}")
    print("诊断完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    diagnose_ground_base()
