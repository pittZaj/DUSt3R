#!/usr/bin/env python3
"""
测试改进后的曲面重构 - 验证每个方法是否保持其原有特性
"""

import sys
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from coal_pile_volume_processor import CoalPileVolumeProcessor


def test_methods_individually():
    """单独测试每个方法，验证是否保持原有特性"""

    test_file = "/mnt/data3/clip/DUSt3R/test/coal_pile (4).ply"

    print("=" * 80)
    print("曲面重构方法独立性测试")
    print("=" * 80)

    processor = CoalPileVolumeProcessor()

    # 准备
    print("\n[准备] 加载和预处理...")
    processor.load_point_cloud(test_file)
    processor.preprocess_point_cloud(voxel_size=0.01)
    processor.segment_ground_plane(method="deterministic")
    processor.pile_clouds = [processor.processed_cloud]

    # 测试几个代表性方法
    methods = [
        "alpha_shape",
        "bpa",
        "poisson",
        "convex_hull",
    ]

    results = {}

    for method in methods:
        print(f"\n{'='*80}")
        print(f"测试方法: {method}")
        print(f"{'='*80}")

        try:
            result = processor.reconstruct_surface(pile_index=0, method=method)

            results[method] = {
                "顶点数": result["顶点数"],
                "三角面数": result["三角面数"],
                "是否水密": result["是否水密"],
            }

            print(f"  顶点数: {result['顶点数']}")
            print(f"  三角面数: {result['三角面数']}")
            print(f"  是否封闭: {result['是否水密']}")

            # 保存网格以便检查
            output_dir = Path("output_method_test")
            output_dir.mkdir(exist_ok=True)
            mesh_file = output_dir / f"mesh_{method}.ply"
            o3d.io.write_triangle_mesh(str(mesh_file), processor.mesh)
            print(f"  已保存: {mesh_file}")

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    # 检查结果是否相同
    print(f"\n{'='*80}")
    print("结果对比")
    print(f"{'='*80}")

    print(f"\n{'方法':<20} {'顶点数':<10} {'三角面数':<10} {'封闭':<8}")
    print("-" * 60)

    for method, result in results.items():
        vertices = result["顶点数"]
        triangles = result["三角面数"]
        watertight = "✓" if result["是否水密"] else "✗"
        print(f"{method:<20} {vertices:<10} {triangles:<10} {watertight:<8}")

    # 检查是否所有方法都产生相同结果
    print(f"\n{'='*80}")
    vertices_list = [r["顶点数"] for r in results.values()]
    triangles_list = [r["三角面数"] for r in results.values()]

    if len(set(vertices_list)) == 1 and len(set(triangles_list)) == 1:
        print("⚠️ 警告：所有方法产生了相同的结果！")
        print("   这表明所有方法都被替换成了同一个算法。")
    else:
        print("✓ 各方法产生了不同的结果，保持了算法独立性。")

    print(f"{'='*80}")


if __name__ == "__main__":
    test_methods_individually()
