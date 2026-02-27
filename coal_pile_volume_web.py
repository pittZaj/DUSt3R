#!/usr/bin/env python3
"""
煤堆点云体积测量Web界面
基于Gradio的友好图形界面，用于点云处理和体积计算
"""

import os
import gradio as gr
import numpy as np
import open3d as o3d
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil

# 导入点云处理模块
from coal_pile_volume_processor import CoalPileVolumeProcessor


class CoalPileVolumeWebApp:
    """煤堆点云体积测量Web应用"""

    def __init__(self):
        """初始化Web应用"""
        self.processor = None
        self.current_file = None
        self.output_dir = Path("./volume_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_analyze_cloud(self, file_obj):
        """
        加载并分析点云文件

        Args:
            file_obj: Gradio文件对象

        Returns:
            分析结果文本和可视化图像
        """
        if file_obj is None:
            return "请先上传点云文件", None

        try:
            # 保存上传的文件
            self.current_file = file_obj.name
            self.processor = CoalPileVolumeProcessor()

            # 加载点云
            info = self.processor.load_point_cloud(self.current_file)

            # 生成分析报告
            report = f"""
## 点云文件分析结果

**文件路径**: `{self.current_file}`

### 基本信息
- **点数量**: {info['点数量']:,} 个点
- **是否有颜色**: {'是' if info['是否有颜色'] else '否'}
- **是否有法向量**: {'是' if info['是否有法向量'] else '否'}

### 边界框信息
- **最小值**: [{info['边界框']['最小值'][0]:.3f}, {info['边界框']['最小值'][1]:.3f}, {info['边界框']['最小值'][2]:.3f}]
- **最大值**: [{info['边界框']['最大值'][0]:.3f}, {info['边界框']['最大值'][1]:.3f}, {info['边界框']['最大值'][2]:.3f}]
- **尺寸**: [{info['边界框']['尺寸'][0]:.3f}, {info['边界框']['尺寸'][1]:.3f}, {info['边界框']['尺寸'][2]:.3f}] 米

✅ 点云加载成功，可以进行下一步处理
"""

            # 生成可视化图像
            vis_image = self._visualize_point_cloud(self.processor.point_cloud)

            return report, vis_image

        except Exception as e:
            return f"❌ 加载点云失败: {str(e)}", None

    def preprocess_cloud(self, voxel_size, nb_neighbors, std_ratio):
        """
        预处理点云

        Args:
            voxel_size: 体素下采样大小
            nb_neighbors: 邻居数量
            std_ratio: 标准差比率

        Returns:
            处理结果文本和可视化图像
        """
        if self.processor is None or self.processor.point_cloud is None:
            return "请先加载点云文件", None

        try:
            result = self.processor.preprocess_point_cloud(
                voxel_size=voxel_size,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )

            report = f"""
## 点云预处理结果

### 处理统计
- **原始点数**: {result['原始点数']:,} 个点
- **下采样后点数**: {result['下采样后点数']:,} 个点
- **去除离群点后点数**: {result['去除离群点后点数']:,} 个点
- **保留比例**: {result['保留比例']}

### 处理参数
- **体素大小**: {voxel_size} 米
- **邻居数量**: {nb_neighbors}
- **标准差比率**: {std_ratio}

✅ 预处理完成，可以进行地面分割
"""

            vis_image = self._visualize_point_cloud(self.processor.processed_cloud)

            return report, vis_image

        except Exception as e:
            return f"❌ 预处理失败: {str(e)}", None

    def segment_ground(self, distance_threshold, ransac_n, num_iterations):
        """
        分割地面平面

        Args:
            distance_threshold: 距离阈值
            ransac_n: RANSAC采样点数
            num_iterations: 迭代次数

        Returns:
            分割结果文本和可视化图像
        """
        if self.processor is None or self.processor.processed_cloud is None:
            return "请先执行预处理", None

        try:
            result = self.processor.segment_ground_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )

            plane = result['地面平面方程']
            report = f"""
## 地面分割结果

### 分割统计
- **地面点数**: {result['地面点数']:,} 个点
- **料堆点数**: {result['料堆点数']:,} 个点
- **料堆占比**: {result['料堆占比']}

### 地面平面方程
```
{plane[0]:.4f}x + {plane[1]:.4f}y + {plane[2]:.4f}z + {plane[3]:.4f} = 0
```

### 处理参数
- **距离阈值**: {distance_threshold} 米
- **RANSAC采样点数**: {ransac_n}
- **迭代次数**: {num_iterations}

✅ 地面分割完成，可以进行体积计算
"""

            # 可视化分割结果（地面+料堆）
            vis_image = self._visualize_segmented_cloud()

            return report, vis_image

        except Exception as e:
            return f"❌ 地面分割失败: {str(e)}", None

    def cluster_piles(self, eps, min_points):
        """
        聚类分割多个料堆

        Args:
            eps: DBSCAN邻域半径
            min_points: 最小点数

        Returns:
            聚类结果文本和可视化图像
        """
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行地面分割", None

        try:
            result = self.processor.cluster_piles(eps=eps, min_points=min_points)

            report = f"""
## 料堆聚类结果

### 聚类统计
- **检测到的料堆数量**: {result['料堆数量']}

### 各料堆信息
"""

            for pile_info in result['料堆信息']:
                report += f"""
#### 料堆 #{pile_info['簇编号']}
- **点数**: {pile_info['点数']:,} 个点
- **中心位置**: [{pile_info['中心'][0]:.3f}, {pile_info['中心'][1]:.3f}, {pile_info['中心'][2]:.3f}]
- **边界框最小值**: [{pile_info['边界框']['最小值'][0]:.3f}, {pile_info['边界框']['最小值'][1]:.3f}, {pile_info['边界框']['最小值'][2]:.3f}]
- **边界框最大值**: [{pile_info['边界框']['最大值'][0]:.3f}, {pile_info['边界框']['最大值'][1]:.3f}, {pile_info['边界框']['最大值'][2]:.3f}]
"""

            report += f"""
### 处理参数
- **DBSCAN邻域半径**: {eps} 米
- **最小点数**: {min_points}

✅ 聚类完成，可以进行体积计算
"""

            vis_image = self._visualize_clustered_piles()

            return report, vis_image

        except Exception as e:
            return f"❌ 聚类失败: {str(e)}", None

    def calculate_volume(self, pile_index, method):
        """
        计算料堆体积

        Args:
            pile_index: 料堆索引
            method: 计算方法

        Returns:
            体积计算结果文本
        """
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行地面分割或聚类"

        try:
            result = self.processor.calculate_pile_volume(
                pile_index=pile_index,
                method=method
            )

            report = f"""
## 料堆体积计算结果

### 料堆 #{result['料堆索引']}

#### 基本信息
- **点数**: {result['点数']:,} 个点
- **堆顶位置**: [{result['堆顶位置'][0]:.3f}, {result['堆顶位置'][1]:.3f}, {result['堆顶位置'][2]:.3f}]
- **地面高度**: {result['地面高度']:.3f} 米
- **料堆高度**: {result['料堆高度']:.3f} 米

#### 边界框
- **最小值**: [{result['边界框']['最小值'][0]:.3f}, {result['边界框']['最小值'][1]:.3f}, {result['边界框']['最小值'][2]:.3f}]
- **最大值**: [{result['边界框']['最大值'][0]:.3f}, {result['边界框']['最大值'][1]:.3f}, {result['边界框']['最大值'][2]:.3f}]
- **尺寸**: [{result['边界框']['尺寸'][0]:.3f}, {result['边界框']['尺寸'][1]:.3f}, {result['边界框']['尺寸'][2]:.3f}] 米

#### 体积结果
- **体积**: **{result['体积(立方米)']:.4f} 立方米**
- **计算方法**: {result['计算方法']}

✅ 体积计算完成
"""

            return report

        except Exception as e:
            return f"❌ 体积计算失败: {str(e)}"

    def calculate_all_volumes(self, method):
        """
        计算所有料堆的体积

        Args:
            method: 计算方法

        Returns:
            所有料堆的体积计算结果
        """
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行地面分割或聚类"

        try:
            volume_results = []
            total_volume = 0.0

            for i in range(len(self.processor.pile_clouds)):
                result = self.processor.calculate_pile_volume(i, method=method)
                volume_results.append(result)
                total_volume += result['体积(立方米)']

            report = f"""
## 所有料堆体积计算结果

### 总体统计
- **料堆总数**: {len(volume_results)}
- **总体积**: **{total_volume:.4f} 立方米**
- **计算方法**: {method}

### 各料堆详细信息
"""

            for result in volume_results:
                report += f"""
#### 料堆 #{result['料堆索引']}
- **点数**: {result['点数']:,} 个点
- **料堆高度**: {result['料堆高度']:.3f} 米
- **体积**: **{result['体积(立方米)']:.4f} 立方米**
- **堆顶位置**: [{result['堆顶位置'][0]:.3f}, {result['堆顶位置'][1]:.3f}, {result['堆顶位置'][2]:.3f}]
"""

            report += "\n✅ 所有料堆体积计算完成"

            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"volume_report_{timestamp}.json"
            self.processor.generate_report(str(report_path), volume_results)

            report += f"\n\n📄 详细报告已保存至: `{report_path}`"

            return report

        except Exception as e:
            return f"❌ 体积计算失败: {str(e)}"

    def _visualize_point_cloud(self, cloud):
        """可视化点云并返回图像"""
        try:
            # 创建可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(cloud)

            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)

            # 渲染并保存图像
            vis.poll_events()
            vis.update_renderer()

            temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            vis.capture_screen_image(temp_image.name)
            vis.destroy_window()

            return temp_image.name

        except Exception as e:
            print(f"可视化失败: {str(e)}")
            return None

    def _visualize_segmented_cloud(self):
        """可视化分割后的点云"""
        if not self.processor.pile_clouds:
            return None

        try:
            # 合并料堆点云进行可视化
            combined = self.processor.pile_clouds[0]
            for pile in self.processor.pile_clouds[1:]:
                combined += pile

            return self._visualize_point_cloud(combined)

        except Exception as e:
            print(f"可视化失败: {str(e)}")
            return None

    def _visualize_clustered_piles(self):
        """可视化聚类后的料堆"""
        return self._visualize_segmented_cloud()

    def create_interface(self):
        """创建Gradio界面"""

        with gr.Blocks(title="煤堆点云体积测量系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🏔️ 煤堆点云体积测量系统

                这是一个基于点云处理的煤堆体积测量系统。系统支持点云加载、预处理、地面分割、料堆聚类和体积计算等功能。

                ## 📋 处理流程

                1. **上传点云文件** - 上传.ply格式的点云文件
                2. **点云预处理** - 下采样、去噪、离群点去除
                3. **地面分割** - 使用RANSAC分割地面平面
                4. **料堆聚类** (可选) - 如果有多个料堆，进行聚类分割
                5. **体积计算** - 计算料堆体积

                ✅ **系统状态**: 已就绪
                """
            )

            with gr.Tabs():
                # Tab 1: 点云加载与分析
                with gr.Tab("1️⃣ 点云加载"):
                    gr.Markdown("### 上传点云文件")

                    with gr.Row():
                        with gr.Column(scale=1):
                            file_input = gr.File(
                                label="选择点云文件 (.ply)",
                                file_types=[".ply"]
                            )
                            load_btn = gr.Button("🔍 加载并分析", variant="primary")

                        with gr.Column(scale=1):
                            load_output = gr.Markdown(label="分析结果")

                    load_vis = gr.Image(label="点云可视化")

                    load_btn.click(
                        self.load_and_analyze_cloud,
                        inputs=[file_input],
                        outputs=[load_output, load_vis]
                    )

                # Tab 2: 点云预处理
                with gr.Tab("2️⃣ 点云预处理"):
                    gr.Markdown("### 点云预处理参数")

                    with gr.Row():
                        with gr.Column(scale=1):
                            voxel_size = gr.Slider(
                                minimum=0.001,
                                maximum=0.05,
                                value=0.01,
                                step=0.001,
                                label="体素下采样大小 (米)",
                                info="较小的值保留更多细节，但处理较慢"
                            )

                            nb_neighbors = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=20,
                                step=5,
                                label="统计离群点去除 - 邻居数量",
                                info="用于评估每个点的邻域"
                            )

                            std_ratio = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                value=2.0,
                                step=0.5,
                                label="统计离群点去除 - 标准差比率",
                                info="较大的值保留更多点"
                            )

                            preprocess_btn = gr.Button("🔧 执行预处理", variant="primary")

                        with gr.Column(scale=1):
                            preprocess_output = gr.Markdown(label="预处理结果")

                    preprocess_vis = gr.Image(label="预处理后的点云")

                    preprocess_btn.click(
                        self.preprocess_cloud,
                        inputs=[voxel_size, nb_neighbors, std_ratio],
                        outputs=[preprocess_output, preprocess_vis]
                    )

                # Tab 3: 地面分割
                with gr.Tab("3️⃣ 地面分割"):
                    gr.Markdown("### 地面平面分割参数")

                    with gr.Row():
                        with gr.Column(scale=1):
                            distance_threshold = gr.Slider(
                                minimum=0.005,
                                maximum=0.1,
                                value=0.02,
                                step=0.005,
                                label="距离阈值 (米)",
                                info="点到平面的最大距离"
                            )

                            ransac_n = gr.Slider(
                                minimum=3,
                                maximum=10,
                                value=3,
                                step=1,
                                label="RANSAC采样点数",
                                info="拟合平面所需的最小点数"
                            )

                            num_iterations = gr.Slider(
                                minimum=100,
                                maximum=5000,
                                value=1000,
                                step=100,
                                label="RANSAC迭代次数",
                                info="较多的迭代次数提高准确性"
                            )

                            segment_btn = gr.Button("✂️ 执行地面分割", variant="primary")

                        with gr.Column(scale=1):
                            segment_output = gr.Markdown(label="分割结果")

                    segment_vis = gr.Image(label="分割后的点云 (灰色=地面, 棕色=料堆)")

                    segment_btn.click(
                        self.segment_ground,
                        inputs=[distance_threshold, ransac_n, num_iterations],
                        outputs=[segment_output, segment_vis]
                    )

                # Tab 4: 料堆聚类 (可选)
                with gr.Tab("4️⃣ 料堆聚类 (可选)"):
                    gr.Markdown("### 料堆聚类参数 (如果场景中有多个料堆)")

                    with gr.Row():
                        with gr.Column(scale=1):
                            cluster_eps = gr.Slider(
                                minimum=0.01,
                                maximum=0.5,
                                value=0.05,
                                step=0.01,
                                label="DBSCAN邻域半径 (米)",
                                info="较大的值会合并更多点"
                            )

                            cluster_min_points = gr.Slider(
                                minimum=50,
                                maximum=500,
                                value=100,
                                step=50,
                                label="最小点数",
                                info="形成簇所需的最小点数"
                            )

                            cluster_btn = gr.Button("🔍 执行聚类", variant="primary")

                        with gr.Column(scale=1):
                            cluster_output = gr.Markdown(label="聚类结果")

                    cluster_vis = gr.Image(label="聚类后的料堆 (不同颜色表示不同料堆)")

                    cluster_btn.click(
                        self.cluster_piles,
                        inputs=[cluster_eps, cluster_min_points],
                        outputs=[cluster_output, cluster_vis]
                    )

                # Tab 5: 体积计算
                with gr.Tab("5️⃣ 体积计算"):
                    gr.Markdown("### 料堆体积计算")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 单个料堆计算")

                            pile_index = gr.Number(
                                value=0,
                                label="料堆索引",
                                info="从0开始的料堆编号",
                                precision=0
                            )

                            volume_method_single = gr.Radio(
                                choices=["convex_hull", "grid"],
                                value="convex_hull",
                                label="体积计算方法",
                                info="convex_hull: 凸包法 (快速), grid: 网格法 (精确)"
                            )

                            calc_single_btn = gr.Button("📊 计算单个料堆体积", variant="secondary")

                            single_volume_output = gr.Markdown(label="单个料堆体积结果")

                        with gr.Column(scale=1):
                            gr.Markdown("#### 所有料堆计算")

                            volume_method_all = gr.Radio(
                                choices=["convex_hull", "grid"],
                                value="convex_hull",
                                label="体积计算方法",
                                info="convex_hull: 凸包法 (快速), grid: 网格法 (精确)"
                            )

                            calc_all_btn = gr.Button("📊 计算所有料堆体积", variant="primary")

                            all_volume_output = gr.Markdown(label="所有料堆体积结果")

                    calc_single_btn.click(
                        self.calculate_volume,
                        inputs=[pile_index, volume_method_single],
                        outputs=[single_volume_output]
                    )

                    calc_all_btn.click(
                        self.calculate_all_volumes,
                        inputs=[volume_method_all],
                        outputs=[all_volume_output]
                    )

            gr.Markdown(
                """
                ---
                ### 💡 使用提示

                1. **参数调整建议**:
                   - 体素大小: 0.005-0.01米适合大多数场景
                   - 地面分割阈值: 0.01-0.03米
                   - 聚类半径: 根据料堆间距调整

                2. **体积计算方法**:
                   - **凸包法**: 计算速度快，适合形状规则的料堆
                   - **网格法**: 计算精确，适合形状不规则的料堆

                3. **注意事项**:
                   - 确保点云质量良好，噪声较少
                   - 地面应相对平坦
                   - 料堆之间应有明显间隔（如需聚类）

                ### 📄 输出文件

                - 处理后的点云: `volume_output/processed_pile.ply`
                - 体积报告: `volume_output/volume_report_<timestamp>.json`
                """
            )

        return demo


def main():
    """主函数"""
    try:
        print("🚀 正在初始化煤堆点云体积测量系统...")
        app = CoalPileVolumeWebApp()

        print("🎨 正在创建Web界面...")
        demo = app.create_interface()

        print("🌐 启动Web服务器...")
        print("📱 访问地址: http://localhost:7867")
        print("🔗 如需外部访问，请使用服务器IP地址")

        demo.launch(
            server_name="0.0.0.0",
            server_port=7867,
            share=False,
            debug=False,
            show_error=True
        )

    except Exception as e:
        print(f"❌ 启动Web应用失败: {str(e)}")
        print("\n🔧 可能的解决方案:")
        print("1. 确保已安装gradio: pip install gradio")
        print("2. 确保已安装open3d: pip install open3d")
        print("3. 检查端口7867是否被占用")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
