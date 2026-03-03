#!/usr/bin/env python3
"""
煤堆点云精细处理分析系统
附属界面：上传PLY点云文件，执行完整处理流程
流程：预处理 → 点云拼接 → 地面分割 → 细化处理 → 堆顶计算 → 曲面重构 → 边界计算 → 体积计算
端口：7869
"""

import os
import gradio as gr
import numpy as np
import open3d as o3d
from pathlib import Path
import json
from datetime import datetime
import tempfile
import trimesh

from coal_pile_volume_processor import CoalPileVolumeProcessor


class PLYAnalyzerApp:
    """点云精细处理分析Web应用"""

    def __init__(self):
        self.processor = None
        self.output_dir = Path("./ply_analysis_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_vis_cloud = None  # 当前可视化点云

    # ─────────────────────────────────────────────
    # Step 1: 加载点云
    # ─────────────────────────────────────────────
    def load_cloud(self, file_obj):
        if file_obj is None:
            return "请先上传点云文件（.ply）", None
        try:
            self.processor = CoalPileVolumeProcessor()
            info = self.processor.load_point_cloud(file_obj.name)
            self._current_vis_cloud = self.processor.point_cloud

            report = f"""## 点云加载成功

**文件**: `{os.path.basename(file_obj.name)}`

| 属性 | 值 |
|------|-----|
| 点数量 | {info['点数量']:,} 个点 |
| 含颜色 | {'是' if info['是否有颜色'] else '否'} |
| 含法向量 | {'是' if info['是否有法向量'] else '否'} |
| X 范围 | {info['边界框']['最小值'][0]:.3f} ~ {info['边界框']['最大值'][0]:.3f} m |
| Y 范围 | {info['边界框']['最小值'][1]:.3f} ~ {info['边界框']['最大值'][1]:.3f} m |
| Z 范围 | {info['边界框']['最小值'][2]:.3f} ~ {info['边界框']['最大值'][2]:.3f} m |
| 尺寸 | {info['边界框']['尺寸'][0]:.3f} × {info['边界框']['尺寸'][1]:.3f} × {info['边界框']['尺寸'][2]:.3f} m |

✅ 加载完成，可进行下一步预处理"""
            return report, self._make_glb(self.processor.point_cloud)
        except Exception as e:
            return f"❌ 加载失败: {str(e)}", None

    # ─────────────────────────────────────────────
    # Step 2: 预处理
    # ─────────────────────────────────────────────
    def preprocess(self, voxel_size, nb_neighbors, std_ratio, remove_layers):
        if self.processor is None or self.processor.point_cloud is None:
            return "请先加载点云文件", None
        try:
            result = self.processor.preprocess_point_cloud(
                voxel_size=voxel_size,
                nb_neighbors=int(nb_neighbors),
                std_ratio=std_ratio,
                remove_layers=remove_layers
            )

            layer_info = ""
            if result['分层剔除'] != "未执行":
                layer_info = f"\n| 分层剔除后 | {result['分层剔除']['剔除后点数']:,} |"

            report = f"""## 预处理完成

| 阶段 | 点数 |
|------|------|
| 原始点数 | {result['原始点数']:,} |{layer_info}
| 下采样后 | {result['下采样后点数']:,} |
| 去离群点后 | {result['去除离群点后点数']:,} |
| 保留比例 | {result['保留比例']} |

**参数**: 体素={voxel_size}m, 邻居={int(nb_neighbors)}, 标准差比率={std_ratio}, 分层剔除={'是' if remove_layers else '否'}

✅ 预处理完成，已估计法向量，可进行地面分割"""
            return report, self._make_glb(self.processor.processed_cloud)
        except Exception as e:
            return f"❌ 预处理失败: {str(e)}", None

    # ─────────────────────────────────────────────
    # Step 3: 地面分割
    # ─────────────────────────────────────────────
    def segment_ground(self, distance_threshold, ransac_n, num_iterations, keep_all_points, ground_method="lowest_points"):
        if self.processor is None or self.processor.processed_cloud is None:
            return "请先执行预处理", None
        try:
            result = self.processor.segment_ground_plane(
                distance_threshold=distance_threshold,
                ransac_n=int(ransac_n),
                num_iterations=int(num_iterations),
                keep_all_points=keep_all_points,
                method=ground_method
            )
            plane = result['地面平面方程']
            keep_info = ""
            if result.get('保留所有点'):
                keep_info = "\n\n✅ **保留所有点模式**：只识别地面，不剔除点，保留完整形态用于曲面重构"

            report = f"""## 地面识别完成

| 类别 | 值 |
|------|------|
| 识别方法 | {result.get('识别方法', ground_method)} |
| 地面点 | {result['地面点数']:,} |
| 料堆点 | {result['料堆点数']:,} |
| 地面上方点数 | {result.get('地面上方点数', 'N/A'):,} |
| 料堆占比 | {result['料堆占比']} |

**地面平面方程**:
```
{plane[0]:.4f}x + {plane[1]:.4f}y + {plane[2]:.4f}z + {plane[3]:.4f} = 0
```
{keep_info}

✅ 地面识别完成，绿色平面为识别出的地面（已调整确保所有点在上方）"""

            glb_path = self._make_glb_with_ground_plane(
                self.processor.pile_clouds[0],
                self.processor.ground_plane_mesh
            )
            return report, glb_path
        except Exception as e:
            return f"❌ 地面识别失败: {str(e)}", None

    def _create_coordinate_axes(self, scale=0.1):
        """
        创建三维坐标轴用于可视化

        Args:
            scale: 坐标轴长度比例

        Returns:
            trimesh.Scene: 包含XYZ坐标轴的场景
        """
        import trimesh

        # 创建坐标轴（Z-up坐标系）
        # X轴 - 红色
        x_axis = trimesh.creation.cylinder(radius=scale*0.01, height=scale,
                                          sections=8)
        x_axis.apply_translation([scale/2, 0, 0])
        x_axis.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi/2, [0, 1, 0]))
        x_axis.visual.face_colors = [255, 0, 0, 255]  # 红色

        # Y轴 - 绿色
        y_axis = trimesh.creation.cylinder(radius=scale*0.01, height=scale,
                                          sections=8)
        y_axis.apply_translation([0, scale/2, 0])
        y_axis.apply_transform(trimesh.transformations.rotation_matrix(
            -np.pi/2, [1, 0, 0]))
        y_axis.visual.face_colors = [0, 255, 0, 255]  # 绿色

        # Z轴 - 蓝色
        z_axis = trimesh.creation.cylinder(radius=scale*0.01, height=scale,
                                          sections=8)
        z_axis.apply_translation([0, 0, scale/2])
        z_axis.visual.face_colors = [0, 0, 255, 255]  # 蓝色

        # 创建箭头（指示方向）
        arrow_scale = scale * 0.05

        # X轴箭头
        x_arrow = trimesh.creation.cone(radius=arrow_scale, height=arrow_scale*2,
                                       sections=8)
        x_arrow.apply_translation([scale, 0, 0])
        x_arrow.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi/2, [0, 1, 0]))
        x_arrow.visual.face_colors = [255, 0, 0, 255]

        # Y轴箭头
        y_arrow = trimesh.creation.cone(radius=arrow_scale, height=arrow_scale*2,
                                       sections=8)
        y_arrow.apply_translation([0, scale, 0])
        y_arrow.apply_transform(trimesh.transformations.rotation_matrix(
            -np.pi/2, [1, 0, 0]))
        y_arrow.visual.face_colors = [0, 255, 0, 255]

        # Z轴箭头
        z_arrow = trimesh.creation.cone(radius=arrow_scale, height=arrow_scale*2,
                                       sections=8)
        z_arrow.apply_translation([0, 0, scale])
        z_arrow.visual.face_colors = [0, 0, 255, 255]

        # 合并所有坐标轴
        axes = [x_axis, y_axis, z_axis, x_arrow, y_arrow, z_arrow]

        return axes

    def _make_glb_with_ground_plane(self, cloud: o3d.geometry.PointCloud,
                                     ground_mesh: o3d.geometry.TriangleMesh) -> str:
        """生成包含点云、地面平面和坐标轴的GLB文件"""
        try:
            import trimesh
            import tempfile

            points = np.asarray(cloud.points)
            if len(points) == 0:
                return None

            if cloud.has_colors():
                colors = (np.asarray(cloud.colors) * 255).astype(np.uint8)
            else:
                colors = np.tile([204, 102, 51], (len(points), 1)).astype(np.uint8)

            # 最多显示8000点
            if len(points) > 8000:
                idx = np.random.choice(len(points), 8000, replace=False)
                points = points[idx]
                colors = colors[idx]

            scene = trimesh.Scene()

            # 添加点云（Z-up转Y-up）
            pc = trimesh.PointCloud(self._zup_to_yup(points), colors=colors)
            scene.add_geometry(pc)

            # 添加地面平面网格（Z-up转Y-up）
            if ground_mesh is not None:
                ground_verts = self._zup_to_yup(np.asarray(ground_mesh.vertices))
                ground_faces = np.asarray(ground_mesh.triangles)
                ground_tm = trimesh.Trimesh(vertices=ground_verts, faces=ground_faces)
                ground_tm.visual.face_colors = [50, 200, 50, 150]  # 半透明绿色
                scene.add_geometry(ground_tm)

            # 添加坐标轴（计算合适的尺度）
            point_range = np.ptp(points, axis=0).max()  # 点云最大范围
            axis_scale = point_range * 0.3  # 坐标轴长度为点云范围的30%
            axes = self._create_coordinate_axes(scale=axis_scale)
            for axis in axes:
                # 转换坐标系（Z-up转Y-up）
                axis_verts = self._zup_to_yup(np.asarray(axis.vertices))
                axis_tm = trimesh.Trimesh(vertices=axis_verts,
                                         faces=np.asarray(axis.faces))
                axis_tm.visual.face_colors = axis.visual.face_colors
                scene.add_geometry(axis_tm)

            tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            scene.export(tmp.name)
            return tmp.name
        except Exception as e:
            print(f"GLB生成失败: {e}")
            return None

    # ─────────────────────────────────────────────
    # Step 4: 细化处理（迭代鲁棒性评估）
    # ─────────────────────────────────────────────
    def refine(self, iterations, nb_neighbors, std_ratio, min_retention, skip_refine):
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行地面分割", None
        try:
            result = self.processor.refine_point_cloud(
                pile_index=0,
                iterations=int(iterations),
                nb_neighbors=int(nb_neighbors),
                std_ratio=std_ratio,
                min_retention_ratio=min_retention,
                skip_refine=skip_refine
            )

            skip_info = ""
            if result.get('跳过细化'):
                skip_info = "\n\n✅ **跳过细化模式**：保留所有点用于曲面重构，保持完整形态"

            report = f"""## 细化处理完成

| 阶段 | 点数 |
|------|------|
| 细化前 | {result['细化前点数']:,} |
| 细化后 | {result['细化后点数']:,} |
| 去除点数 | {result['去除点数']:,} |
| 保留比例 | {result['保留比例']} |

**参数**: 迭代={int(iterations)}次, 邻居={int(nb_neighbors)}, 标准差比率={std_ratio}, 最小保留={min_retention*100:.0f}%
{skip_info}

✅ 细化完成，可进行曲面重构"""
            return report, self._make_glb(self.processor.pile_clouds[0])
        except Exception as e:
            return f"❌ 细化处理失败: {str(e)}", None

    # ─────────────────────────────────────────────
    # Step 5: 曲面重构
    # ─────────────────────────────────────────────
    def reconstruct(self, method, depth):
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行细化处理", None
        try:
            result = self.processor.reconstruct_surface(
                pile_index=0,
                method=method,
                depth=int(depth)
            )
            report = f"""## 曲面重构完成

| 属性 | 值 |
|------|-----|
| 重构方法 | {result['重构方法']} |
| 顶点数 | {result['顶点数']:,} |
| 三角面数 | {result['三角面数']:,} |
| 是否水密 | {'✅ 是（可精确计算体积）' if result['是否水密'] else '⚠️ 否（将使用凸包法）'} |

✅ 曲面重构完成，可进行边界计算和体积计算"""

            # 可视化网格
            glb_path = self._make_glb_from_mesh(self.processor.mesh)
            return report, glb_path
        except Exception as e:
            return f"❌ 曲面重构失败: {str(e)}", None

    # ─────────────────────────────────────────────
    # Step 6: 边界与堆顶计算
    # ─────────────────────────────────────────────
    def calc_boundary(self):
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行地面分割"
        try:
            result = self.processor.calculate_boundary(pile_index=0)
            bbox = result['轴对齐边界框']
            obb = result['有向边界框']
            proj = result['底面投影']

            report = f"""## 边界与堆顶计算结果

### 堆顶信息
| 属性 | 值 |
|------|-----|
| 堆顶坐标 | [{result['堆顶位置'][0]:.3f}, {result['堆顶位置'][1]:.3f}, {result['堆顶位置'][2]:.3f}] |
| 地面高度 | {result['地面高度']:.3f} m |
| **料堆高度** | **{result['料堆高度']:.3f} m** |

### 轴对齐边界框
| 属性 | 值 |
|------|-----|
| 最小值 | [{bbox['最小值'][0]:.3f}, {bbox['最小值'][1]:.3f}, {bbox['最小值'][2]:.3f}] |
| 最大值 | [{bbox['最大值'][0]:.3f}, {bbox['最大值'][1]:.3f}, {bbox['最大值'][2]:.3f}] |
| 尺寸 | {bbox['尺寸'][0]:.3f} × {bbox['尺寸'][1]:.3f} × {bbox['尺寸'][2]:.3f} m |

### 有向边界框
| 属性 | 值 |
|------|-----|
| 中心 | [{obb['中心'][0]:.3f}, {obb['中心'][1]:.3f}, {obb['中心'][2]:.3f}] |
| 尺寸 | {obb['尺寸'][0]:.3f} × {obb['尺寸'][1]:.3f} × {obb['尺寸'][2]:.3f} m |

### 底面投影轮廓
| 属性 | 值 |
|------|-----|
| 凸包面积 | {proj['凸包面积(m²)']:.4f} m² |
| 凸包周长 | {proj['凸包周长(m)']:.4f} m |
| 轮廓顶点数 | {proj['轮廓顶点数']} |

✅ 边界计算完成，可进行体积计算"""
            return report
        except Exception as e:
            return f"❌ 边界计算失败: {str(e)}"

    # ─────────────────────────────────────────────
    # Step 7: 体积计算
    # ─────────────────────────────────────────────
    def calc_volume(self, method, coal_density):
        if self.processor is None or not self.processor.pile_clouds:
            return "请先执行地面分割", None, None

        try:
            result = self.processor.calculate_pile_volume(pile_index=0, method=method)
            volume = result['体积(立方米)']
            weight = volume * coal_density / 1000.0
            confidence = result.get('置信度', 0.0)

            multi_methods = ""
            if result.get('多方法结果'):
                multi_methods = "\n### 多方法对比\n| 方法 | 体积(m³) |\n|------|----------|\n"
                for m, v in result['多方法结果'].items():
                    multi_methods += f"| {m} | {v:.4f} |\n"

            report = f"""## 体积计算结果

### 核心指标
| 指标 | 值 |
|------|-----|
| **体积** | **{volume:.4f} m³** |
| **重量** | **{weight:.3f} 吨** |
| 计算方法 | {result['计算方法']} |
| 置信度 | {confidence:.2%} |
{multi_methods}
### 料堆信息
| 属性 | 值 |
|------|-----|
| 点数 | {result['点数']:,} |
| 料堆高度 | {result['料堆高度']:.3f} m |
| 地面高度 | {result['地面高度']:.3f} m |
| 堆顶位置 | [{result['堆顶位置'][0]:.3f}, {result['堆顶位置'][1]:.3f}, {result['堆顶位置'][2]:.3f}] |

### 边界框尺寸
{result['边界框']['尺寸'][0]:.3f} × {result['边界框']['尺寸'][1]:.3f} × {result['边界框']['尺寸'][2]:.3f} m

⚠️ **注意**: 当前结果为相对尺度，需要参考物进行绝对尺度标定

✅ 计算完成"""

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = self.output_dir / f"analysis_{timestamp}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 保存点云
            ply_path = str(out_dir / "refined_pile.ply")
            self.processor.save_processed_cloud(ply_path)

            # 保存网格（如果有）
            if self.processor.mesh is not None:
                mesh_path = str(out_dir / "surface_mesh.ply")
                self.processor.save_mesh(mesh_path)

            # 保存JSON报告
            json_path = str(out_dir / "analysis_report.json")
            self.processor.generate_report(json_path, [result])

            return report, ply_path, json_path

        except Exception as e:
            return f"❌ 体积计算失败: {str(e)}", None, None

    # ─────────────────────────────────────────────
    # 辅助：生成GLB可视化
    # ─────────────────────────────────────────────
    @staticmethod
    def _zup_to_yup(pts: np.ndarray) -> np.ndarray:
        """将Z-up坐标系转换为Y-up（GLB标准）: X→X, Y→-Z, Z→Y"""
        return np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])

    def _make_glb(self, cloud: o3d.geometry.PointCloud) -> str:
        """将点云转为GLB文件用于3D预览（带坐标轴）"""
        try:
            points = np.asarray(cloud.points)
            if len(points) == 0:
                return None

            if cloud.has_colors():
                colors = (np.asarray(cloud.colors) * 255).astype(np.uint8)
            else:
                colors = np.tile([204, 102, 51], (len(points), 1)).astype(np.uint8)

            # 最多显示8000点
            if len(points) > 8000:
                idx = np.random.choice(len(points), 8000, replace=False)
                points = points[idx]
                colors = colors[idx]

            scene = trimesh.Scene()
            pc = trimesh.PointCloud(self._zup_to_yup(points), colors=colors)
            scene.add_geometry(pc)

            # 添加坐标轴
            point_range = np.ptp(points, axis=0).max()
            axis_scale = point_range * 0.3
            axes = self._create_coordinate_axes(scale=axis_scale)
            for axis in axes:
                axis_verts = self._zup_to_yup(np.asarray(axis.vertices))
                axis_tm = trimesh.Trimesh(vertices=axis_verts,
                                         faces=np.asarray(axis.faces))
                axis_tm.visual.face_colors = axis.visual.face_colors
                scene.add_geometry(axis_tm)

            tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            scene.export(tmp.name)
            return tmp.name
        except Exception as e:
            print(f"GLB生成失败: {e}")
            return None

    def _make_glb_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> str:
        """将Open3D网格转为GLB"""
        try:
            vertices = self._zup_to_yup(np.asarray(mesh.vertices))
            faces = np.asarray(mesh.triangles)
            tm = trimesh.Trimesh(vertices=vertices, faces=faces)
            tm.visual.face_colors = [180, 120, 60, 200]

            scene = trimesh.Scene()
            scene.add_geometry(tm)

            tmp = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
            scene.export(tmp.name)
            return tmp.name
        except Exception as e:
            print(f"网格GLB生成失败: {e}")
            return None

    # ─────────────────────────────────────────────
    # 创建界面
    # ─────────────────────────────────────────────
    def create_interface(self):
        with gr.Blocks(title="煤堆点云精细处理分析系统") as demo:
            gr.Markdown("""
# 🔬 煤堆点云精细处理分析系统

上传由主系统生成的 `coal_pile.ply` 点云文件，执行完整的精细化处理流程。

**处理流程**: 加载 → 预处理 → 地面分割 → 细化处理 → 曲面重构 → 边界计算 → 体积计算
""")

            with gr.Tabs():

                # ── Tab 1: 加载 ──────────────────────────────
                with gr.Tab("1️⃣ 加载点云"):
                    gr.Markdown("上传 `.ply` 格式点云文件（来自主系统输出的 `coal_pile.ply`）")
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_input = gr.File(label="选择点云文件 (.ply)", file_types=[".ply"])
                            load_btn = gr.Button("🔍 加载并分析", variant="primary")
                        with gr.Column(scale=1):
                            load_out = gr.Markdown()
                    load_vis = gr.Model3D(label="点云预览", height=400, clear_color=[0.1, 0.1, 0.15, 1.0])
                    load_btn.click(self.load_cloud, inputs=[file_input], outputs=[load_out, load_vis])

                # ── Tab 2: 预处理 ─────────────────────────────
                with gr.Tab("2️⃣ 预处理"):
                    gr.Markdown("""
体素下采样 + 统计离群点去除 + 法向量估计

**默认参数已优化为煤堆最佳配置（往大估算）**：
- 体素=0.002m：保留更多细节
- 标准差=4.0：宽松标准，减少误删
- 开启分层剔除：只保留最外层表面点
""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            voxel_size = gr.Slider(0.001, 0.01, value=0.002, step=0.001,
                                                   label="体素下采样大小 (m)",
                                                   info="默认0.002m，保留更多点")
                            nb_neighbors = gr.Slider(10, 50, value=15, step=5,
                                                     label="统计邻居数量",
                                                     info="默认15")
                            std_ratio = gr.Slider(1.0, 5.0, value=4.0, step=0.5,
                                                  label="标准差比率",
                                                  info="默认4.0（宽松，往大估算）")
                            remove_layers = gr.Checkbox(value=True, label="剔除分层叠加点（保留最外层）",
                                                       info="✅ 推荐开启，保留煤堆表面形态")
                            pre_btn = gr.Button("🔧 执行预处理", variant="primary")
                        with gr.Column(scale=1):
                            pre_out = gr.Markdown()
                    pre_vis = gr.Model3D(label="预处理后点云", height=400, clear_color=[0.1, 0.1, 0.15, 1.0])
                    pre_btn.click(self.preprocess,
                                  inputs=[voxel_size, nb_neighbors, std_ratio, remove_layers],
                                  outputs=[pre_out, pre_vis])

                # ── Tab 3: 地面分割 ───────────────────────────
                with gr.Tab("3️⃣ 地面识别"):
                    gr.Markdown("""
识别地面位置，并自动调整确保所有点都在地面上方

**⚠️ 重大更新：基于物料堆真空特性的智能算法！**

**最新推荐方法（v4.0）**：
- **pile_aware（强烈推荐，NEW！）**：物料堆感知算法
  - ✅ 专门针对DUSt3R点云设计（地面点极少）
  - ✅ 基于物料堆内部真空特性
  - ✅ 分析不同高度层的XY投影面积
  - ✅ 自动判断地面方向（避免误判侧面）
  - ✅ 适应"顶端尖、地面宽"的物料堆特征
  - 原理：如果面积随高度递增 → 地面在底部（正确）

**核心洞察**（来自用户反馈）：
1. DUSt3R生成的点云主要是物料堆表面点，地面点极少
2. 物料堆内部是真空的（没有点）
3. 如果地面朝上（正确）：点云呈现凹陷，面积随高度递增
4. 如果地面朝下（错误）：点云呈现凸起，面积随高度递减

**其他方法**：
- **csf**：CSF布料模拟（适合大规模LiDAR点云）
- **deterministic**：完全确定性算法（快速、稳定）
- **coordinate_correction**：排除侧面策略
- **region_growing**：区域生长
- **max_cross_section**：多候选平面+水平度优先
- **adaptive**：多层次分析
- **lowest_points**：底部点直接拟合
- **ransac**：全局RANSAC（已固定随机种子）
- **convex_hull_base**：3D凸包底面

**核心改进**：
1. 新增pile_aware算法（基于物料堆真空特性）
2. 自动判断地面方向（避免误判侧面）
3. 专门优化DUSt3R点云特征

**自动调整**：识别后自动向下平移地面，确保所有点都在地面上方
""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ground_method = gr.Radio(
                                choices=["pile_aware", "csf", "deterministic", "coordinate_correction", "region_growing", "max_cross_section", "lowest_points", "adaptive", "ransac", "convex_hull_base"],
                                value="pile_aware",
                                label="地面识别方法",
                                info="强烈推荐pile_aware（基于物料堆真空特性，专为DUSt3R设计）"
                            )
                            dist_thresh = gr.Slider(0.005, 0.05, value=0.01, step=0.005,
                                                    label="距离阈值 (m)",
                                                    info="默认0.01m")
                            ransac_n = gr.Slider(3, 10, value=3, step=1,
                                                 label="RANSAC采样点数",
                                                 info="默认3")
                            num_iter = gr.Slider(100, 5000, value=1000, step=100,
                                                 label="迭代次数",
                                                 info="默认1000")
                            keep_all_points = gr.Checkbox(value=True, label="✅ 保留所有点（推荐）",
                                                         info="只识别地面，不剔除点，保留完整形态")
                            seg_btn = gr.Button("✂️ 执行地面识别", variant="primary")
                        with gr.Column(scale=1):
                            seg_out = gr.Markdown()
                    seg_vis = gr.Model3D(label="点云 + 绿色地面平面", height=400, clear_color=[0.1, 0.1, 0.15, 1.0])
                    seg_btn.click(self.segment_ground,
                                  inputs=[dist_thresh, ransac_n, num_iter, keep_all_points, ground_method],
                                  outputs=[seg_out, seg_vis])

                # ── Tab 4: 细化处理 ───────────────────────────
                with gr.Tab("4️⃣ 细化处理"):
                    gr.Markdown("""
**迭代鲁棒性评估**：通过统计滤波去除噪声点和离群点

**推荐策略**：✅ 跳过细化（保留所有点用于曲面重构）

如果点云质量较好，建议跳过细化，保留所有点以获得更真实的曲面重构和更大的体积估算。

> 参考方案：基于滤波的方式，将三维点云数据通过迭代过程对噪声点和离群点进行鲁棒性评估。
""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            skip_refine = gr.Checkbox(value=True, label="✅ 跳过细化（推荐）",
                                                     info="保留所有点，获得更真实的重构")
                            refine_iter = gr.Slider(1, 5, value=1, step=1,
                                                    label="迭代次数",
                                                    info="默认1次（如不跳过）")
                            refine_nb = gr.Slider(10, 50, value=20, step=5,
                                                  label="邻居数量",
                                                  info="默认20")
                            refine_std = gr.Slider(1.0, 4.0, value=3.0, step=0.5,
                                                   label="标准差比率",
                                                   info="默认3.0（宽松）")
                            min_retention = gr.Slider(0.5, 0.95, value=0.85, step=0.05,
                                                     label="最小保留比例",
                                                     info="默认0.85")
                            refine_btn = gr.Button("🎯 执行细化处理", variant="primary")
                        with gr.Column(scale=1):
                            refine_out = gr.Markdown()
                    refine_vis = gr.Model3D(label="处理后点云", height=400, clear_color=[0.1, 0.1, 0.15, 1.0])
                    refine_btn.click(self.refine,
                                     inputs=[refine_iter, refine_nb, refine_std, min_retention, skip_refine],
                                     outputs=[refine_out, refine_vis])

                # ── Tab 5: 曲面重构 ───────────────────────────
                with gr.Tab("5️⃣ 曲面重构"):
                    gr.Markdown("""
从点云重建三角网格曲面

**方法说明**：
- **pile_convex（推荐）**：堆料凸包重构，专为堆料设计
  - 基于高度图，每个位置取Z最大值（表面点）
  - 空缺处用邻域最大值填充（往大估算，不产生凹陷）
  - 底部封闭到地面，生成水密网格
  - 完全符合"垂直落布"原理
- **poisson**：泊松重构，点云空缺处可能产生大凹陷
- **bpa**：球旋转算法，空缺处曲面不完整
""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            recon_method = gr.Radio(
                                choices=["pile_convex", "poisson", "bpa"],
                                value="pile_convex",
                                label="重构方法",
                                info="pile_convex: 堆料专用（推荐）| poisson: 泊松 | bpa: 球旋转"
                            )
                            recon_depth = gr.Slider(6, 12, value=7, step=1,
                                                    label="泊松重构深度",
                                                    info="仅poisson方法有效，默认7")
                            recon_btn = gr.Button("🏗️ 执行曲面重构", variant="primary")
                        with gr.Column(scale=1):
                            recon_out = gr.Markdown()
                    recon_vis = gr.Model3D(label="重构曲面预览", height=400, clear_color=[0.1, 0.1, 0.15, 1.0])
                    recon_btn.click(self.reconstruct,
                                    inputs=[recon_method, recon_depth],
                                    outputs=[recon_out, recon_vis])

                # ── Tab 6: 边界与堆顶 ─────────────────────────
                with gr.Tab("6️⃣ 边界与堆顶"):
                    gr.Markdown("计算料堆的3D边界框、底面投影轮廓和堆顶位置")
                    with gr.Row():
                        with gr.Column(scale=1):
                            boundary_btn = gr.Button("📐 计算边界与堆顶", variant="primary")
                        with gr.Column(scale=1):
                            boundary_out = gr.Markdown()
                    boundary_btn.click(self.calc_boundary, outputs=[boundary_out])

                # ── Tab 7: 体积计算 ───────────────────────────
                with gr.Tab("7️⃣ 体积计算"):
                    gr.Markdown("""
基于精细处理后的点云计算最终体积和重量

**体积计算原理**（"垂直落布"法）：
1. 将煤堆点云投影到地面
2. 计算煤堆表面和地面之间的体积
3. 就像从上方垂直落下一块布，布和地面之间的空间就是煤堆体积

**默认方法**：convex_hull（凸包法，往大估算）

**方法选择**：
- **convex_hull（推荐）**：凸包法，体积略大（保守估计）
- **auto**：自动选择最佳方法
- **multi**：多方法融合，提供置信度评估
- **mesh**：需先完成曲面重构
""")
                    with gr.Row():
                        with gr.Column(scale=1):
                            vol_method = gr.Radio(
                                choices=["convex_hull", "auto", "multi", "mesh", "grid"],
                                value="convex_hull",
                                label="体积计算方法",
                                info="默认convex_hull（往大估算）"
                            )
                            coal_density = gr.Slider(1000, 1600, value=1300, step=10,
                                                     label="煤炭密度 (kg/m³)",
                                                     info="原煤: 1200-1400 | 精煤: 1300-1500")
                            vol_btn = gr.Button("📊 计算体积与重量", variant="primary", size="lg")
                        with gr.Column(scale=1):
                            vol_out = gr.Markdown()

                    with gr.Row():
                        ply_out = gr.File(label="📦 精细处理后点云 (.ply)")
                        json_out = gr.File(label="📄 分析报告 (.json)")

                    vol_btn.click(self.calc_volume,
                                  inputs=[vol_method, coal_density],
                                  outputs=[vol_out, ply_out, json_out])

            gr.Markdown("""
---
### 💡 煤堆测量最佳实践（保留完整形态）

#### 推荐流程（保留所有点，往大估算）
1. **预处理**：体素=0.002m, 标准差=4.0, ✅开启分层剔除
2. **地面识别**：✅保留所有点（只识别地面，不剔除）
3. **细化处理**：✅跳过细化（保留所有点用于重构）
4. **曲面重构**：泊松深度=7-8
5. **体积计算**：凸包法或网格法

#### 核心策略："垂直落布"法

**原理**：
1. 识别地面平面（不剔除点）
2. 保留所有点进行曲面重构
3. 就像从上方垂直落下一块布，贴合煤堆表面
4. 布和地面之间的空间 = 煤堆体积

**优势**：
- ✅ 保留完整形态，曲面重构更真实
- ✅ 不剔除点，体积更大（往大估算）
- ✅ 符合保守估计原则

#### ⚠️ 尺度标定（必须！）
- DUSt3R输出的是相对尺度，需要参考物标定
- 建议在现场放置1m标尺
- 真实体积 = 点云体积 × (缩放因子)³

### 📁 输出文件
- 精细处理后点云：`ply_analysis_output/analysis_<时间戳>/refined_pile.ply`
- 曲面网格：`ply_analysis_output/analysis_<时间戳>/surface_mesh.ply`
- 分析报告：`ply_analysis_output/analysis_<时间戳>/analysis_report.json`
""")

        return demo


def main():
    print("=" * 60)
    print("🔬 煤堆点云精细处理分析系统 - 启动中...")
    print("=" * 60)

    app = PLYAnalyzerApp()
    demo = app.create_interface()

    print("🌐 启动Web服务器...")
    print("📱 访问地址: http://localhost:7869")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7869,
        share=False,
        debug=False,
        show_error=True,
        inbrowser=False
    )


if __name__ == "__main__":
    main()
