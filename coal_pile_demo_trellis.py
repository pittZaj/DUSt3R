#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤堆体积测算Demo - TRELLIS.2版本
使用TRELLIS.2模型从单张图片生成3D模型并计算体积
"""

import gradio as gr
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用RTX 5880 Ada (GPU 0)
os.environ["ATTN_BACKEND"] = "sdpa"  # 使用PyTorch原生的scaled_dot_product_attention

import sys
sys.path.insert(0, '/mnt/data3/clip/DUSt3R/TRELLIS.2-main')

import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import trimesh
import torch
import plotly.graph_objects as go

print("正在加载TRELLIS.2模型...")
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel
    print("✓ TRELLIS.2模块导入成功")
    print("✓ o-voxel模块导入成功")
except Exception as e:
    print(f"✗ TRELLIS.2模块导入失败: {e}")
    print("将使用简化的深度估计方法")


class TrellisCoalPileEstimator:
    """使用TRELLIS.2的煤堆体积估算器"""

    def __init__(self):
        print("=" * 60)
        print("初始化 TrellisCoalPileEstimator")
        print("=" * 60)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ 使用设备: {self.device}")

        if torch.cuda.is_available():
            print(f"✓ GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # 尝试加载TRELLIS.2模型
        self.pipeline = None
        try:
            print("\n" + "=" * 60)
            print("开始加载TRELLIS.2-4B模型...")
            print("=" * 60)
            print("步骤1/3: 读取配置文件...")
            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("/mnt/data3/clip/DUSt3R/TRELLIS.2-4B")
            print("✓ 配置文件读取成功")

            print("步骤2/3: 加载模型权重到GPU...")
            self.pipeline.cuda()
            print("✓ 模型已加载到GPU")

            print("步骤3/3: 模型初始化完成")
            print("=" * 60)
            print("✓ TRELLIS.2模型加载成功！")
            print("=" * 60)
        except Exception as e:
            print("=" * 60)
            print(f"✗ TRELLIS.2模型加载失败: {e}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            print("将使用备用方法")
            print("=" * 60)

    def generate_3d_from_image_trellis(self, image):
        """使用TRELLIS.2从图片生成3D模型"""
        if self.pipeline is None:
            raise Exception("TRELLIS.2模型未加载")

        print("使用TRELLIS.2生成3D模型...")
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image

        # 运行TRELLIS.2
        mesh = self.pipeline.run(pil_image)[0]
        mesh.simplify(16777216)  # nvdiffrast limit

        # 返回TRELLIS mesh对象（包含vertices, faces, attrs等）
        return mesh

    def fallback_depth_estimation(self, image):
        """备用深度估计方法"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        depth = (255 - gray).astype(np.float32) / 255.0
        depth = cv2.GaussianBlur(depth, (5, 5), 0)
        return depth

    def create_point_cloud_from_depth(self, image, depth, step=2):
        """从深度图创建点云"""
        h, w = depth.shape
        points = []
        colors = []

        for y in range(0, h, step):
            for x in range(0, w, step):
                z = depth[y, x]
                if z > 0.01:
                    px = (x / w - 0.5) * 2
                    py = (y / h - 0.5) * 2
                    pz = z
                    points.append([px, py, pz])

                    if len(image.shape) == 3:
                        color = image[y, x] / 255.0
                        colors.append(color)
                    else:
                        gray_val = image[y, x] / 255.0
                        colors.append([gray_val, gray_val, gray_val])

        return np.array(points), np.array(colors)

    def create_mesh_from_points(self, points, colors):
        """从点云创建网格"""
        try:
            cloud = trimesh.points.PointCloud(points, colors=colors)
            mesh = cloud.convex_hull
            return mesh
        except Exception as e:
            print(f"网格创建失败: {e}")
            mesh = trimesh.creation.box(extents=[1, 1, 1])
            return mesh

    def export_glb(self, mesh, output_path):
        """导出GLB格式（使用o-voxel）"""
        try:
            import o_voxel
            print("导出GLB格式...")

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000,
                texture_size=4096,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True
            )
            glb.export(output_path, extension_webp=True)
            print(f"✓ GLB文件已保存: {output_path}")
            return output_path
        except Exception as e:
            print(f"✗ GLB导出失败: {e}")
            return None

    def calculate_volume(self, mesh):
        """计算网格体积"""
        try:
            # 如果是TRELLIS mesh对象，转换为trimesh
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                if torch.is_tensor(mesh.vertices):
                    vertices = mesh.vertices.cpu().numpy()
                    faces = mesh.faces.cpu().numpy()
                    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                else:
                    trimesh_mesh = mesh
            else:
                trimesh_mesh = mesh

            if trimesh_mesh.is_watertight:
                volume = trimesh_mesh.volume
            else:
                trimesh_mesh.fill_holes()
                volume = trimesh_mesh.volume if trimesh_mesh.is_watertight else trimesh_mesh.convex_hull.volume
            return abs(volume)
        except Exception as e:
            print(f"体积计算失败: {e}")
            return 0.0

    def create_interactive_3d(self, mesh, output_path):
        """创建可交互的3D模型"""
        try:
            # 如果是TRELLIS mesh对象，转换为numpy
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                if torch.is_tensor(mesh.vertices):
                    vertices = mesh.vertices.cpu().numpy()
                    faces = mesh.faces.cpu().numpy()
                else:
                    vertices = mesh.vertices
                    faces = mesh.faces
            else:
                vertices = mesh.vertices
                faces = mesh.faces

            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color='brown',
                    opacity=0.8,
                    name='Coal Pile',
                    hoverinfo='text',
                    text=f'煤堆3D模型 (TRELLIS.2生成)'
                )
            ])

            fig.update_layout(
                title='煤堆3D重建模型 - TRELLIS.2生成（可交互）',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z (高度)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='data'
                ),
                width=800,
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            html_path = output_path.replace('.png', '_interactive.html')
            fig.write_html(html_path, include_plotlyjs='cdn', auto_open=False)

            return html_path

        except Exception as e:
            print(f"交互式3D创建失败: {e}")
            return None

    def estimate_coal_pile(self, image, coal_density=1.3, scale_factor=1.0, use_trellis=True):
        """估算煤堆体积和重量"""
        print("开始3D重建...")

        try:
            if use_trellis and self.pipeline is not None:
                # 使用TRELLIS.2生成3D模型
                trellis_mesh = self.generate_3d_from_image_trellis(image)
                method = "TRELLIS.2"

                # 转换为trimesh用于体积计算
                vertices = trellis_mesh.vertices.cpu().numpy()
                faces = trellis_mesh.faces.cpu().numpy()
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                point_count = len(vertices)

                # 保存TRELLIS mesh用于GLB导出
                trellis_mesh_obj = trellis_mesh
            else:
                # 使用备用方法
                print("使用备用深度估计方法...")
                depth = self.fallback_depth_estimation(image)
                points, colors = self.create_point_cloud_from_depth(image, depth, step=2)
                mesh = self.create_mesh_from_points(points, colors)
                method = "Fallback"
                point_count = len(points)
                trellis_mesh_obj = None

            print(f"生成的网格: {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面")

        except Exception as e:
            print(f"TRELLIS.2生成失败: {e}")
            print("回退到备用方法...")
            depth = self.fallback_depth_estimation(image)
            points, colors = self.create_point_cloud_from_depth(image, depth, step=2)
            mesh = self.create_mesh_from_points(points, colors)
            method = "Fallback"
            point_count = len(points)
            trellis_mesh_obj = None

        # 计算体积
        print("计算体积...")
        volume_model_units = self.calculate_volume(mesh)
        volume_m3 = volume_model_units * (scale_factor ** 3)
        weight_tons = volume_m3 * coal_density

        print(f"体积: {volume_m3:.2f} 立方米")
        print(f"重量: {weight_tons:.2f} 吨")

        # 保存输出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/mnt/data3/clip/DUSt3R/coal_pile_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # 保存网格
        mesh_path = os.path.join(output_dir, "coal_pile.ply")
        mesh.export(mesh_path)

        # 如果使用TRELLIS.2，尝试导出GLB
        glb_path = None
        if trellis_mesh_obj is not None:
            glb_path = os.path.join(output_dir, "coal_pile.glb")
            glb_result = self.export_glb(trellis_mesh_obj, glb_path)
            if glb_result is None:
                glb_path = None

        # 创建可交互的3D模型
        print("生成可交互3D模型...")
        vis_path = os.path.join(output_dir, "visualization.png")
        html_path = self.create_interactive_3d(mesh, vis_path)

        return {
            "volume_m3": volume_m3,
            "weight_tons": weight_tons,
            "mesh_path": mesh_path,
            "glb_path": glb_path,
            "html_path": html_path,
            "output_dir": output_dir,
            "point_count": point_count,
            "method": method
        }


def create_demo():
    """创建Gradio演示界面"""

    estimator = TrellisCoalPileEstimator()

    def process_image(image, coal_density, scale_factor, use_trellis):
        """处理图像并返回结果"""
        try:
            if image is None:
                return None, "请上传图像"

            if isinstance(image, Image.Image):
                image = np.array(image)

            # 估算体积
            result = estimator.estimate_coal_pile(
                image,
                coal_density=coal_density,
                scale_factor=scale_factor,
                use_trellis=use_trellis
            )

            # 准备输出信息
            glb_info = ""
            if result.get('glb_path'):
                glb_info = f"- **GLB模型**: `{result['glb_path']}`\n"

            info = f"""
## 煤堆体积测算结果 - TRELLIS.2版本

### 测算数据
- **体积**: {result['volume_m3']:.2f} 立方米
- **预估重量**: {result['weight_tons']:.2f} 吨
- **顶点/点数**: {result['point_count']}
- **使用方法**: {result['method']}

### 参数设置
- **煤炭密度**: {coal_density} 吨/立方米
- **尺度因子**: {scale_factor}

### 输出文件
- 3D网格 (PLY): `{result['mesh_path']}`
{glb_info}- **可交互3D模型 (HTML)**: `{result['html_path']}`

### 关于TRELLIS.2
TRELLIS.2是微软开发的4B参数大型3D生成模型，可以从单张图片生成高质量的3D模型。
- 支持复杂拓扑结构
- 高分辨率重建（512³-1536³）
- 完整的PBR材质
- 支持导出GLB格式（含纹理）

### 使用建议
1. 下载HTML文件在浏览器中打开，可以旋转、缩放3D模型
2. GLB文件可以在Blender、Unity等3D软件中打开
3. 体积测算结果为预估值，需要现场校准
4. 建议使用已知参照物校准尺度因子
            """

            html_file = result['html_path'] if result['html_path'] else None

            return html_file, info

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg

    # 创建界面
    with gr.Blocks(title="煤堆体积测算Demo - TRELLIS.2版本") as demo:
        gr.Markdown("""
        # 🏭 煤堆体积测算系统 Demo - TRELLIS.2版本

        ## ✨ 特点
        - ✅ **TRELLIS.2模型**: 使用微软4B参数的大型3D生成模型
        - ✅ **单张图片生成3D**: 从单张煤堆图片生成高质量3D模型
        - ✅ **可交互3D模型**: 使用Plotly生成可旋转、缩放的3D模型
        - ✅ **自动体积计算**: 基于生成的3D模型自动计算体积

        ## 使用说明
        1. 上传煤堆图像
        2. 设置参数
        3. 选择是否使用TRELLIS.2（如果模型加载成功）
        4. 点击"开始测算"
        5. 查看结果并下载可交互的HTML文件
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="上传煤堆图像", type="numpy")

                coal_density = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.3,
                    step=0.1,
                    label="煤炭密度 (吨/立方米)"
                )

                scale_factor = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="尺度因子 (模型单位→米)"
                )

                use_trellis = gr.Checkbox(
                    value=True,
                    label="使用TRELLIS.2模型（如果可用）"
                )

                submit_btn = gr.Button("🚀 开始测算", variant="primary")

            with gr.Column():
                output_html = gr.File(label="可交互3D模型（下载后在浏览器打开）")
                output_info = gr.Markdown(label="测算结果")

        submit_btn.click(
            fn=process_image,
            inputs=[input_image, coal_density, scale_factor, use_trellis],
            outputs=[output_html, output_info]
        )

        gr.Markdown("""
        ---
        ### 💡 关于TRELLIS.2模型

        **TRELLIS.2的优势**:
        1. 从单张图片生成高质量3D模型
        2. 支持复杂拓扑结构（开放表面、非流形几何）
        3. 高分辨率重建能力
        4. 完整的PBR材质支持

        **系统要求**:
        - NVIDIA GPU with 24GB+ VRAM
        - CUDA 12.4+
        - Python 3.10+

        ### 🔄 技术说明
        - 本demo使用TRELLIS.2-4B模型
        - 模型会自动从Hugging Face下载（首次运行需要时间）
        - 如果TRELLIS.2不可用，会自动回退到简化的深度估计方法
        """)

    return demo


if __name__ == "__main__":
    print("="*60)
    print("煤堆体积测算Demo启动中 - TRELLIS.2版本")
    print("="*60)

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7871,
        share=False
    )

