#!/usr/bin/env python3
"""
Qwen3-VL-4B-Instruct Web演示界面
基于Gradio的友好图形界面，使用官方推荐的模型加载和推理方法
"""

import os
# 设置环境变量优化CUDA内存分配
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# 注释掉固定的GPU设置，让命令行参数生效
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gradio as gr
import torch
import gc
import time
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

# 修复torch.compiler兼容性问题
import torch.compiler
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

# 添加更多兼容性修复
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

# 全局性能优化开关（若GPU支持TF32，将显著加速）
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    print("⚙️ 已启用TF32/高精度矩阵乘加速（如GPU支持）")
except Exception:
    pass

class Qwen3VL4BWebDemo:
    """Qwen3-VL-4B Web演示类"""
    
    def __init__(self, model_path=None):
        """初始化Web演示"""
        self.model_path = model_path or self._find_model_path()
        self.model = None
        self.processor = None
        
        # 安防告警编号字典，将编号映射为具体的检测问题
        self.alarm_codes = {
            "ET02006": "人员跌倒",
            "ET02007": "打瞌睡检测",
            "ET02009": "打架告警",
            "ET03011": "人员攀爬",
            "ET03013": "违规跨越",
            "ET03001": "接打电话",
            "ET03002": "违规抽烟",
            "ET03009": "使用手机",
            "ET05001": "明火告警",
            "ET05002": "烟雾告警"
        }
        
        self._load_model()
    
    def _find_model_path(self):
        """自动查找模型路径"""
        # 允许使用环境变量显式指定
        env_path = os.environ.get("QWEN_VL_MODEL_PATH")
        if env_path and os.path.exists(os.path.join(env_path, "config.json")):
            print(f"✓ 使用环境变量模型路径: {env_path}")
            return env_path

        possible_paths = [
            "/mnt/data3/clip/Qwen3-VL-8B-Instruct/Qwen/Qwen3-VL-4B-Instruct",
            "/mnt/data3/clip/Qwen3-VL-8B-Instruct/Qwen/Qwen/Qwen3-VL-4B-Instruct",  # 兼容多一层Qwen目录
            "./Qwen/Qwen3-VL-4B-Instruct",
            "./Qwen3-VL-4B-Instruct",
            os.path.expanduser("~/.cache/modelscope/hub/Qwen/Qwen3-VL-4B-Instruct"),
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                print(f"✓ 找到模型路径: {path}")
                return path

        # 广泛扫描父目录，寻找包含config.json的Qwen3-VL-4B-Instruct目录
        scan_roots = [
            "/mnt/data3/clip/Qwen3-VL-8B-Instruct/Qwen",
            "/mnt/data3/clip/Qwen3-VL-8B-Instruct",
        ]
        for root in scan_roots:
            for dirpath, dirnames, filenames in os.walk(root):
                if dirpath.endswith("Qwen3-VL-4B-Instruct") and "config.json" in filenames:
                    print(f"✓ 扫描到模型路径: {dirpath}")
                    return dirpath

        raise FileNotFoundError("❌ 未找到Qwen3-VL-4B-Instruct模型")
    
    def _load_model(self):
        """加载模型和处理器"""
        print(f"🔄 正在加载模型: {self.model_path}")
        
        try:
            # 彻底清理所有GPU内存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                gc.collect()
            
            # 确保只使用指定的GPU
            torch.cuda.set_device(0)  # 在CUDA_VISIBLE_DEVICES=1的环境下，这实际是GPU 1
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 选择注意力实现：优先使用FlashAttention 2（如果可用）
            attn_impl = "eager"
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                print("⚡ 检测到flash-attn，使用FlashAttention 2")
            except Exception:
                print("ℹ️ 未检测到flash-attn，使用eager注意力实现")

            # 加载模型 - 使用官方推荐的AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # 明确指定float16以节省内存
                device_map="cuda:0",  # 指定使用当前可见的GPU（实际是GPU 1）
                trust_remote_code=True,
                attn_implementation=attn_impl,
                low_cpu_mem_usage=True  # 降低CPU内存使用
            )
            
            # 设置模型为评估模式
            self.model.eval()
            
            # ⚠️ 禁用torch.compile以避免首次推理延迟
            # torch.compile在首次推理时需要1-2分钟编译时间，不适合Web演示场景
            # 如需启用，请取消以下注释（仅在离线批处理场景下推荐）
            # try:
            #     if hasattr(torch, 'compile') and torch.cuda.is_available():
            #         print("🚀 正在编译模型以优化性能...")
            #         self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            #         print("✅ 模型编译完成！")
            # except Exception as e:
            #     print(f"⚠️ 模型编译失败，将使用未编译版本: {str(e)}")
            print("ℹ️ torch.compile已禁用以避免首次推理延迟")
            
            print("✅ 模型和处理器加载成功！")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    
    def inference_text_only(self, text_query, max_new_tokens=512, temperature=0.7):
        """纯文本推理
        添加详细的分步计时与GPU内存统计，帮助定位耗时瓶颈。
        """
        start_t = time.perf_counter()
        def _gpu_mem_info():
            try:
                device = self.model.device
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                return f"GPU内存: allocated={allocated/1024**2:.2f}MiB, reserved={reserved/1024**2:.2f}MiB"
            except Exception:
                return "GPU内存: N/A"
        # 处理告警编号转换
        processed_query = self._process_alarm_code(text_query)
        
        # 构建消息 - 复检场景优化（在问题中嵌入指导）
        # 将指导信息直接嵌入用户问题中，避免系统角色兼容性问题
        enhanced_query = f"请简短回答（直接给出是/否答案，再简短说明理由）：{processed_query}"
        messages = [
            {
                "role": "user", 
                "content": enhanced_query
            }
        ]

        t0 = time.perf_counter()
        # 应用聊天模板并准备输入
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        t1 = time.perf_counter()
        print(f"⏱️ [文本] 模板与编码耗时: {(t1 - t0)*1000:.2f} ms | {_gpu_mem_info()}")
        
        # 移动到模型设备
        t2 = time.perf_counter()
        inputs = inputs.to(self.model.device)
        t3 = time.perf_counter()
        print(f"⏱️ [文本] 数据转移到GPU耗时: {(t3 - t2)*1000:.2f} ms | {_gpu_mem_info()}")
        
        try:
            # 生成回复 - 优化参数
            with torch.inference_mode():
                t4 = time.perf_counter()
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 64),  # 复检场景限制长度
                    temperature=0.1,  # 低温度确保一致性
                    do_sample=False,  # 禁用采样，使用贪婪搜索
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # ⚠️ 关键优化：启用KV缓存
                    num_beams=1,
                    repetition_penalty=1.05,
                )
                torch.cuda.synchronize()
                t5 = time.perf_counter()
                print(f"⏱️ [文本] 生成阶段耗时: {(t5 - t4):.3f} s | {_gpu_mem_info()}")
            
            # 解码生成的token
            t6 = time.perf_counter()
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            t7 = time.perf_counter()
            print(f"⏱️ [文本] 截取新tokens耗时: {(t7 - t6)*1000:.2f} ms")
            
            t8 = time.perf_counter()
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            t9 = time.perf_counter()
            print(f"⏱️ [文本] 解码耗时: {(t9 - t8)*1000:.2f} ms")
            print(f"⏱️ [文本] 总耗时: {(t9 - start_t):.3f} s")
            
            return output_text
            
        finally:
            # 清理GPU内存（减少不必要的清理）
            del inputs
            if 'generated_ids' in locals():
                del generated_ids
            if 'generated_ids_trimmed' in locals():
                del generated_ids_trimmed
            torch.cuda.empty_cache()
    
    def inference_with_image(self, text_query, image, max_new_tokens=512, temperature=0.7):
        """图像+文本推理
        添加详细的分步计时与GPU内存统计，帮助定位耗时瓶颈。
        """
        start_t = time.perf_counter()
        def _gpu_mem_info():
            try:
                device = self.model.device
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                return f"GPU内存: allocated={allocated/1024**2:.2f}MiB, reserved={reserved/1024**2:.2f}MiB"
            except Exception:
                return "GPU内存: N/A"
        # 限制图像尺寸以节省内存和加速处理（进一步下调以降延迟）
        max_size = 672  # 最大边长限制
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"🔧 图像已调整大小至: {new_size}")
        t_resize_done = time.perf_counter()
        print(f"⏱️ [图像] 尺寸调整耗时: {(t_resize_done - start_t)*1000:.2f} ms")
        
        # 处理告警编号转换
        processed_query = self._process_alarm_code(text_query)
        
        # 构建消息 - 复检场景优化（在问题中嵌入指导）
        t_msg0 = time.perf_counter()
        # 将指导信息直接嵌入用户问题中，避免系统角色兼容性问题
        enhanced_query = f"请简短回答（直接给出是/否答案，再简短说明理由）：{processed_query}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # 直接使用PIL Image对象
                    {"type": "text", "text": enhanced_query}
                ]
            }
        ]
        t_msg1 = time.perf_counter()
        print(f"⏱️ [图像] 构建消息耗时: {(t_msg1 - t_msg0)*1000:.2f} ms")
        
        try:
            
            # 应用聊天模板并准备输入
            t_apply0 = time.perf_counter()
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            t_apply1 = time.perf_counter()
            print(f"⏱️ [图像] 模板与编码耗时: {(t_apply1 - t_apply0)*1000:.2f} ms | {_gpu_mem_info()}")
            
            # 移动到模型设备
            t_to0 = time.perf_counter()
            inputs = inputs.to(self.model.device)
            t_to1 = time.perf_counter()
            print(f"⏱️ [图像] 数据转移到GPU耗时: {(t_to1 - t_to0)*1000:.2f} ms | {_gpu_mem_info()}")
            
            try:
                # 生成回复 - 优化参数以提升性能
                with torch.inference_mode():
                    t_gen0 = time.perf_counter()
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_new_tokens, 64),  # 复检场景限制长度
                        temperature=0.1,  # 低温度确保一致性
                        do_sample=False,  # 禁用采样，使用贪婪搜索
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,  # ⚠️ 关键优化：启用KV缓存可大幅提升速度（2-5倍）
                        num_beams=1,  # 使用贪婪搜索以加快速度
                        repetition_penalty=1.05,  # 轻微惩罚重复
                    )
                    torch.cuda.synchronize()  # 只在最后同步一次
                    t_gen1 = time.perf_counter()
                    print(f"⏱️ [图像] 生成阶段耗时: {(t_gen1 - t_gen0):.3f} s | {_gpu_mem_info()}")
                
                # 解码生成的token
                t_trim0 = time.perf_counter()
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                t_trim1 = time.perf_counter()
                print(f"⏱️ [图像] 截取新tokens耗时: {(t_trim1 - t_trim0)*1000:.2f} ms")
                
                t_dec0 = time.perf_counter()
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                t_dec1 = time.perf_counter()
                print(f"⏱️ [图像] 解码耗时: {(t_dec1 - t_dec0)*1000:.2f} ms")
                print(f"⏱️ [图像] 总耗时: {(t_dec1 - start_t):.3f} s")
                
                return output_text
                
            finally:
                # 清理GPU内存（延迟清理以减少开销）
                t_clean0 = time.perf_counter()
                del inputs
                if 'generated_ids' in locals():
                    del generated_ids
                if 'generated_ids_trimmed' in locals():
                    del generated_ids_trimmed
                # 只在必要时清理缓存
                torch.cuda.empty_cache()
                t_clean1 = time.perf_counter()
                print(f"⏱️ [图像] 后处理清理耗时: {(t_clean1 - t_clean0)*1000:.2f} ms | {_gpu_mem_info()}")
            
        except Exception as e:
            error_msg = f"❌ 图像推理过程中出现错误: {str(e)}"
            print(error_msg)
            raise
    
    def _process_alarm_code(self, text_query):
        """
        处理告警编号，将编号转换为对应的检测问题
        
        Args:
            text_query: 用户输入的文本（可能是编号或普通文本）
            
        Returns:
            处理后的问题文本
        """
        # 去除空格并转为大写
        query_clean = text_query.strip().upper()
        
        # 检查是否为告警编号
        if query_clean in self.alarm_codes:
            alarm_type = self.alarm_codes[query_clean]
            # 转换为检测问题
            detection_question = f"图中是否存在{alarm_type}？"
            print(f"📝 告警编号转换: {query_clean} -> {detection_question}")
            return detection_question
        else:
            # 不是编号，直接返回原文本
            return text_query
    
    def chat(self, message, image, history, max_tokens, temperature):
        """
        聊天函数
        
        Args:
            message: 用户输入的文本
            image: 上传的图像
            history: 对话历史
            max_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            更新后的历史记录
        """
        if not message.strip():
            return history
        
        try:
            if image is not None:
                # 多模态推理
                response = self.inference_with_image(message, image, max_tokens, temperature)
            else:
                # 纯文本推理
                response = self.inference_text_only(message, max_tokens, temperature)
            
            # 更新历史记录
            history.append([message, response])
            
            return history
            
        except Exception as e:
            error_msg = f"❌ 生成回复时出现错误: {str(e)}"
            print(error_msg)
            history.append([message, error_msg])
            return history
    
    def create_interface(self):
        """创建Gradio界面"""
        
        with gr.Blocks(title="Qwen3-VL-4B-Instruct 演示", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🤖 Qwen3-VL-4B-Instruct 多模态对话演示
                
                这是一个基于Qwen3-VL-4B-Instruct的多模态对话系统演示。
                您可以上传图片并与AI进行对话，AI能够理解图片内容并回答相关问题。
                
                ✅ **模型状态**: 已成功加载并可正常使用
                """
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    # 聊天界面
                    chatbot = gr.Chatbot(
                        label="💬 对话历史",
                        height=400,
                        show_label=True,
                        avatar_images=("👤", "🤖")
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="输入消息",
                            placeholder="请输入您的问题...",
                            scale=4,
                            lines=2
                        )
                        submit_btn = gr.Button("🚀 发送", variant="primary", scale=1)
                    
                    # 图片上传
                    image_input = gr.Image(
                        label="📷 上传图片 (可选)",
                        type="pil",
                        height=200
                    )
                    
                    # 控制按钮
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清除对话", variant="secondary")
                        clear_image_btn = gr.Button("🖼️ 清除图片", variant="secondary")
                
                with gr.Column(scale=1):
                    # 参数设置
                    gr.Markdown("### ⚙️ 生成参数")
                    
                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=128,
                        value=64,
                        step=16,
                        label="最大生成长度",
                        info="复检场景建议64以内，简洁快速"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.1,
                        label="生成温度",
                        info="复检场景建议低温度(0.1)确保一致性"
                    )
                    
                    gr.Markdown(
                        """
                        ### 📝 使用说明
                        
                        **🔍 安防告警检测**
                        - 上传图片，输入告警编号进行复检
                        - 支持的告警编号：
                        
                        **🚨 人员行为类**
                        - ET02006: 人员跌倒
                        - ET02007: 打瞌睡检测
                        - ET02009: 打架告警
                        - ET03011: 人员攀爬
                        - ET03013: 违规跨越
                        
                        **📱 违规行为类**
                        - ET03001: 接打电话
                        - ET03002: 违规抽烟
                        - ET03009: 使用手机
                        
                        **🔥 火灾安全类**
                        - ET05001: 明火告警
                        - ET05002: 烟雾告警
                        
                        **⚙️ 参数说明**
                        - **最大生成长度**: 复检场景建议64以内
                        - **生成温度**: 低温度(0.1)确保一致性
                        
                        ### 💡 使用示例
                        
                        **📷 告警复检**
                        1. 上传待检测图片
                        2. 输入告警编号（如：ET05001）
                        3. 点击发送，获得简洁的是/否判断
                        
                        **💬 普通问答**
                        - 也支持直接输入文本问题
                        - 如："图中是否有人员跌倒？"
                        """
                    )
            
            # 事件绑定
            def respond(message, image, history, max_tokens, temperature):
                return self.chat(message, image, history, max_tokens, temperature), ""
            
            def clear_history():
                return []
            
            def clear_image():
                return None
            
            # 提交事件
            submit_btn.click(
                respond,
                inputs=[msg, image_input, chatbot, max_tokens, temperature],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                respond,
                inputs=[msg, image_input, chatbot, max_tokens, temperature],
                outputs=[chatbot, msg]
            )
            
            # 清除事件
            clear_btn.click(
                clear_history,
                outputs=[chatbot]
            )
            
            clear_image_btn.click(
                clear_image,
                outputs=[image_input]
            )
        
        return demo

def main():
    """主函数"""
    try:
        print("🚀 正在初始化Qwen3-VL-4B Web演示...")
        demo_app = Qwen3VL4BWebDemo()
        
        print("🎨 正在创建Web界面...")
        demo = demo_app.create_interface()
        
        print("🌐 启动Web服务器...")
        print("📱 访问地址: http://localhost:7866")
        print("🔗 如需外部访问，请使用服务器IP地址")
        print("🚨 支持的告警编号: ET02006, ET02007, ET02009, ET03011, ET03013, ET03001, ET03002, ET03009, ET05001, ET05002")
        
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=7866,       # 端口号
            share=False,            # 不创建公共链接
            debug=False,            # 调试模式
            show_error=True,        # 显示错误信息
            inbrowser=True          # 自动打开浏览器
        )
        
    except Exception as e:
        print(f"❌ 启动Web演示失败: {str(e)}")
        print("\n🔧 可能的解决方案:")
        print("1. 确保已安装gradio: pip install gradio")
        print("2. 确保模型已正确下载和配置")
        print("3. 检查端口7861是否被占用")
        print("4. 确保transformers版本 >= 4.57.0")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()