# MiniGPT-4实践

## 简介
MiniGPT-4 是一个集成了视觉和语言模型的系统，能够通过图像和文本交互生成自然语言响应。本项目使用 **MiniGPT-4**、**BLIP2** 图像处理器和 **Vicuna13B** 语言模型，允许用户通过上传图像并提问，获取关于图像内容的详细描述。系统还支持异步生成文本，实时输出语言模型的响应。

## 使用说明
- 首先，下载 [MiniGPT-4](https://huggingface.co/catid/minigpt4) 模型文件，并将其放入到 `models` 文件夹中。
- 然后，运行 `main.py` 加载和启动模型，使用图片和问题进行测试。
```
python main.py
```