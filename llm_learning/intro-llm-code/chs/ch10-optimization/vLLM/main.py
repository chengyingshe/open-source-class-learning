'''2.5 vLLM 推理框架实践'''

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from vllm import LLM, SamplingParams

# 给定提示样例
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 创建sampling参数对象
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建大语言模型
llm = LLM(model="facebook/opt-125m")

# 从提示中生成文本。输出是一个包含提示、生成的文本和其他信息的RequestOutput对象列表
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
