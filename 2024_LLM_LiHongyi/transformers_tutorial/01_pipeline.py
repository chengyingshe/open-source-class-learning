from transformers import pipeline
# import os
# os.environ['HF_HUB_OFFLINE'] = '1'  # Set offline mode

# 1. task
pipe = pipeline(task="automatic-speech-recognition")  # ASR
output = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(output)

# 2. model
pipe = pipeline(model="FacebookAI/roberta-large-mnli")
pipe("This restaurant is awesome")
print(output)

# 3. multi-input
pipe = pipeline(model="FacebookAI/roberta-large-mnli")
output = pipe(["This restaurant is awesome", "It is ugly"])
print(output)

# 4. with gradio
import gradio as gr
pipe = pipeline(task="sentiment-analysis", model="FacebookAI/roberta-large-mnli")
gr.Interface.from_pipeline(pipe).launch(server_port=8888)
