import gradio as gr
from openai import OpenAI

client = OpenAI(api_key="sk-7ee6d041ef2f446993987ed59cf64d54", base_url="https://api.deepseek.com")


history = []

def bot(message, history):
    history.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=True
    )

    answer = ''
    for chunk in stream:
        answer += chunk.choices[0].delta.content or ""
        yield answer

gr.ChatInterface(
    fn=bot,
    type='messages',
    title='hw1_chatbot',
    textbox=gr.Textbox(placeholder='Input your query')
).launch(debug=True)
