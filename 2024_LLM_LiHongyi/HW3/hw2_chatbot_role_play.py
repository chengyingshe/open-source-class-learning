import gradio as gr
from openai import OpenAI

client = OpenAI(api_key="sk-7ee6d041ef2f446993987ed59cf64d54", base_url="https://api.deepseek.com")
role = None

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

with gr.Blocks() as demo:
    with gr.Row(variant='panel'):
        role_text = gr.Textbox(placeholder='Input the role to play', show_label=False, container=False)
        role_set_btn = gr.Button('Set Role')

    title = gr.Markdown("""
        <h1 style='text-align: center'>hw2_chatbot_role_play</h1>
        """)
    
    chat = gr.ChatInterface(
        fn=bot,
        type='messages'
    )
    chat.textbox.placeholder = 'Input your query'
    chat.chatbot.label = 'deepseek-chat'
    
    def set_role(role_text):
        global history
        if role_text != '':
            history = [{"role": "system", "content": f"You are a helpful assistant, now please act as a {role_text} and play a game with the user."}]
            return None
        return chat.chatbot
    
    role_set_btn.click(fn=set_role, inputs=role_text, outputs=chat.chatbot)
    
demo.launch(debug=True)
