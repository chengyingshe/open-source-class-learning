import time
import gradio as gr
import random

def random_response(message, history):
    return random.choice(["Yes", "No"])

demo01 = gr.ChatInterface(
    fn=random_response, 
    type="messages"
)

def alternatingly_agree(message, history):
    if len([h for h in history if h['role'] == "assistant"]) % 2 == 0:
        return f"Yes, I do think that: {message}"
    else:
        return "I don't think so"

demo02 = gr.ChatInterface(
    fn=alternatingly_agree, 
    type="messages"
)

"""streaming chatbots"""
def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i+1]

# customizing the chatbot
demo03 = gr.ChatInterface(
    fn=slow_echo, 
    type="messages"
)

def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"

demo04 = gr.ChatInterface(
    yes_man,
    type="messages",
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Yes Man",
    description="Ask Yes Man any question",
    # theme="ocean",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    # cache_examples=True,
)

# multimodal chat interface
def count_images(message, history):
    num_images = len(message["files"])
    total_images = 0
    for message in history:
        if isinstance(message["content"], tuple):
            total_images += 1
    return f"You just uploaded {num_images} images, total uploaded: {total_images+num_images}"

demo05 = gr.ChatInterface(
    fn=count_images, 
    type="messages", 
    examples=[
        {"text": "No files", "files": []}
    ], 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"])
)

demo05.launch(debug=True)