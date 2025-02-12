import random
import time
import gradio as gr
import plotly.express as px

"""Chatbot and Streaming Chatbot"""
def bot(history):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    time.sleep(2)
    history[-1][1] = bot_message
    return history

def streaming_bot(history):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    history[-1][1] = ''
    for c in bot_message:
        history[-1][1] += c
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo01:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder='Input your question')
    clear = gr.Button('Clear')
    
    def user(user_message, history):
        return "", history + [[user_message, None]]

    msg.submit(
        user, 
        [msg, chatbot], 
        [msg, chatbot], 
        queue=False
    ).then(  # .then() function executed consecutively
        streaming_bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

"""liking / disliking chat messages"""
def greet(history, input):
    return history + [(input, "Hello, " + input)]

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)

with gr.Blocks() as demo02:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    textbox.submit(greet, [chatbot, textbox], [chatbot])
    chatbot.like(vote, None, None)  # Adding this line causes the like/dislike icons to appear in your chatbot
    

"""MultimodalTextbox"""
def random_plot():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    size='petal_length', hover_data=['petal_width'])
    return fig

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    history[-1][1] = "Cool!"
    return history

fig = random_plot()

with gr.Blocks(fill_height=True) as demo03:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(interactive=True,
                                      file_count="multiple",
                                      placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)
    
demo01.launch(debug=True)
    