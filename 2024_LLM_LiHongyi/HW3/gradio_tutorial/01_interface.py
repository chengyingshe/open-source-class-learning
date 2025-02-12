import gradio as gr
import cv2

def greet(name, intensity):
    return 'Hello, ' + name + '!' * int(intensity)

# gr.Interface()是一个已经实现的接口
# Demo01
demo01 = gr.Interface(
    fn=greet,
    inputs=['text', 'slider'],
    outputs=['text']
)

# Demo02
demo02 = gr.Interface(
    fn=lambda s: f'Hello, {s}',
    inputs=[gr.Textbox(label='Name')],
    outputs=[gr.Textbox(label='Greet')],
    # live=True,  # run on live
    examples=['Max', 'Smith', 'Sora']
)

# Demo03
def cvt_img_color_map(img, cmap):
    new_img = img
    if cmap == 'rgb':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cmap == 'hsv':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif cmap == 'hls':
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return new_img

demo03 = gr.Interface(
    fn=cvt_img_color_map,
    inputs=[
        gr.Image(label='Input'),
        gr.Radio(
            ['bgr', 'rgb', 'hsv', 'hls'], 
            value='bgr', label='Color Map'
        )
    ],
    outputs=[gr.Image(label='Output')],
    live=True
)

demo03.launch(debug=True, 
              server_port=8888)