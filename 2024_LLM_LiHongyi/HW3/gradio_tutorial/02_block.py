import gradio as gr

def greet(name):
    return 'Hello, ' + name + '!'

with gr.Blocks() as demo01:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

with gr.Blocks() as demo02:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(
        label="Output Box", 
        interactive=True  # whether could be edited
                        )
    greet_btn = gr.Button("Greet")
    @greet_btn.click(inputs=name, outputs=output)
    def greet(name):
        return f'Hello, {name}!'

def increase(num):
    return num + 1

with gr.Blocks() as demo03:
    a = gr.Number(label="a")
    b = gr.Number(label="b")
    atob = gr.Button("a > b")
    btoa = gr.Button("b > a")
    atob.click(increase, a, b)
    btoa.click(increase, b, a)


# blocks using markdown text
with gr.Blocks() as demo04:
    gr.Markdown(
        """
        ## Hello Gradio!
        This is a demo testing markdown.
        """
    )
    inp = gr.Textbox(placeholder='Input your name')
    out = gr.Textbox()
    inp.change(fn=greet, inputs=inp, outputs=out)  # change function (similar to `interative=True`)

# Function return Dict
with gr.Blocks() as demo05:
    food_box = gr.Number(value=10, label="Food Count")
    status_box = gr.Textbox()

    def eat(food):
        if food > 0:
            return {  # return dict (key should be the component variable name)
                    food_box: food - 1, 
                    status_box: "full"
                }
        else:
            return {status_box: "hungry"}

    gr.Button("Eat").click(
        fn=eat,
        inputs=food_box,
        outputs=[food_box, status_box]
    )


demo05.launch(debug=True)