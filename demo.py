import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from utils import load_json, init_logger
from demo import ConversationalAgent, CustomTheme

GEOM_EXAMPLES = "demo/geometry_for_demo.json"
MODEL_PATH = "/root/MPO1/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5_mpo/Internvl2_5-1B-MPO"
OUTPUT_PATH = "./outputs"

def setup_seeds():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    setup_seeds()
    # logging
    init_logger(OUTPUT_PATH)
    # food examples
    geom_examples = load_json(GEOM_EXAMPLES)
    
    agent = ConversationalAgent(model_path=MODEL_PATH,
                                outputs_dir=OUTPUT_PATH)
    
    theme = CustomTheme()
    
    titles = [
        """<center><B><font face="Comic Sans MS" size=10>书生大模型实战营</font></B></center>"""  ## Kalam:wght@700
        """<center><B><font face="Courier" size=5>「彩蛋岛」InternVL中RLHF及MPO技术的原理与实践</font></B></center>"""
    ]
    
    language = """Language: 中文 and English"""
    with gr.Blocks(theme) as demo_chatbot:
        for title in titles:
            gr.Markdown(title)
        # gr.Markdown(article)
        gr.Markdown(language)
        
        with gr.Row():
            with gr.Column(scale=3):
                start_btn = gr.Button("Start Chat", variant="primary", interactive=True)
                clear_btn = gr.Button("Clear Context", interactive=False)
                image = gr.Image(type="pil", interactive=False)
                upload_btn = gr.Button("🖼️ Upload Image", interactive=False)
                
                with gr.Accordion("Generation Settings"):
                    temperature = gr.Slider(minimum=0, maximum=1.5, step=0.1,
                                            value=0.0,
                                            interactive=True,
                                            label='temperature',
                                            visible=True)
                    
            with gr.Column(scale=7):
                chat_state = gr.State()
                chatbot = gr.Chatbot(label='InternVL2.5', height=800, avatar_images=((os.path.join(os.path.dirname(__file__), 'demo/user.png')), (os.path.join(os.path.dirname(__file__), "demo/bot.png"))))
                text_input = gr.Textbox(label='User', placeholder="Please click the <Start Chat> button to start chat!", interactive=False)
                gr.Markdown("### 输入示例")
                def on_text_change(text):
                    return gr.update(interactive=True)
                text_input.change(fn=on_text_change, inputs=text_input, outputs=text_input)
                gr.Examples(
                    examples=[["Your task is to answer the question below. Give step by step reasoning before you answer, and when you\\'re ready to answer, please use the format \"Final answer: ..\"\\n\\nQuestion:\\n\\nThe diameters of \\\\odot A, \\\\odot B, and \\\\odot C are 8 inches, 18 inches, and 11 inches, respectively. Find F G.\\nA. 14\\nB. 16\\nC. 7\\nD. 8"],
                              ["Your task is to answer the question below. Give step by step reasoning before you answer, and when you\\'re ready to answer, please use the format \"Final answer: ..\"\\n\\nQuestion:\\n\\nFind x."],
                              ["Your task is to answer the question below. Give step by step reasoning before you answer, and when you\\'re ready to answer, please use the format \"Final answer: ..\"\\n\\nQuestion:\\n\\nUse parallelogram to, find x.\\nA. 5\\nB. 10\\nC. 11\\nD. 17"]],
                    inputs=[text_input]
                )
        
        with gr.Row():
            gr.Markdown("### 测试图片")
        with gr.Row():
            example_geometry3k = gr.Examples(examples=geom_examples["geometry3k"], inputs=image, label="geometry3k", examples_per_page=5)
        with gr.Row():
            gr.Markdown("### 正确答案：")
            gr.Markdown("#### 1. A  2. 18  3. A")
            
                
        start_btn.click(agent.start_chat, [chat_state], [text_input, start_btn, clear_btn, image, upload_btn, chat_state])
        clear_btn.click(agent.restart_chat, [chat_state], [chatbot, text_input, start_btn, clear_btn, image, upload_btn, chat_state], queue=False)
        upload_btn.click(agent.upload_image, [image, chatbot, chat_state], [image, chatbot, chat_state])
        text_input.submit(
            agent.respond,
            inputs=[text_input, image, chatbot, temperature, chat_state], 
            outputs=[text_input, image, chatbot, chat_state]
        )

    demo_chatbot.launch(share=True, server_name="127.0.0.1", server_port=1099, allowed_paths=['./'])
    demo_chatbot.queue()
    

if __name__ == "__main__":
    main()