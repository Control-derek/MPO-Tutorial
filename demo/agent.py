import os
import logging
from datetime import datetime

import gradio as gr
from PIL import Image

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

class ConversationalAgent:
    def __init__(self, 
                 model_path,
                 outputs_dir) -> None:
        self.pipe = pipeline(model_path,
                        chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2'),
                        backend_config=TurbomindEngineConfig(session_len=16384))
        self.uploaded_images_storage = os.path.join(outputs_dir, "uploaded")
        self.uploaded_images_storage = os.path.abspath(self.uploaded_images_storage)
        os.makedirs(self.uploaded_images_storage, exist_ok=True)
        self.sess = None
        
    def start_chat(self, chat_state):
        self.sess = None
        self.context = ""
        self.current_image_id = -1
        self.image_list = []
        self.pixel_values_list = []
        self.seen_image_idx = []
        logging.info("=" * 30 + "Start Chat" + "=" * 30)
        
        return (
            #gr.update(interactive=False),  # [image] Image
            gr.update(interactive=True, placeholder='input the text.'),  # [input_text] Textbox
            gr.update(interactive=False),  # [start_btn] Button
            gr.update(interactive=True),  # [clear_btn] Button
            gr.update(interactive=True),  # [image] Image
            gr.update(interactive=True),  # [upload_btn] Button
            chat_state  # [chat_state] State
        )
        
    def restart_chat(self, chat_state):
        self.sess = None
        self.context = ""
        self.current_image_id = -1
        self.image_list = []
        self.pixel_values_list = []
        self.seen_image_idx = []
        
        logging.info("=" * 30 + "End Chat" + "=" * 30)
        
        return (
            None,  # [chatbot] Chatbot
            #gr.update(value=None, interactive=True),  # [image] Image
            gr.update(interactive=False, placeholder="Please click the <Start Chat> button to start chat!"),  # [input_text] Textbox
            gr.update(interactive=True),  # [start] Button
            gr.update(interactive=False),  # [clear] Button
            gr.update(value=None, interactive=False),  # [image] Image
            gr.update(interactive=False),  # [upload_btn] Button
            chat_state  # [chat_state] State
        )
        
    def upload_image(self, image: Image.Image, chat_history: gr.Chatbot, chat_state: gr.State):
        logging.info(f"type(image): {type(image)}")
        
        self.image_list.append(image)        
        save_image_path = os.path.join(self.uploaded_images_storage, "{}.jpg".format(len(os.listdir(self.uploaded_images_storage))))
        image.save(save_image_path)
        logging.info(f"image save path: {save_image_path}")
        chat_history.append((gr.HTML(f'<img src="./file={save_image_path}" style="width: 200px; height: auto; display: inline-block;">'), "Received."))
        
        return None, chat_history, chat_state
    
    def respond(
        self,
        message,
        image,
        chat_history: gr.Chatbot,
        temperature,
        chat_state,
    ):
        current_time = datetime.now().strftime("%b%d-%H:%M:%S")
        logging.info(f"Time: {current_time}")
        logging.info(f"User: {message}")
        gen_config = GenerationConfig(temperature=temperature)
        chat_input = message
        if image is not None:
            save_image_path = os.path.join(self.uploaded_images_storage, "{}.jpg".format(len(os.listdir(self.uploaded_images_storage))))
            image.save(save_image_path)
            logging.info(f"image save path: {save_image_path}")
            chat_input = (message, image)
        if self.sess is None:
            self.sess = self.pipe.chat(chat_input, gen_config=gen_config)
        else:
            self.sess = self.pipe.chat(chat_input, session=self.sess, gen_config=gen_config)
        response = self.sess.response.text
        if image is not None:
            chat_history.append((gr.HTML(f'{message}\n\n<img src="./file={save_image_path}" style="width: 200px; height: auto; display: inline-block;">'), response))
        else:
            chat_history.append((message, response))
        
        logging.info(f"generated text = \n{response}")        
        
        return "", None, chat_history, chat_state
