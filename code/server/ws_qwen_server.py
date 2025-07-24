# CUDA_VISIBLE_DEVICES=0 python test_qwen_sparse.py

import os
import numpy as np
import glob
from PIL import Image
import io
import cv2
os.environ["WANDB_DISABLED"] = "true"
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
import warnings
import ast







warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 参数定义=======================================
# MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MAX_SEQ_LEN = 1024
# 参数定义=======================================









# 系统消息定义=======================================
system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data. 
Your task is to process and extract meaningful insights from images, videos, and visual patterns, 
leveraging multimodal understanding to provide accurate and contextually relevant information."""
# 系统消息定义=======================================


# prompt定义
query_text = """Analyze the image carefully and determine if the test tube is placed in the test tube rack on the table. 
Provide your final answer by stating only 'Yes' or 'No'. Note: The test tube lying horizontally on the rack (even if touching it) should be counted as 'No'. 
"""








def convert_to_png(image_path):
    """
    将指定路径的图片转换为 PNG 格式的图片对象。
    """
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        png_image = Image.open(buffer)
        png_image.filename = image_path
    return png_image






def convert_np_to_png_image(np_image: np.ndarray):
    """
    将 NumPy 图像数组 (RGB 格式) 转换为 Qwen 接受的 PNG 格式 PIL 图像对象。
    """
    if np_image.dtype != np.uint8:
        np_image = (np_image * 255).astype(np.uint8)

    pil_image = Image.fromarray(np_image)  # np.array -> PIL.Image
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    png_image = Image.open(buffer)
    return png_image






def extract_list(var_name, text):
    lines = [line.strip() for line in text.split('\n')]
    for line in lines:
        if line.startswith(f"{var_name} ="):
            try:
                return ast.literal_eval(line.split('=', 1)[1].strip())
            except (SyntaxError, ValueError):
                return None
    return None






with open('prompt_template_sparse.txt', 'r') as f:
    prompt_template_sparse = f.read()







def build_prompt_sparse(image_wrist, instruction=None):
    # print("!!!!!!!!!!!!", image_wrist, instruction)
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
        "role": "user",
        "content": [           
            {
                "type": "text",
                "text": f"{instruction}"
            },
            {"type": "image", "image": convert_np_to_png_image(image_wrist)},       
        ]
    }
]
    

    return messages







if device == "cuda":
    
    # quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        low_cpu_mem_usage=False,
        # quantization_config=bnb_config,
        # use_cache=True
    )
    print("model.hf_device_map", model.hf_device_map)

else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        use_cache=True    )










# instruction = "place the test tube in the test tube rack"
# "Insert the burette into the beaker."
# "Grab the test tube from the test tube rack and insert it into another well of the test tube rack."
# "Grab the test tube from the test tube rack and press it upside down on the test tube rack."
# "place the beaker on the plate."
# "place the petri dish on the plate."
# "press the black button"
# "Grab the test tube and lift it up from the test tube rack"
# "Push the cube to the center of the circle."
# "Open the drawer."







def run_qwen_inference(image_wrist, prompt, weight_path):

    

    # model.load_adapter(weight_path)
    
    adapter_name = weight_path

    # if adapter_name not in model.adapters:
    #     model.load_adapter(weight_path, adapter_name=adapter_name)

    # try:
    #     model.load_adapter(weight_path, adapter_name=adapter_name)
    #     # 加载新技能时可能要先卸载原来技能的权重
    # except ValueError as e:
    #     if "already exists" in str(e):
    #         print(f"Adapter '{adapter_name}' already loaded.")
    #     else:
    #         raise e


    # model.set_adapter(adapter_name)



    processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)  # 比如有的分词器需要训练，这个是加载训练好的分词器
    processor.tokenizer.padding_side = "right"



    def text_generator1(sample_data):            # 在执行完format_data函数后进一步处理输入内容，然后此函数可以得到qwen的输出内容


        text = processor.apply_chat_template(
            sample_data[0:2], tokenize=False, add_generation_prompt=True             # 需要改
        )

        # image_inputs = sample_data[1]["content"][3:3+count*1]["image"]                       # 需要改
        image_inputs = [
            item["image"] for item in sample_data[1]["content"] 
            if item["type"] == "image" and "image" in item
            ]

        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        del inputs  

        return output_text[0]



    # print(prompt, weight_path, image_wrist)

    messages1 = build_prompt_sparse(image_wrist, prompt) 
    generated_text1 = text_generator1(messages1)

    with open('output_sparse.txt', 'w') as f:
        f.write(generated_text1)

    # print("generated_text =========== **************")
    # print(generated_text1)

    assistant_index = generated_text1.find("assistant")
    if assistant_index == -1:
        raise ValueError("'assistant' not found in the text.")

    final_answer = generated_text1[assistant_index+10:]  
    # print("!!!!!!!", extract_list("execute_state", final_answer))
    print("assistant: ", final_answer)

    return final_answer






