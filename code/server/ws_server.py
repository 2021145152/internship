import asyncio
import websockets
import base64
import json
import numpy as np
import cv2
from ws_qwen_server import run_qwen_inference
import ast
from datetime import datetime
import os







def decode_base64_image_to_ndarray(base64_string):
    """
    将 base64 编码的 PNG 图像字符串还原为 OpenCV 的 np.ndarray 格式。
    """
    image_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR 格式
    return image_np






def extract_list(var_name, text):
    lines = [line.strip() for line in text.split('\n')]
    for line in lines:
        if line.startswith(f"{var_name} ="):
            try:
                return ast.literal_eval(line.split('=', 1)[1].strip())
            except (SyntaxError, ValueError):
                return None
    return None







def parse_prompt_dict(file_path):
    result = {}
    
    with open(file_path, 'r') as file:
        current_verb = None
        current_noun = None
        current_prompt_lines = []
        current_weights_path = None
        in_prompt = False
        
        for line in file:
            stripped_line = line.strip()
            
            if not stripped_line or stripped_line.startswith('#'):
                continue
                
            if stripped_line == '---':
                if current_verb and current_noun:
                    prompt = ''.join(current_prompt_lines).strip() if current_prompt_lines else ""
                    if current_verb not in result:
                        result[current_verb] = {}
                    result[current_verb][current_noun] = {
                        "prompt": prompt,
                        "weights_path": current_weights_path
                    }
                current_verb = None
                current_noun = None
                current_prompt_lines = []
                current_weights_path = None
                in_prompt = False
                continue
                
            if line.startswith('verb:'):
                current_verb = line.split(':', 1)[1].strip()
            elif line.startswith('noun:'):
                current_noun = line.split(':', 1)[1].strip()
            elif line.startswith('weights_path:'):
                current_weights_path = line.split(':', 1)[1].strip()
            elif line.startswith('prompt:'):
                prompt_part = line.split(':', 1)[1]
                current_prompt_lines.append(prompt_part)
                in_prompt = True
            elif in_prompt:
                current_prompt_lines.append(line)
    return result












latest_data = {
    "instruction": "",
    "image_np": None,
    "verb_list_left": None,
    "noun_list_left": None,
    "verb_list_right": None,
    "noun_list_right": None
}


lock = asyncio.Lock()  # 保证并发安全



async def handle_connection(websocket):
    print("Client connected")

    async def receive_loop():
        try:
            async for message in websocket:
                data = json.loads(message)
                instruction = data.get("instruction", "")
                image_base64 = data.get("image", "")

                image_np = decode_base64_image_to_ndarray(image_base64)

                # 解析当前原子技能
                verb_list_left = extract_list('verb_list_left', instruction)
                noun_list_left = extract_list('noun_list_left', instruction)
                verb_list_right = extract_list('verb_list_right', instruction)
                noun_list_right = extract_list('noun_list_right', instruction)


                # os.makedirs("receive_debug_images", exist_ok=True)
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # image_filename = f"receive_debug_images/frame_{timestamp}.png"
                # cv2.imwrite(image_filename, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))




                async with lock:
                    latest_data["instruction"] = instruction
                    latest_data["image_np"] = image_np
                    latest_data["verb_list_left"] = verb_list_left
                    latest_data["noun_list_left"] = noun_list_left
                    latest_data["verb_list_right"] = verb_list_right
                    latest_data["noun_list_right"] = noun_list_right

        except websockets.ConnectionClosed:
            print("Client disconnected")


    async def inference_loop():

        consecutive_yes_count = 0   # 连续Yes的计数器
        required_yes_count = 3      # 设定阈值，比如连续3次Yes就终止

        while True:
            await asyncio.sleep(1.0)  # 每秒执行一次推理
            async with lock:
                instruction = latest_data["instruction"]
                image_np = latest_data["image_np"]
                verb_list_left = latest_data["verb_list_left"]
                noun_list_left = latest_data["noun_list_left"]
                verb_list_right = latest_data["verb_list_right"]
                noun_list_right = latest_data["noun_list_right"]

            if instruction and image_np is not None and verb_list_left and noun_list_left:
                try:
                    prompt_dict = parse_prompt_dict('prompt_dict_final.txt')
                    # === 左臂判别 ===
                    print("server端收到的verb_list_left:", verb_list_left)
                    print("server端收到的noun_list_left:", noun_list_left)
                    if verb_list_left and noun_list_left and len(verb_list_left) > 0 and len(noun_list_left) > 0:
                        if str(verb_list_left[0]).lower() == "null" or str(noun_list_left[0]).lower() == "null":
                            left_result = "yes"
                        else:
                            # # === 自动推理（如需恢复可取消注释） ===
                            # query_text_left = prompt_dict[verb_list_left[0]][noun_list_left[0]]['prompt']
                            # weights_path_left = prompt_dict[verb_list_left[0]][noun_list_left[0]]['weights_path']
                            # left_result = run_qwen_inference(image_np, query_text_left, weights_path_left)
                            # print(query_text_left,weights_path_left,left_result)
                            # left_result = "yes" if left_result.strip().lower().startswith("yes") else "no"
                            # === 手动输入判别 ===
                            left_result = input(f"[人工判别] 左臂: 动作={verb_list_left[0]}, 物体={noun_list_left[0]}，请输入yes或no（回车默认yes）: ").strip().lower()
                            if left_result == "":
                                left_result = "yes"
                            elif not left_result.startswith("y"):
                                left_result = "no"
                    else:
                        left_result = "no"

                    # === 右臂判别 ===
                    if verb_list_right and noun_list_right and len(verb_list_right) > 0 and len(noun_list_right) > 0:
                        if str(verb_list_right[0]).lower() == "null" or str(noun_list_right[0]).lower() == "null":
                            right_result = "yes"
                        else:
                            # # === 自动推理（如需恢复可取消注释） ===
                            # query_text_right = prompt_dict[verb_list_right[0]][noun_list_right[0]]['prompt']
                            # weights_path_right = prompt_dict[verb_list_right[0]][noun_list_right[0]]['weights_path']
                            # right_result = run_qwen_inference(image_np, query_text_right, weights_path_right)
                            # right_result = "yes" if right_result.strip().lower().startswith("yes") else "no"
                            # === 手动输入判别 ===
                            right_result = input(f"[人工判别] 右臂: 动作={verb_list_right[0]}, 物体={noun_list_right[0]}，请输入yes或no（回车默认yes）: ").strip().lower()
                            if right_result == "":
                                right_result = "yes"
                            elif not right_result.startswith("y"):
                                right_result = "no"
                    else:
                        right_result = "no"

                    print(f"Qwen Output: left={left_result}, right={right_result}")

                    
                    if left_result == "yes" and right_result == "yes":
                        consecutive_yes_count += 1
                        print(f"Consecutive YES count: {consecutive_yes_count}")
                    else:
                        consecutive_yes_count = 0

                    if consecutive_yes_count >= required_yes_count:
                        print(f"✅ Detected {required_yes_count} consecutive 'Yes' for both arms. Stopping inference for this skill.")
                        await websocket.send(json.dumps({"left": "yes", "right": "yes"}))
                        consecutive_yes_count = 0  # reset for next skill
                        # 不再break，继续等待下一个技能
                    else:
                        await websocket.send(json.dumps({"left": left_result, "right": right_result}))
                except Exception as e:
                    print(f"Error in inference: {e}")
                    await websocket.send(json.dumps({"left": "error", "right": "error"}))


    # 异步并发执行：同时启动并并发运行 receive_loop() 和 inference_loop() 两个协程函数，直到它们都完成（或其中一个崩溃）为止
    await asyncio.gather(receive_loop(), inference_loop())




async def main():
    print("Server listening on port 8010...")
    async with websockets.serve(handle_connection, "0.0.0.0", 8000, max_size=20 * 1024 * 1024):
        await asyncio.Future()




if __name__ == "__main__":
    asyncio.run(main())






