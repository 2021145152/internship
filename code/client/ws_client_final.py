import asyncio
import websockets
import base64
import cv2
import numpy as np
import json
from orbbec_sensor import OrbbecCamera
import rospy
import time
import ast
import requests
# 集成任务拆分模块
from single_view2all_subtasks import ConstraintGenerator, get_verified_bboxes









def encode_image(image_np):
    _, buffer = cv2.imencode('.png', image_np)
    return base64.b64encode(buffer).decode('utf-8')






async def main_loop():

    rospy.init_node("camera_client")
    camera = OrbbecCamera()

    uri = "ws://zz.irmv.top:8010"

    # === 1. 采集一帧图像并保存为png文件 ===
    obs = camera.get_obs()
    if obs is not None:
        rgb = obs["rgb"]
        split_img_path = "tmp_task_split.png"
        cv2.imwrite(split_img_path, rgb)
    else:
        print("Failed to get image for task split.")
        return

    # === 2. 任务指令字符串（ ===
    instruction = "Weigh the beaker and then place it on the plate. Only use the left arm to perform the task."  

    # === 3. 调用任务拆分，获得动名词列表 ===
    config = {
        'model': 'gpt-4o',
        'temperature': 0,
        'max_tokens': 1000
    }
    generator = ConstraintGenerator(config, rag_enabled=False)
    _, _, verb_list_left, noun_list_left, verb_list_right, noun_list_right = generator.generate_highlevel(split_img_path, instruction, stage=0)
    verb_list_left = [['grasp'], ['place'], ['grasp'], ['place']]
    noun_list_left = [['transparent beaker'], ['weight', 'transparent beaker'], ['transparent beaker'], ['square plate', 'transparent beaker']]
    verb_list_right = [['null'], ['null'], ['null'], ['null']]
    noun_list_right = [['null'], ['null'], ['null'], ['null']]
    print("[Task Split] verb_list_left:", verb_list_left)
    print("[Task Split] noun_list_left:", noun_list_left)
    print("[Task Split] verb_list_right:", verb_list_right)
    print("[Task Split] noun_list_right:", noun_list_right)


    num_skills = len(verb_list_left)
    current_skill_idx = 0

    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as websocket:
        for skill_idx in range(num_skills):
            # 构造名词列表，若为null/None则置空
            nl_left = noun_list_left[skill_idx]
            nl_right = noun_list_right[skill_idx]
            def is_null_or_none(lst):
                if isinstance(lst, str):
                    return lst.lower() in ("null", "none")
                if isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], str):
                    return lst[0].lower() in ("null", "none")
                return False
            if is_null_or_none(nl_left):
                nl_left = []
            if is_null_or_none(nl_right):
                nl_right = []

            # 采集当前帧图片
            obs = camera.get_obs()
            if obs is not None:
                rgb = obs["rgb"]
                rgb = cv2.resize(rgb, (640,360))
                current_img_path = f"tmp_skill_{skill_idx}.png"
                cv2.imwrite(current_img_path, rgb)
            else:
                print("Failed to get image for bbox.")
                return

            # 合并名词列表，去除null/None
            current_nouns = []
            if isinstance(nl_left, list):
                current_nouns += [n for n in nl_left if n not in ("null", "None")]
            if isinstance(nl_right, list):
                current_nouns += [n for n in nl_right if n not in ("null", "None")]
            current_nouns = list(set(current_nouns))

            # 获取高质量bbox
            final_bbox = get_verified_bboxes(current_img_path, current_nouns)

            payload_skill = {
                "verb_list_left": verb_list_left[skill_idx],
                "noun_list_left": nl_left,
                "verb_list_right": verb_list_right[skill_idx],
                "noun_list_right": nl_right,
                "bbox": final_bbox
            }
            print(payload_skill)
            try:
                r = requests.post("http://localhost:5000/dual_skills", json=payload_skill)
                print(f"[Skill] Sent /dual_skills, response: {r.text}")
                if r.text.strip() != "ok":
                    print("=========",r.text.strip())
                    print("[Skill] /dual_skills failed, aborting.")
                    return
            except Exception as e:
                print(f"[Skill] Failed to send /dual_skills: {e}")
                return

            # 2. 只有收到ok才发送 /start
            try:
                r = requests.post("http://localhost:5000/start")
                print(f"[Skill] Sent /start, response: {r.text}")
                if r.text.strip() != "ok":
                    print("[Skill] /start failed, aborting.")
                    return
            except Exception as e:
                print(f"[Skill] Failed to send /start: {e}")
                return

            # 3. 持续向 server 端发送图像+当前技能，直到 server 返回 yes
            print(f"[Server] Executing skill {skill_idx+1}/{num_skills} ...")
            while not rospy.is_shutdown():
                obs = camera.get_obs()
                if obs is not None:
                    rgb = obs["rgb"]
                    img_base64 = encode_image(rgb)
                    instruction_str = (
                        f"verb_list_left = {verb_list_left[skill_idx]}\n"
                        f"noun_list_left = {noun_list_left[skill_idx]}\n"
                        f"verb_list_right = {verb_list_right[skill_idx]}\n"
                        f"noun_list_right = {noun_list_right[skill_idx]}"
                    )
                    payload = {
                        "instruction": instruction_str,
                        "image": img_base64
                    }
                    await websocket.send(json.dumps(payload))
                    print("Sent image + current skill")

                    try:
                        response = await websocket.recv()
                        print("Server response:", response)
                        # === 双臂判别：只有left和right都为yes才break ===
                        result = json.loads(response)
                        if result.get("left", "no").strip().lower() == "yes" and result.get("right", "no").strip().lower() == "yes":
                            print(f"[Server] Skill {skill_idx+1} completed (both arms yes).")
                            break
                    except Exception as e:
                        print("Failed to receive server response:", e)
                else:
                    print("Failed to get image data.")

                await asyncio.sleep(0.2)

            # 4. 向 skill 端发送 /pause
            try:
                r = requests.post("http://localhost:5000/pause")
                print(f"[Skill] Sent /pause, response: {r.text}")
                if r.text.strip() != "ok":
                    print("[Skill] /pause failed, aborting.")
                    return
            except Exception as e:
                print(f"[Skill] Failed to send /pause: {e}")
                return

        try:
            r = requests.post("http://localhost:5000/end")
            print(f"[Skill] Sent /end, response: {r.text}")
        except Exception as e:
            print(f"[Skill] Failed to send /end: {e}")





if __name__ == "__main__":
    asyncio.run(main_loop())





