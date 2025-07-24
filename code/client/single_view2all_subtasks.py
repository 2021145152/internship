import base64
from openai import OpenAI
import os
import json
import parse
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import ast
import tempfile
from groundingdino_detector import GroundingDINODetector







# def extract_list(var_name, text):
#     pattern = rf'{var_name}\s*=\s*(\[.*?\])'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         try:
#             return eval(match.group(1))
#         except Exception:
#             return None
#     return None



def extract_list(var_name, text):
    lines = [line.strip() for line in text.split('\n')]
    for line in lines:
        if line.startswith(f"{var_name} ="):
            try:
                return ast.literal_eval(line.split('=', 1)[1].strip())
            except (SyntaxError, ValueError):
                return None
    return None

def extract_dict(var_name, text):
    """
    专门用于提取字典格式的变量
    """
    lines = [line.strip() for line in text.split('\n')]
    start_line = None
    end_line = None
    
    # 找到开始行
    for i, line in enumerate(lines):
        if line.startswith(f"{var_name} = {{"):
            start_line = i
            break
    
    if start_line is None:
        return None
    
    # 找到结束行（匹配大括号）
    brace_count = 0
    for i in range(start_line, len(lines)):
        line = lines[i]
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0 and i > start_line:
            end_line = i
            break
    
    if end_line is None:
        return None
    
    # 提取字典文本
    dict_text = '\n'.join(lines[start_line:end_line + 1])
    dict_text = dict_text.replace(f"{var_name} = ", "")
    
    try:
        return ast.literal_eval(dict_text)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing dict: {e}")
        print(f"Dict text: {dict_text}")
        return None




class BioChemRAG:
    def __init__(self, index_path: str, doc_path: str):
       
        with open(doc_path, 'r') as f:
            self.documents = json.load(f)
            
        self.texts = []
        self.doc_ids = []
        for doc_id, doc in self.documents.items():
            text = f"{doc['task']} {' '.join(doc['steps'])} {' '.join(doc['equipment'])}"
            self.texts.append(text)
            self.doc_ids.append(doc_id)
            
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.texts)
    
    def retrieve(self, query: str, top_k: int = 1):
        query_vector = self.vectorizer.transform([query])
        
        similarities = (query_vector * self.vectors.T).toarray()[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_docs = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id].copy()
            doc['score'] = float(similarities[idx])
            retrieved_docs.append(doc)
        
        return retrieved_docs
    




def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')





class ConstraintGenerator:
    def __init__(self, config, rag_enabled=True):

        self.config = config
        self.client = OpenAI(
            api_key="sk-ZbYwaaskxCMrUDE9AxbYjExPgIUTw09eqhOhAtucAHj3Hz2b",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.rag_enabled = rag_enabled
        
        # 加载提示模板
        with open(os.path.join(self.base_dir, 'prompt_single2all.txt'), 'r', encoding='utf-8') as f:
            self.prompt_template_high = f.read()
        
        # 初始化知识检索系统
        if self.rag_enabled:
            self.rag = BioChemRAG(
                index_path=None,            # os.path.join(self.base_dir, "knowledge/biochem_index.faiss")
                doc_path=os.path.join(self.base_dir, "knowledge/biochem_docs.json")
            )

    def _query_bio_knowledge(self, instruction):
        """
        查询相关的实验知识
        
        Args:
            instruction: 任务指令
        
        Returns:
            list: 包含最相似实验记录的列表
        """
        if not self.rag_enabled:
            return None
        
        # 只检索最相似的一个实验
        return self.rag.retrieve(instruction, top_k=1)

    def _build_prompt_high(self, image_path_recorder, instruction, stage, 
                         last_subtask=None, last_env_image=None, last_success=None, 
                         last_failure_reason=None):
        """构建提示信息"""
        img_base64_recorder = self._encode_image(image_path_recorder)
        
        knowledge = self._query_bio_knowledge(instruction)
        
        retrieved_task_type = ""
        retrieved_steps = ""
        retrieved_instruments = ""
        
        if knowledge and len(knowledge) > 0:
            k = knowledge[0]  
            retrieved_task_type = k['task']
            retrieved_steps = "\n".join(k['steps']) if isinstance(k['steps'], list) else k['steps']
            retrieved_instruments = ", ".join(k['equipment']) if 'equipment' in k else ""
        
        # 构建基础提示，填充知识库内容
        prompt_text = self.prompt_template_high.format(
            instruction=instruction,
            retrieved_task_type=retrieved_task_type,
            retrieved_steps=retrieved_steps,
            retrieved_instruments=retrieved_instruments
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent assistant that can control a robot to perform manipulation tasks. The manipulation task is given as image of the environment, along with a text task instruction."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The current high level stage is {stage}."
                    },
                    {
                        "type": "text",
                        "text": prompt_text  
                    }
                ]
            }
        ]
        
        # 添加任务状态信息
        if stage >= 1:
            if last_subtask:
                status = "successfully completed" if last_success else f"failed. Reason: {last_failure_reason}"
                task_status = f"The last subtask was: '{last_subtask}'. It was {status}."
                messages[1]["content"].append({
                    "type": "text",
                    "text": task_status
                })
                
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64_recorder}"
            }
        })
        
        return messages




    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_object_descriptions(self, image_path, object_categories):
        """
        根据物体列表和图片生成详细的物体描述
        
        Args:
            image_path: 图片路径
            object_categories: 物体类别列表，如 ["beaker", "plate", "scale"]
        
        Returns:
            tuple: (详细描述字典, 简洁描述字典)
        """
        if not object_categories:
            return {}, {}
        
        # 构建物体描述生成的prompt
        object_list_text = ", ".join(object_categories)
        
        prompt_text = f"""
        You are an expert at describing objects in laboratory and kitchen environments. Given a list of object categories and an image, please generate TWO types of descriptions for each object:

        1. DETAILED descriptions for GPT-4O verification (longer, more descriptive)
        2. CONCISE descriptions for GroundingDINO detection (shorter, key visual features only)

        Object categories to describe: {object_list_text}

        For each object, provide both descriptions that include:
        - Color (if visible/applicable)
        - Material (if visible/applicable) 
        - Size/shape characteristics (if visible/applicable)
        - Any distinctive features or markings
        - Position or context clues (if relevant)

        Please respond in the following format:
        detailed_descriptions = {{
            "object_name": "detailed descriptive phrase for GPT-4O verification",
            ...
        }}

        concise_descriptions = {{
            "object_name": "concise phrase for GroundingDINO detection",
            ...
        }}

        Example format:
        detailed_descriptions = {{
            "beaker": "a clear glass beaker with white measurement markings, positioned centrally on the wooden table",
            "plate": "a white ceramic plate with floral designs, situated on the left side of the table"
        }}

        concise_descriptions = {{
            "beaker": "clear glass beaker with markings",
            "plate": "white ceramic plate"
        }}

        IMPORTANT: 
        - Detailed descriptions should be comprehensive for verification
        - Concise descriptions should be short (3-5 words max) for GroundingDINO
        - Focus on the most distinctive visual features for concise descriptions
        """
        
        img_base64 = self._encode_image(image_path)
        
        messages = [
            {
                "role": "system", 
                "content": "You are an expert at describing objects for computer vision tasks. Provide clear, visual descriptions that help identify objects in images."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
        
        # 调用API获取描述
        response = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=0.1,  # 较低的温度以获得更一致的描述
            max_tokens=800
        )
        
        output = response.choices[0].message.content
        
        # 保存原始输出用于调试
        with open(f"src_reward/tests/object_descriptions_raw.txt", 'w') as f:
            f.write(output)
        
        # 提取详细描述和简洁描述字典
        detailed_descriptions = extract_dict('detailed_descriptions', output)
        concise_descriptions = extract_dict('concise_descriptions', output)
        
        if not detailed_descriptions or not concise_descriptions:
            print("Warning: Failed to extract descriptions from model output")
            print("Raw output:")
            print(output)
            # 如果提取失败，返回简单的描述
            detailed_descriptions = {obj: f"a {obj}" for obj in object_categories}
            concise_descriptions = {obj: obj for obj in object_categories}
        
        return detailed_descriptions, concise_descriptions

    def verify_detection_with_gpt4o(self, image_path, object_detections, detailed_descriptions, annotated_image_path=None, all_candidates=None):
        """
        使用 GPT-4O 验证检测结果，包含三次验证机制
        
        Args:
            image_path: 图片路径
            object_detections: GroundingDINO 检测结果，格式为 {"object_name": [x1, y1, x2, y2]}
            detailed_descriptions: 详细描述字典，用于验证
            annotated_image_path: 已标注的图片路径（如果为None则重新生成）
            all_candidates: 所有候选bbox字典，格式为 {"object_name": [(bbox1, logit1), (bbox2, logit2), ...]}
        
        Returns:
            dict: 验证后的有效检测结果，格式为 {"object_name": [x1, y1, x2, y2]}
        """
        if not object_detections:
            return {}
        
        # 构建验证提示
        verification_prompt = self._build_verification_prompt(object_detections, detailed_descriptions)
        
        # 使用已有的标注图片，如果没有则重新生成
        if annotated_image_path is None or not os.path.exists(annotated_image_path):
            annotated_image_path = self._create_annotated_image(image_path, object_detections)
        
        # 编码标注后的图像
        img_base64 = self._encode_image(annotated_image_path)
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at verifying object detection results. Your task is to validate whether the detected bounding boxes correctly identify the objects described in the image."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": verification_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
        
        try:
            # 调用 GPT-4O 进行验证
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )
            
            output = response.choices[0].message.content
            
            # 保存原始输出用于调试
            with open(f"src_reward/tests/verification_raw.txt", 'w') as f:
                f.write(output)
            
            # 解析验证结果
            verified_detections = self._parse_verification_result(output, object_detections)
            
            # 三次验证：对invalid的物体尝试top3候选bbox
            if all_candidates:
                final_detections = self._perform_third_verification(
                    image_path, verified_detections, detailed_descriptions, all_candidates, annotated_image_path, object_detections
                )
            else:
                # 如果没有候选bbox，直接根据验证状态转换结果
                final_detections = {}
                for obj_name, status in verified_detections.items():
                    if status == "valid":
                        final_detections[obj_name] = object_detections[obj_name]
                    else:
                        final_detections[obj_name] = None
            
            # 清理临时文件（只清理我们自己创建的临时文件，不清理GroundingDINO生成的文件）
            if annotated_image_path and os.path.exists(annotated_image_path):
                # 检查是否是临时文件（我们自己创建的）
                if 'tmp' in annotated_image_path or tempfile.gettempdir() in annotated_image_path:
                    os.remove(annotated_image_path)
            
            return final_detections
            
        except Exception as e:
            print(f"Error in GPT-4O verification: {e}")
            # 如果验证失败，返回原始检测结果
            return object_detections

    def _build_verification_prompt(self, object_detections, detailed_descriptions):
        """构建验证提示"""
        
        # 构建检测结果描述
        detection_info = []
        for obj_name, bbox in object_detections.items():
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                description = detailed_descriptions.get(obj_name, f"a {obj_name}")
                detection_info.append(f"- {obj_name}: bbox [{x1}, {y1}, {x2}, {y2}], description: {description}")
        
        detection_text = "\n".join(detection_info)
        
        prompt = f"""
        Please verify the object detection results shown in the annotated image. 

        The image shows bounding boxes around detected objects. For each detected object, please verify:
        1. Does the bounding box correctly contain the described object?
        2. Is the object actually present in the image at that location?
        3. Does the object match the detailed description?

        Detected objects and their descriptions:
        {detection_text}

        Please respond in the following format:
        verified_detections = {{
            "object_name": "valid" or "invalid",
            ...
        }}

        Example:
        verified_detections = {{
            "beaker": "valid",
            "plate": "invalid",
            "scale": "valid"
        }}

        If a detection is "valid", it means the bounding box correctly identifies the object. If "invalid", it means the detection is incorrect or the object is not present.

        Focus on accuracy and be strict in your verification. Look carefully at the bounding boxes in the image.
        """
        
        return prompt

    def _create_annotated_image(self, image_path, object_detections):
        """
        创建标注了边界框的图片
        
        Args:
            image_path: 原图路径
            object_detections: 检测结果字典
            
        Returns:
            str: 标注图片的临时路径
        """
        try:
            import cv2
            import tempfile
            
            # 读取原图
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # 定义颜色列表
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 洋红色
                (0, 255, 255),  # 黄色
            ]
            
            # 绘制每个检测结果
            for i, (object_name, bbox) in enumerate(object_detections.items()):
                if bbox is not None:
                    # 获取颜色
                    color = colors[i % len(colors)]
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label = f"{object_name}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # 绘制标签背景
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # 绘制标签文字
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # 保存标注后的图片
            cv2.imwrite(temp_path, image)
            
            return temp_path
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            # 如果创建失败，返回原图路径
            return image_path

    def _parse_verification_result(self, output, original_detections):
        """解析验证结果"""
        
        # 提取验证结果
        verified_detections = extract_dict('verified_detections', output)
        
        if not verified_detections:
            print("Warning: Failed to parse verification result, using original detections")
            # 如果解析失败，假设所有检测都是valid
            return {obj_name: "valid" for obj_name in original_detections.keys()}
        
        # 返回验证状态，不进行过滤
        verification_status = {}
        for obj_name, status in verified_detections.items():
            if obj_name in original_detections and original_detections[obj_name] is not None:
                verification_status[obj_name] = status.lower()
                if status.lower() == "valid":
                    print(f"  ✓ Verified {obj_name}: {original_detections[obj_name]}")
                else:
                    print(f"  ✗ Rejected {obj_name}: {status}")
            else:
                print(f"  ⚠ Warning: {obj_name} not found in original detections")
        
        return verification_status

    def _perform_third_verification(self, image_path, verification_status, detailed_descriptions, all_candidates, annotated_image_path, original_detections):
        """
        对无效检测进行第三次验证，尝试使用top3候选bbox
        
        Args:
            image_path: 图片路径
            verification_status: 二次验证状态 {"object_name": "valid"/"invalid"}
            detailed_descriptions: 详细描述字典
            all_candidates: 所有候选bbox字典
            annotated_image_path: 标注图片路径
            original_detections: 原始检测结果
            
        Returns:
            dict: 最终验证结果 {"object_name": bbox 或 None}
        """
        final_detections = {}
        
        # 处理所有物体
        for obj_name in verification_status.keys():
            status = verification_status[obj_name]
            
            if status == "valid":
                # 如果二次验证为valid，直接使用原始检测结果
                final_detections[obj_name] = original_detections[obj_name]
            elif status == "invalid":
                # 如果二次验证为invalid，进行三次验证
                print(f"\nAttempting third verification for {obj_name}...")
                
                # 获取该物体的候选bbox
                candidates = all_candidates.get(obj_name, [])
                if not candidates:
                    print(f"  ⚠ No candidates found for {obj_name}, keeping as invalid")
                    final_detections[obj_name] = None
                    continue
                
                # 获取原始检测的bbox（已经在二次验证中被判定为invalid）
                original_bbox = original_detections[obj_name]
                
                # 尝试所有候选bbox（跳过已经在二次验证中被判定为invalid的bbox）
                valid_candidates = []
                print(f"  Trying {len(candidates)} candidates for {obj_name}")
                
                for i, (bbox, logit) in enumerate(candidates):
                    print(f"    Candidate {i+1}: bbox {bbox}, logit {logit:.3f}")
                    
                    # 跳过已经在二次验证中被判定为invalid的bbox
                    if original_bbox is not None and np.array_equal(bbox, original_bbox):
                        print(f"      Skipping candidate {i+1} (already invalidated in second verification)")
                        continue
                    
                    # 构建单个bbox的验证prompt
                    verification_result = self._verify_single_bbox(
                        image_path, obj_name, bbox, detailed_descriptions, annotated_image_path, i+1
                    )
                    
                    if verification_result == "valid":
                        valid_candidates.append((bbox, logit))
                        print(f"      ✓ Candidate {i+1} verified as valid")
                    else:
                        print(f"      ✗ Candidate {i+1} verified as invalid")
                
                # 选择logit最高的valid候选框
                if valid_candidates:
                    # 按logit排序，选择最高的
                    valid_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_bbox, best_logit = valid_candidates[0]
                    final_detections[obj_name] = best_bbox
                    print(f"  ✓ Third verification successful for {obj_name} with best candidate (logit: {best_logit:.3f})")
                else:
                    print(f"  ⚠ No valid candidates found for {obj_name}, keeping as invalid")
                    final_detections[obj_name] = None
            else:
                # 未知状态，保持为None
                print(f"  ⚠ Unknown verification status for {obj_name}: {status}")
                final_detections[obj_name] = None
        
        return final_detections

    def _verify_single_bbox(self, image_path, obj_name, bbox, detailed_descriptions, annotated_image_path, candidate_idx):
        """
        验证单个bbox
        
        Args:
            image_path: 图片路径
            obj_name: 物体名称
            bbox: 边界框 [x1, y1, x2, y2]
            detailed_descriptions: 详细描述字典
            annotated_image_path: 标注图片路径
            candidate_idx: 候选索引（用于文件命名）
            
        Returns:
            str: "valid" 或 "invalid"
        """
        try:
            x1, y1, x2, y2 = bbox
            description = detailed_descriptions.get(obj_name, f"a {obj_name}")
            
            # 构建单个bbox的验证prompt
            prompt = f"""
            Please verify this specific object detection result.
            
            Object: {obj_name}
            Description: {description}
            Bounding box: [{x1}, {y1}, {x2}, {y2}]
            
            Look carefully at the bounding box in the image and determine if it correctly identifies the described object.
            
            Please respond with ONLY "valid" or "invalid".
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at verifying object detection results. Respond with only 'valid' or 'invalid'."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._encode_image(annotated_image_path)}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用GPT-4O进行验证
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                temperature=0.1,
                max_tokens=50  # 只需要简短回答
            )
            
            output = response.choices[0].message.content.strip().lower()
            
            # 保存原始输出用于调试
            with open(f"src_reward/tests/third_verification_{obj_name}_candidate_{candidate_idx}.txt", 'w') as f:
                f.write(f"Object: {obj_name}\n")
                f.write(f"Bbox: {bbox}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Response: {output}\n")
            
            # 解析结果
            if "valid" in output:
                return "valid"
            elif "invalid" in output:
                return "invalid"
            else:
                print(f"    ⚠ Unexpected response for {obj_name}: {output}")
                return "invalid"  # 默认判定为invalid
                
        except Exception as e:
            print(f"    ✗ Error in single bbox verification for {obj_name}: {e}")
            return "invalid"

    def detect_objects_with_groundingdino(self, image_path, concise_descriptions, stage=0):
        """
        使用 GroundingDINO 检测物体
        
        Args:
            image_path: 图片路径
            concise_descriptions: 简洁描述字典，用于 GroundingDINO 检测
            stage: 当前阶段（用于文件命名）
        
        Returns:
            tuple: (检测结果字典, 标注图片路径, 候选bbox字典, detector实例)
        """
        try:
            # 初始化 GroundingDINO 检测器
            detector = GroundingDINODetector(device="cuda")
            
            # 进行检测，获取候选bbox
            detections, all_candidates = detector.detect_objects(
                image_path=image_path,
                object_descriptions=concise_descriptions,
                box_threshold=0.35,
                text_threshold=0.25,
                return_candidates=True
            )
            
            # 保存可视化结果（如果失败不影响检测结果）
            annotated_image_path = None
            try:
                output_path = f"/home/detection_visualization_{stage}.jpg"
                detector.visualize_detections(image_path, detections, output_path)
                annotated_image_path = output_path
            except Exception as viz_error:
                print(f"Warning: Visualization failed: {viz_error}")
            
            return detections, annotated_image_path, all_candidates, detector
            
        except Exception as e:
            print(f"Error in GroundingDINO detection: {e}")
            # 返回空的检测结果
            empty_candidates = {obj: [] for obj in concise_descriptions.keys()}
            return {obj: None for obj in concise_descriptions.keys()}, None, empty_candidates, None

    def generate_highlevel(self, test_img_path_recorder, instruction, stage, 
                          last_subtask=None, last_env_image=None, 
                          last_subtask_success=None, failure_reason=None):
        """
        生成高层任务规划
        """
        self.stage = stage
        self.task_dir = f"src_reward/tests/constraints_{self.stage}"     # %%%
        os.makedirs(self.task_dir, exist_ok=True)
        
        messages = self._build_prompt_high(
            test_img_path_recorder, 
            instruction, 
            stage, 
            last_subtask, 
            last_env_image, 
            last_subtask_success, 
            failure_reason
        )
        
        stream = self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            stream=True
        )
        
        output = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                output += delta.content


        with open(f"src_reward/tests/output_raw_highlevel_{stage}.txt", 'w') as f:
            f.write(output)
        
        all_objects_categories = extract_list('all_objects_categories', output)
        The_subtasks = extract_list('The_subtasks', output)
        verb_list_left = extract_list('verb_list_left', output)
        noun_list_left = extract_list('noun_list_left', output)
        verb_list_right = extract_list('verb_list_right', output)
        noun_list_right = extract_list('noun_list_right', output)

        
        return all_objects_categories, The_subtasks, verb_list_left, noun_list_left, verb_list_right, noun_list_right


def get_verified_bboxes(image_path, object_names, config=None):
    """
    给定图片路径和物体名词列表，返回每个物体的高质量bbox（经过二次验证）。
    返回格式: {object_name: [x1, y1, x2, y2], ...}
    """
    if config is None:
        config = {
            'model': 'gpt-4o',
            'temperature': 0,
            'max_tokens': 1000
        }
    generator = ConstraintGenerator(config, rag_enabled=False)
    # 1. 生成物体描述
    detailed_descriptions, concise_descriptions = generator.generate_object_descriptions(image_path, object_names)
    # 2. GroundingDINO 检测
    object_detections, annotated_image_path, all_candidates, detector = generator.detect_objects_with_groundingdino(image_path, concise_descriptions)
    # 3. 二次验证
    final_bboxes = generator.verify_detection_with_gpt4o(image_path, object_detections, detailed_descriptions, annotated_image_path, all_candidates)
    # 转为标准list，兼容json
    for k, v in final_bboxes.items():
        if isinstance(v, np.ndarray):
            final_bboxes[k] = v.tolist()
    return final_bboxes





if __name__ == "__main__":
    config = {
        'model': 'gpt-4o',
        'temperature': 0,
        'max_tokens': 1000
    }
    
    generator = ConstraintGenerator(config, False)
    test_img_path = "/home/task_split/split/1.png"            # %%%
    instruction = "Weigh the tube and then place it on the plate. Only use the left arm to perform the task."
    # "Weigh the beaker and then place it on the plate. Only use the left arm to perform the task."      
    # "Add solution to test tube"  tube_liquid1
    # "Insert the test tube into the test tube rack"  tube_rack2.jpg
    # "Insert the test tube into the test tube rack"  000011
    # "Press the red button."  
    # "Prepare breakfast for me and serve it to me on a tray."
    # "Prepare me a cup of coffee using the grind-and-brew machine and give it to me."
    # "Press the red button."
    # "Open the drawer."
    # "Insert the pipette into the beaker."
    # "Insert the pipette into the test tube."
    # "Insert the test tube into the test tube rack."
    # "Place the beaker on the plate."
    # "Place the petri dish on the plate."
    # "Pour liquid from the test tube into the beaker."
    # "Weigh the beaker and then place it on the plate." 
    # "Weigh the beaker and then place it on the plate. Only use the right arm to perform the task." 
    all_objects_categories, The_subtasks, verb_list_left, noun_list_left, verb_list_right, noun_list_right = generator.generate_highlevel(test_img_path, instruction, stage=0)
    print('all_objects_categories =', all_objects_categories)
    print('The_subtasks =', The_subtasks)
    print('verb_list_left =', verb_list_left)
    print('noun_list_left =', noun_list_left)
    print('verb_list_right =', verb_list_right)
    print('noun_list_right =', noun_list_right)

    # 第一步：生成物体描述
    print("\n=== Step 1: Generating Object Descriptions ===")
    if all_objects_categories:
        detailed_descriptions, concise_descriptions = generator.generate_object_descriptions(test_img_path, all_objects_categories)
        print('detailed_descriptions =', detailed_descriptions)
        print('concise_descriptions =', concise_descriptions)
    else:
        print("No objects found to describe")
        detailed_descriptions = {}
        concise_descriptions = {}

    # 第二步：使用 GroundingDINO 检测物体
    print("\n=== Step 2: GroundingDINO Object Detection ===")
    if concise_descriptions:
        object_detections, annotated_image_path, all_candidates, detector = generator.detect_objects_with_groundingdino(test_img_path, concise_descriptions, stage=0)
        print('object_detections =', object_detections)
        print('annotated_image_path =', annotated_image_path)
        print('all_candidates =', all_candidates)
    else:
        print("No objects to detect")
        object_detections = {}
        annotated_image_path = None
        all_candidates = {}
        detector = None

    # 第三步：使用 GPT-4O 验证检测结果
    print("\n=== Step 3: GPT-4O Verification ===")
    if object_detections and detailed_descriptions:
        final_detections = generator.verify_detection_with_gpt4o(test_img_path, object_detections, detailed_descriptions, annotated_image_path, all_candidates)
        print('final_detections =', final_detections)
        # 重新绘制最终结果图片，路径与annotated_image_path相同
        if annotated_image_path and final_detections and detector is not None:
            detector.visualize_detections(test_img_path, final_detections, annotated_image_path)
            print(f"Final detection visualization saved to: {annotated_image_path}")
    else:
        print("No detections to verify")
        final_detections = object_detections  # 如果没有检测结果，使用原始结果




    import requests  # 放在文件顶部导入

    # 向服务器发送数据
    try:
        payload = {
            "all_objects_categories": all_objects_categories,
            "The_subtasks": The_subtasks,
            "verb_list_left": verb_list_left,
            "noun_list_left": noun_list_left,
            "verb_list_right": verb_list_right,
            "noun_list_right": noun_list_right,
            "final_detections": final_detections             # 最终验证后的检测结果（物体+bbox坐标）
        }
        response = requests.post("http://localhost:5000/dual_skills", json=payload)
        print("Server responded:", response.text)
    except Exception as e:
        print("Error sending data to server:", e)





