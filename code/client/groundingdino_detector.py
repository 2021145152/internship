#!/usr/bin/env python3
"""
GroundingDINO 物体检测模块
用于根据物体描述生成边界框
"""

import os
import sys
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# 添加 GroundingDINO 路径
# GROUNDINGDINO_PATH =  "/home/GroundingDINO"
GROUNDINGDINO_PATH =  "data/keymanip"
sys.path.append(GROUNDINGDINO_PATH)

from inference import Model


class GroundingDINODetector:
    """GroundingDINO 物体检测器"""
    
    def __init__(self, device: str = "cuda"):
        """
        初始化 GroundingDINO 检测器
        
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
        """
        self.device = device
        self.model = None
        # self.config_path = os.path.join(GROUNDINGDINO_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        # self.checkpoint_path = os.path.join(GROUNDINGDINO_PATH, "weights/groundingdino_swint_ogc.pth")
        self.config_path =  "/data/keymanip/deps/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.checkpoint_path = "/data/keymanip/real_world_refined/checkpoints/groundingdino_swint_ogc.pth"
        
        # 检查文件是否存在
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        self._load_model()
    
    def _load_model(self):
        """加载 GroundingDINO 模型"""
        try:
            print(f"Loading GroundingDINO model from {self.checkpoint_path}")
            self.model = Model(
                model_config_path=self.config_path,
                model_checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            print("GroundingDINO model loaded successfully!")
        except Exception as e:
            print(f"Error loading GroundingDINO model: {e}")
            raise
    
    def detect_objects(self, 
                      image_path: str, 
                      object_descriptions: Dict[str, str],
                      box_threshold: float = 0.35,
                      text_threshold: float = 0.25,
                      return_candidates: bool = False) -> Dict[str, Optional[np.ndarray]]:
        """
        检测图像中的物体
        
        Args:
            image_path: 图像路径
            object_descriptions: 物体描述字典，格式为 {"object_name": "description"}
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配阈值
            return_candidates: 是否返回候选bbox（用于三次验证）
            
        Returns:
            Dict[str, Optional[np.ndarray]]: 物体名称到边界框的映射
                边界框格式为 [x1, y1, x2, y2] (左上角和右下角坐标)
                如果未检测到物体，值为 None
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {}
        all_candidates = {}  # 存储所有候选bbox
        
        # 对每个物体进行检测
        for object_name, description in object_descriptions.items():
            print(f"Detecting: {object_name} with description: '{description}'")
            
            try:
                # 使用 GroundingDINO 进行检测
                detections, phrases = self.model.predict_with_caption(
                    image=image_rgb,
                    caption=description,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                
                # 添加调试信息
                print(f"    Raw detections: {len(detections)}")
                print(f"    Raw phrases: {phrases}")
                if len(detections) > 0:
                    print(f"    Detection confidences: {detections.confidence}")
                
                # 获取候选bbox（top3或更少）
                candidates = self._get_candidate_bboxes(detections, phrases, object_name, description)
                all_candidates[object_name] = candidates
                
                # 获取最佳匹配的边界框（保持向后兼容）
                best_bbox = candidates[0][0] if candidates else None
                results[object_name] = best_bbox
                
                if best_bbox is not None:
                    print(f"  ✓ Found {object_name} at bbox: {best_bbox}")
                else:
                    print(f"  ✗ No detection for {object_name}")
                    
            except Exception as e:
                print(f"  ✗ Error detecting {object_name}: {e}")
                results[object_name] = None
                all_candidates[object_name] = []
        
        # 如果请求返回候选bbox，则返回包含候选信息的字典
        if return_candidates:
            return results, all_candidates
        else:
            return results
    
    def _get_candidate_bboxes(self, 
                             detections, 
                             phrases: List[str], 
                             object_name: str, 
                             description: str,
                             top_k: int = 3) -> List[Tuple[np.ndarray, float]]:
        """
        从检测结果中获取top-k候选边界框
        
        Args:
            detections: GroundingDINO 检测结果
            phrases: 检测到的短语列表
            object_name: 目标物体名称
            description: 物体描述
            top_k: 返回的候选数量
            
        Returns:
            List[Tuple[np.ndarray, float]]: 候选边界框列表，每个元素为 (bbox, logit)
                边界框格式为 [x1, y1, x2, y2]，如果没有匹配则返回空列表
        """
        if len(detections) == 0:
            return []
        
        # 计算每个检测结果与目标物体的相似度和置信度
        candidates = []
        for i, phrase in enumerate(phrases):
            # 计算短语与物体名称和描述的相似度
            name_similarity = self._calculate_similarity(phrase.lower(), object_name.lower())
            desc_similarity = self._calculate_similarity(phrase.lower(), description.lower())
            
            # 综合相似度（物体名称权重更高）
            similarity = 0.7 * name_similarity + 0.3 * desc_similarity
            
            # 获取置信度
            confidence = float(detections.confidence[i]) if hasattr(detections, 'confidence') else 0.0
            
            # 综合分数（相似度 + 置信度）
            combined_score = 0.6 * similarity + 0.4 * confidence
            
            # 如果相似度太低，跳过
            if similarity < 0.05:  # 降低阈值，使其更宽松
                continue
            
            # 获取边界框
            bbox = detections.xyxy[i].astype(np.int32)
            candidates.append((bbox, combined_score))
        
        # 按综合分数排序，取top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]
        
        print(f"    Found {len(top_candidates)} candidates for {object_name}")
        for i, (bbox, score) in enumerate(top_candidates):
            print(f"      Candidate {i+1}: bbox {bbox}, score {score:.3f}")
        
        return top_candidates
    
    def _get_best_bbox(self, 
                      detections, 
                      phrases: List[str], 
                      object_name: str, 
                      description: str) -> Optional[np.ndarray]:
        """
        从检测结果中选择最佳匹配的边界框
        
        Args:
            detections: GroundingDINO 检测结果
            phrases: 检测到的短语列表
            object_name: 目标物体名称
            description: 物体描述
            
        Returns:
            Optional[np.ndarray]: 最佳匹配的边界框 [x1, y1, x2, y2]，如果没有匹配则返回 None
        """
        if len(detections) == 0:
            return None
        
        # 计算每个检测结果与目标物体的相似度
        similarities = []
        for i, phrase in enumerate(phrases):
            # 计算短语与物体名称和描述的相似度
            name_similarity = self._calculate_similarity(phrase.lower(), object_name.lower())
            desc_similarity = self._calculate_similarity(phrase.lower(), description.lower())
            
            # 综合相似度（物体名称权重更高）
            similarity = 0.7 * name_similarity + 0.3 * desc_similarity
            similarities.append(similarity)
        
        # 选择相似度最高的检测结果
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # 如果最佳相似度太低，返回 None
        if best_similarity < 0.1:  # 降低阈值，使其更宽松
            print(f"    Similarity too low: {best_similarity}")
            return None
        
        # 返回最佳匹配的边界框
        best_bbox = detections.xyxy[best_idx].astype(np.int32)
        return best_bbox
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 简单的词汇重叠相似度计算
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def visualize_detections(self, 
                           image_path: str, 
                           detections: Dict[str, Optional[np.ndarray]],
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image_path: 图像路径
            detections: 检测结果字典
            output_path: 输出图像路径（可选）
            
        Returns:
            np.ndarray: 标注后的图像
        """
        try:
            # 读取图像
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
            for i, (object_name, bbox) in enumerate(detections.items()):
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
            
            # 保存结果
            if output_path:
                cv2.imwrite(output_path, image)
                print(f"Visualization saved to: {output_path}")
            
            return image
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            # 如果可视化失败，返回原图
            image = cv2.imread(image_path)
            if output_path:
                cv2.imwrite(output_path, image)
            return image


def test_groundingdino_detector():
    """测试 GroundingDINO 检测器"""
    
    # 测试配置
    image_path = "1.png"  # 根据实际情况调整
    object_descriptions = {
        "beaker": "clear glass beaker with markings",
        "plate": "white ceramic plate",
        "scale": "digital weighing scale"
    }
    
    try:
        # 初始化检测器
        detector = GroundingDINODetector(device="cuda")
        
        # 进行检测
        print("=== Testing GroundingDINO Detection ===")
        detections, all_candidates = detector.detect_objects(image_path, object_descriptions, return_candidates=True)
        
        # 打印结果
        print("\nDetection Results:")
        for obj_name, bbox in detections.items():
            if bbox is not None:
                print(f"  {obj_name}: {bbox}")
            else:
                print(f"  {obj_name}: Not detected")
        
        # 可视化结果
        output_path = "detection_visualization.jpg"
        detector.visualize_detections(image_path, detections, output_path)
        
        return detections
        
    except Exception as e:
        print(f"Error in test: {e}")
        return None


if __name__ == "__main__":
    test_groundingdino_detector() 