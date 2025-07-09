import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict
import os

class PoseEstimator:
    """基于HRNet的姿态估计器，用于第二阶段重排序"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device
        self.transform = self._get_transform()
        
        # 初始化HRNet模型
        self.model = self._load_hrnet_model()
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
        
        # COCO关键点定义
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def _get_transform(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((256, 192)),  # HRNet标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_hrnet_model(self):
        """加载HRNet模型（简化版本）"""
        # 这里使用简化的HRNet结构，实际使用时可以加载预训练模型
        class SimpleHRNet(nn.Module):
            def __init__(self, num_keypoints=17):
                super(SimpleHRNet, self).__init__()
                # 简化的HRNet结构
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
                
                self.head = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_keypoints, 1)
                )
                
            def forward(self, x):
                x = self.backbone(x)
                x = self.head(x)
                return x
        
        return SimpleHRNet()
    
    def extract_keypoints(self, image: np.ndarray) -> np.ndarray:
        """提取单张图像的关键点"""
        # 预处理
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 前向传播
            heatmaps = self.model(input_tensor)
            
            # 从热力图中提取关键点坐标
            keypoints = self._extract_keypoints_from_heatmaps(heatmaps[0])
            
        return keypoints
    
    def _extract_keypoints_from_heatmaps(self, heatmaps: torch.Tensor) -> np.ndarray:
        """从热力图中提取关键点坐标"""
        keypoints = []
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i].cpu().numpy()
            
            # 找到热力图的峰值
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            
            # 转换为原始图像坐标
            x = x * 4  # 根据下采样倍数调整
            y = y * 4
            
            keypoints.extend([x, y, 1.0])  # 添加置信度
        
        return np.array(keypoints).reshape(-1, 3)
    
    def compute_pose_similarity(self, query_keypoints: np.ndarray, 
                               candidate_keypoints: np.ndarray) -> float:
        """计算姿态相似性"""
        if query_keypoints is None or candidate_keypoints is None:
            return 0.0
        
        # 计算关键点距离
        distances = []
        for i in range(len(query_keypoints)):
            if query_keypoints[i][2] > 0.1 and candidate_keypoints[i][2] > 0.1:  # 置信度阈值
                dist = np.linalg.norm(query_keypoints[i][:2] - candidate_keypoints[i][:2])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # 计算平均距离并转换为相似性分数
        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance / 50.0)  # 距离越小，相似性越高
        
        return float(similarity)
    
    def batch_extract_keypoints(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """批量提取关键点"""
        keypoints_list = []
        for image in images:
            try:
                keypoints = self.extract_keypoints(image)
                keypoints_list.append(keypoints)
            except Exception as e:
                print(f"提取关键点失败: {e}")
                keypoints_list.append(None)
        
        return keypoints_list

class TwoStageReID:
    """两阶段ReID系统"""
    
    def __init__(self, mmn_model_path: str, pose_model_path: str = None):
        self.mmn_model = self._load_mmn_model(mmn_model_path)
        self.pose_estimator = PoseEstimator(pose_model_path)
        
    def _load_mmn_model(self, model_path: str):
        """加载MMN模型"""
        # 这里应该加载你训练好的MMN模型
        # 简化实现，实际使用时需要加载真实的模型
        return None
    
    def first_stage_ranking(self, query_image: np.ndarray, 
                           gallery_images: List[np.ndarray]) -> List[int]:
        """第一阶段：MMN粗筛选，返回Top-10索引"""
        # 这里应该使用你的MMN模型进行推理
        # 简化实现，返回前10个索引
        return list(range(min(10, len(gallery_images))))
    
    def second_stage_reranking(self, query_image: np.ndarray,
                              candidate_images: List[np.ndarray],
                              initial_scores: List[float]) -> List[float]:
        """第二阶段：姿态估计重排序"""
        
        # 提取查询图像的关键点
        query_keypoints = self.pose_estimator.extract_keypoints(query_image)
        
        # 提取候选图像的关键点
        candidate_keypoints = self.pose_estimator.batch_extract_keypoints(candidate_images)
        
        # 计算姿态相似性
        pose_scores = []
        for candidate_kp in candidate_keypoints:
            pose_similarity = self.pose_estimator.compute_pose_similarity(
                query_keypoints, candidate_kp)
            pose_scores.append(pose_similarity)
        
        # 融合分数
        final_scores = []
        for i in range(len(initial_scores)):
            # 加权融合：MMN分数 + 姿态分数
            final_score = 0.7 * initial_scores[i] + 0.3 * pose_scores[i]
            final_scores.append(final_score)
        
        return final_scores
    
    def rerank(self, query_image: np.ndarray, 
               gallery_images: List[np.ndarray]) -> List[int]:
        """完整的两阶段重排序"""
        
        # 第一阶段：MMN粗筛选
        top10_indices = self.first_stage_ranking(query_image, gallery_images)
        candidate_images = [gallery_images[i] for i in top10_indices]
        
        # 获取初始分数（这里需要你的MMN模型输出）
        initial_scores = [1.0] * len(candidate_images)  # 占位符
        
        # 第二阶段：姿态重排序
        final_scores = self.second_stage_reranking(
            query_image, candidate_images, initial_scores)
        
        # 重新排序
        reranked_indices = [top10_indices[i] for i in 
                           np.argsort(final_scores)[::-1]]
        
        return reranked_indices

# 使用示例
def test_pose_estimator():
    """测试姿态估计器"""
    pose_estimator = PoseEstimator()
    
    # 模拟图像数据
    dummy_image = np.random.randint(0, 255, (384, 192, 3), dtype=np.uint8)
    
    # 提取关键点
    keypoints = pose_estimator.extract_keypoints(dummy_image)
    print(f"提取到 {len(keypoints)} 个关键点")
    
    # 测试相似性计算
    similarity = pose_estimator.compute_pose_similarity(keypoints, keypoints)
    print(f"自相似性: {similarity:.4f}")

if __name__ == "__main__":
    test_pose_estimator() 