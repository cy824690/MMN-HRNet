import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional
import os
import time
import scipy.io

# 导入原始模块
from data_loader import TestData
from data_manager import process_query_sysu, process_gallery_sysu
from model import embed_net
from eval_metrics import eval_sysu
from pose_estimator_mmpose import MMPoseHRNetEstimator  # 导入mmpose集成

# 兼容 Pillow 新旧版本的 resample 参数
try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = getattr(Image, 'ANTIALIAS', 3)

# 注释掉原有导入
# class PoseEstimator:
#     """基于HRNet的姿态估计器，完全匹配原始MMN模块"""
    
#     def __init__(self, device: str = 'cuda'):
#         self.device = device
#         self.transform = self._get_transform()
        
#         # 初始化HRNet模型
#         self.model = self._load_hrnet_model()
#         self.model.to(device)
#         self.model.eval()
        
#         # COCO关键点定义
#         self.keypoint_names = [
#             'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
#             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
#             'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
#             'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
#         ]
        
#     def _get_transform(self):
#         """获取图像预处理变换，匹配原始MMN"""
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                        std=[0.229, 0.224, 0.225])
#         return transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((384, 192)),  # 匹配原始尺寸
#             transforms.ToTensor(),
#             normalize,
#         ])
    
#     def _load_hrnet_model(self):
#         """加载HRNet模型（简化版本）"""
#         class SimpleHRNet(nn.Module):
#             def __init__(self, num_keypoints=17):
#                 super(SimpleHRNet, self).__init__()
#                 # 简化的HRNet结构
#                 self.backbone = nn.Sequential(
#                     nn.Conv2d(3, 64, 3, padding=1),
#                     nn.BatchNorm2d(64),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(64, 64, 3, padding=1),
#                     nn.BatchNorm2d(64),
#                     nn.ReLU(inplace=True),
#                     nn.MaxPool2d(2, 2)
#                 )
                
#                 self.head = nn.Sequential(
#                     nn.Conv2d(64, 128, 3, padding=1),
#                     nn.BatchNorm2d(128),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(128, num_keypoints, 1)
#                 )
                
#             def forward(self, x):
#                 x = self.backbone(x)
#                 x = self.head(x)
#                 return x
        
#         return SimpleHRNet()
    
#     def extract_keypoints(self, image_tensor: torch.Tensor) -> np.ndarray:
#         """提取关键点，输入为torch.Tensor格式"""
#         with torch.no_grad():
#             # 前向传播
#             heatmaps = self.model(image_tensor.to(self.device))
            
#             # 从热力图中提取关键点坐标
#             keypoints = self._extract_keypoints_from_heatmaps(heatmaps[0])
            
#         return keypoints
    
#     def _extract_keypoints_from_heatmaps(self, heatmaps: torch.Tensor) -> np.ndarray:
#         """从热力图中提取关键点坐标"""
#         keypoints = []
#         for i in range(heatmaps.shape[0]):
#             heatmap = heatmaps[i].cpu().numpy()
            
#             # 找到热力图的峰值
#             y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            
#             # 转换为原始图像坐标
#             x = x * 4  # 根据下采样倍数调整
#             y = y * 4
            
#             keypoints.extend([x, y, 1.0])  # 添加置信度
        
#         return np.array(keypoints).reshape(-1, 3)
    
#     def compute_pose_similarity(self, query_keypoints: np.ndarray, 
#                                candidate_keypoints: np.ndarray) -> float:
#         """计算姿态相似性"""
#         if query_keypoints is None or candidate_keypoints is None:
#             return 0.0
        
#         # 计算关键点距离
#         distances = []
#         for i in range(len(query_keypoints)):
#             if query_keypoints[i][2] > 0.1 and candidate_keypoints[i][2] > 0.1:  # 置信度阈值
#                 dist = np.linalg.norm(query_keypoints[i][:2] - candidate_keypoints[i][:2])
#                 distances.append(dist)
        
#         if not distances:
#             return 0.0
        
#         # 计算平均距离并转换为相似性分数
#         avg_distance = np.mean(distances)
#         similarity = np.exp(-avg_distance / 50.0)  # 距离越小，相似性越高
        
#         return float(similarity)

def preprocess_gallery_img(tensor_img):
    """
    将DataLoader输出的tensor图像转为numpy格式（[H, W, 3]，uint8），只保留RGB图像。
    """
    # tensor: [C, H, W], float, 0-1 or 0-255
    img = tensor_img.detach().cpu().numpy()
    if img.max() <= 1.0:
        img = img * 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    # 剔除红外图（单通道）
    if img.shape[2] == 1:
        return None  # 红外图直接剔除
    # 若不是3通道，跳过
    if img.shape[2] != 3:
        return None
    return img

class TwoStageReIDSystem:
    """两阶段ReID系统，完全匹配原始MMN模块"""
    
    def __init__(self, mmn_model_path: str, device: str = 'cuda'):
        self.device = device
        
        # 加载MMN模型
        self.mmn_net = self._load_mmn_model(mmn_model_path)
        # === 替换为mmpose HRNet ===
        config_file = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        checkpoint_file = 'checkpoint/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        self.pose_estimator = MMPoseHRNetEstimator(config_file, checkpoint_file, device)
        # =========================
        
        # 数据预处理 - 完全匹配原始test.py
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 192)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
    def _load_mmn_model(self, model_path: str):
        """加载MMN模型，完全匹配原始test.py"""
        n_class = 395  # SYSU-MM01的类别数
        net = embed_net(n_class, no_local='on', gm_pool='on', arch='resnet50')
        
        if os.path.isfile(model_path):
            print(f'==> loading checkpoint {model_path}')
            checkpoint = torch.load(model_path, map_location=self.device)
            net.load_state_dict(checkpoint['net'])
            print(f'==> loaded checkpoint (epoch {checkpoint["epoch"]})')
        
        net.to(self.device)
        net.eval()
        return net
    
    def extract_query_feat(self, query_loader):
        """提取查询特征，完全匹配原始test.py"""
        self.mmn_net.eval()
        print('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        nquery = len(query_loader.dataset)
        pool_dim = 8192
        
        query_feat_pool = np.zeros((nquery, pool_dim))
        query_feat_fc = np.zeros((nquery, pool_dim))
        Xquery_feat_pool = np.zeros((nquery, pool_dim))
        Xquery_feat_fc = np.zeros((nquery, pool_dim))
        
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = input.to(self.device)
                feat_pool, feat_fc = self.mmn_net(input, input, 2)  # thermal mode
                
                query_feat_pool[ptr:ptr+batch_num, :] = feat_pool[:batch_num].detach().cpu().numpy()
                query_feat_fc[ptr:ptr+batch_num, :] = feat_fc[:batch_num].detach().cpu().numpy()
                Xquery_feat_pool[ptr:ptr+batch_num, :] = feat_pool[batch_num:].detach().cpu().numpy()
                Xquery_feat_fc[ptr:ptr+batch_num, :] = feat_fc[batch_num:].detach().cpu().numpy()
                ptr = ptr + batch_num
                
        print('Extracting Time:\t {:.3f}'.format(time.time()-start))
        return query_feat_pool, query_feat_fc, Xquery_feat_pool, Xquery_feat_fc
    
    def extract_gall_feat(self, gall_loader):
        """提取图库特征，完全匹配原始test.py"""
        self.mmn_net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        ngall = len(gall_loader.dataset)
        pool_dim = 8192
        
        gall_feat_pool = np.zeros((ngall, pool_dim))
        gall_feat_fc = np.zeros((ngall, pool_dim))
        Xgall_feat_pool = np.zeros((ngall, pool_dim))
        Xgall_feat_fc = np.zeros((ngall, pool_dim))
        
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = input.to(self.device)
                feat_pool, feat_fc = self.mmn_net(input, input, 1)  # visible mode
                
                gall_feat_pool[ptr:ptr+batch_num, :] = feat_pool[:batch_num].detach().cpu().numpy()
                gall_feat_fc[ptr:ptr+batch_num, :] = feat_fc[:batch_num].detach().cpu().numpy()
                Xgall_feat_pool[ptr:ptr+batch_num, :] = feat_pool[batch_num:].detach().cpu().numpy()
                Xgall_feat_fc[ptr:ptr+batch_num, :] = feat_fc[batch_num:].detach().cpu().numpy()
                ptr = ptr + batch_num
                
        print('Extracting Time:\t {:.3f}'.format(time.time()-start))
        return gall_feat_pool, gall_feat_fc, Xgall_feat_pool, Xgall_feat_fc
    
    def first_stage_ranking(self, query_feat_fc: np.ndarray, 
                           gall_feat_fc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """第一阶段：MMN粗筛选，返回距离矩阵和Top-10索引"""
        # 计算距离矩阵，匹配原始test.py
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        
        # 获取Top-10索引
        top10_indices = np.argsort(-distmat, axis=1)[:, :10]
        
        return distmat, top10_indices
    
    def second_stage_reranking(self, query_loader, gall_loader, 
                              top10_indices: np.ndarray, query_idx: int = 0) -> List[float]:
        """第二阶段：姿态估计重排序（仅对RGB gallery，剔除红外）"""
        # 获取查询图像的关键点
        query_keypoints: Optional[np.ndarray] = None
        for batch_idx, (input, label) in enumerate(query_loader):
            if batch_idx * query_loader.batch_size <= query_idx < (batch_idx + 1) * query_loader.batch_size:
                local_idx = query_idx - batch_idx * query_loader.batch_size
                if local_idx < input.size(0):
                    img = preprocess_gallery_img(input[local_idx])
                    if img is not None:
                        assert isinstance(img, np.ndarray), f"img is not np.ndarray, but {type(img)}"
                        assert img.shape[2] == 3, f"img shape is {img.shape}, expected 3 channels"
                        assert img.dtype == np.uint8, f"img dtype is {img.dtype}, expected uint8"
                        query_keypoints = self.pose_estimator.extract_keypoints([img])[0]
                break
        # 获取候选图像的关键点
        candidate_keypoints = []
        for idx in top10_indices:
            found = False
            for batch_idx, (input, label) in enumerate(gall_loader):
                if batch_idx * gall_loader.batch_size <= idx < (batch_idx + 1) * gall_loader.batch_size:
                    local_idx = idx - batch_idx * gall_loader.batch_size
                    if local_idx < input.size(0):
                        img = preprocess_gallery_img(input[local_idx])
                        if img is not None:
                            assert isinstance(img, np.ndarray), f"img is not np.ndarray, but {type(img)}"
                            assert img.shape[2] == 3, f"img shape is {img.shape}, expected 3 channels"
                            assert img.dtype == np.uint8, f"img dtype is {img.dtype}, expected uint8"
                            candidate_kp = self.pose_estimator.extract_keypoints([img])[0]
                            candidate_keypoints.append(candidate_kp)
                            found = True
                        else:
                            # 红外图剔除，填充全零关键点
                            candidate_keypoints.append(np.zeros((17, 3)))
                            found = True
                    break
            if not found:
                dummy_keypoints = np.zeros((17, 3))
                candidate_keypoints.append(dummy_keypoints)
        # 计算姿态相似性
        pose_scores = []
        for candidate_kp in candidate_keypoints:
            if query_keypoints is not None:
                pose_similarity = self.pose_estimator.compute_pose_similarity(
                    query_keypoints, candidate_kp)
                pose_scores.append(pose_similarity)
            else:
                pose_scores.append(0.0)
        return pose_scores
    
    def evaluate_with_pose_reranking(self, data_path: str, mode: str = 'all', 
                                   trial: int = 0, test_batch: int = 64):
        """完整的评估流程，集成姿态重排序，完全匹配原始test.py"""
        
        print("=== 两阶段ReID评估 ===")
        
        # 加载数据，匹配原始test.py
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        
        nquery = len(query_label)
        ngall = len(gall_label)
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  ------------------------------")
        
        # 创建数据加载器，匹配原始test.py
        queryset = TestData(query_img, query_label, transform=self.transform_test, 
                           img_size=(192, 384))
        gallset = TestData(gall_img, gall_label, transform=self.transform_test, 
                          img_size=(192, 384))
        
        query_loader = data.DataLoader(queryset, batch_size=test_batch, 
                                     shuffle=False, num_workers=4)
        gall_loader = data.DataLoader(gallset, batch_size=test_batch, 
                                    shuffle=False, num_workers=4)
        
        # 第一阶段：MMN特征提取
        print("第一阶段：MMN特征提取...")
        query_feat_pool, query_feat_fc, Xquery_feat_pool, Xquery_feat_fc = self.extract_query_feat(query_loader)
        gall_feat_pool, gall_feat_fc, Xgall_feat_pool, Xgall_feat_fc = self.extract_gall_feat(gall_loader)
        
        # 计算所有距离矩阵，匹配原始test.py
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        distmat1 = np.matmul(query_feat_fc, np.transpose(Xgall_feat_fc))
        distmat2 = np.matmul(Xquery_feat_fc, np.transpose(gall_feat_fc))
        distmat3 = np.matmul(Xquery_feat_fc, np.transpose(Xgall_feat_fc))
        
        # 融合距离矩阵
        distmat_fused = distmat + distmat1 + distmat2 + distmat3
        
        # 获取Top-10索引
        top10_indices = np.argsort(-distmat_fused, axis=1)[:, :10]
        
        # 第二阶段：姿态重排序（仅对第一个查询进行演示）
        print("第二阶段：姿态重排序...")
        pose_scores = self.second_stage_reranking(query_loader, gall_loader, 
                                                 top10_indices[0], query_idx=0)
        
        # 融合分数
        mmn_scores = distmat_fused[0, top10_indices[0]]
        final_scores = 0.7 * mmn_scores + 0.3 * np.array(pose_scores)
        
        # 重新排序
        reranked_indices = top10_indices[0][np.argsort(-final_scores)]
        
        # 评估结果
        cmc, mAP, mINP = eval_sysu(-distmat_fused, query_label, gall_label, query_cam, gall_cam)
        
        print(f"原始MMN结果:")
        print(f"Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | mAP: {mAP:.2%}")
        print(f"原始Top-1索引: {top10_indices[0][0]}")
        print(f"重排序后Top-1索引: {reranked_indices[0]}")
                # 1. 未使用姿态估计的评测
        cmc, mAP, mINP = eval_sysu(-distmat_fused, query_label, gall_label, query_cam, gall_cam)
        print("【未使用姿态估计】")
        print(f"Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | mAP: {mAP:.2%}")
        
        # 2. 批量对所有query做姿态重排序
        print("批量进行姿态重排序...")
        distmat_pose = distmat_fused.copy()
        for q_idx in range(nquery):
            top10 = top10_indices[q_idx]
            # 计算当前query的Top-10 gallery的姿态分数
            pose_scores = self.second_stage_reranking(query_loader, gall_loader, top10, query_idx=q_idx)
            mmn_scores = distmat_fused[q_idx, top10]
            final_scores = 0.7 * mmn_scores + 0.3 * np.array(pose_scores)
            reranked = top10[np.argsort(-final_scores)]
            # 用重排后的顺序更新距离矩阵
            distmat_pose[q_idx, top10] = distmat_fused[q_idx, reranked]
            if q_idx % 10 == 0:
                print(f"已处理 {q_idx}/{nquery} 个query...")
        
        # 3. 使用姿态估计的评测
        cmc_pose, mAP_pose, mINP_pose = eval_sysu(-distmat_pose, query_label, gall_label, query_cam, gall_cam)
        print("【使用姿态估计】")
        print(f"Rank-1: {cmc_pose[0]:.2%} | Rank-5: {cmc_pose[4]:.2%} | Rank-10: {cmc_pose[9]:.2%} | mAP: {mAP_pose:.2%}")
        
        # 4. 返回结果
        return {
            'cmc': cmc,
            'mAP': mAP,
            'mINP': mINP,
            'cmc_pose': cmc_pose,
            'mAP_pose': mAP_pose,
            'mINP_pose': mINP_pose,
        }

# 使用示例
def test_integration():
    """测试集成效果"""
    print("开始测试两阶段ReID集成...")
    
    # 初始化系统
    model_path = "save_model/sysu_agw_p4_n4_lr_0.1_seed_0_best.t"
    data_path = "./Datasets/SYSU-MM01/"
    
    if os.path.exists(model_path):
        system = TwoStageReIDSystem(model_path)
        
        # 执行评估
        results = system.evaluate_with_pose_reranking(data_path, mode='all', trial=0)
        
        print("✅ 集成测试完成!")
        print(f"姿态重排序分数: {results['pose_scores']}")
        return results
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None

if __name__ == "__main__":
    test_integration() 