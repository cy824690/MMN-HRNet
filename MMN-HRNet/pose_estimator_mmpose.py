import torch
import numpy as np
from mmpose.apis import init_model, inference_topdown
from typing import List

class MMPoseHRNetEstimator:
    """
    使用mmpose HRNet进行人体关键点检测的姿态估计器
    """
    def __init__(self, config_file: str, checkpoint_file: str, device: str = 'cuda:0'):
        self.device = device
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.model.cfg.visualizer = None  # 关闭可视化

    def extract_keypoints(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量提取关键点
        imgs: List of np.ndarray (H, W, 3), BGR格式
        返回: List of np.ndarray (17, 3)
        """
        keypoints = []
        for img in imgs:
            # mmpose要求BGR格式
            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()
            if img.shape[2] == 3 and img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            print(f"img.shape: {img.shape}, img.dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
            bbox = np.array([0.0, 0.0, float(img.shape[1]), float(img.shape[0])], dtype=np.float32)
            bbox = np.ascontiguousarray(bbox, dtype=np.float32).reshape(4)
            print("DEBUG: bbox in mmpose:", bbox, type(bbox), getattr(bbox, 'shape', None), getattr(bbox, 'dtype', None))
            if bbox is None:
                raise RuntimeError("bbox is None in mmpose inference_topdown!")
            assert bbox is not None, "bbox is None!"
            assert isinstance(bbox, np.ndarray), f"bbox is not np.ndarray, but {type(bbox)}"
            assert bbox.shape == (4,), f"bbox shape is {bbox.shape}, expected (4,)"
            assert bbox.dtype == np.float32, f"bbox dtype is {bbox.dtype}, expected float32"
            person_results = [{
                'bbox': bbox,
                'bbox_score': float(1.0),
                'label': int(0),
                'track_id': 0
            }]
            print(f"person_results: {person_results}, type: {type(person_results[0]['bbox'])}, shape: {person_results[0]['bbox'].shape}, dtype: {person_results[0]['bbox'].dtype}")
            result = inference_topdown(self.model, img, person_results, bbox_format='xyxy')
            if len(result['pred_instances']['keypoints']) > 0:
                kp = result['pred_instances']['keypoints'][0]  # (17, 2)
                score = result['pred_instances']['keypoint_scores'][0]  # (17,)
                kp = np.concatenate([kp, score[:, None]], axis=1)  # (17, 3)
                keypoints.append(kp)
            else:
                keypoints.append(np.zeros((17, 3)))
        return keypoints

    @staticmethod
    def compute_pose_similarity(query_kp: np.ndarray, gallery_kp: np.ndarray) -> float:
        """
        计算两组关键点的姿态相似性（欧氏距离+exp映射）
        """
        valid = (query_kp[:, 2] > 0.1) & (gallery_kp[:, 2] > 0.1)
        if not np.any(valid):
            return 0.0
        dist = np.linalg.norm(query_kp[valid, :2] - gallery_kp[valid, :2], axis=1)
        avg_dist = np.mean(dist)
        similarity = np.exp(-avg_dist / 50.0)
        return float(similarity) 