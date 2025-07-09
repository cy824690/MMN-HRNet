#!/usr/bin/env python3
"""
测试姿态估计重排序效果
"""

import numpy as np
import torch
from pose_estimator import PoseEstimator, TwoStageReID
import time

def test_pose_estimation():
    """测试姿态估计功能"""
    print("=== 测试姿态估计功能 ===")
    
    # 初始化姿态估计器
    pose_estimator = PoseEstimator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成测试图像
    test_images = []
    for i in range(5):
        # 模拟不同姿态的图像
        img = np.random.randint(0, 255, (384, 192, 3), dtype=np.uint8)
        test_images.append(img)
    
    print(f"生成了 {len(test_images)} 张测试图像")
    
    # 提取关键点
    start_time = time.time()
    keypoints_list = pose_estimator.batch_extract_keypoints(test_images)
    end_time = time.time()
    
    print(f"关键点提取耗时: {end_time - start_time:.3f}秒")
    print(f"成功提取关键点的图像数: {sum(1 for kp in keypoints_list if kp is not None)}")
    
    # 测试相似性计算
    if keypoints_list[0] is not None:
        similarity = pose_estimator.compute_pose_similarity(
            keypoints_list[0], keypoints_list[0])
        print(f"自相似性测试: {similarity:.4f}")
        
        if len(keypoints_list) > 1 and keypoints_list[1] is not None:
            cross_similarity = pose_estimator.compute_pose_similarity(
                keypoints_list[0], keypoints_list[1])
            print(f"跨图像相似性测试: {cross_similarity:.4f}")

def test_two_stage_reranking():
    """测试两阶段重排序"""
    print("\n=== 测试两阶段重排序 ===")
    
    # 初始化两阶段系统
    two_stage_system = TwoStageReID(mmn_model_path=None)
    
    # 生成模拟数据
    query_image = np.random.randint(0, 255, (384, 192, 3), dtype=np.uint8)
    gallery_images = [np.random.randint(0, 255, (384, 192, 3), dtype=np.uint8) 
                     for _ in range(50)]
    
    print(f"查询图像: {query_image.shape}")
    print(f"画廊图像: {len(gallery_images)} 张")
    
    # 执行两阶段重排序
    start_time = time.time()
    reranked_indices = two_stage_system.rerank(query_image, gallery_images)
    end_time = time.time()
    
    print(f"重排序耗时: {end_time - start_time:.3f}秒")
    print(f"重排序结果: {reranked_indices[:10]}")  # 显示前10个结果

def benchmark_performance():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    pose_estimator = PoseEstimator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试不同数量的图像
    for num_images in [10, 20, 50]:
        test_images = [np.random.randint(0, 255, (384, 192, 3), dtype=np.uint8) 
                      for _ in range(num_images)]
        
        start_time = time.time()
        keypoints_list = pose_estimator.batch_extract_keypoints(test_images)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_images
        print(f"{num_images} 张图像: 平均 {avg_time*1000:.2f}ms/张")

def simulate_improvement():
    """模拟性能提升"""
    print("\n=== 性能提升模拟 ===")
    
    # 模拟当前MMN的性能
    current_rank1 = 71.55  # 当前Rank-1准确率
    current_rank10 = 94.40  # 当前Rank-10准确率
    
    # 模拟姿态估计的改进效果
    pose_improvement_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # 姿态估计的改进率
    
    print("不同姿态估计改进率下的预期Rank-1准确率:")
    for improvement in pose_improvement_rates:
        # 假设姿态估计能正确识别出前10名中的正确匹配
        potential_improvement = current_rank10 * improvement
        expected_rank1 = current_rank1 + potential_improvement
        print(f"改进率 {improvement*100}%: Rank-1 = {expected_rank1:.2f}%")

if __name__ == "__main__":
    print("开始姿态估计重排序测试...")
    
    try:
        test_pose_estimation()
        test_two_stage_reranking()
        benchmark_performance()
        simulate_improvement()
        
        print("\n✅ 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 