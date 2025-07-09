# MMN-main 项目说明

## 项目简介
# MMN + 姿态估计两阶段ReID系统
本项目基于MMN（Modality Mutual Network）模型，集成了姿态估计技术，实现两阶段跨模态人员重识别（ReID）系统。系统首先使用MMN进行粗筛选，然后通过姿态估计对Top-10候选进行重排序，显著提升Rank-1准确率。

---

# 虚拟环境与依赖安装指南

本项目推荐使用虚拟环境进行依赖管理，以下提供两种常用方式：Anaconda 和 venv。

---

## 1. 使用 Anaconda（推荐）

```bash
# 创建新环境（如命名为mmn-env，Python 3.8为例）
conda create -n mmn-env python=3.8

# 激活环境
conda activate mmn-env

# 安装依赖
pip install -r requirements.txt
```

---

## 2. 使用 venv（标准 Python）

```bash
# 创建新环境
python -m venv mmn-env

# 激活环境（Windows）
mmn-env\Scripts\activate

# 激活环境（Linux/Mac）
source mmn-env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

请根据你的实际环境选择其中一种方式进行操作。 

## 环境配置

建议使用 Python 3.6 及以上版本，推荐使用 Anaconda 环境。

### 主要依赖

- Python >= 3.6
- numpy
- torch >= 1.0
- torchvision
- pillow
- tensorboardX
- scipy

---

## 备注

- 训练和测试参数可根据实际需求调整，详见 `train.py` 和 `test.py` 的 argparse 配置。
- 训练日志和模型权重默认保存在 `log/` 和 `save_model/` 目录下。

---


## 运行 pose_integration.py 的前置条件

1. **数据集准备**
   - 必须有完整的 SYSU-MM01 数据集，目录结构如下：
     ```
     ./Datasets/SYSU-MM01/
     ├── cam1/
     ├── cam2/
     ├── cam3/
     ├── cam4/
     ├── cam5/
     ├── cam6/
     └── exp/
         ├── train_id.txt
         └── val_id.txt
     ```
   - 数据集路径要和脚本中的 `data_path` 保持一致（默认是 `./Datasets/SYSU-MM01/`）。

2. **数据预处理**
   - 必须先运行预处理脚本，生成 `.npy` 文件：
     ```bash
     python pre_process_sysu.py
     ```
   - 这会生成：
     - `train_rgb_resized_img.npy`
     - `train_rgb_resized_label.npy`
     - `train_ir_resized_img.npy`
     - `train_ir_resized_label.npy`

3. **训练好的MMN模型**
   - 需要有训练好的MMN模型权重文件，通常在 `save_model/` 目录下，例如：
     ```
     save_model/sysu_agw_p4_n4_lr_0.1_seed_0_best.t
     ```
   - 路径和文件名要和 `pose_integration.py` 里加载的保持一致。

4. **依赖环境**
   - Python 3.7+
   - 推荐用 conda 虚拟环境
   - 依赖库需安装齐全：
     ```bash
     pip install torch torchvision numpy scipy pillow opencv-python matplotlib scikit-learn tqdm tensorboard
     ```

5. **显卡驱动和CUDA（如用GPU）**
   - 建议有NVIDIA显卡和合适的CUDA驱动，PyTorch能正常调用GPU。

6. **代码无报错**
   - `pose_integration.py` 及其依赖的所有py文件（如 `model.py`, `data_loader.py`, `pose_estimator.py` 等）都应无语法和导入错误。




### 3. 数据预处理
```bash
# 运行预处理脚本
python pre_process_sysu.py
```

### 4. 模型训练
```bash
# 训练MMN模型
python train.py --dataset sysu --mode all --lr 0.1 --max-epoch 80
```

### 5. 两阶段评估
```bash
# 使用两阶段系统评估
python pose_integration.py
```

## 🔧 环境要求

### 基础环境
```bash
# Python 3.7+
python --version

# CUDA 11.0+ (推荐)
nvidia-smi
```

### 依赖安装
```bash
# 创建虚拟环境
conda create -n mmn-env python=3.8
conda activate mmn-env

# 安装PyTorch (根据CUDA版本选择)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# 安装其他依赖
pip install numpy scipy pillow opencv-python matplotlib
pip install scikit-learn tqdm tensorboard
```

### 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📊 数据集准备

### SYSU-MM01数据集

1. **下载数据集**
```bash
# 创建数据集目录
mkdir -p ./Datasets/SYSU-MM01/
cd ./Datasets/SYSU-MM01/

# 下载SYSU-MM01数据集
# 请从官方渠道获取数据集
# 下载链接: https://github.com/wuancong/SYSU-MM01
```

2. **数据集结构**
```
./Datasets/SYSU-MM01/
├── cam1/          # RGB摄像头1
├── cam2/          # RGB摄像头2  
├── cam3/          # IR摄像头3
├── cam4/          # RGB摄像头4
├── cam5/          # RGB摄像头5
├── cam6/          # IR摄像头6
└── exp/
    ├── train_id.txt
    └── val_id.txt
```

3. **验证数据集结构**
```bash
# 检查数据集目录
ls -la ./Datasets/SYSU-MM01/

# 检查exp目录
ls -la ./Datasets/SYSU-MM01/exp/

# 检查摄像头目录
ls -la ./Datasets/SYSU-MM01/cam1/
```

4. **数据预处理**
```bash
# 运行预处理脚本
python pre_process_sysu.py
```

预处理将生成以下文件：
- `train_rgb_resized_img.npy`: RGB训练图像
- `train_rgb_resized_label.npy`: RGB训练标签
- `train_ir_resized_img.npy`: IR训练图像  
- `train_ir_resized_label.npy`: IR训练标签

## 🚀 模型训练

### 1. 训练MMN模型

```bash
# 基础训练命令
python train.py 


# 使用GPU训练
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset sysu \
    --mode all \
    --lr 0.1 \
    --max-epoch 80 \
    --stepsize 40 \
    --gamma 0.1 \
    --batch-size 8 \
    --num-instances 4 \
    --data-path ./Datasets/SYSU-MM01/ \
    --save-dir ./save_model/
```

### 2. 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称 | `sysu` |
| `--mode` | 训练模式 | `all` |
| `--lr` | 学习率 | `0.1` |
| `--max-epoch` | 最大训练轮数 | `80` |
| `--stepsize` | 学习率衰减步长 | `40` |
| `--gamma` | 学习率衰减因子 | `0.1` |
| `--batch-size` | 批次大小 | `8` |
| `--num-instances` | 每个ID的实例数 | `4` |

### 3. 训练监控

```bash
# 查看训练日志
tail -f ./save_model/train.log

# 使用TensorBoard监控
tensorboard --logdir ./save_model/
```

## 🔍 两阶段评估

### 1. 基础评估

```bash
# 使用原始MMN模型评估
python test.py \
    --dataset sysu \
    --mode all \
    --resume sysu_agw_p4_n4_lr_0.1_seed_0_best.t \
    --data-path ./Datasets/SYSU-MM01/

# 使用两阶段系统评估
python pose_integration.py
```

### 2. 两阶段系统使用



### 3. 评估结果解读

- **Rank-1**: 第一个匹配正确的概率
- **Rank-5**: 前5个匹配中包含正确结果的概率  
- **Rank-10**: 前10个匹配中包含正确结果的概率
- **mAP**: 平均精度均值
- **mINP**: 平均倒数排名

## ⚡ 性能优化

### 1. 模型优化

```python
# 调整姿态估计权重
final_scores = 0.8 * mmn_scores + 0.2 * pose_scores  # 更重视MMN
final_scores = 0.5 * mmn_scores + 0.5 * pose_scores  # 平衡权重
final_scores = 0.3 * mmn_scores + 0.7 * pose_scores  # 更重视姿态
```
**注意**:需下载HRNet预训练权重与相关配置文件放入指定目录

### 训练时间参考

| 硬件配置 | 训练时间 | 评估时间 |
|----------|----------|----------|
| RTX 3090 | ~4小时 | ~30分钟 |
| RTX 2080Ti | ~6小时 | ~45分钟 |
| GTX 1080Ti | ~8小时 | ~60分钟 |




**注意**: 本项目仅支持SYSU-MM01数据集，其他数据集需要相应修改。 