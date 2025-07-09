import numpy as np
from PIL import Image
import pdb
import os

# 兼容 Pillow 新旧版本的 resample 参数
try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = getattr(Image, 'ANTIALIAS', 3)  # 3是BICUBIC的整数值

data_path = './Datasets/SYSU-MM01/'

rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras = ['cam3','cam6']

# load id info
file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]
    
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    
# combine train and val split   
id_train.extend(id_val) 

files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)
            
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel
pid_container = set()
for img_path in files_ir:
    # 从路径中提取ID，更安全的方式
    path_parts = img_path.split('/')
    if len(path_parts) >= 2:
        try:
            # 尝试从倒数第二个部分提取ID（通常是文件夹名）
            pid_str = path_parts[-2]
            pid = int(pid_str)
            pid_container.add(pid)
        except (ValueError, IndexError):
            # 如果失败，尝试从文件名中提取
            filename = path_parts[-1]
            # 查找连续的数字
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                try:
                    pid = int(numbers[0])
                    pid_container.add(pid)
                except ValueError:
                    print(f"Warning: Could not extract PID from {img_path}")
                    continue
            else:
                print(f"Warning: No numbers found in {img_path}")
                continue
    else:
        print(f"Warning: Invalid path format {img_path}")
        continue

pid2label = {pid:label for label, pid in enumerate(pid_container)}
fix_image_width = 192
fix_image_height = 384
def read_imgs(train_image):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), resample)
        pix_array = np.array(img)

        train_img.append(pix_array) 
        
        # label
        # 从路径中提取ID，更安全的方式
        path_parts = img_path.split('/')
        if len(path_parts) >= 2:
            try:
                # 尝试从倒数第二个部分提取ID（通常是文件夹名）
                pid_str = path_parts[-2]
                pid = int(pid_str)
                if pid in pid2label:
                    pid = pid2label[pid]
                    train_label.append(pid)
                else:
                    print(f"Warning: PID {pid} not found in pid2label for {img_path}")
                    continue
            except (ValueError, IndexError):
                # 如果失败，尝试从文件名中提取
                filename = path_parts[-1]
                # 查找连续的数字
                import re
                numbers = re.findall(r'\d+', filename)
                if numbers:
                    try:
                        pid = int(numbers[0])
                        if pid in pid2label:
                            pid = pid2label[pid]
                            train_label.append(pid)
                        else:
                            print(f"Warning: PID {pid} not found in pid2label for {img_path}")
                            continue
                    except ValueError:
                        print(f"Warning: Could not extract PID from {img_path}")
                        continue
                else:
                    print(f"Warning: No numbers found in {img_path}")
                    continue
        else:
            print(f"Warning: Invalid path format {img_path}")
            continue
    return np.array(train_img), np.array(train_label)
       
# rgb imges
train_img, train_label = read_imgs(files_rgb)
np.save(data_path + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# ir imges
train_img, train_label = read_imgs(files_ir)
np.save(data_path + 'train_ir_resized_img.npy', train_img)
np.save(data_path + 'train_ir_resized_label.npy', train_label)
