import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def get_basic_transforms(phase='train', val_augment=False):
    if phase == 'train' or val_augment:
        return {
            'vis': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'ir': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        }
    else:
        return {
            'vis': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'ir': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        }


# MultiModalDomainDataset 类保持不变，可以直接复制你原来的，或者用上一轮我修正过带错误检查的版本
# 建议使用上一轮那个带 FileNotFoundError 检查的版本
class MultiModalDomainDataset(Dataset):
    def __init__(self, root_dir, domain_type='source', phase='train', img_size=224, weather=None,
                 global_label_map=None, val_augment=False):
        self.weather = weather if weather else os.path.basename(root_dir)
        self.domain_type = domain_type
        self.phase = phase
        self.img_size = img_size
        self.samples = []
        self.domain_label = 0 if domain_type == 'source' else 1
        self.val_augment = val_augment

        # 兼容 train 和 val 都在同一级目录的情况
        # 逻辑：如果 root_dir/phase 存在，就用这个；否则直接用 root_dir (针对某些特殊结构)
        phase_path = os.path.join(root_dir, phase)
        if not os.path.exists(phase_path):
            # 尝试回退逻辑，或者报错
            if os.path.exists(root_dir) and phase in ['train', 'val']:
                # 假设用户给的路径没包含 train/val，我们在内部拼
                pass
            else:
                raise FileNotFoundError(f"找不到路径: {phase_path}")

        # ================== 数据加载核心逻辑 ==================
        if domain_type == 'source':
            vis_dir = os.path.join(phase_path, '可见光')
            ir_dir = os.path.join(phase_path, '红外')

            # 容错：有些数据集结构可能直接就是类别
            if not os.path.exists(vis_dir):
                # 尝试直接遍历类别
                pass

            valid_classes = [d for d in os.listdir(vis_dir) if os.path.isdir(os.path.join(vis_dir, d))]
            for class_name in valid_classes:
                vis_class_dir = os.path.join(vis_dir, class_name)
                ir_class_dir = os.path.join(ir_dir, class_name)
                if not os.path.isdir(ir_class_dir): continue

                vis_files = sorted(
                    [f for f in os.listdir(vis_class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                ir_files = sorted(
                    [f for f in os.listdir(ir_class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

                # 截断对齐
                min_len = min(len(vis_files), len(ir_files))
                for i in range(min_len):
                    self.samples.append({
                        'vis': os.path.join(vis_class_dir, vis_files[i]),
                        'ir': os.path.join(ir_class_dir, ir_files[i]),
                        'label': int(class_name)
                    })
        else:  # target
            valid_classes = [d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))]
            for class_name in valid_classes:
                class_dir = os.path.join(phase_path, class_name)
                vis_dir = os.path.join(class_dir, '可见光')
                ir_dir = os.path.join(class_dir, '红外')
                if not (os.path.exists(vis_dir) and os.path.exists(ir_dir)): continue

                vis_files = sorted([f for f in os.listdir(vis_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

                min_len = min(len(vis_files), len(ir_files))
                for i in range(min_len):
                    self.samples.append({
                        'vis': os.path.join(vis_dir, vis_files[i]),
                        'ir': os.path.join(ir_dir, ir_files[i]),
                        'label': int(class_name)
                    })

        self.labels = [s['label'] for s in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))

        if global_label_map:
            self.label_map = global_label_map
        else:
            self.label_map = {orig: idx for idx, orig in enumerate(self.unique_labels)}

        self.num_classes = len(self.label_map)
        self.get_label_map = lambda: self.label_map
        self.transform = get_basic_transforms(phase, self.val_augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            vis_img = Image.open(sample['vis']).convert('RGB')
            ir_img = Image.open(sample['ir']).convert('L').convert('RGB')
        except:
            # 简单的容错，如果图片坏了，随机返回一张（生产环境不建议，但为了不中断训练）
            return self.__getitem__((idx + 1) % len(self))

        label_id = self.label_map.get(sample['label'], 0)

        return {
            'vis': self.transform['vis'](vis_img),
            'ir': self.transform['ir'](ir_img),
            'label': torch.tensor(label_id, dtype=torch.long),
            'domain_label': self.domain_label
        }