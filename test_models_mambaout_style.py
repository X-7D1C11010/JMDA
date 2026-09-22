"""
使用MambaOut-main中的方法测试ResNet、ConvNeXt、ViT三种模型
支持三个单模态：可见光、红外、AIS信号

测试流程:
1. 从晴天数据集获取global_label_map（确保标签一致性）
2. 使用timm库直接创建预训练模型
3. 在目标域天气数据上直接测试（每种天气5次，不同随机种子）
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from datetime import datetime
import json
import logging

sys.path.insert(0, r'D:\Code\JMDA-Net\MambaOut-main')

from timm.models import create_model

MODALITIES = {
    'vis': {'name': '可见光', 'channels': 3},
    'ir': {'name': '红外', 'channels': 3},
    'ais': {'name': 'AIS', 'channels': 1}
}


class WeatherDataset(Dataset):
    """天气图像数据集（支持可见光和红外）"""
    def __init__(self, root_dir, modality='vis', phase='test', transform=None, 
                 global_label_map=None):
        self.root_dir = root_dir
        self.modality = modality
        self.phase = phase
        self.transform = transform
        self.samples = []
        self.label_map = {}
        self.num_classes = 0
        
        self._load_data(global_label_map)
        
    def _load_data(self, global_label_map=None):
        modality_dir = MODALITIES[self.modality]['name']
        
        phase_path = os.path.join(self.root_dir, self.phase)
        
        # 结构1: phase/可见光/{类别}/ (晴天)
        modal_first_path = os.path.join(phase_path, modality_dir)
        if os.path.isdir(modal_first_path):
            for class_id in os.listdir(modal_first_path):
                class_path = os.path.join(modal_first_path, class_id)
                if not os.path.isdir(class_path):
                    continue
                
                try:
                    class_id_int = int(class_id)
                except ValueError:
                    continue
                
                if class_id_int not in self.label_map:
                    self.label_map[class_id_int] = len(self.label_map)
                label = self.label_map[class_id_int]
                
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.bmp')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, class_id_int))  # 保存原始类别ID
        
        # 结构2: phase/{类别}/可见光/ (雨天、逆光等)
        elif os.path.isdir(phase_path):
            for class_id in os.listdir(phase_path):
                class_path = os.path.join(phase_path, class_id)
                if not os.path.isdir(class_path):
                    continue
                
                modal_path = os.path.join(class_path, modality_dir)
                if not os.path.isdir(modal_path):
                    continue
                
                try:
                    class_id_int = int(class_id)
                except ValueError:
                    continue
                
                if class_id_int not in self.label_map:
                    self.label_map[class_id_int] = len(self.label_map)
                label = self.label_map[class_id_int]
                
                for img_name in os.listdir(modal_path):
                    if img_name.endswith(('.jpg', '.png', '.bmp')):
                        img_path = os.path.join(modal_path, img_name)
                        self.samples.append((img_path, class_id_int))  # 保存原始类别ID
        
        # 结构3: {类别}/phase/可见光/
        else:
            for class_id in os.listdir(self.root_dir):
                class_phase_path = os.path.join(self.root_dir, class_id, self.phase)
                if not os.path.isdir(class_phase_path):
                    continue
                
                modal_path = os.path.join(class_phase_path, modality_dir)
                if not os.path.exists(modal_path):
                    continue
                
                try:
                    class_id_int = int(class_id)
                except ValueError:
                    continue
                
                if class_id_int not in self.label_map:
                    self.label_map[class_id_int] = len(self.label_map)
                label = self.label_map[class_id_int]
                
                for img_name in os.listdir(modal_path):
                    if img_name.endswith(('.jpg', '.png', '.bmp')):
                        img_path = os.path.join(modal_path, img_name)
                        self.samples.append((img_path, class_id_int))  # 保存原始类别ID
        
        # 使用全局标签映射（从晴天数据集获取的完整映射）
        if global_label_map is not None:
            self.label_map = global_label_map
            self.num_classes = len(self.label_map)
        else:
            self.num_classes = len(self.label_map)
        
        print(f"[{modality_dir}] 数据集加载完成: {len(self.samples)}张图片, {self.num_classes}个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, original_label = self.samples[idx]
        
        # 使用全局标签映射获取标签ID
        label = self.label_map.get(original_label, 0)
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            return torch.randn(3, 224, 224), label


class AISDataset(Dataset):
    """AIS信号数据集"""
    def __init__(self, data_root, phase='test', global_label_map=None):
        self.data_root = data_root
        self.phase = phase
        self.samples = []
        self.label_map = {}
        self.num_classes = 0
        
        self._load_data(global_label_map)
        
    def _load_data(self, global_label_map=None):
        ais_file = os.path.join(self.data_root, 'AIS', 'balanced_AIS-dataset_16classes_100persample.mat')
        
        if not os.path.exists(ais_file):
            print(f"AIS数据文件不存在: {ais_file}")
            return
        
        try:
            import scipy.io as sio
            mat_data = sio.loadmat(ais_file)
            
            data = mat_data['dataset']
            labels = mat_data['labels'].flatten()
            
            n_samples = data.shape[0]
            train_ratio = 0.8
            train_size = int(n_samples * train_ratio)
            
            if self.phase == 'train':
                data = data[:train_size]
                labels = labels[:train_size]
            else:
                data = data[train_size:]
                labels = labels[train_size:]
            
            for i in range(data.shape[0]):
                sample = data[i].astype(np.float32)
                label = int(labels[i])
                
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                mapped_label = self.label_map[label]
                
                self.samples.append((sample, mapped_label))
            
            # 使用全局标签映射
            if global_label_map is not None:
                self.label_map = global_label_map
                self.num_classes = len(self.label_map)
            else:
                self.num_classes = len(self.label_map)
            
            print(f"[AIS] 数据集加载完成: {len(self.samples)}个样本, {self.num_classes}个类别")
            
        except Exception as e:
            print(f"加载AIS数据失败: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        return torch.from_numpy(sample), label


def get_transforms(phase='test', modality='vis'):
    """获取数据变换（参考MambaOut-main的validate.py）"""
    if modality == 'ais':
        return None
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


def create_model_by_name(model_name, num_classes, pretrained=True, modality='vis'):
    """使用timm库创建模型（参考MambaOut-main的validate.py）"""
    model_mapping = {
        'resnet': 'resnet50',
        'convnext': 'convnext_tiny',
        'vit': 'vit_base_patch16_224'
    }
    
    timm_name = model_mapping.get(model_name, model_name)
    
    model = create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=MODALITIES[modality]['channels']
    )
    
    return model


def test_model(model, loader, device):
    """测试模型（参考MambaOut-main的predict.py）"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    accuracy = correct / total
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def setup_logger(log_dir, model_name, modality, weather):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{modality}_{weather}_{timestamp}.log")
    
    logger = logging.getLogger(f"{model_name}_{modality}_{weather}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    data_root = r'D:\Code\JMDA-Net\Data'
    log_dir = r'D:\Code\JMDA-Net\experiment_logs_mambaout'
    os.makedirs(log_dir, exist_ok=True)
    
    models_list = ['resnet', 'convnext', 'vit']
    modalities_list = ['vis', 'ir', 'ais']
    weather_list = ['雨天', '逆光', '黑天', '雾天']
    test_iterations = 5
    
    # ========== 关键：从晴天数据集获取global_label_map ==========
    # 确保所有数据集使用相同的标签映射，标签ID与模型输出类别数一致
    sunny_root = os.path.join(data_root, '晴天')
    
    global_label_map = None
    num_classes = None
    
    # 首先尝试加载晴天可见光数据集获取标签映射
    try:
        sunny_train_ds = WeatherDataset(sunny_root, modality='vis', phase='train',
                                        transform=None, global_label_map=None)
        global_label_map = sunny_train_ds.label_map
        num_classes = sunny_train_ds.num_classes
        print(f"\n从晴天可见光数据集获取标签映射:")
        print(f"  标签映射: {global_label_map}")
        print(f"  类别数量: {num_classes}")
    except Exception as e:
        print(f"加载晴天可见光数据集失败: {e}")
    
    all_results = {}
    
    for modality in modalities_list:
        print(f"\n{'='*80}")
        print(f"测试模态: {MODALITIES[modality]['name']}")
        print(f"{'='*80}")
        
        modality_results = {}
        
        if modality == 'ais':
            weather_list_ais = ['AIS']
        else:
            weather_list_ais = weather_list
        
        for weather in weather_list_ais:
            print(f"\n{'='*60}")
            print(f"测试数据: {weather}")
            print(f"{'='*60}")
            
            weather_results = {}
            
            if modality == 'ais':
                test_ds = AISDataset(data_root, phase='test', global_label_map=global_label_map)
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
                actual_num_classes = num_classes if num_classes else test_ds.num_classes
            else:
                weather_root = os.path.join(data_root, weather)
                if not os.path.exists(weather_root):
                    print(f"跳过: {weather} 数据不存在")
                    continue
                
                try:
                    test_ds = WeatherDataset(weather_root, modality=modality, phase='val',
                                              transform=get_transforms('test', modality),
                                              global_label_map=global_label_map)
                    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
                    actual_num_classes = num_classes if num_classes else test_ds.num_classes
                except Exception as e:
                    print(f"数据加载失败: {e}")
                    continue
            
            print(f"测试集: {len(test_ds)}个样本")
            print(f"使用类别数量: {actual_num_classes}")
            print(f"使用全局标签映射: {global_label_map}")
            
            for model_name in models_list:
                print(f"\n--- 测试模型: {model_name} ---")
                
                model_metrics = []
                
                for iter_idx in range(test_iterations):
                    seed = 42 + iter_idx
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    
                    print(f"  迭代 {iter_idx+1}/{test_iterations}, 随机种子: {seed}")
                    
                    if modality == 'ais':
                        model = create_model_by_name(model_name, actual_num_classes, 
                                                     pretrained=False, modality=modality)
                        # 调整最后一层全连接层
                        if hasattr(model, 'fc'):
                            model.fc = nn.Linear(model.fc.in_features, actual_num_classes)
                    else:
                        model = create_model_by_name(model_name, actual_num_classes, 
                                                     pretrained=True, modality=modality)
                    
                    model = model.to(device)
                    
                    metrics = test_model(model, test_loader, device)
                    model_metrics.append(metrics)
                    
                    print(f"    Accuracy: {metrics['accuracy']:.4f}")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall: {metrics['recall']:.4f}")
                    print(f"    F1: {metrics['f1']:.4f}")
                
                stats = {}
                for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    values = [m[metric_name] for m in model_metrics]
                    stats[metric_name] = {
                        'values': values,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'max': np.max(values),
                        'min': np.min(values)
                    }
                
                weather_results[model_name] = stats
            
            modality_results[weather] = weather_results
        
        all_results[modality] = modality_results
    
    report_path = os.path.join(log_dir, "test_summary.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("测试完成!")
    print(f"汇总报告已保存到: {report_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()