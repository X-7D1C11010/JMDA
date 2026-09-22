import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, ConvNeXt_Small_Weights
from torchvision import transforms
import numpy as np
import time
import random
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class WeatherDataset(Dataset):
    def __init__(self, root_dir, modality='vis', phase='val', global_label_map=None):
        self.modality = modality
        self.phase = phase
        self.samples = []
        self.ais_input_dim = 128
        
        phase_path = os.path.join(root_dir, phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"找不到路径: {phase_path}")
        
        if modality in ['vis', 'ir']:
            self._load_image_data(phase_path)
        elif modality == 'ais':
            self._load_ais_data(root_dir)
        else:
            raise ValueError(f"未知模态: {modality}")
        
        self.labels = [s['label'] for s in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))
        
        if global_label_map:
            self.label_map = global_label_map
        else:
            self.label_map = {orig: idx for idx, orig in enumerate(self.unique_labels)}
        
        self.num_classes = len(self.label_map)
        self.transform = self._get_transform()
    
    def _load_image_data(self, phase_path):
        class_dirs = [d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))]
        
        for class_name in class_dirs:
            class_path = os.path.join(phase_path, class_name)
            modality_dir = os.path.join(class_path, '可见光' if self.modality == 'vis' else '红外')
            
            if not os.path.exists(modality_dir):
                continue
            
            files = sorted([f for f in os.listdir(modality_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for f in files:
                self.samples.append({
                    'path': os.path.join(modality_dir, f),
                    'label': int(class_name)
                })
        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"找不到模态目录或目录为空: {phase_path}")
    
    def _load_ais_data(self, root_dir):
        ais_path = os.path.join(root_dir, 'AIS', 'balanced_AIS-dataset_16classes_100persample.mat')
        if not os.path.exists(ais_path):
            ais_path = os.path.join(os.path.dirname(root_dir), 'AIS', 'balanced_AIS-dataset_16classes_100persample.mat')
        
        if not os.path.exists(ais_path):
            raise FileNotFoundError(f"找不到AIS数据文件: {ais_path}")
        
        if not HAS_H5PY:
            raise ImportError("需要安装h5py来读取matlab v7.3格式文件")
        
        with h5py.File(ais_path, 'r') as f:
            keys = list(f.keys())
            
            rcv_i_key = None
            rcv_q_key = None
            label_key = None
            
            for key in keys:
                if 'rcv_i' in key.lower():
                    rcv_i_key = key
                elif 'rcv_q' in key.lower():
                    rcv_q_key = key
                elif 'label' in key.lower():
                    label_key = key
            
            rcv_i = np.array(f[rcv_i_key])
            rcv_q = np.array(f[rcv_q_key])
            labels = np.array(f[label_key])
            
            if len(labels.shape) == 2 and labels.shape[1] == 1:
                labels = labels.flatten()
            
            num_samples = len(labels)
            self.ais_input_dim = rcv_i.shape[0] * 2
        
        for i in range(num_samples):
            if rcv_i.shape[1] == num_samples:
                i_feature = rcv_i[:, i]
                q_feature = rcv_q[:, i]
            else:
                i_feature = rcv_i[i]
                q_feature = rcv_q[i]
            
            combined_feature = np.concatenate([i_feature, q_feature])
            self.samples.append({
                'feature': combined_feature,
                'label': int(labels[i])
            })
    
    def _get_transform(self):
        if self.modality == 'vis':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.modality == 'ir':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.modality in ['vis', 'ir']:
            try:
                if self.modality == 'vis':
                    img = Image.open(sample['path']).convert('RGB')
                else:
                    img = Image.open(sample['path']).convert('L').convert('RGB')
                data = self.transform(img)
            except Exception as e:
                print(f"警告: 无法加载图片 {sample['path']}")
                return self.__getitem__((idx + 1) % len(self))
        else:
            data = torch.tensor(sample['feature'], dtype=torch.float32)
        
        label_id = self.label_map.get(sample['label'], 0)
        
        return {
            'data': data,
            'label': torch.tensor(label_id, dtype=torch.long)
        }


class AISFeatureExtractor(nn.Module):
    def __init__(self, input_dim=23040, output_dim=512):
        super(AISFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, x):
        return self.features(x)


def create_model(model_name, modality, num_classes, ais_input_dim=23040):
    if modality == 'ais':
        backbone = AISFeatureExtractor(input_dim=ais_input_dim, output_dim=512)
        classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        model = nn.Sequential(backbone, classifier)
        return model
    
    if model_name == 'resnet':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vit':
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'convnext':
        model = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    return model


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs = data['data'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0.0
    
    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) > 1:
        precision_macro = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
        precision_micro = precision_score(all_labels, all_predicted, average='micro', zero_division=0)
        recall_macro = recall_score(all_labels, all_predicted, average='macro', zero_division=0)
        recall_micro = recall_score(all_labels, all_predicted, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_predicted, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_predicted, average='micro', zero_division=0)
    else:
        precision_macro = 0.0
        precision_micro = 0.0
        recall_macro = 0.0
        recall_micro = 0.0
        f1_macro = 0.0
        f1_micro = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }
    
    return metrics


def test_model_on_weathers(model_name, modality, data_root, num_iterations=5):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"test_{model_name}_{modality}_{timestamp}.log")
    
    logger = logging.getLogger(f'test_{model_name}_{modality}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info(f"测试配置:")
    logger.info(f"  模型: {model_name}")
    logger.info(f"  模态: {modality}")
    logger.info(f"  设备: {DEVICE}")
    logger.info(f"  测试迭代次数: {num_iterations}")
    logger.info("=" * 80)
    
    weathers = ['黑天', '逆光', '雾天', '雨天']
    weather_datasets = {}
    all_labels = set()
    
    for weather in weathers:
        weather_path = os.path.join(data_root, weather)
        if os.path.exists(weather_path):
            try:
                ds = WeatherDataset(weather_path, modality=modality, phase='val')
                loader = DataLoader(ds, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
                weather_datasets[weather] = {'dataset': ds, 'loader': loader}
                all_labels.update(ds.unique_labels)
                logger.info(f"{weather}测试集大小: {len(ds)}")
            except FileNotFoundError as e:
                logger.info(f"{weather}数据集加载失败: {e}")
    
    if len(weather_datasets) == 0:
        logger.info("没有找到任何天气数据集")
        return None
    
    global_label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
    num_classes = len(global_label_map)
    
    logger.info(f"总类别数量: {num_classes}")
    logger.info(f"类别映射: {global_label_map}")
    
    weather_results = {weather: [] for weather in weathers}
    
    for iteration in range(num_iterations):
        seed = 42 + iteration
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"测试迭代 {iteration + 1}/{num_iterations}")
        logger.info(f"随机种子: {seed}")
        logger.info(f"{'='*80}")
        
        ais_input_dim = 0
        for weather, data in weather_datasets.items():
            if modality == 'ais' and hasattr(data['dataset'], 'ais_input_dim'):
                ais_input_dim = data['dataset'].ais_input_dim
                break
        
        if modality == 'ais' and ais_input_dim > 0:
            model = create_model(model_name, modality, num_classes, ais_input_dim).to(DEVICE)
        else:
            model = create_model(model_name, modality, num_classes).to(DEVICE)
        
        for weather, data in weather_datasets.items():
            test_metrics = evaluate(model, data['loader'], DEVICE, num_classes)
            weather_results[weather].append(test_metrics)
            
            logger.info(f"{weather}测试结果:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Precision (Macro): {test_metrics['precision_macro']:.4f}")
            logger.info(f"  Recall (Macro): {test_metrics['recall_macro']:.4f}")
            logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("各天气数据集统计结果")
    logger.info("=" * 80)
    
    metrics_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                     'precision_micro', 'recall_micro', 'f1_micro']
    
    for weather, results in weather_results.items():
        if len(results) == 0:
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"天气: {weather}")
        logger.info(f"{'='*60}")
        
        for metric_name in metrics_names:
            values = np.array([r[metric_name] for r in results])
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            logger.info(f"{metric_name}:")
            logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
            logger.info(f"  均值: {mean_val:.4f}")
            logger.info(f"  标准差: {std_val:.4f}")
    
    logger.info(f"\n测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return weather_results


def main():
    models_list = ['resnet', 'vit', 'convnext']
    modalities = ['vis', 'ir', 'ais']
    data_root = r'D:\Code\JMDA-Net\Data'
    
    for model_name in models_list:
        for modality in modalities:
            print(f"\n{'='*60}")
            print(f"开始测试: {model_name} | {modality}")
            print(f"测试集: 黑天、逆光、雾天、雨天")
            print(f"{'='*60}")
            
            try:
                test_model_on_weathers(model_name, modality, data_root, num_iterations=5)
            except Exception as e:
                print(f"测试失败: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()