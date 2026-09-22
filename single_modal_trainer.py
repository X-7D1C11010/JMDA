import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, ConvNeXt_Small_Weights
from torchvision import transforms
import numpy as np
import time
import random
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class SingleModalDataset(Dataset):
    def __init__(self, root_dir, modality='vis', phase='train', weather=None, global_label_map=None, 
                 structure_type='weather'):
        self.modality = modality
        self.phase = phase
        self.samples = []
        self.weather = weather if weather else os.path.basename(root_dir)
        self.ais_input_dim = 128
        self.structure_type = structure_type
        
        if modality in ['vis', 'ir']:
            if structure_type == 'weather':
                phase_path = os.path.join(root_dir, phase)
                if not os.path.exists(phase_path):
                    phase_path = root_dir
                self._load_weather_image_data(phase_path)
            else:
                self._load_sunny_image_data(root_dir, phase)
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
    
    def _load_weather_image_data(self, phase_path):
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
    
    def _load_sunny_image_data(self, root_dir, phase):
        phase_path = os.path.join(root_dir, phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"找不到路径: {phase_path}")
        
        class_dirs = [d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))]
        
        for class_name in class_dirs:
            class_path = os.path.join(phase_path, class_name)
            modality_dir = os.path.join(class_path, '可见光' if self.modality == 'vis' else '红外')
            
            if not os.path.exists(modality_dir):
                modality_dir = class_path
            
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
            ais_path = os.path.join(root_dir, 'balanced_AIS-dataset_16classes_100persample.mat')
        
        if not os.path.exists(ais_path):
            raise FileNotFoundError(f"找不到AIS数据文件: {ais_path}")
        
        if not HAS_H5PY:
            raise ImportError("需要安装h5py来读取matlab v7.3格式文件，请运行: pip install h5py")
        
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
            
            if rcv_i_key is None:
                rcv_i_key = keys[0] if len(keys) > 0 else None
            if rcv_q_key is None:
                rcv_q_key = keys[1] if len(keys) > 1 else None
            if label_key is None:
                label_key = keys[2] if len(keys) > 2 else None
            
            if rcv_i_key is None or rcv_q_key is None or label_key is None:
                raise ValueError(f"无法找到所需的键，文件中的键: {keys}")
            
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
            if self.phase == 'train':
                return transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        elif self.modality == 'ir':
            if self.phase == 'train':
                return transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])
            else:
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
                print(f"警告: 无法加载图片 {sample['path']}, 错误: {e}")
                return self.__getitem__((idx + 1) % len(self))
        else:
            data = torch.tensor(sample['feature'], dtype=torch.float32)
        
        label_id = self.label_map.get(sample['label'], 0)
        
        return {
            'data': data,
            'label': torch.tensor(label_id, dtype=torch.long)
        }


class AISFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3144, output_dim=512):
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


def create_model(model_name, modality, num_classes, ais_input_dim=3144):
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


def evaluate_with_confusion(model, dataloader, device, num_classes):
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
        cm = confusion_matrix(all_labels, all_predicted, labels=np.arange(num_classes))
    else:
        precision_macro = 0.0
        precision_micro = 0.0
        recall_macro = 0.0
        recall_micro = 0.0
        f1_macro = 0.0
        f1_micro = 0.0
        cm = np.zeros((num_classes, num_classes))
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm
    }
    
    return metrics


def train_on_sunny_test_on_weather(model_name, modality, data_root, num_iterations=5, epochs=5, batch_size=32,
                                     target_acc_min=0.0, target_acc_max=1.0):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sunny_root = os.path.join(data_root, '晴天')
    
    if not os.path.exists(sunny_root):
        print(f"警告: 晴天数据集路径不存在: {sunny_root}")
        return None
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    model_save_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"sunny_train_{model_name}_{modality}_{timestamp}.log")
    
    logger = logging.getLogger(f'sunny_train_{model_name}_{modality}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info(f"训练配置:")
    logger.info(f"  模型: {model_name}")
    logger.info(f"  模态: {modality}")
    logger.info(f"  设备: {DEVICE}")
    logger.info(f"  迭代次数: {num_iterations}")
    logger.info(f"  每迭代轮数: {epochs}")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  训练集: {sunny_root}")
    logger.info("=" * 80)
    
    try:
        train_ds = SingleModalDataset(sunny_root, modality=modality, phase='train', 
                                     structure_type='sunny')
    except FileNotFoundError as e:
        logger.info(f"训练集加载失败: {e}")
        return None
    
    try:
        val_ds = SingleModalDataset(sunny_root, modality=modality, phase='val', 
                                   global_label_map=train_ds.label_map, 
                                   structure_type='sunny')
    except FileNotFoundError as e:
        logger.info(f"验证集加载失败: {e}")
        return None
    
    logger.info(f"晴天训练集大小: {len(train_ds)}")
    logger.info(f"晴天验证集大小: {len(val_ds)}")
    logger.info(f"类别数量: {train_ds.num_classes}")
    logger.info(f"类别映射: {train_ds.label_map}")
    if modality == 'ais':
        logger.info(f"AIS输入维度: {train_ds.ais_input_dim}")
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.info("数据集为空，跳过此训练")
        return None
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    
    best_models = []
    
    for iteration in range(num_iterations):
        iter_start_time = time.time()
        
        seed = 42 + iteration
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"迭代 {iteration + 1}/{num_iterations}")
        logger.info(f"随机种子: {seed}")
        logger.info(f"{'='*80}")
        
        if modality == 'ais':
            model = create_model(model_name, modality, train_ds.num_classes, train_ds.ais_input_dim).to(DEVICE)
        else:
            model = create_model(model_name, modality, train_ds.num_classes).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            
            loss_accum = 0.0
            train_correct = 0
            train_total = 0
            
            for data in train_loader:
                inputs = data['data'].to(DEVICE)
                labels = data['label'].to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                loss_accum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            val_metrics = evaluate_with_confusion(model, val_loader, DEVICE, train_ds.num_classes)
            val_acc = val_metrics['accuracy']
            
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            avg_loss = loss_accum / len(train_loader) if len(train_loader) > 0 else 0.0
            epoch_time = time.time() - epoch_start_time
            
            log_msg = (f"Epoch [{epoch + 1}/{epochs}] | "
                       f"Loss: {avg_loss:.4f} | "
                       f"Train Acc: {train_acc:.4f} | "
                       f"Val Acc: {val_acc:.4f} | "
                       f"Val F1 (Macro): {val_metrics['f1_macro']:.4f} | "
                       f"LR: {current_lr:.6f} | "
                       f"Time: {epoch_time:.2f}s")
            
            logger.info(log_msg)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
        
        iter_time = time.time() - iter_start_time
        
        logger.info(f"\n{'='*80}")
        logger.info(f"迭代 {iteration + 1}/{num_iterations} 训练完成")
        logger.info(f"最佳验证准确率: {best_val_acc:.4f}")
        logger.info(f"迭代耗时: {iter_time:.2f}秒")
        
        if best_model_state is not None:
            best_models.append(best_model_state)
            
            model_save_path = os.path.join(model_save_dir, f"{model_name}_{modality}_iter{iteration+1}.pth")
            torch.save(best_model_state, model_save_path)
            logger.info(f"  >>> 最佳模型已保存到: {model_save_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("所有迭代训练完成，开始从文件加载模型在晴天验证集上测试")
    logger.info("=" * 80)
    
    saved_model_paths = [os.path.join(model_save_dir, f"{model_name}_{modality}_iter{i+1}.pth") 
                         for i in range(num_iterations)]
    
    val_results = []
    
    for iter_idx, model_path in enumerate(saved_model_paths):
        if not os.path.exists(model_path):
            logger.info(f"  跳过: 模型文件不存在 {model_path}")
            continue
        
        if modality == 'ais':
            model = create_model(model_name, modality, train_ds.num_classes, train_ds.ais_input_dim).to(DEVICE)
        else:
            model = create_model(model_name, modality, train_ds.num_classes).to(DEVICE)
        
        model.load_state_dict(torch.load(model_path))
        logger.info(f"  已从文件加载模型: {model_path}")
        
        test_metrics = evaluate_with_confusion(model, val_loader, DEVICE, train_ds.num_classes)
        
        val_results.append(test_metrics)
        
        logger.info(f"  迭代 {iter_idx + 1} 验证集测试结果:")
        logger.info(f"    Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"    Precision (Macro): {test_metrics['precision_macro']:.4f}")
        logger.info(f"    Recall (Macro): {test_metrics['recall_macro']:.4f}")
        logger.info(f"    F1 (Macro): {test_metrics['f1_macro']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("晴天验证集统计结果")
    logger.info("=" * 80)
    
    metrics_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                     'precision_micro', 'recall_micro', 'f1_micro']
    
    for metric_name in metrics_names:
        values = np.array([r[metric_name] for r in val_results])
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        logger.info(f"{metric_name}:")
        logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
        logger.info(f"  均值: {mean_val:.4f}")
        logger.info(f"  标准差: {std_val:.4f}")
    
    logger.info(f"\n训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return val_results


def main():
    models_list = ['resnet', 'vit', 'convnext']
    modalities = ['vis', 'ir', 'ais']
    data_root = r'D:\Code\JMDA-Net\Data'
    
    for model_name in models_list:
        for modality in modalities:
            print(f"\n{'='*60}")
            print(f"开始训练: {model_name} | {modality}")
            print(f"训练集: 晴天/train")
            print(f"测试集: 晴天/val")
            print(f"训练轮数: 10轮/迭代")
            print(f"迭代次数: 5次")
            print(f"{'='*60}")
            
            try:
                train_on_sunny_test_on_weather(model_name, modality, data_root, 
                                               num_iterations=5, epochs=3, batch_size=32)
            except Exception as e:
                print(f"训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()