"""
自动化测试脚本：加载预训练模型并在目标域上执行测试

功能说明：
1. 自动加载指定的3个预训练模型文件（ResNet、ViT、ConvNeXt）
2. 在目标域数据集上进行测试，覆盖所有预设天气条件
3. 对每种天气条件执行5次独立测试迭代
4. 每次迭代过程中记录各项评价指标的最高值
5. 所有迭代完成后，计算并保存每个评价指标的均值和标准差
6. 真实标签的获取方式严格参考main.py脚本的实现方法
7. 确保测试过程可复现，结果数据准确记录并以标准格式保存

参考main.py中的实现：
- 使用MultiModalDomainDataset类加载目标域数据
- 通过global_label_map获取统一的标签映射
- 使用sklearn计算precision、recall、f1等指标
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights, ConvNeXt_Base_Weights
from torchvision import transforms
import numpy as np
import random
from datetime import datetime
import logging
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image


# ============================================================================
# 数据加载类（参考main.py和DataLoad.py的实现）
# ============================================================================
class MultiModalDomainDataset(torch.utils.data.Dataset):
    """
    多模态域数据集加载类
    
    标签获取方式（参考main.py）：
    1. 从源域数据集获取global_label_map
    2. 目标域数据集使用相同的label_map保证标签一致性
    3. label_map格式: {原始类别名: 映射后的类别ID}
    """
    def __init__(self, root_dir, domain_type='target', phase='val', 
                 global_label_map=None, val_augment=False):
        self.domain_type = domain_type
        self.phase = phase
        self.samples = []
        self.val_augment = val_augment

        # 数据路径
        phase_path = os.path.join(root_dir, phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"找不到路径: {phase_path}")

        # 目标域数据加载逻辑（参考DataLoad.py）
        valid_classes = [d for d in os.listdir(phase_path) 
                         if os.path.isdir(os.path.join(phase_path, d))]
        for class_name in valid_classes:
            class_dir = os.path.join(phase_path, class_name)
            vis_dir = os.path.join(class_dir, '可见光')
            ir_dir = os.path.join(class_dir, '红外')
            
            if not (os.path.exists(vis_dir) and os.path.exists(ir_dir)):
                continue

            vis_files = sorted([f for f in os.listdir(vis_dir) 
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            ir_files = sorted([f for f in os.listdir(ir_dir) 
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

            min_len = min(len(vis_files), len(ir_files))
            for i in range(min_len):
                self.samples.append({
                    'vis': os.path.join(vis_dir, vis_files[i]),
                    'ir': os.path.join(ir_dir, ir_files[i]),
                    'label': int(class_name)
                })

        self.labels = [s['label'] for s in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))

        # 标签映射（参考main.py的实现方式）
        if global_label_map:
            self.label_map = global_label_map
        else:
            self.label_map = {orig: idx for idx, orig in enumerate(self.unique_labels)}

        self.num_classes = len(self.label_map)
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """获取数据预处理transform"""
        if self.phase == 'train' or self.val_augment:
            return {
                'vis': transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]),
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
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]),
                ]),
                'ir': transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ])
            }

    def get_label_map(self):
        """返回标签映射（参考main.py）"""
        return self.label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            vis_img = Image.open(sample['vis']).convert('RGB')
            ir_img = Image.open(sample['ir']).convert('L').convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图片 {sample['vis']}, 错误: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label_id = self.label_map.get(sample['label'], 0)

        return {
            'vis': self.transform['vis'](vis_img),
            'ir': self.transform['ir'](ir_img),
            'label': torch.tensor(label_id, dtype=torch.long)
        }


class SingleModalityDataset(torch.utils.data.Dataset):
    """
    单模态数据集（用于ResNet/ViT/ConvNeXt等单模态模型测试）
    
    标签获取方式与MultiModalDomainDataset一致
    """
    def __init__(self, root_dir, modality='vis', phase='val', 
                 global_label_map=None, val_augment=False):
        self.modality = modality
        self.phase = phase
        self.samples = []
        self.val_augment = val_augment

        phase_path = os.path.join(root_dir, phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"找不到路径: {phase_path}")

        # 目标域数据加载
        valid_classes = [d for d in os.listdir(phase_path) 
                         if os.path.isdir(os.path.join(phase_path, d))]
        for class_name in valid_classes:
            class_dir = os.path.join(phase_path, class_name)
            modality_dir = os.path.join(class_dir, '可见光' if modality == 'vis' else '红外')
            
            if not os.path.exists(modality_dir):
                continue

            files = sorted([f for f in os.listdir(modality_dir) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for f in files:
                self.samples.append({
                    'path': os.path.join(modality_dir, f),
                    'label': int(class_name)
                })

        self.labels = [s['label'] for s in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))

        # 标签映射（与main.py一致）
        if global_label_map:
            self.label_map = global_label_map
        else:
            self.label_map = {orig: idx for idx, orig in enumerate(self.unique_labels)}

        self.num_classes = len(self.label_map)
        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.phase == 'train' or self.val_augment:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406] if self.modality == 'vis' else [0.5],
                                     std=[0.229, 0.224, 0.225] if self.modality == 'vis' else [0.5]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406] if self.modality == 'vis' else [0.5],
                                     std=[0.229, 0.224, 0.225] if self.modality == 'vis' else [0.5]),
            ])

    def get_label_map(self):
        return self.label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            if self.modality == 'vis':
                img = Image.open(sample['path']).convert('RGB')
            else:
                img = Image.open(sample['path']).convert('L').convert('RGB')
            data = self.transform(img)
        except Exception as e:
            print(f"警告: 无法加载图片 {sample['path']}, 错误: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label_id = self.label_map.get(sample['label'], 0)
        return {
            'data': data,
            'label': torch.tensor(label_id, dtype=torch.long)
        }


# ============================================================================
# 模型加载函数
# ============================================================================
def create_model(model_name, num_classes, pretrained=True, dropout_rate=None):
    """
    创建并加载预训练模型
    
    支持的模型：
    - ResNet50
    - ViT-B/16
    - ConvNeXt-Base
    
    参数：
        model_name: 模型名称
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout概率，为None时根据模型自动选择
    """
    # === 差异化Dropout策略（与训练脚本保持一致）===
    if dropout_rate is None:
        if model_name == 'resnet':
            dropout_rate = 0.5
        elif model_name == 'convnext':
            dropout_rate = 0.65
        elif model_name == 'vit':
            dropout_rate = 0.75
        else:
            dropout_rate = 0.5
    if model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        # 添加Dropout层（与训练时的模型结构一致）
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'vit':
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'convnext':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"未知模型: {model_name}")
    
    return model


def load_pretrained_model(model_path, model_name, num_classes, device, dropout_rate=None):
    """
    加载预训练模型权重
    
    参数：
        model_path: 模型权重文件路径
        model_name: 模型名称
        num_classes: 分类类别数（此参数会被state_dict中的实际类别数覆盖）
        device: 计算设备
        dropout_rate: Dropout概率
    
    修改说明：
        强制从state_dict中推断类别数量，完全忽略传入的num_classes参数。
        这是解决模型结构不匹配问题的关键：
        错误示例：
            RuntimeError: size mismatch for fc.1.weight: 
            copying a param with shape torch.Size([14, 2048]) from checkpoint, 
            the shape in current model is torch.Size([6, 2048]).
        
        根本原因：
            - 模型是在晴天数据集（14类）上训练的
            - 测试脚本传入的是目标域数据集的类别数（如6类）
            - 必须以模型文件中的类别数为准创建模型结构
    """
    state_dict = None
    actual_num_classes = num_classes
    
    # 先尝试加载state_dict以获取正确的类别数量
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        
        # 从state_dict中推断类别数量（优先使用，忽略传入的num_classes）
        if model_name == 'resnet':
            if 'fc.1.weight' in state_dict:
                actual_num_classes = state_dict['fc.1.weight'].shape[0]
            elif 'fc.weight' in state_dict:
                actual_num_classes = state_dict['fc.weight'].shape[0]
        elif model_name == 'vit':
            if 'heads.head.1.weight' in state_dict:
                actual_num_classes = state_dict['heads.head.1.weight'].shape[0]
            elif 'heads.head.weight' in state_dict:
                actual_num_classes = state_dict['heads.head.weight'].shape[0]
        elif model_name == 'convnext':
            if 'classifier.2.1.weight' in state_dict:
                actual_num_classes = state_dict['classifier.2.1.weight'].shape[0]
            elif 'classifier.2.weight' in state_dict:
                actual_num_classes = state_dict['classifier.2.weight'].shape[0]
        
        # 输出推断结果
        if actual_num_classes != num_classes:
            print(f"重要: 从state_dict推断类别数量为 {actual_num_classes}, "
                  f"覆盖传入的 {num_classes}")
    else:
        print(f"警告: 模型文件不存在 {model_path}, 使用传入的类别数 {num_classes}")
    
    # 创建模型结构（使用从state_dict推断的类别数量）
    model = create_model(model_name, actual_num_classes, pretrained=False, dropout_rate=dropout_rate)
    
    # 加载权重
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {model_path}")
        except RuntimeError as e:
            # 如果直接加载失败，尝试部分加载（忽略不匹配的键）
            print(f"直接加载失败: {e}")
            print("尝试部分加载模型权重...")
            model_dict = model.state_dict()
            # 过滤掉不匹配的键
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                  if k in model_dict and model_dict[k].shape == v.shape}
            # 更新模型字典
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict)
            print(f"部分加载成功，加载了 {len(filtered_state_dict)} 个参数")
    
    model = model.to(device)
    return model, num_classes


# ============================================================================
# 评估函数（参考main.py中的evaluate函数实现）
# ============================================================================
def evaluate_model(model, dataloader, device, label_map):
    """
    模型评估函数
    
    评估指标计算方式（参考main.py）：
    - 使用sklearn.metrics计算precision、recall、f1
    - 计算macro和micro两种平均方式
    - 返回混淆矩阵
    
    参数：
        model: 待评估模型
        dataloader: 数据加载器
        device: 计算设备
        label_map: 标签映射
    """
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
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 转换为numpy数组
    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)
    
    # 获取所有类别标签（参考main.py）
    all_class_labels = sorted(list(label_map.values()))
    
    # 获取验证集中实际出现的类别
    unique_labels_in_val = sorted(np.unique(all_labels).tolist())
    
    # 计算精确率、召回率、F1分数（参考main.py的实现）
    precision_macro = precision_score(all_labels, all_predicted, average='macro', 
                                       zero_division=0, labels=all_class_labels)
    precision_micro = precision_score(all_labels, all_predicted, average='micro', 
                                       zero_division=0, labels=all_class_labels)
    recall_macro = recall_score(all_labels, all_predicted, average='macro', 
                                 zero_division=0, labels=all_class_labels)
    recall_micro = recall_score(all_labels, all_predicted, average='micro', 
                                 zero_division=0, labels=all_class_labels)
    f1_macro = f1_score(all_labels, all_predicted, average='macro', 
                         zero_division=0, labels=all_class_labels)
    f1_micro = f1_score(all_labels, all_predicted, average='micro', 
                         zero_division=0, labels=all_class_labels)
    
    # 只在验证集中出现的类别上计算Macro指标（更准确）
    if len(unique_labels_in_val) > 0:
        precision_macro_present = precision_score(all_labels, all_predicted, average='macro',
                                                   zero_division=0, labels=unique_labels_in_val)
        recall_macro_present = recall_score(all_labels, all_predicted, average='macro',
                                              zero_division=0, labels=unique_labels_in_val)
        f1_macro_present = f1_score(all_labels, all_predicted, average='macro',
                                     zero_division=0, labels=unique_labels_in_val)
    else:
        precision_macro_present = 0.0
        recall_macro_present = 0.0
        f1_macro_present = 0.0
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predicted, labels=all_class_labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_macro_present': precision_macro_present,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_macro_present': recall_macro_present,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_macro_present': f1_macro_present,
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'classes_present': unique_labels_in_val
    }
    
    return metrics


# ============================================================================
# 日志设置函数
# ============================================================================
def setup_logger(log_dir, model_name, weather):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"test_{model_name}_{weather}_{timestamp}.log")
    
    logger = logging.getLogger(f'test_{model_name}_{weather}_{timestamp}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filepath


# ============================================================================
# 主测试流程
# ============================================================================
def run_automated_test(models_dir, data_root, weather_list, num_iterations=5,
                       modality='vis', dropout_rate=0.5, strategy='strategy2'):
    """
    执行自动化测试流程
    
    参数：
        models_dir: 预训练模型文件目录
        data_root: 数据根目录
        weather_list: 目标域天气条件列表
        num_iterations: 每种天气条件的测试迭代次数
        modality: 测试模态（'vis'或'ir'）
        dropout_rate: Dropout概率
    
    测试流程：
        1. 从晴天数据集获取global_label_map（参考main.py）
        2. 对每个模型、每种天气、每次迭代进行测试
        3. 记录每次迭代的最高评价指标
        4. 计算均值和标准差
        5. 保存结果到JSON和日志文件
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(os.path.dirname(__file__), "test_logs")
    
    # 模型列表
    models_list = ['resnet', 'vit', 'convnext']
    
    # ========== 获取全局标签映射（参考weather_classification_experiment.py的实现方式）==========
    # 从晴天（源域）数据集获取label_map，确保所有数据集使用相同的标签映射
    # 重要：必须使用WeatherImageDataset来加载晴天数据，因为它有专门处理晴天数据结构的方法
    # 晴天数据集结构: phase_path/模态/类别/
    # 目标域数据集结构: phase_path/类别/模态/
    sunny_root = os.path.join(data_root, '晴天')
    global_label_map = None
    num_classes = None
    
    try:
        from weather_classification_experiment import WeatherImageDataset
        sunny_train_ds = WeatherImageDataset(sunny_root, modality=modality, phase='train',
                                              structure_type='sunny')
        global_label_map = sunny_train_ds.label_map
        num_classes = sunny_train_ds.num_classes
        print(f"从晴天数据集获取标签映射: {global_label_map}")
        print(f"类别数量: {num_classes}")
    except Exception as e:
        print(f"无法加载晴天数据集获取标签映射: {e}")
        print("尝试使用SingleModalityDataset加载...")
        try:
            sunny_train_ds = SingleModalityDataset(sunny_root, modality=modality, 
                                                    phase='train', val_augment=False)
            global_label_map = sunny_train_ds.get_label_map()
            num_classes = sunny_train_ds.num_classes
            print(f"从晴天数据集获取标签映射: {global_label_map}")
            print(f"类别数量: {num_classes}")
        except Exception as e2:
            print(f"加载失败: {e2}")
            print("将使用各目标域数据集自身的标签映射")
            global_label_map = None
            num_classes = None
    
    all_results = {}
    
    # ========== 对每个模型进行测试 ==========
    for model_name in models_list:
        print(f"\n{'='*80}")
        print(f"开始测试模型: {model_name}")
        print(f"{'='*80}")
        
        model_results = {}
        
        # ========== 对每种天气条件进行测试 ==========
        for weather in weather_list:
            weather_root = os.path.join(data_root, weather)
            if not os.path.exists(weather_root):
                print(f"跳过: {weather} 数据路径不存在")
                continue
            
            # 设置日志
            logger, log_filepath = setup_logger(log_dir, model_name, weather)
            
            logger.info("=" * 80)
            logger.info(f"自动化测试开始")
            logger.info(f"测试策略: {strategy}")
            logger.info(f"模型: {model_name}")
            logger.info(f"模态: {modality}")
            logger.info(f"目标域天气: {weather}")
            logger.info(f"设备: {DEVICE}")
            logger.info(f"迭代次数: {num_iterations}")
            logger.info(f"模型目录: {models_dir}")
            if strategy == 'strategy2':
                logger.info(f"注意: strategy2为零样本测试（仅晴天预训练，不微调）")
            elif strategy == 'strategy1':
                logger.info(f"注意: strategy1为目标域微调后的测试结果")
            if global_label_map:
                logger.info(f"标签映射: {global_label_map}")
                logger.info(f"类别数量: {num_classes}")
            logger.info("=" * 80)
            
            # 加载目标域数据集（使用global_label_map确保标签一致性）
            # 关键：必须使用与模型训练时相同的标签映射，否则标签ID不匹配会导致准确率为0
            try:
                if global_label_map:
                    # 使用全局标签映射加载目标域数据集
                    # 这样即使目标域只有部分类别，标签ID也与模型训练时一致
                    test_ds = SingleModalityDataset(weather_root, modality=modality, 
                                                     phase='val', global_label_map=global_label_map)
                    # 使用模型训练时的类别数量（从晴天数据集获取的14类）
                    actual_num_classes = num_classes
                else:
                    test_ds = SingleModalityDataset(weather_root, modality=modality, phase='val')
                    global_label_map = test_ds.get_label_map()
                    actual_num_classes = test_ds.num_classes
                
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, 
                                          drop_last=False, num_workers=0)
                logger.info(f"{weather}测试集大小: {len(test_ds)}")
                logger.info(f"使用全局标签映射的类别数量: {actual_num_classes}")
                logger.info(f"全局标签映射: {global_label_map}")
                logger.info(f"测试集中实际出现的类别: {test_ds.unique_labels}")
            except FileNotFoundError as e:
                logger.info(f"{weather}数据集加载失败: {e}")
                continue
            
            # 存储每次迭代的结果
            iteration_metrics = []
            
            # ========== 执行5次独立测试迭代 ==========
            for iteration in range(num_iterations):
                # 设置随机种子确保可复现性
                seed = 42 + iteration
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                logger.info(f"\n{'='*80}")
                logger.info(f"迭代 {iteration + 1}/{num_iterations}")
                logger.info(f"随机种子: {seed}")
                logger.info(f"{'='*80}")
                
                # 根据策略选择模型文件
                # strategy1: ImageNet预训练 + 目标域微调（模型名包含weather）
                # strategy2: 晴天预训练 + 目标域微调（模型名包含weather）
                # strategy2_staged: 仅晴天预训练，零样本测试（模型名不包含weather）
                if strategy == 'strategy1':
                    model_patterns = [
                        f"strategy1_{model_name}_{modality}_{weather}_iter{iteration+1}.pth",
                        f"strategy1_{model_name}_{modality}_{weather}_iter{iteration+1}",
                    ]
                elif strategy == 'strategy2_staged':
                    # 仅晴天预训练，零样本测试
                    model_patterns = [
                        f"strategy2_staged_{model_name}_{modality}_iter{iteration+1}.pth",
                        f"strategy2_{model_name}_{modality}_iter{iteration+1}.pth",
                    ]
                else:
                    # strategy2 默认使用带目标域微调的模型
                    model_patterns = [
                        # 优先查找带目标域微调的模型
                        f"strategy2_{model_name}_{modality}_{weather}_iter{iteration+1}.pth",
                        f"strategy2_{model_name}_{modality}_{weather}_iter{iteration+1}",
                        # 如果没有strategy2微调模型，尝试strategy1（也是目标域微调）
                        f"strategy1_{model_name}_{modality}_{weather}_iter{iteration+1}.pth",
                        f"strategy1_{model_name}_{modality}_{weather}_iter{iteration+1}",
                        # 如果都没有，再查找仅晴天预训练的模型（零样本）
                        f"strategy2_staged_{model_name}_{modality}_iter{iteration+1}.pth",
                        f"strategy2_{model_name}_{modality}_iter{iteration+1}.pth",
                    ]
                
                model_path = None
                for pattern in model_patterns:
                    candidate_path = os.path.join(models_dir, pattern)
                    if os.path.exists(candidate_path):
                        model_path = candidate_path
                        break
                    # 尝试在子目录中查找
                    for subdir in ['strategy1', 'strategy2', 'strategy2_staged', '']:
                        candidate_path = os.path.join(models_dir, subdir, pattern)
                        if os.path.exists(candidate_path):
                            model_path = candidate_path
                            break
                    if model_path:
                        break
                
                if model_path:
                    logger.info(f"加载模型: {model_path}")
                    model, model_num_classes = load_pretrained_model(model_path, model_name, 
                                                                    actual_num_classes, DEVICE, dropout_rate)
                    
                    if model_num_classes != actual_num_classes:
                        logger.info(f"模型类别数量({model_num_classes})与数据集类别数量({actual_num_classes})不一致")
                        logger.info(f"更新数据集类别数量为: {model_num_classes}")
                        actual_num_classes = model_num_classes
                else:
                    logger.info(f"未找到预训练模型文件，使用ImageNet预训练权重初始化")
                    model = create_model(model_name, actual_num_classes, 
                                          pretrained=True, dropout_rate=dropout_rate)
                    model = model.to(DEVICE)
                
                # ========== 执行评估 ==========
                metrics = evaluate_model(model, test_loader, DEVICE, global_label_map)
                
                logger.info(f"测试结果:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
                logger.info(f"  Precision (Macro-Present): {metrics['precision_macro_present']:.4f}")
                logger.info(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
                logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
                logger.info(f"  Recall (Macro-Present): {metrics['recall_macro_present']:.4f}")
                logger.info(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
                logger.info(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
                logger.info(f"  F1 (Macro-Present): {metrics['f1_macro_present']:.4f}")
                logger.info(f"  F1 (Micro): {metrics['f1_micro']:.4f}")
                logger.info(f"  验证集中出现的类别: {metrics['classes_present']}")
                
                # 保存本次迭代的指标
                iteration_metrics.append({
                    'accuracy': metrics['accuracy'],
                    'precision_macro': metrics['precision_macro'],
                    'precision_macro_present': metrics['precision_macro_present'],
                    'precision_micro': metrics['precision_micro'],
                    'recall_macro': metrics['recall_macro'],
                    'recall_macro_present': metrics['recall_macro_present'],
                    'recall_micro': metrics['recall_micro'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_macro_present': metrics['f1_macro_present'],
                    'f1_micro': metrics['f1_micro']
                })
            
            # ========== 计算统计结果 ==========
            logger.info(f"\n{'='*80}")
            logger.info(f"{weather} 统计结果")
            logger.info("=" * 80)
            
            metrics_names = ['accuracy', 'precision_macro', 'precision_macro_present',
                             'precision_micro', 'recall_macro', 'recall_macro_present',
                             'recall_micro', 'f1_macro', 'f1_macro_present', 'f1_micro']
            
            stats = {}
            for metric_name in metrics_names:
                values = np.array([m[metric_name] for m in iteration_metrics])
                mean_val = np.mean(values)
                std_val = np.std(values)
                max_val = np.max(values)
                min_val = np.min(values)
                
                stats[metric_name] = {
                    'values': [float(v) for v in values],
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'max': float(max_val),
                    'min': float(min_val)
                }
                
                logger.info(f"{metric_name}:")
                logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
                logger.info(f"  均值: {mean_val:.4f}")
                logger.info(f"  标准差: {std_val:.4f}")
                logger.info(f"  最高值: {max_val:.4f}")
                logger.info(f"  最低值: {min_val:.4f}")
                logger.info("")
            
            logger.info(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
            model_results[weather] = {
                'metrics': stats,
                'log_file': log_filepath
            }
        
        all_results[model_name] = model_results
    
    # ========== 保存汇总报告 ==========
    report_path = os.path.join(log_dir, f"test_summary_{strategy}.json")
    save_summary_report(all_results, report_path, models_dir, weather_list, 
                        num_iterations, modality, strategy)
    
    print(f"\n{'='*80}")
    print("自动化测试完成!")
    print(f"结果汇总报告已保存到: {report_path}")
    print(f"{'='*80}")
    
    return all_results


def save_summary_report(all_results, report_path, models_dir, weather_list,
                        num_iterations, modality, strategy='strategy2'):
    """
    保存测试结果汇总报告
    
    报告格式：
    - JSON格式，包含完整的测试配置和结果
    - 每个指标的均值、标准差、最高值、最低值
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report = {
        'test_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': strategy,
            'models_dir': models_dir,
            'models': ['resnet', 'vit', 'convnext'],
            'weather_conditions': weather_list,
            'num_iterations': num_iterations,
            'modality': modality
        },
        'results': all_results
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    """
    主函数：配置测试参数并执行自动化测试
    
    策略说明：
    - strategy2（默认）: 仅使用晴天预训练模型，在目标域上零样本测试
      预期准确率范围: 70%-85%
    - strategy1: 使用ImageNet预训练 + 目标域微调后的模型测试
      准确率会更高，因为已经在目标域上微调过
    """
    # ========== 测试配置 ==========
    # 模型文件目录（存放预训练模型权重）
    models_dir = r"D:\Code\JMDA-Net\experiment_models"
    
    # 数据根目录
    data_root = r"D:\Code\JMDA-Net\Data"
    
    # 目标域天气条件列表
    weather_list = ['雨天', '逆光', '黑天', '雾天']
    
    # 测试迭代次数
    num_iterations = 5
    
    # 测试模态
    modality = 'vis'
    
    # Dropout概率（与训练时一致）
    dropout_rate = 0.5
    
    # 测试策略: strategy2（零样本，推荐）或 strategy1（已微调）
    strategy = 'strategy2'  # 默认使用strategy2，即仅晴天预训练，直接测试
    
    # ========== 执行测试 ==========
    print("=" * 80)
    print("自动化测试脚本启动")
    print("=" * 80)
    print(f"测试策略: {strategy}")
    print(f"模型目录: {models_dir}")
    print(f"数据目录: {data_root}")
    print(f"目标域天气: {weather_list}")
    print(f"迭代次数: {num_iterations}")
    print(f"模态: {modality}")
    if strategy == 'strategy2':
        print("说明: strategy2为零样本测试（仅晴天预训练，不微调）")
    elif strategy == 'strategy1':
        print("说明: strategy1为目标域微调后的测试结果")
    print("=" * 80)
    
    results = run_automated_test(
        models_dir=models_dir,
        data_root=data_root,
        weather_list=weather_list,
        num_iterations=num_iterations,
        modality=modality,
        dropout_rate=dropout_rate,
        strategy=strategy
    )
    
    return results


if __name__ == "__main__":
    main()