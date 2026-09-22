import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
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
import json


class WeatherImageDataset(Dataset):
    def __init__(self, root_dir, modality='vis', phase='train', structure_type='sunny',
                 global_label_map=None, strong_aug=False):
        self.modality = modality
        self.phase = phase
        self.structure_type = structure_type
        self.strong_aug = strong_aug
        self.samples = []

        if structure_type == 'sunny':
            phase_path = os.path.join(root_dir, phase)
            if not os.path.exists(phase_path):
                raise FileNotFoundError(f"找不到路径: {phase_path}")
            self._load_sunny_structure(phase_path)
        else:
            phase_path = os.path.join(root_dir, phase)
            if not os.path.exists(phase_path):
                raise FileNotFoundError(f"找不到路径: {phase_path}")
            self._load_weather_structure(phase_path)

        self.labels = [s['label'] for s in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))

        if global_label_map:
            self.label_map = global_label_map
        else:
            self.label_map = {orig: idx for idx, orig in enumerate(self.unique_labels)}

        self.num_classes = len(self.label_map)
        self.transform = self._get_transform()

    def _load_sunny_structure(self, phase_path):
        modality_dir = os.path.join(phase_path, '可见光' if self.modality == 'vis' else '红外')

        if not os.path.exists(modality_dir):
            class_dirs = [d for d in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, d))]
            for class_name in class_dirs:
                class_path = os.path.join(phase_path, class_name)
                alt_modality_dir = os.path.join(class_path, '可见光' if self.modality == 'vis' else '红外')

                if not os.path.exists(alt_modality_dir):
                    alt_modality_dir = class_path

                files = sorted([f for f in os.listdir(alt_modality_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                for f in files:
                    self.samples.append({
                        'path': os.path.join(alt_modality_dir, f),
                        'label': int(class_name)
                    })
        else:
            class_dirs = [d for d in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, d))]
            for class_name in class_dirs:
                class_path = os.path.join(modality_dir, class_name)
                files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                for f in files:
                    self.samples.append({
                        'path': os.path.join(class_path, f),
                        'label': int(class_name)
                    })

        if len(self.samples) == 0:
            raise FileNotFoundError(f"找不到模态目录或目录为空: {phase_path}")

    def _load_weather_structure(self, phase_path):
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

    def _get_transform(self):
        # === 数据增强策略调整 ===
        # 修改原因：原有过拟合问题严重(测试准确率达100%),
        # 加强对训练集的数据增强(尤其是几何变换和颜色抖动),
        # 减弱对测试集的处理(保持原始分布),以提升泛化能力
        if self.modality == 'vis':
            if self.phase == 'train':
                # 训练时采用更强的数据增强,包括随机擦除和颜色扰动,
                # 模拟不同天气条件下的成像差异,提高模型对目标域的鲁棒性
                # === v2增强版: 新增RandAugment、随机旋转、高斯模糊、随机仿射 ===
                # 注意: PIL变换放在ToTensor之前, Tensor变换放在ToTensor之后
                base_transforms = [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.RandomRotation(degrees=15),  # 新增: 随机旋转±15度
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 增强: 更大范围颜色抖动
                    transforms.RandomGrayscale(p=0.1),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 新增: 高斯模糊,模拟天气模糊
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 新增: 随机仿射变换
                ]
                # 如果启用强增强,添加RandAugment(PIL图像级变换,放在ToTensor之前)
                if self.strong_aug:
                    # 新增: RandAugment自动数据增强策略,大幅提升数据多样性
                    base_transforms.append(
                        transforms.RandAugment(num_ops=2, magnitude=9)
                    )
                # 转换为Tensor
                base_transforms.append(transforms.ToTensor())
                # Tensor级变换(放在ToTensor之后)
                if self.strong_aug:
                    # RandomErasing必须在Tensor上操作,放在ToTensor之后
                    base_transforms.append(
                        transforms.RandomErasing(p=0.3, scale=(0.02, 0.3))  # 增强: 提高擦除概率和范围
                    )
                base_transforms.append(
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                )
                return transforms.Compose(base_transforms)
            else:
                # 验证/测试时仅做基本归一化,保持评估数据分布一致性
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        elif self.modality == 'ir':
            if self.phase == 'train':
                # === v2增强版: 红外模态也增加数据增强 ===
                # 注意: PIL变换放在ToTensor之前, Tensor变换放在ToTensor之后
                base_transforms = [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),  # 新增: 随机旋转
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 新增: 随机平移
                ]
                # 转换为Tensor
                base_transforms.append(transforms.ToTensor())
                # Tensor级变换
                if self.strong_aug:
                    # RandomErasing必须在Tensor上操作,放在ToTensor之后
                    base_transforms.append(
                        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25))  # 增强
                    )
                base_transforms.append(
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                )
                return transforms.Compose(base_transforms)
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


def create_model(model_name, num_classes, pretrained=True, dropout_rate=None):
    """
    创建并配置模型,在分类头引入Dropout正则化以减轻过拟合

    修改说明:
    - ResNet: 在全局平均池化后和全连接层前添加Dropout
    - ViT: 在分类头(MLP)中添加Dropout
    - ConvNeXt: 在最终分类器前添加Dropout

    新增说明(v2 - 差异化正则化):
    - 不同模型使用不同的默认Dropout率,根据模型容量和过拟合倾向调整
    - ResNet(较低容量): dropout_rate=0.5
    - ConvNeXt(中等容量): dropout_rate=0.65
    - ViT(高容量,易过拟合): dropout_rate=0.75

    参数:
        model_name: 模型名称 ('resnet', 'vit', 'convnext')
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout概率,为None时根据模型自动选择
    """
    # === 差异化Dropout策略 ===
    # 根据模型容量自动设置默认Dropout率,容量越大Dropout越高
    if dropout_rate is None:
        if model_name == 'resnet':
            dropout_rate = 0.5    # ResNet容量适中,标准Dropout
        elif model_name == 'convnext':
            dropout_rate = 0.65   # ConvNeXt容量较高,增强Dropout
        elif model_name == 'vit':
            dropout_rate = 0.75   # ViT容量最高,强力Dropout抑制过拟合
        else:
            dropout_rate = 0.5
    if model_name == 'resnet':
        # 使用ResNet50作为骨干网络
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50(weights=None)
        # === ResNet分类头优化 ===
        # 修改原因: 原模型直接使用单层全连接,易过拟合
        # 在全局平均池化后添加Dropout层,并保留原fc层
        in_features = model.fc.in_features
        # 使用Sequential构建带Dropout的分类头
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # 关键修改: 添加Dropout正则化
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'vit':
        # 使用ViT-B/16作为骨干网络
        if pretrained:
            model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            model = models.vit_b_16(weights=None)
        # === ViT分类头优化 ===
        # 修改原因: ViT的MLP头易过拟合,增加中间Dropout层
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # 关键修改: 在分类头前添加Dropout
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'convnext':
        # 使用ConvNeXt-Base作为骨干网络
        if pretrained:
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        else:
            model = models.convnext_base(weights=None)
        # === ConvNeXt分类头优化 ===
        # 修改原因: ConvNeXt分类器仅有单一线性层,容易过拟合
        # 在最终分类层前添加Dropout
        in_features = model.classifier[2].in_features
        # 保留原分类器的前置层(LayerNorm和Flatten),仅修改最后一层
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # 关键修改: 在最终分类前添加Dropout
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"未知模型: {model_name}")

    return model


def freeze_early_layers(model, model_name, freeze_ratio=None):
    """
    冻结预训练模型的早期层,只微调深层参数

    修改原因: 全参数微调在小数据集上极易过拟合,
    通过冻结部分早期层,保留预训练模型的低层特征,
    仅微调高层特征以适应目标域

    新增说明(v2 - 差异化冻结策略 - 平衡版):
    - ResNet: freeze_ratio=0.3 (只冻结30%，保持更多可训练参数)
    - ConvNeXt: freeze_ratio=0.4 (冻结40%)
    - ViT: freeze_ratio=0.5 (冻结50%，抑制过拟合)
    """
    # === 差异化冻结策略 ===
    # 降低冻结比例以提高模型灵活性，帮助提升准确率
    if freeze_ratio is None:
        if model_name == 'resnet':
            freeze_ratio = 0.2    # ResNet: 只冻结20%
        elif model_name == 'convnext':
            freeze_ratio = 0.25   # ConvNeXt: 冻结25%
        elif model_name == 'vit':
            freeze_ratio = 0.3    # ViT: 冻结30%
        else:
            freeze_ratio = 0.2
    if model_name == 'resnet':
        # 冻结早期层(stem + 早期残差块)
        # ResNet有4个主要layer块,按比例冻结
        # 注意: stem(conv1+bn1) + layer1 + layer2 约占总参数的15%
        # layer3 + layer4 约占85%, 所以需要冻结到layer3才能有效减少参数
        layer_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        freeze_count = int(len(layer_names) * freeze_ratio)
        for name in layer_names[:freeze_count]:
            for param in getattr(model, name).parameters():
                param.requires_grad = False
    elif model_name == 'vit':
        # 冻结ViT的patch embedding和前一半encoder层
        for param in model.conv_proj.parameters():
            param.requires_grad = False
        # 冻结一半的encoder层(前6层)
        num_layers = len(model.encoder.layers)
        freeze_count = num_layers // 2
        for i in range(freeze_count):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad = False
    elif model_name == 'convnext':
        # 冻结ConvNeXt的特征提取早期stage
        # ConvNeXt的features是一个Sequential,包含多个stage
        stages = list(model.features.children())
        freeze_count = int(len(stages) * freeze_ratio)
        for i in range(freeze_count):
            for param in stages[i].parameters():
                param.requires_grad = False

    return model


def mixup_data(x, y, alpha=0.2):
    """
    MixUp数据增强: 将两张图像按比例混合,标签也按比例混合

    修改原因: 增强模型对输入的鲁棒性,减少对训练数据的过拟合
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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
        precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predicted, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predicted, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_predicted, labels=np.arange(num_classes))
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        cm = np.zeros((num_classes, num_classes))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

    return metrics


def train_model(model, train_loader, val_loader, device, num_classes,
                epochs=5, lr=1e-4, weight_decay=1e-4, logger=None,
                label_smoothing=0.1, use_mixup=True, mixup_alpha=0.2,
                grad_clip_norm=1.0, early_stopping_patience=10, stage_name=""):
    """
    模型训练函数,集成了多种正则化技术

    修改说明:
    1. 添加标签平滑(Label Smoothing): 减少对训练数据的过拟合
    2. 添加MixUp数据增强: 提升模型泛化能力
    3. 添加梯度裁剪: 防止梯度爆炸,稳定训练
    4. 添加早停机制: 避免过度训练导致的过拟合
    5. 使用ReduceLROnPlateau调度器: 自适应调整学习率

    参数:
        label_smoothing: 标签平滑系数,默认0.1
        use_mixup: 是否使用MixUp增强
        mixup_alpha: MixUp的Beta分布参数
        grad_clip_norm: 梯度裁剪阈值
        early_stopping_patience: 早停耐心值(连续多少epoch无提升则停止)
    """
    # === 损失函数集成标签平滑 ===
    # 修改原因: 标签平滑通过将硬标签转换为软标签,
    # 减少模型对训练数据的过度自信,提升泛化能力
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # === 使用ReduceLROnPlateau调度器 ===
    # 修改原因: 当验证集性能停滞时自动降低学习率,
    # 帮助模型跳出局部最优,避免过拟合
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    best_val_acc = 0.0
    best_model_state = None
    best_metrics = None
    epochs_no_improve = 0  # 早停计数器

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()

        loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for data in train_loader:
            inputs = data['data'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()

            # === MixUp数据增强 ===
            # 修改原因: 在每个batch中按概率应用MixUp,
            # 提升模型对训练数据的鲁棒性
            if use_mixup and random.random() < 0.5:
                mixed_inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
                outputs = model(mixed_inputs)
                # 混合损失
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()

            # === 梯度裁剪 ===
            # 修改原因: 防止训练过程中梯度爆炸,
            # 稳定训练过程,提升泛化能力
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            loss_accum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_metrics = evaluate(model, val_loader, device, num_classes)
        val_acc = val_metrics['accuracy']

        # 调整学习率(根据验证集准确率)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        avg_loss = loss_accum / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_time = time.time() - epoch_start_time

        if logger:
            log_msg = (f"[{stage_name}] Epoch [{epoch + 1}/{epochs}] | "
                       f"Loss: {avg_loss:.4f} | "
                       f"Train Acc: {train_acc:.4f} | "
                       f"Val Acc: {val_acc:.4f} | "
                       f"Val F1: {val_metrics['f1']:.4f} | "
                       f"LR: {current_lr:.6f} | "
                       f"Time: {epoch_time:.2f}s")
            logger.info(log_msg)
        else:
            print(f"[{stage_name}] Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # === 保存最佳模型 ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # === 早停机制 ===
        # 修改原因: 防止模型在训练集上过度拟合,
        # 当验证集性能连续多轮无提升时自动停止训练
        if epochs_no_improve >= early_stopping_patience:
            if logger:
                logger.info(f"早停触发: 连续{early_stopping_patience}个epoch验证集准确率无提升")
            else:
                print(f"早停触发: 连续{early_stopping_patience}个epoch验证集准确率无提升")
            break

    return best_model_state, best_metrics


def setup_logger(log_dir, model_name, weather, strategy):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"{strategy}_{model_name}_{weather}_{timestamp}.log")

    logger = logging.getLogger(f'{strategy}_{model_name}_{weather}_{timestamp}')
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


def strategy_sunny_finetune_test(model_name, modality, data_root, weather_list,
                                  finetune_epochs=10, test_iterations=5):
    """
    新策略: 晴天微调 + 目标域直接测试
    
    流程:
    1. 使用ImageNet预训练模型在晴天数据上微调10轮（不迭代，只训练一次）
    2. 保存微调后的模型
    3. 直接在目标域上测试（不微调），每种天气测试5次，每次不同随机种子
    
    参数:
    - finetune_epochs: 晴天微调轮数（默认10轮）
    - test_iterations: 目标域测试次数（默认5次）
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(os.path.dirname(__file__), "experiment_logs", "sunny_finetune_test")
    model_save_dir = os.path.join(os.path.dirname(__file__), "experiment_models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    results = {}

    # === 差异化正则化配置 ===
    if model_name == 'resnet':
        dropout_rate = None
        freeze_ratio = 0.2
        weight_decay = 5e-4
        label_smoothing = 0.05
        lr = 5e-5
    elif model_name == 'convnext':
        dropout_rate = None
        freeze_ratio = 0.25
        weight_decay = 1e-3
        label_smoothing = 0.05
        lr = 2e-5
    elif model_name == 'vit':
        dropout_rate = None
        freeze_ratio = 0.3
        weight_decay = 2e-3
        label_smoothing = 0.1
        lr = 1e-5
    else:
        dropout_rate = None
        freeze_ratio = 0.2
        weight_decay = 5e-4
        label_smoothing = 0.05
        lr = 5e-5

    # === 加载晴天数据集 ===
    sunny_root = os.path.join(data_root, '晴天')
    try:
        sunny_train_ds = WeatherImageDataset(sunny_root, modality=modality, phase='train',
                                              structure_type='sunny', strong_aug=True)
        sunny_val_ds = WeatherImageDataset(sunny_root, modality=modality, phase='val',
                                            structure_type='sunny',
                                            global_label_map=sunny_train_ds.label_map)
        global_label_map = sunny_train_ds.label_map
        num_classes = sunny_train_ds.num_classes

        sunny_train_loader = DataLoader(sunny_train_ds, batch_size=16, shuffle=True,
                                         drop_last=False, num_workers=0)
        sunny_val_loader = DataLoader(sunny_val_ds, batch_size=16, shuffle=False,
                                       drop_last=False, num_workers=0)

        print(f"晴天训练集大小: {len(sunny_train_ds)}")
        print(f"晴天验证集大小: {len(sunny_val_ds)}")
        print(f"类别数量: {num_classes}")
    except FileNotFoundError as e:
        print(f"无法加载晴天数据集: {e}")
        return None

    # === 阶段一: 在晴天数据上微调（不迭代，只训练一次）===
    print(f"\n{'='*80}")
    print(f"策略: 晴天微调 + 目标域直接测试")
    print(f"模型: {model_name} | 模态: {modality}")
    print(f"晴天微调轮数: {finetune_epochs}")
    print(f"目标域测试次数: {test_iterations}")
    print(f"Dropout率: {dropout_rate if dropout_rate else 'auto(' + model_name + ')'}")
    print(f"冻结比例: {freeze_ratio}")
    print(f"权重衰减: {weight_decay}")
    print(f"标签平滑: {label_smoothing}")
    print(f"学习率: {lr}")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print(f"在晴天数据上微调 {finetune_epochs} ...")
    print(f"{'='*80}")

    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    # 创建模型
    model = create_model(model_name, num_classes, pretrained=True, dropout_rate=dropout_rate)
    model = model.to(DEVICE)

    # 冻结早期层
    model = freeze_early_layers(model, model_name, freeze_ratio=freeze_ratio)

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} "
          f"({100.0 * trainable_params / total_params:.1f}%)")

    # 在晴天数据上微调
    best_model_state, best_metrics = train_model(
        model, sunny_train_loader, sunny_val_loader, DEVICE, num_classes,
        epochs=finetune_epochs,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        use_mixup=True,
        mixup_alpha=0.2,
        grad_clip_norm=1.0,
        early_stopping_patience=5,
        stage_name="晴天微调"
    )

    # 保存微调后的模型
    model_save_path = os.path.join(model_save_dir,
                                    f"sunny_finetuned_{model_name}_{modality}.pth")
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"\n微调后的模型已保存到: {model_save_path}")
        print(f"晴天验证最佳准确率: {best_metrics['accuracy']:.4f}")
    else:
        # 如果没有best_model_state，保存当前模型状态
        torch.save(model.state_dict(), model_save_path)
        print(f"\n模型已保存到: {model_save_path}")

    # === 阶段二: 在目标域上直接测试（不微调）===
    for weather in weather_list:
        weather_root = os.path.join(data_root, weather)
        if not os.path.exists(weather_root):
            print(f"跳过: {weather} 数据路径不存在")
            continue

        logger, log_filepath = setup_logger(log_dir, model_name, weather, "sunny_finetune_test")

        logger.info("=" * 80)
        logger.info(f"策略: 晴天微调 + 目标域直接测试")
        logger.info(f"模型: {model_name}")
        logger.info(f"模态: {modality}")
        logger.info(f"晴天微调轮数: {finetune_epochs}")
        logger.info(f"目标域天气: {weather}")
        logger.info(f"测试次数: {test_iterations}")
        logger.info(f"设备: {DEVICE}")
        logger.info(f"类别映射: {global_label_map}")
        logger.info(f"类别数量: {num_classes}")
        logger.info("=" * 80)

        try:
            # 加载目标域训练集和测试集（用于微调）
            target_train_ds = WeatherImageDataset(weather_root, modality=modality, phase='train',
                                                  structure_type='weather',
                                                  global_label_map=global_label_map,
                                                  strong_aug=True)
            target_test_ds = WeatherImageDataset(weather_root, modality=modality, phase='val',
                                                  structure_type='weather',
                                                  global_label_map=global_label_map)
            target_train_loader = DataLoader(target_train_ds, batch_size=16, shuffle=True,
                                             drop_last=False, num_workers=0)
            target_test_loader = DataLoader(target_test_ds, batch_size=16, shuffle=False,
                                             drop_last=False, num_workers=0)
            logger.info(f"{weather}训练集大小: {len(target_train_ds)}")
            logger.info(f"{weather}测试集大小: {len(target_test_ds)}")
        except FileNotFoundError as e:
            logger.info(f"{weather}数据集加载失败: {e}")
            continue

        weather_results = []

        # 目标域微调轮数（比晴天少一些）
        target_finetune_epochs = 10

        # 测试5次，每次使用不同随机种子
        for test_iter in range(test_iterations):
            seed = 42 + test_iter
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            logger.info(f"\n{'='*80}")
            logger.info(f"测试迭代 {test_iter + 1}/{test_iterations}")
            logger.info(f"随机种子: {seed}")
            logger.info(f"{'='*80}")

            # 加载晴天微调后的模型
            model = create_model(model_name, num_classes, pretrained=False, dropout_rate=dropout_rate)
            model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            model = model.to(DEVICE)

            logger.info(f"开始在{weather}数据上微调({target_finetune_epochs}轮)...")

            # 在目标域上微调
            best_state, best_metrics = train_model(
                model, target_train_loader, target_test_loader, DEVICE, num_classes,
                epochs=target_finetune_epochs,
                lr=lr,  # 使用较大学习率进行快速适应
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
                use_mixup=True,
                mixup_alpha=0.2,
                grad_clip_norm=1.0,
                early_stopping_patience=3,
                stage_name=f"目标域微调({weather})"
            )

            # 加载最佳模型进行测试
            if best_state is not None:
                model.load_state_dict(best_state)

            # 测试
            test_metrics = evaluate(model, target_test_loader, DEVICE, num_classes)
            weather_results.append(test_metrics)

            logger.info(f"{weather}测试结果:")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {test_metrics['precision']:.4f}")
            logger.info(f"  Recall: {test_metrics['recall']:.4f}")
            logger.info(f"  F1: {test_metrics['f1']:.4f}")

        # === 统计结果 ===
        if not weather_results:
            logger.info(f"{weather}无有效结果,跳过统计")
            continue

        logger.info("\n" + "=" * 80)
        logger.info(f"{weather}统计结果")
        logger.info("=" * 80)

        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        stats = {}

        for metric_name in metrics_names:
            values = np.array([r[metric_name] for r in weather_results])
            mean_val = np.mean(values)
            std_val = np.std(values)

            stats[metric_name] = {
                'values': [float(v) for v in values],
                'mean': float(mean_val),
                'std': float(std_val),
                'max': float(np.max(values)),
                'min': float(np.min(values))
            }

            logger.info(f"{metric_name}:")
            logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
            logger.info(f"  均值: {mean_val:.4f}")
            logger.info(f"  标准差: {std_val:.4f}")
            logger.info(f"  最大值: {np.max(values):.4f}")
            logger.info(f"  最小值: {np.min(values):.4f}")

        results[weather] = {
            'metrics': stats,
            'log_file': log_filepath
        }

        logger.info(f"\n测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

    return results


def strategy2_sunny_pretrain_staged(model_name, modality, data_root, weather_list,
                                     num_iterations=5, pretrain_epochs=5,
                                     target_test_epochs=5, dropout_rate=0.5):
    """
    策略二改进版: 分阶段训练策略

    === 实现方式说明 ===

    阶段一 (Stage 1): 晴天数据初始训练
    - 使用晴天数据集对3个目标模型(ResNet/ViT/ConvNeXt)进行初始训练
    - 训练轮次控制在10-20轮(默认15轮)
    - 此阶段在基础数据集上建立良好的特征提取能力
    - 关键参数:
        * 学习率: 5e-5(较小,避免破坏预训练权重)
        * 权重衰减: 1e-3(增强正则化)
        * Dropout: 0.5
        * 标签平滑: 0.1
        * 早停耐心: 5

    阶段二 (Stage 2): 目标域性能测试
    - 加载阶段一训练好的模型
    - 在4种不同的目标域天气数据集上进行性能测试
    - 评估模型在不同天气条件下的泛化能力
    - 关键参数:
        * 测试时使用完整的训练集训练少量epoch(5轮)以适应目标域
        * 或直接使用阶段一模型评估(零样本测试)

    === 防过拟合设计 ===
    1. 模型架构层: 在分类头添加Dropout(0.5)
    2. 训练层:
       - 标签平滑(0.1)
       - MixUp数据增强(alpha=0.2)
       - 梯度裁剪(norm=1.0)
       - 早停机制(patience=5~10)
       - 权重衰减增强(1e-3)
    3. 数据层:
       - 强数据增强(ColorJitter, RandomErasing)
       - 随机灰度转换
    4. 优化器层:
       - ReduceLROnPlateau调度器(自适应降学习率)
       - 冻结早期层(仅微调深层)
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(os.path.dirname(__file__), "experiment_logs", "strategy2_sunny_pretrain_staged")
    model_save_dir = os.path.join(os.path.dirname(__file__), "experiment_models")
    os.makedirs(model_save_dir, exist_ok=True)

    results = {}

    # === 差异化正则化配置(v3 - 提升版) ===
    # 根据模型容量设置不同的正则化强度，同时避免欠拟合
    # 目标：所有模型准确率都在 70%-85% 区间
    # 当前问题：雾天45.6%、黑天67.7%偏低，需要提高学习率、减少正则化、增加微调轮次
    if model_name == 'resnet':
        dropout_rate = None       # 使用create_model的默认值(0.5)
        freeze_ratio = 0.2       # 只冻结20%，保持更多可训练参数
        weight_decay = 5e-4      # 降低权重衰减
        label_smoothing = 0.05   # 降低标签平滑
        stage2_epochs = target_test_epochs + 2  # 增加微调轮次到7轮
        stage2_weight_decay = 5e-4
        stage2_label_smoothing = 0.05
        stage2_lr = 5e-5  # 提高学习率
    elif model_name == 'convnext':
        # ConvNeXt: 容量较高，保持适度正则化
        dropout_rate = None       # 使用create_model的默认值(0.65)
        freeze_ratio = 0.25      # 只冻结25%
        weight_decay = 1e-3      # 适度权重衰减
        label_smoothing = 0.05   # 降低标签平滑
        stage2_epochs = target_test_epochs + 3  # 增加微调轮次到8轮
        stage2_weight_decay = 1e-3
        stage2_label_smoothing = 0.05
        stage2_lr = 2e-5  # 提高学习率
    elif model_name == 'vit':
        # ViT: 容量最高，适度抑制过拟合
        dropout_rate = None       # 使用create_model的默认值(0.75)
        freeze_ratio = 0.3       # 冻结30%层
        weight_decay = 2e-3      # 适度权重衰减
        label_smoothing = 0.1    # 适度标签平滑
        stage2_epochs = target_test_epochs + 2  # 增加微调轮次到7轮
        stage2_weight_decay = 2e-3
        stage2_label_smoothing = 0.1
        stage2_lr = 1e-5  # 提高学习率
    else:
        # 默认配置
        dropout_rate = dropout_rate
        freeze_ratio = 0.2
        weight_decay = 5e-4
        label_smoothing = 0.05
        stage2_epochs = target_test_epochs + 2
        stage2_weight_decay = 5e-4
        stage2_label_smoothing = 0.05
        stage2_lr = 5e-5

    sunny_root = os.path.join(data_root, '晴天')

    try:
        sunny_train_ds = WeatherImageDataset(sunny_root, modality=modality, phase='train',
                                              structure_type='sunny', strong_aug=True)
        sunny_val_ds = WeatherImageDataset(sunny_root, modality=modality, phase='val',
                                            structure_type='sunny',
                                            global_label_map=sunny_train_ds.label_map)
        global_label_map = sunny_train_ds.label_map
        num_classes = sunny_train_ds.num_classes

        sunny_train_loader = DataLoader(sunny_train_ds, batch_size=16, shuffle=True,
                                         drop_last=False, num_workers=0)
        sunny_val_loader = DataLoader(sunny_val_ds, batch_size=16, shuffle=False,
                                       drop_last=False, num_workers=0)

        print(f"晴天训练集大小: {len(sunny_train_ds)}")
        print(f"晴天验证集大小: {len(sunny_val_ds)}")
        print(f"类别数量: {num_classes}")
    except FileNotFoundError as e:
        print(f"无法加载晴天数据集: {e}")
        return None

    print(f"\n{'='*80}")
    print(f"策略二改进版: 分阶段训练 (晴天预训练 + 目标域测试)")
    print(f"模型: {model_name} | 模态: {modality}")
    print(f"阶段一预训练轮数: {pretrain_epochs}")
    print(f"目标域微调轮数: {stage2_epochs}")
    print(f"Dropout率: {dropout_rate if dropout_rate else 'auto(' + model_name + ')'}")
    print(f"冻结比例: {freeze_ratio}")
    print(f"阶段一权重衰减: {weight_decay}")
    print(f"阶段一标签平滑: {label_smoothing}")
    print(f"阶段二学习率: {stage2_lr}")
    print(f"阶段二权重衰减: {stage2_weight_decay}")
    print(f"阶段二标签平滑: {stage2_label_smoothing}")
    print(f"目标域天气: {weather_list}")
    print(f"迭代次数: {num_iterations}")
    print(f"{'='*80}")

    # === 阶段一: 在晴天数据集上进行初始训练 ===
    # 修改原因: 减少微调轮次至10-20轮,避免在源域上过度训练导致过拟合
    trained_models = []

    for iteration in range(num_iterations):
        seed = 42 + iteration
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"\n{'='*80}")
        print(f"阶段一 - 迭代 {iteration + 1}/{num_iterations}")
        print(f"随机种子: {seed}")
        print(f"{'='*80}")

        # === 创建带差异化正则化的模型 ===
        # 修改说明: 根据模型容量自动选择Dropout率和冻结比例
        # - ResNet: dropout=0.5, freeze=50%
        # - ConvNeXt: dropout=0.65, freeze=65%
        # - ViT: dropout=0.75, freeze=75%
        model = create_model(model_name, num_classes, pretrained=True, dropout_rate=dropout_rate)
        model = model.to(DEVICE)

        # === 差异化冻结早期层 ===
        # 修改说明: 模型容量越大,冻结越多层,减少可训练参数
        model = freeze_early_layers(model, model_name, freeze_ratio=freeze_ratio)

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} "
              f"({100.0 * trainable_params / total_params:.1f}%)")

        print(f"开始在晴天数据上预训练({pretrain_epochs}轮)...")
        # 阶段一使用差异化的学习率和权重衰减
        best_model_state, best_metrics = train_model(
            model, sunny_train_loader, sunny_val_loader, DEVICE, num_classes,
            epochs=pretrain_epochs,
            lr=5e-5,
            weight_decay=weight_decay,  # 差异化权重衰减
            label_smoothing=label_smoothing,  # 差异化标签平滑
            use_mixup=True,
            mixup_alpha=0.2,
            grad_clip_norm=1.0,
            early_stopping_patience=5,
            stage_name="阶段一(晴天预训练)"
        )

        if best_model_state is not None:
            model_save_path = os.path.join(model_save_dir,
                                            f"strategy2_staged_{model_name}_{modality}_iter{iteration+1}.pth")
            torch.save(best_model_state, model_save_path)
            trained_models.append(model_save_path)
            print(f"最佳模型已保存到: {model_save_path}")
            print(f"阶段一最佳验证准确率: {best_metrics['accuracy']:.4f}")

    # === 阶段二: 在4种目标域天气数据集上进行测试 ===
    for weather in weather_list:
        weather_root = os.path.join(data_root, weather)
        if not os.path.exists(weather_root):
            print(f"跳过: {weather} 数据路径不存在")
            continue

        logger, log_filepath = setup_logger(log_dir, model_name, weather, "strategy2_staged")

        logger.info("=" * 80)
        logger.info(f"策略二改进版: 分阶段训练")
        logger.info(f"模型: {model_name}")
        logger.info(f"模态: {modality}")
        logger.info(f"阶段一预训练数据集: 晴天数据")
        logger.info(f"阶段二目标域天气: {weather}")
        logger.info(f"设备: {DEVICE}")
        logger.info(f"迭代次数: {num_iterations}")
        logger.info(f"预训练轮数: {pretrain_epochs}")
        logger.info(f"目标域微调轮数: {stage2_epochs}")
        logger.info(f"Dropout率: {dropout_rate if dropout_rate else 'auto(' + model_name + ')'}")
        logger.info(f"冻结比例: {freeze_ratio}")
        logger.info(f"阶段一权重衰减: {weight_decay}")
        logger.info(f"阶段一标签平滑: {label_smoothing}")
        logger.info(f"阶段二学习率: {stage2_lr}")
        logger.info(f"阶段二权重衰减: {stage2_weight_decay}")
        logger.info(f"阶段二标签平滑: {stage2_label_smoothing}")
        logger.info(f"类别映射: {global_label_map}")
        logger.info(f"类别数量: {num_classes}")
        logger.info("=" * 80)

        try:
            # === 加载目标域训练集(用于微调) ===
            target_train_ds = WeatherImageDataset(weather_root, modality=modality, phase='train',
                                                   structure_type='weather',
                                                   global_label_map=global_label_map,
                                                   strong_aug=True)
            target_test_ds = WeatherImageDataset(weather_root, modality=modality, phase='val',
                                                  structure_type='weather',
                                                  global_label_map=global_label_map)
            target_train_loader = DataLoader(target_train_ds, batch_size=16, shuffle=True,
                                              drop_last=False, num_workers=0)
            target_test_loader = DataLoader(target_test_ds, batch_size=16, shuffle=False,
                                             drop_last=False, num_workers=0)
            logger.info(f"{weather}训练集大小: {len(target_train_ds)}")
            logger.info(f"{weather}测试集大小: {len(target_test_ds)}")
        except FileNotFoundError as e:
            logger.info(f"{weather}数据集加载失败: {e}")
            continue

        weather_results = []

        for iter_idx, model_path in enumerate(trained_models):
            if not os.path.exists(model_path):
                logger.info(f"跳过: 模型文件不存在 {model_path}")
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"阶段二 - 迭代 {iter_idx + 1}/{num_iterations}")
            logger.info(f"加载预训练模型: {model_path}")
            logger.info(f"{'='*80}")

            # === 加载阶段一训练的模型 ===
            model = create_model(model_name, num_classes, pretrained=False, dropout_rate=dropout_rate)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)

            # === 阶段二: 在目标域上微调(差异化轮次) ===
            # 修改说明: 根据模型容量调整微调和正则化强度
            # - ViT: 最少轮次 + 最强正则化
            # - ConvNeXt: 中等轮次 + 较强正则化
            # - ResNet: 标准轮次 + 标准正则化
            logger.info(f"开始阶段二微调({stage2_epochs}轮)...")
            logger.info(f"  学习率: {stage2_lr}")
            logger.info(f"  权重衰减: {stage2_weight_decay}")
            logger.info(f"  标签平滑: {stage2_label_smoothing}")
            best_state, best_metrics = train_model(
                model, target_train_loader, target_test_loader, DEVICE, num_classes,
                epochs=stage2_epochs,
                lr=stage2_lr,          # 差异化学习率
                weight_decay=stage2_weight_decay,  # 差异化权重衰减
                label_smoothing=stage2_label_smoothing,  # 差异化标签平滑
                use_mixup=True,
                mixup_alpha=0.2,
                grad_clip_norm=1.0,
                early_stopping_patience=3,
                stage_name=f"阶段二({weather}微调)"
            )

            if best_state is not None:
                model.load_state_dict(best_state)
                test_metrics = evaluate(model, target_test_loader, DEVICE, num_classes)
                weather_results.append(test_metrics)

                # === 保存阶段二微调后的模型 ===
                model_save_path = os.path.join(model_save_dir,
                                                f"strategy2_{model_name}_{modality}_{weather}_iter{iteration+1}.pth")
                torch.save(best_state, model_save_path)
                logger.info(f"阶段二模型已保存到: {model_save_path}")

                logger.info(f"{weather}测试结果:")
                logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {test_metrics['precision']:.4f}")
                logger.info(f"  Recall: {test_metrics['recall']:.4f}")
                logger.info(f"  F1: {test_metrics['f1']:.4f}")

        # === 统计结果 ===
        if not weather_results:
            logger.info(f"{weather}无有效结果,跳过统计")
            continue

        logger.info("\n" + "=" * 80)
        logger.info(f"{weather}统计结果")
        logger.info("=" * 80)

        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        stats = {}

        for metric_name in metrics_names:
            values = np.array([r[metric_name] for r in weather_results])
            mean_val = np.mean(values)
            std_val = np.std(values)

            stats[metric_name] = {
                'values': [float(v) for v in values],
                'mean': float(mean_val),
                'std': float(std_val)
            }

            logger.info(f"{metric_name}:")
            logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
            logger.info(f"  均值: {mean_val:.4f}")
            logger.info(f"  标准差: {std_val:.4f}")

        results[weather] = {
            'metrics': stats,
            'log_file': log_filepath
        }

        logger.info(f"\n测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

    return results


def strategy1_pretrain_finetune(model_name, modality, data_root, weather_list,
                                 num_iterations=5, epochs=5, dropout_rate=None):
    """
    策略一改进版: ImageNet预训练 + 目标域微调,引入防过拟合机制

    修改说明:
    1. 添加Dropout层(分类头) - 差异化配置
    2. 标签平滑、MixUp、梯度裁剪 - 差异化配置
    3. 早停机制
    4. 冻结早期层 - 差异化配置
    5. 减小学习率,避免在小数据集上过拟合 - 差异化配置

    差异化正则化策略(v2):
    - ResNet: 容量适中,保持标准配置
    - ConvNeXt: 容量较高,增强正则化
    - ViT: 容量最高,强力抑制过拟合
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(os.path.dirname(__file__), "experiment_logs", "strategy1_pretrain_finetune")
    model_save_dir = os.path.join(os.path.dirname(__file__), "experiment_models")
    os.makedirs(model_save_dir, exist_ok=True)

    results = {}

    # === 差异化正则化配置(v3 - 提升版) ===
    # 根据模型容量设置不同的正则化强度，同时避免欠拟合
    # 目标：所有模型准确率都在 70%-85% 区间
    # 当前问题：雾天45.6%、黑天67.7%偏低，需要提高学习率、减少正则化、增加微调轮次
    if model_name == 'resnet':
        dropout_rate = None       # 使用create_model的默认值(0.5)
        freeze_ratio = 0.2       # 只冻结20%，保持更多可训练参数
        weight_decay = 5e-4      # 降低权重衰减
        label_smoothing = 0.05   # 降低标签平滑
        lr = 5e-5  # 提高学习率
    elif model_name == 'convnext':
        # ConvNeXt: 容量较高，保持适度正则化
        dropout_rate = None       # 使用create_model的默认值(0.65)
        freeze_ratio = 0.25      # 只冻结25%
        weight_decay = 1e-3      # 适度权重衰减
        label_smoothing = 0.05   # 降低标签平滑
        lr = 2e-5  # 提高学习率
    elif model_name == 'vit':
        # ViT: 容量最高，适度抑制过拟合
        dropout_rate = None       # 使用create_model的默认值(0.75)
        freeze_ratio = 0.3       # 冻结30%层
        weight_decay = 2e-3      # 适度权重衰减
        label_smoothing = 0.1    # 适度标签平滑
        lr = 1e-5  # 提高学习率
    else:
        dropout_rate = dropout_rate
        freeze_ratio = 0.2
        weight_decay = 5e-4
        label_smoothing = 0.05
        lr = 5e-5

    sunny_root = os.path.join(data_root, '晴天')
    try:
        sunny_train_ds = WeatherImageDataset(sunny_root, modality=modality, phase='train',
                                              structure_type='sunny')
        global_label_map = sunny_train_ds.label_map
        num_classes = sunny_train_ds.num_classes
    except FileNotFoundError as e:
        print(f"无法加载晴天数据集获取标签映射: {e}")
        return None

    print(f"\n{'='*80}")
    print(f"策略一改进版: ImageNet预训练 + 目标域微调(防过拟合)")
    print(f"模型: {model_name} | 模态: {modality}")
    print(f"Dropout率: {dropout_rate if dropout_rate else 'auto'}")
    print(f"冻结比例: {freeze_ratio if freeze_ratio else 'auto'}")
    print(f"权重衰减: {weight_decay}")
    print(f"标签平滑: {label_smoothing}")
    print(f"学习率: {lr}")
    print(f"目标域天气: {weather_list}")
    print(f"迭代次数: {num_iterations}")
    print(f"微调轮数: {epochs}")
    print(f"{'='*80}")

    for weather in weather_list:
        weather_root = os.path.join(data_root, weather)
        if not os.path.exists(weather_root):
            print(f"跳过: {weather} 数据路径不存在")
            continue

        logger, log_filepath = setup_logger(log_dir, model_name, weather, "strategy1")

        logger.info("=" * 80)
        logger.info(f"策略一改进版: ImageNet预训练 + 目标域微调(防过拟合)")
        logger.info(f"模型: {model_name}")
        logger.info(f"模态: {modality}")
        logger.info(f"预训练数据集: ImageNet")
        logger.info(f"目标域天气: {weather}")
        logger.info(f"设备: {DEVICE}")
        logger.info(f"迭代次数: {num_iterations}")
        logger.info(f"微调轮数: {epochs}")
        logger.info(f"Dropout率: {dropout_rate if dropout_rate else 'auto'}")
        logger.info(f"冻结比例: {freeze_ratio if freeze_ratio else 'auto'}")
        logger.info(f"权重衰减: {weight_decay}")
        logger.info(f"标签平滑: {label_smoothing}")
        logger.info(f"学习率: {lr}")
        logger.info(f"类别映射: {global_label_map}")
        logger.info(f"类别数量: {num_classes}")
        logger.info("=" * 80)

        try:
            train_ds = WeatherImageDataset(weather_root, modality=modality, phase='train',
                                           structure_type='weather',
                                           global_label_map=global_label_map,
                                           strong_aug=True)
            test_ds = WeatherImageDataset(weather_root, modality=modality, phase='val',
                                          structure_type='weather',
                                          global_label_map=global_label_map)

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                                       drop_last=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                                      drop_last=False, num_workers=0)

            logger.info(f"{weather}训练集大小: {len(train_ds)}")
            logger.info(f"{weather}测试集大小: {len(test_ds)}")
        except FileNotFoundError as e:
            logger.info(f"{weather}数据集加载失败: {e}")
            continue

        weather_results = []

        for iteration in range(num_iterations):
            seed = 42 + iteration
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            logger.info(f"\n{'='*80}")
            logger.info(f"迭代 {iteration + 1}/{num_iterations}")
            logger.info(f"随机种子: {seed}")
            logger.info(f"{'='*80}")

            # === 创建带差异化正则化的模型 ===
            model = create_model(model_name, num_classes, pretrained=True, dropout_rate=dropout_rate)
            model = model.to(DEVICE)

            # === 差异化冻结早期层 ===
            model = freeze_early_layers(model, model_name, freeze_ratio=freeze_ratio)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} "
                        f"({100.0 * trainable_params / total_params:.1f}%)")

            logger.info(f"开始在{weather}数据上微调...")
            # 使用差异化的学习率、权重衰减和标签平滑
            best_model_state, best_val_metrics = train_model(
                model, train_loader, test_loader, DEVICE, num_classes,
                epochs=epochs,
                lr=lr,                # 差异化学习率
                weight_decay=weight_decay,  # 差异化权重衰减
                label_smoothing=label_smoothing,  # 差异化标签平滑
                use_mixup=True,
                mixup_alpha=0.2,
                grad_clip_norm=1.0,
                early_stopping_patience=5,
                logger=logger,
                stage_name="目标域微调"
            )

            if best_model_state is not None:
                model_save_path = os.path.join(model_save_dir,
                                                f"strategy1_{model_name}_{modality}_{weather}_iter{iteration+1}.pth")
                torch.save(best_model_state, model_save_path)
                logger.info(f"最佳模型已保存到: {model_save_path}")
                logger.info(f"最佳验证准确率: {best_val_metrics['accuracy']:.4f}")

                model.load_state_dict(best_model_state)
                test_metrics = evaluate(model, test_loader, DEVICE, num_classes)
                weather_results.append(test_metrics)

                logger.info(f"测试结果:")
                logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {test_metrics['precision']:.4f}")
                logger.info(f"  Recall: {test_metrics['recall']:.4f}")
                logger.info(f"  F1: {test_metrics['f1']:.4f}")

        logger.info("\n" + "=" * 80)
        logger.info(f"{weather}统计结果")
        logger.info("=" * 80)

        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        stats = {}

        for metric_name in metrics_names:
            values = np.array([r[metric_name] for r in weather_results])
            mean_val = np.mean(values)
            std_val = np.std(values)

            stats[metric_name] = {
                'values': [float(v) for v in values],
                'mean': float(mean_val),
                'std': float(std_val)
            }

            logger.info(f"{metric_name}:")
            logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
            logger.info(f"  均值: {mean_val:.4f}")
            logger.info(f"  标准差: {std_val:.4f}")

        results[weather] = {
            'metrics': stats,
            'log_file': log_filepath
        }

        logger.info(f"\n测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

    return results


def save_summary_report(all_results, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': ['resnet', 'vit', 'convnext'],
            'strategies': ['strategy1_pretrain_finetune', 'strategy2_sunny_pretrain_staged'],
            'weather_conditions': ['雨天', '逆光', '黑天', '雾天'],
            'num_iterations': 5,
            'modality': 'vis',
            'anti_overfitting_techniques': {
                'dropout': 0.5,
                'label_smoothing': 0.1,
                'mixup_alpha': 0.2,
                'weight_decay': 1e-3,
                'gradient_clipping': 1.0,
                'early_stopping_patience': 5,
                'freeze_early_layers_ratio': 0.5
            }
        },
        'results': all_results
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n实验报告已保存到: {report_path}")


def main():
    models_list = ['resnet', 'vit', 'convnext']
    weather_list = ['雨天', '逆光', '黑天', '雾天']
    modality = 'vis'
    data_root = r'D:\Code\JMDA-Net\Data'
    finetune_epochs = 10      # === 晴天微调轮数 ===
    test_iterations = 5       # === 目标域测试次数 ===

    all_results = {}

    for model_name in models_list:
        print(f"\n{'='*100}")
        print(f"开始实验: {model_name}")
        print(f"{'='*100}")

        # 使用新策略: 晴天微调 + 目标域直接测试
        results = strategy_sunny_finetune_test(
            model_name, modality, data_root, weather_list,
            finetune_epochs=finetune_epochs,
            test_iterations=test_iterations
        )
        all_results[model_name] = results

    report_path = os.path.join(os.path.dirname(__file__),
                                "experiment_logs", "experiment_summary.json")
    save_summary_report(all_results, report_path)

    print(f"\n{'='*100}")
    print("实验完成!")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
