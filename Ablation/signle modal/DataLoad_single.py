import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, RandomErasing, RandomHorizontalFlip, RandomRotation
import scipy.io as sio
try:
    import h5py
except ImportError:
    h5py = None


def get_advanced_transforms(phase='train', modality='vis'):
    """
    获取数据增强策略

    Args:
        phase: 'train' 或 'val'
        modality: 'vis' (可见光), 'ir' (红外), 'ais' (AIS信号)
    """
    if modality == 'ais':
        # AIS信号不需要图像变换
        return None

    if phase == 'train':
        if modality == 'vis':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:  # ir
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
    else:  # val
        if modality == 'vis':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:  # ir
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])


class SingleModalityDataset(Dataset):
    """
    单模态数据集 - 用于消融实验

    支持三种模态：
    1. 'vis': 可见光图像
    2. 'ir': 红外图像
    3. 'ais': AIS信号 (从.mat文件加载)
    """

    def __init__(self, root_dir, modality='vis', domain_type='source', phase='train',
                 img_size=224, weather=None, global_label_map=None, ais_data_path=None):
        """
        Args:
            root_dir: 数据根目录
            modality: 模态类型 ('vis', 'ir', 'ais')
            domain_type: 域类型 ('source', 'target')
            phase: 阶段 ('train', 'val')
            img_size: 图像大小
            weather: 天气类型
            global_label_map: 全局标签映射
            ais_data_path: AIS数据路径 (.mat文件)
        """
        self.modality = modality
        if weather is not None:
            self.weather = weather
        elif root_dir is not None:
            self.weather = os.path.basename(root_dir)
        else:
            # AIS 模态可不依赖图像目录，给一个稳定的默认天气标签
            self.weather = "ais"
        self.domain_type = domain_type
        self.phase = phase
        self.img_size = img_size
        self.samples = []
        self.domain_label = 0 if domain_type == 'source' else 1
        self.ais_data_path = ais_data_path

        # 加载数据
        if modality == 'ais':
            self._load_ais_data()
        else:
            if root_dir is None:
                raise ValueError("非AIS模态必须提供有效的 root_dir")
            self._load_image_data(root_dir)

        # 构建标签映射
        self.labels = [s['label'] for s in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))

        if global_label_map:
            self.label_map = global_label_map
        else:
            self.label_map = {orig: idx for idx, orig in enumerate(self.unique_labels)}

        self.num_classes = len(self.label_map)
        self.get_label_map = lambda: self.label_map

        # 获取变换
        self.transform = get_advanced_transforms(phase, modality)

        print(f"[{modality.upper()}] 加载完成: {len(self.samples)} 个样本, {self.num_classes} 个类别")

    def _load_ais_data(self):
        """加载AIS信号数据"""
        if not self.ais_data_path or not os.path.exists(self.ais_data_path):
            raise FileNotFoundError(f"AIS数据文件不存在: {self.ais_data_path}")

        try:
            # v7.2 及以下
            mat_data = sio.loadmat(self.ais_data_path)

            # 假设数据格式: 'data' 字段存储特征, 'labels' 字段存储标签
            if 'data' in mat_data and 'labels' in mat_data:
                ais_features = mat_data['data']
                ais_labels = mat_data['labels'].flatten()
            else:
                # 尝试其他可能的字段名
                possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                print(f"可用的MAT文件键: {possible_keys}")
                data_key = possible_keys[0] if possible_keys else None
                if data_key:
                    ais_data_array = np.array(mat_data[data_key])
                    ais_features = ais_data_array[:, :-1]
                    ais_labels = ais_data_array[:, -1].flatten()
                else:
                    raise ValueError("无法从MAT文件中提取数据")
        except (NotImplementedError, ValueError, OSError) as mat_error:
            if h5py is None:
                raise ImportError(
                    "AIS MAT file appears to be MATLAB v7.3/HDF5, but h5py is not installed. "
                    "Install h5py to run AIS modality experiments."
                )
            # v7.3: scipy 不支持，使用 h5py 读取
            with h5py.File(self.ais_data_path, 'r') as f:
                keys = list(f.keys())
                print(f"检测到MAT v7.3，HDF5键: {keys}")

                data_key_candidates = ['data', 'X', 'features', 'feat']
                label_key_candidates = ['labels', 'label', 'y', 'Y', 'target', 'targets']

                data_key = next((k for k in data_key_candidates if k in f), None)
                label_key = next((k for k in label_key_candidates if k in f), None)

                if data_key is not None and label_key is not None:
                    ais_features = np.array(f[data_key])
                    ais_labels = np.array(f[label_key]).reshape(-1)
                else:
                    # 兜底：选一个二维数组作为特征，另一个一维/单列数组作为标签
                    array_map = {}
                    for k in keys:
                        obj = f[k]
                        if hasattr(obj, 'shape'):
                            array_map[k] = np.array(obj)

                    two_d = [(k, v) for k, v in array_map.items() if v.ndim == 2]
                    one_d = [(k, v) for k, v in array_map.items() if v.ndim == 1 or (v.ndim == 2 and 1 in v.shape)]

                    if len(two_d) == 0:
                        raise ValueError(f"无法从MAT v7.3文件中找到二维特征数组，可用键: {keys}")

                    if len(one_d) > 0:
                        data_key, ais_features = two_d[0]
                        label_key, ais_labels_arr = one_d[0]
                        ais_labels = ais_labels_arr.reshape(-1)
                    else:
                        # 最后兜底：二维数组最后一列作为标签
                        data_key, full_arr = two_d[0]
                        if full_arr.shape[1] < 2:
                            raise ValueError(f"二维数组列数不足，无法拆分特征和标签: key={data_key}, shape={full_arr.shape}")
                        ais_features = full_arr[:, :-1]
                        ais_labels = full_arr[:, -1].reshape(-1)
                        label_key = f"{data_key}[:,-1]"

                    print(f"自动推断键: data={data_key}, labels={label_key}")

                if all(k in f for k in ('balanced_rcv_I', 'balanced_rcv_Q', 'new_balanced_label')):
                    # Balanced AIS files store I/Q components separately.
                    # Concatenate both components so AIS uses the full signal.
                    ais_i = np.array(f['balanced_rcv_I'])
                    ais_q = np.array(f['balanced_rcv_Q'])
                    ais_labels = np.array(f['new_balanced_label']).reshape(-1)

                    def align_iq_features(arr, labels, key_name):
                        arr = np.asarray(arr, dtype=np.float32)
                        labels_len = len(labels)
                        if arr.ndim == 2:
                            if arr.shape[0] == labels_len:
                                return arr
                            if arr.shape[1] == labels_len:
                                return arr.T
                        elif arr.ndim > 2:
                            sample_axis = next(
                                (axis for axis, size in enumerate(arr.shape) if size == labels_len),
                                None,
                            )
                            if sample_axis is not None:
                                return np.moveaxis(arr, sample_axis, 0).reshape(labels_len, -1)
                        raise ValueError(
                            f"AIS HDF5 key {key_name} does not match label count: "
                            f"shape={arr.shape}, labels={ais_labels.shape}"
                        )

                    ais_i = align_iq_features(ais_i, ais_labels, 'balanced_rcv_I')
                    ais_q = align_iq_features(ais_q, ais_labels, 'balanced_rcv_Q')
                    if ais_i.shape[0] != ais_q.shape[0]:
                        raise ValueError(
                            f"AIS I/Q sample count mismatch: I={ais_i.shape}, Q={ais_q.shape}"
                        )
                    ais_features = np.concatenate([ais_i, ais_q], axis=1)
                    print("AIS HDF5 keys: data=balanced_rcv_I+balanced_rcv_Q, labels=new_balanced_label")

                # MATLAB/HDF5 常见存储方向修正：若是 [feat_dim, n_samples]，转置为 [n_samples, feat_dim]
                if ais_features.ndim == 2 and len(ais_labels) == ais_features.shape[0]:
                    pass
                elif ais_features.ndim == 2 and len(ais_labels) == ais_features.shape[1]:
                    ais_features = ais_features.T
                else:
                    raise ValueError(
                        f"特征与标签长度不匹配: features.shape={ais_features.shape}, labels.shape={ais_labels.shape}"
                    )

        ais_features = np.asarray(ais_features, dtype=np.float32)
        ais_labels = np.asarray(ais_labels).reshape(-1)

        # 根据phase划分数据 (简单的8:2划分)
        n_samples = len(ais_labels)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)

        split_idx = int(0.8 * n_samples)
        if self.phase == 'train':
            selected_indices = indices[:split_idx]
        else:  # val
            selected_indices = indices[split_idx:]

        for idx in selected_indices:
            self.samples.append({
                'data': ais_features[idx],
                'label': int(ais_labels[idx])
            })

    def _load_image_data(self, root_dir):
        """加载图像数据 (可见光或红外)"""
        phase_path = os.path.join(root_dir, self.phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"找不到路径: {phase_path}")

        # 确定模态文件夹名称
        modality_folder = '可见光' if self.modality == 'vis' else '红外'

        if self.domain_type == 'source':
            # 源域结构: root/phase/可见光(或红外)/class_id/
            mod_dir = os.path.join(phase_path, modality_folder)

            if not os.path.exists(mod_dir):
                raise FileNotFoundError(f"找不到模态文件夹: {mod_dir}")

            valid_classes = [d for d in os.listdir(mod_dir) if os.path.isdir(os.path.join(mod_dir, d))]

            for class_name in valid_classes:
                class_dir = os.path.join(mod_dir, class_name)
                img_files = sorted([f for f in os.listdir(class_dir)
                                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

                for img_file in img_files:
                    self.samples.append({
                        'path': os.path.join(class_dir, img_file),
                        'label': int(class_name)
                    })

        else:  # target
            # 目标域结构: root/phase/class_id/可见光(或红外)/
            valid_classes = [d for d in os.listdir(phase_path)
                           if os.path.isdir(os.path.join(phase_path, d))]

            for class_name in valid_classes:
                class_dir = os.path.join(phase_path, class_name)
                mod_dir = os.path.join(class_dir, modality_folder)

                if not os.path.exists(mod_dir):
                    continue

                img_files = sorted([f for f in os.listdir(mod_dir)
                                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

                for img_file in img_files:
                    self.samples.append({
                        'path': os.path.join(mod_dir, img_file),
                        'label': int(class_name)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label_id = self.label_map.get(sample['label'], 0)

        if self.modality == 'ais':
            # AIS信号数据
            data = torch.tensor(sample['data'], dtype=torch.float32)
            return {
                'data': data,
                'label': torch.tensor(label_id, dtype=torch.long),
                'domain_label': self.domain_label
            }
        else:
            # 图像数据
            try:
                if self.modality == 'vis':
                    img = Image.open(sample['path']).convert('RGB')
                else:  # ir
                    img = Image.open(sample['path']).convert('L').convert('RGB')
            except:
                # 容错：图片损坏时返回下一张
                return self.__getitem__((idx + 1) % len(self))

            return {
                'data': self.transform(img),
                'label': torch.tensor(label_id, dtype=torch.long),
                'domain_label': self.domain_label
            }


class PairedSingleModalitySampler:
    """
    配对采样器 - 单模态版本
    确保源域和目标域的每个batch中的样本属于相同类别
    """

    def __init__(self, src_ds, tgt_ds, batch_size):
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = batch_size

        # 构建类别索引
        self.src_indices = self._build_index(src_ds)
        self.tgt_indices = self._build_index(tgt_ds)

        # 找到共有类别
        common_classes = set(self.src_indices.keys()) & set(self.tgt_indices.keys())
        self.classes = sorted(list(common_classes))
        print(f"配对采样器已初始化，共 {len(self.classes)} 个共有类别。")

    def _build_index(self, dataset):
        from collections import defaultdict
        indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            indices[label].append(idx)
        return indices

    def __iter__(self):
        import random
        # 打乱各类别内部的索引
        for c in self.classes:
            random.shuffle(self.src_indices[c])
            random.shuffle(self.tgt_indices[c])

        # 计算batch数量
        min_len = min(len(self.src_ds), len(self.tgt_ds))
        n_batches = min_len // self.batch_size

        for _ in range(n_batches):
            batch_src_idxs = []
            batch_tgt_idxs = []

            # 随机选择类别
            batch_classes = random.choices(self.classes, k=self.batch_size)

            for c in batch_classes:
                s_idx = random.choice(self.src_indices[c])
                t_idx = random.choice(self.tgt_indices[c])

                batch_src_idxs.append(s_idx)
                batch_tgt_idxs.append(t_idx)

            # 打包成batch
            src_batch = torch.utils.data.default_collate([self.src_ds[i] for i in batch_src_idxs])
            tgt_batch = torch.utils.data.default_collate([self.tgt_ds[i] for i in batch_tgt_idxs])

            yield src_batch, tgt_batch

    def __len__(self):
        return min(len(self.src_ds), len(self.tgt_ds)) // self.batch_size
