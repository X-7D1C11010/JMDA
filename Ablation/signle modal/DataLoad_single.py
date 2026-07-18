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
                 img_size=224, weather=None, global_label_map=None, ais_data_path=None,
                 ais_allowed_labels=None, ais_split_seed=42, ais_augment=True):
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
            ais_allowed_labels: 当前天气中实际存在的AIS类别；None表示保留全部类别
            ais_split_seed: AIS分层训练/验证划分的固定随机种子
            ais_augment: 是否在AIS训练样本上应用相位/噪声增强
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
        self.ais_allowed_labels = (
            None if ais_allowed_labels is None
            else {int(label) for label in ais_allowed_labels}
        )
        self.ais_split_seed = int(ais_split_seed)
        self.ais_augment = bool(ais_augment)

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
        """加载AIS信号数据，兼容 MAT/HDF5/NPY/NPZ/ASCII 数值矩阵。"""
        if not self.ais_data_path or not os.path.exists(self.ais_data_path):
            raise FileNotFoundError(f"AIS数据文件不存在: {self.ais_data_path}")

        def is_git_lfs_pointer(path):
            try:
                with open(path, 'rb') as f:
                    head = f.read(80)
                return head.startswith(b'version https://git-lfs.github.com/spec')
            except OSError:
                return False

        def read_git_lfs_oid(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('oid sha256:'):
                            return line.split(':', 1)[1]
            except OSError:
                pass
            return None

        def find_real_ais_file(pointer_path):
            pointer_path = os.path.abspath(pointer_path)
            search_roots = []
            ais_dir = os.path.dirname(pointer_path)
            data_root = os.path.dirname(ais_dir)
            for root in (ais_dir, data_root, os.getcwd()):
                if root and os.path.isdir(root) and root not in search_roots:
                    search_roots.append(root)

            oid = read_git_lfs_oid(pointer_path)
            if oid and len(oid) >= 4:
                ancestors = []
                for root in (pointer_path, os.getcwd()):
                    cur = os.path.abspath(root if os.path.isdir(root) else os.path.dirname(root))
                    while cur and cur not in ancestors:
                        ancestors.append(cur)
                        parent = os.path.dirname(cur)
                        if parent == cur:
                            break
                        cur = parent
                for ancestor in ancestors:
                    lfs_object = os.path.join(ancestor, '.git', 'lfs', 'objects', oid[:2], oid[2:4], oid)
                    if os.path.isfile(lfs_object) and not is_git_lfs_pointer(lfs_object):
                        return lfs_object

            exts = ('.mat', '.h5', '.hdf5', '.npz', '.npy', '.csv', '.txt')
            candidates = []
            for root in search_roots:
                for dirpath, _, filenames in os.walk(root):
                    if os.path.abspath(dirpath).startswith(os.path.abspath(os.path.join(data_root, 'train'))):
                        continue
                    for filename in filenames:
                        if not filename.lower().endswith(exts):
                            continue
                        candidate = os.path.abspath(os.path.join(dirpath, filename))
                        if candidate == pointer_path:
                            continue
                        if 'ais' not in candidate.lower() and 'balanced' not in filename.lower():
                            continue
                        if not is_git_lfs_pointer(candidate):
                            candidates.append(candidate)
                if candidates:
                    break
            return sorted(candidates, key=lambda p: (0 if 'balanced' in os.path.basename(p).lower() else 1, p))[0] if candidates else None

        if is_git_lfs_pointer(self.ais_data_path):
            real_path = find_real_ais_file(self.ais_data_path)
            if real_path:
                print(f"AIS路径是Git LFS指针，自动切换到真实数据文件: {real_path}")
                self.ais_data_path = real_path
            else:
                raise FileNotFoundError(
                    "AIS数据文件是Git LFS指针文件，不包含真实数据。请在服务器执行 git lfs pull，"
                    "或通过 --ais_data_path / JMDA_AIS_DATA_PATH 指向真实AIS数据文件。"
                )

        def is_label_like(values):
            values = np.asarray(values).reshape(-1)
            if values.size == 0 or not np.all(np.isfinite(values)):
                return False
            if not np.allclose(values, np.round(values), atol=1e-4):
                return False
            unique_count = len(np.unique(values.astype(np.int64)))
            return 1 < unique_count <= min(200, max(2, values.size // 2))

        def align_features(features, labels, source_name):
            features = np.asarray(features, dtype=np.float32)
            labels = np.asarray(labels).reshape(-1)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            elif features.ndim > 2:
                sample_axis = next((i for i, s in enumerate(features.shape) if s == labels.size), None)
                if sample_axis is None:
                    raise ValueError(f"{source_name}: cannot infer sample axis from {features.shape}")
                features = np.moveaxis(features, sample_axis, 0).reshape(labels.size, -1)

            if features.ndim != 2:
                raise ValueError(f"{source_name}: features must be 2D after reshape, got {features.shape}")
            if features.shape[0] == labels.size:
                return features, labels
            if features.shape[1] == labels.size:
                return features.T, labels
            raise ValueError(
                f"{source_name}: feature/label length mismatch: "
                f"features.shape={features.shape}, labels.shape={labels.shape}"
            )

        def split_packed_array(arr, source_name):
            arr = np.asarray(arr)
            if arr.ndim != 2:
                raise ValueError(f"{source_name}: packed AIS array must be 2D, got {arr.shape}")
            if arr.shape[0] < 2 or arr.shape[1] < 2:
                raise ValueError(f"{source_name}: packed AIS array is too small: {arr.shape}")

            candidates = [
                ("last_col", arr[:, :-1], arr[:, -1]),
                ("first_col", arr[:, 1:], arr[:, 0]),
                ("last_row", arr[:-1, :].T, arr[-1, :]),
                ("first_row", arr[1:, :].T, arr[0, :]),
            ]
            for name, features, labels in candidates:
                if is_label_like(labels):
                    print(f"AIS packed array inferred as {source_name}:{name}")
                    return align_features(features, labels, f"{source_name}:{name}")
            raise ValueError(f"{source_name}: cannot infer label column/row from shape {arr.shape}")

        def load_balanced_iq(container, source_name):
            keys = ('balanced_rcv_I', 'balanced_rcv_Q', 'new_balanced_label')
            if not all(k in container for k in keys):
                return None
            labels = np.asarray(container['new_balanced_label']).reshape(-1)
            i_features, labels = align_features(container['balanced_rcv_I'], labels, f"{source_name}:I")
            q_features, labels_q = align_features(container['balanced_rcv_Q'], labels, f"{source_name}:Q")
            if labels_q.shape[0] != labels.shape[0]:
                raise ValueError(f"{source_name}: I/Q labels mismatch")
            # Receiver gain differs considerably between records. Normalize one
            # I/Q record by its joint RMS while preserving phase and waveform
            # structure; this prevents the high-dimensional MLP/CNN from using
            # amplitude scale as a sample identifier.
            rms = np.sqrt(np.mean(i_features ** 2 + q_features ** 2, axis=1, keepdims=True) + 1e-8)
            i_features = i_features / rms
            q_features = q_features / rms
            print(f"AIS {source_name} keys: data=balanced_rcv_I+balanced_rcv_Q, labels=new_balanced_label")
            return np.concatenate([i_features, q_features], axis=1), labels

        def load_scipy_mat():
            mat_data = sio.loadmat(self.ais_data_path)
            balanced = load_balanced_iq(mat_data, "MAT")
            if balanced is not None:
                return balanced
            if 'data' in mat_data and 'labels' in mat_data:
                return align_features(mat_data['data'], mat_data['labels'], "MAT:data/labels")

            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"可用的MAT文件键: {keys}")
            label_key = next((k for k in keys if 'label' in k.lower() or k.lower() in {'y', 'target'}), None)
            data_key = next((k for k in keys if k != label_key), None)
            if data_key and label_key:
                return align_features(mat_data[data_key], mat_data[label_key], f"MAT:{data_key}/{label_key}")
            if data_key:
                return split_packed_array(mat_data[data_key], f"MAT:{data_key}")
            raise ValueError("无法从MAT文件中提取AIS数据")

        def load_hdf5_mat():
            if h5py is None:
                raise ImportError("h5py is not installed")
            with h5py.File(self.ais_data_path, 'r') as f:
                keys = list(f.keys())
                print(f"检测到HDF5/MAT键: {keys}")
                balanced = load_balanced_iq(f, "HDF5")
                if balanced is not None:
                    return balanced

                data_keys = ['data', 'X', 'features', 'feat']
                label_keys = ['labels', 'label', 'y', 'Y', 'target', 'targets']
                data_key = next((k for k in data_keys if k in f), None)
                label_key = next((k for k in label_keys if k in f), None)
                if data_key and label_key:
                    return align_features(np.array(f[data_key]), np.array(f[label_key]), f"HDF5:{data_key}/{label_key}")

                arrays = {k: np.array(f[k]) for k in keys if hasattr(f[k], 'shape')}
                one_d = [(k, v) for k, v in arrays.items() if v.ndim == 1 or (v.ndim == 2 and 1 in v.shape)]
                two_d = [(k, v) for k, v in arrays.items() if v.ndim == 2]
                if one_d and two_d:
                    label_key, labels = sorted(one_d, key=lambda kv: 0 if 'label' in kv[0].lower() else 1)[0]
                    data_candidates = [(k, v) for k, v in two_d if k != label_key]
                    if data_candidates:
                        data_key, features = data_candidates[0]
                        return align_features(features, labels, f"HDF5:{data_key}/{label_key}")
                if two_d:
                    key, arr = two_d[0]
                    return split_packed_array(arr, f"HDF5:{key}")
                raise ValueError(f"无法从HDF5文件中提取AIS数据，可用键: {keys}")

        def load_numpy_or_text():
            path_lower = self.ais_data_path.lower()
            if path_lower.endswith('.npz'):
                data = np.load(self.ais_data_path)
                keys = list(data.keys())
                balanced = load_balanced_iq(data, "NPZ")
                if balanced is not None:
                    return balanced
                label_key = next((k for k in keys if 'label' in k.lower() or k.lower() in {'y', 'target'}), None)
                data_key = next((k for k in keys if k != label_key), None)
                if data_key and label_key:
                    return align_features(data[data_key], data[label_key], f"NPZ:{data_key}/{label_key}")
                if data_key:
                    return split_packed_array(data[data_key], f"NPZ:{data_key}")
            if path_lower.endswith('.npy'):
                return split_packed_array(np.load(self.ais_data_path), "NPY")

            delimiters = [None, ',', '\t', ';']
            text_errors = []
            for delimiter in delimiters:
                try:
                    arr = np.genfromtxt(self.ais_data_path, delimiter=delimiter)
                    if arr.ndim == 2 and arr.size > 0 and np.isfinite(arr).any():
                        return split_packed_array(arr, f"TEXT delimiter={delimiter!r}")
                except Exception as exc:
                    text_errors.append(f"{delimiter!r}: {exc}")
            with open(self.ais_data_path, 'rb') as f:
                head = f.read(80)
            raise ValueError(
                "文本/CSV/NPY兜底读取失败; "
                f"file_head={head!r}; errors={text_errors}"
            )

        errors = []
        ais_features = None
        ais_labels = None
        for loader in (load_scipy_mat, load_hdf5_mat, load_numpy_or_text):
            try:
                ais_features, ais_labels = loader()
                print(f"AIS loader used: {loader.__name__}")
                break
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")

        if ais_features is None or ais_labels is None:
            raise ValueError("无法加载AIS数据；已尝试 MAT/HDF5/NPY/NPZ/TEXT。 " + " | ".join(errors))

        ais_features = np.asarray(ais_features, dtype=np.float32)
        ais_labels = np.asarray(ais_labels).reshape(-1)

        if not np.all(np.isfinite(ais_features)):
            raise ValueError("AIS特征包含NaN或Inf")
        if not np.allclose(ais_labels, np.round(ais_labels), atol=1e-4):
            raise ValueError("AIS标签必须为整数编码")
        ais_labels = np.round(ais_labels).astype(np.int64)

        # Perform a deterministic stratified 80/20 split. The previous global
        # shuffle could omit rare classes from validation and made weather-wise
        # comparisons unstable. Train and validation indices remain disjoint.
        rng = np.random.RandomState(self.ais_split_seed)
        train_indices = []
        val_indices = []
        for label in sorted(np.unique(ais_labels).tolist()):
            class_indices = np.flatnonzero(ais_labels == label)
            rng.shuffle(class_indices)
            val_count = max(1, int(round(0.2 * len(class_indices))))
            if len(class_indices) > 1:
                val_count = min(val_count, len(class_indices) - 1)
            train_indices.extend(class_indices[:-val_count].tolist())
            val_indices.extend(class_indices[-val_count:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        selected_indices = train_indices if self.phase == 'train' else val_indices
        if self.ais_allowed_labels is not None:
            selected_indices = [
                idx for idx in selected_indices
                if int(ais_labels[idx]) in self.ais_allowed_labels
            ]
        if not selected_indices:
            raise ValueError(
                f"AIS {self.phase} split is empty after weather label filtering: "
                f"allowed={sorted(self.ais_allowed_labels or [])}"
            )

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
            if self.phase == 'train' and self.ais_augment and data.numel() % 2 == 0:
                # Apply the same temporal offset and phase rotation to both I/Q
                # halves, followed by weak receiver noise. Independent I/Q
                # transforms would destroy the underlying complex waveform.
                half = data.numel() // 2
                i_part, q_part = data[:half], data[half:]
                max_shift = min(256, max(1, half // 32))
                shift = int(torch.randint(-max_shift, max_shift + 1, ()).item())
                i_part = torch.roll(i_part, shifts=shift, dims=0)
                q_part = torch.roll(q_part, shifts=shift, dims=0)
                theta = torch.rand((), dtype=data.dtype) * (2.0 * np.pi)
                cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                data = torch.cat([
                    cos_theta * i_part - sin_theta * q_part,
                    sin_theta * i_part + cos_theta * q_part,
                ])
                data = data + 0.005 * torch.randn_like(data)
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

        # Guarantee enough updates for small targets without making each of the
        # five repeated runs unnecessarily long.
        smaller_batches = max(
            1,
            (min(len(self.src_ds), len(self.tgt_ds)) + self.batch_size - 1)
            // self.batch_size,
        )
        larger_batches = max(
            1,
            (max(len(self.src_ds), len(self.tgt_ds)) + self.batch_size - 1)
            // self.batch_size,
        )
        n_batches = min(larger_batches, max(8, smaller_batches))

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
        smaller_batches = max(
            1,
            (min(len(self.src_ds), len(self.tgt_ds)) + self.batch_size - 1)
            // self.batch_size,
        )
        larger_batches = max(
            1,
            (max(len(self.src_ds), len(self.tgt_ds)) + self.batch_size - 1)
            // self.batch_size,
        )
        return min(larger_batches, max(8, smaller_batches))
