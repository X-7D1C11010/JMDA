import random
import torch
from collections import defaultdict


class PairedClassSampler:
    def __init__(self, src_ds, tgt_ds, batch_size):
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = batch_size

        self.src_indices = self._build_index(src_ds)
        self.tgt_indices = self._build_index(tgt_ds)
        self.classes = sorted(list(self.src_indices.keys()))

        common_classes = set(self.src_indices.keys()) & set(self.tgt_indices.keys())
        self.classes = sorted(list(common_classes))
        print(f"配对采样器已初始化，共 {len(self.classes)} 个共有类别。")

    def _build_index(self, dataset):
        indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            indices[label].append(idx)
        return indices

    def __iter__(self):
        for c in self.classes:
            random.shuffle(self.src_indices[c])
            random.shuffle(self.tgt_indices[c])

        min_len = min(len(self.src_ds), len(self.tgt_ds))
        n_batches = min_len // self.batch_size

        for _ in range(n_batches):
            batch_src_idxs = []
            batch_tgt_idxs = []

            batch_classes = random.choices(self.classes, k=self.batch_size)

            for c in batch_classes:
                s_idx = random.choice(self.src_indices[c])
                t_idx = random.choice(self.tgt_indices[c])

                batch_src_idxs.append(s_idx)
                batch_tgt_idxs.append(t_idx)

            src_batch = torch.utils.data.default_collate([self.src_ds[i] for i in batch_src_idxs])
            tgt_batch = torch.utils.data.default_collate([self.tgt_ds[i] for i in batch_tgt_idxs])

            yield src_batch, tgt_batch

    def __len__(self):
        return min(len(self.src_ds), len(self.tgt_ds)) // self.batch_size