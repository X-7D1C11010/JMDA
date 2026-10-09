"""Epoch-covering supervised source/target class pairing."""

import math
import random
from collections import defaultdict

import torch


class PairedClassSampler:
    """Cover every target sample once and pair it with same-class source data.

    The previous sampler drew both domains with replacement.  With the small
    target weather sets this left roughly one third of the labeled target
    observations unseen in an epoch.  Target indices are now shuffled without
    replacement; source indices cycle within the corresponding class.
    """

    def __init__(self, src_ds, tgt_ds, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = int(batch_size)
        self.src_indices = self._build_index(src_ds)
        self.tgt_indices = self._build_index(tgt_ds)
        self.classes = sorted(set(self.src_indices) & set(self.tgt_indices))
        if not self.classes:
            raise ValueError("Source and target domains have no common classes.")

        missing_target_classes = sorted(set(self.tgt_indices) - set(self.src_indices))
        if missing_target_classes:
            raise ValueError(
                "Supervised pairing cannot cover target-only classes: "
                f"{missing_target_classes}"
            )
        self.target_indices = [
            index
            for class_label in self.classes
            for index in self.tgt_indices[class_label]
        ]
        self.samples_per_class = max(
            len(self.tgt_indices[class_label]) for class_label in self.classes
        )
        self.epoch_sample_count = self.samples_per_class * len(self.classes)
        print(
            "配对采样器已初始化，共 "
            f"{len(self.classes)} 个共有类别；每个epoch覆盖全部 "
            f"{len(self.target_indices)} 个目标样本，并按类别平衡扩展为 "
            f"{self.epoch_sample_count} 个训练观测。"
        )

    @staticmethod
    def _build_index(dataset):
        indices = defaultdict(list)
        for index, label in enumerate(dataset.labels):
            indices[int(label)].append(index)
        return indices

    def __iter__(self):
        target_order = []
        for class_label in self.classes:
            pool = list(self.tgt_indices[class_label])
            random.shuffle(pool)
            repeats, remainder = divmod(self.samples_per_class, len(pool))
            target_order.extend(pool * repeats + pool[:remainder])
        random.shuffle(target_order)

        source_pools = {}
        source_offsets = {}
        for class_label in self.classes:
            pool = list(self.src_indices[class_label])
            random.shuffle(pool)
            source_pools[class_label] = pool
            source_offsets[class_label] = 0

        for start in range(0, len(target_order), self.batch_size):
            target_batch_indices = target_order[start:start + self.batch_size]
            source_batch_indices = []
            for target_index in target_batch_indices:
                class_label = int(self.tgt_ds.labels[target_index])
                pool = source_pools[class_label]
                offset = source_offsets[class_label]
                if offset >= len(pool):
                    random.shuffle(pool)
                    offset = 0
                source_batch_indices.append(pool[offset])
                source_offsets[class_label] = offset + 1

            source_batch = torch.utils.data.default_collate(
                [self.src_ds[index] for index in source_batch_indices]
            )
            target_batch = torch.utils.data.default_collate(
                [self.tgt_ds[index] for index in target_batch_indices]
            )
            target_batch["sample_index"] = torch.tensor(
                target_batch_indices,
                dtype=torch.long,
            )
            yield source_batch, target_batch

    def __len__(self):
        return math.ceil(self.epoch_sample_count / self.batch_size)


__all__ = ["PairedClassSampler"]
