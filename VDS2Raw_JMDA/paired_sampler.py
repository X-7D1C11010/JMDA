"""Supervised cross-domain pairing without class rebalancing."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, List, Sequence, Tuple

from torch.utils.data import default_collate


class SameClassPairedBatcher:
    """Cover each target sample once and pair it with a same-class source."""

    def __init__(self, source_dataset, target_dataset, batch_size: int, seed: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.source_by_class = self._index(source_dataset.labels)
        self.target_by_class = self._index(target_dataset.labels)
        missing = sorted(set(self.target_by_class) - set(self.source_by_class))
        if missing:
            raise ValueError(f"Target classes absent from source domain: {missing}")
        if not self.target_by_class:
            raise ValueError("Target training dataset is empty")

    @staticmethod
    def _index(labels: Sequence[int]) -> Dict[int, List[int]]:
        result: DefaultDict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            result[int(label)].append(index)
        return dict(result)

    @property
    def epoch_sample_count(self) -> int:
        return len(self.target_dataset)

    def __len__(self) -> int:
        return math.ceil(self.epoch_sample_count / self.batch_size)

    def pairs_for_epoch(self, epoch: int) -> List[Tuple[int, int]]:
        rng = random.Random(self.seed + int(epoch))
        source_pools = {
            label: list(indices) for label, indices in self.source_by_class.items()
        }
        source_offsets = {label: 0 for label in source_pools}
        for pool in source_pools.values():
            rng.shuffle(pool)
        target_indices = list(range(len(self.target_dataset)))
        rng.shuffle(target_indices)
        pairs: List[Tuple[int, int]] = []
        for target_index in target_indices:
            label = int(self.target_dataset.labels[target_index])
            pool = source_pools[label]
            offset = source_offsets[label]
            if offset >= len(pool):
                rng.shuffle(pool)
                offset = 0
            pairs.append((pool[offset], target_index))
            source_offsets[label] = offset + 1
        return pairs

    def iter_epoch(self, epoch: int) -> Iterator[Tuple[dict, dict]]:
        pairs = self.pairs_for_epoch(epoch)
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            source_batch = default_collate(
                [self.source_dataset[source_index] for source_index, _ in batch_pairs]
            )
            target_batch = default_collate(
                [self.target_dataset[target_index] for _, target_index in batch_pairs]
            )
            yield source_batch, target_batch


__all__ = ["SameClassPairedBatcher"]
