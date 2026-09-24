"""Epoch-boundary feature collection for Tensor SVD projection updates."""

from collections import defaultdict
import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import default_collate


def _indices_by_class(dataset) -> Dict[int, List[int]]:
    indices: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(dataset.labels):
        indices[int(label)].append(index)
    return indices


def build_epoch_pairs(
    source_dataset,
    target_dataset,
    seed: int,
    class_paired: bool,
) -> List[Tuple[int, int]]:
    """Return deterministic pairs while covering each target sample once.

    In supervised adaptation, every target sample is paired with a source
    sample of the same class.  Source samples cycle within a class when the
    domain sizes differ.  The unpaired fallback never inspects class labels.
    """
    rng = random.Random(int(seed))

    if not class_paired:
        source_indices = list(range(len(source_dataset)))
        target_indices = list(range(len(target_dataset)))
        if not source_indices or not target_indices:
            raise ValueError("Source and target datasets must both be non-empty.")
        rng.shuffle(source_indices)
        rng.shuffle(target_indices)
        return [
            (source_indices[position % len(source_indices)], target_index)
            for position, target_index in enumerate(target_indices)
        ]

    source_by_class = _indices_by_class(source_dataset)
    target_by_class = _indices_by_class(target_dataset)
    target_classes = set(target_by_class)
    missing_classes = sorted(target_classes - set(source_by_class))
    if missing_classes:
        raise ValueError(
            "Cannot construct supervised SVD statistics because target "
            f"classes are absent from the source domain: {missing_classes}."
        )

    pairs: List[Tuple[int, int]] = []
    for class_label in sorted(target_classes):
        source_indices = list(source_by_class[class_label])
        target_indices = list(target_by_class[class_label])
        rng.shuffle(source_indices)
        rng.shuffle(target_indices)
        pairs.extend(
            (
                source_indices[position % len(source_indices)],
                target_index,
            )
            for position, target_index in enumerate(target_indices)
        )

    rng.shuffle(pairs)
    if not pairs:
        raise ValueError("No source-target pairs were available for the SVD update.")
    return pairs


@torch.no_grad()
def update_epoch_projections(
    tal_module,
    encoders: Sequence[torch.nn.Module],
    modality_keys: Sequence[str],
    source_dataset,
    target_dataset,
    batch_size: int,
    device: torch.device,
    seed: int,
    class_paired: bool = True,
):
    """Extract a complete feature bank and perform one projection update."""
    if len(encoders) != len(modality_keys):
        raise ValueError("encoders and modality_keys must have the same length.")
    if len(encoders) != tal_module.num_modalities:
        raise ValueError(
            "The number of encoders must match tal_module.num_modalities."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    pairs = build_epoch_pairs(
        source_dataset,
        target_dataset,
        seed=seed,
        class_paired=class_paired,
    )
    previous_training_states = [encoder.training for encoder in encoders]
    for encoder in encoders:
        encoder.eval()

    source_features: List[List[torch.Tensor]] = [[] for _ in encoders]
    target_features: List[List[torch.Tensor]] = [[] for _ in encoders]

    try:
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            source_batch = default_collate(
                [source_dataset[source_index] for source_index, _ in batch_pairs]
            )
            target_batch = default_collate(
                [target_dataset[target_index] for _, target_index in batch_pairs]
            )

            for mode, (encoder, key) in enumerate(zip(encoders, modality_keys)):
                source_input = source_batch[key].to(device, non_blocking=True)
                target_input = target_batch[key].to(device, non_blocking=True)
                source_features[mode].append(encoder(source_input).float())
                target_features[mode].append(encoder(target_input).float())
    finally:
        for encoder, was_training in zip(encoders, previous_training_states):
            encoder.train(was_training)

    source_bank = [torch.cat(chunks, dim=0) for chunks in source_features]
    target_bank = [torch.cat(chunks, dim=0) for chunks in target_features]
    update_info = tal_module.update_projections(source_bank, target_bank)
    update_info["class_paired"] = bool(class_paired)
    return update_info


def format_svd_update(update_info: Dict[str, object]) -> str:
    """Create a compact, reproducible projection-update log line."""
    relative_change = float(update_info["relative_singular_change"])
    change_text = "inf" if relative_change == float("inf") else f"{relative_change:.3e}"
    return (
        f"SVD update #{update_info['update_count']}: "
        f"pairs={update_info['sample_count']}, sweeps={update_info['sweeps']}, "
        f"converged={update_info['converged']}, rel_change={change_text}, "
        f"effective_ranks={update_info['effective_ranks']}, "
        f"class_paired={update_info['class_paired']}"
    )


__all__ = [
    "build_epoch_pairs",
    "update_epoch_projections",
    "format_svd_update",
]
