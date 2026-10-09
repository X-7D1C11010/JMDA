"""Strictly paired VIS/IR dataset with synchronized augmentation."""

import os
import hashlib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from DataLoad import MultiModalDomainDataset


class PairedModalTransform:
    """Sample crop/flip once and apply it to both registered modalities."""

    def __init__(self, phase="train", val_augment=False):
        self.augment = phase == "train" or bool(val_augment)
        self.vis_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.ir_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

    def __call__(self, vis_img, ir_img):
        if self.augment:
            vis_img = TF.resize(vis_img, [256, 256])
            ir_img = TF.resize(ir_img, [256, 256])
            top, left, height, width = transforms.RandomCrop.get_params(
                vis_img,
                output_size=(224, 224),
            )
            vis_img = TF.crop(vis_img, top, left, height, width)
            ir_img = TF.crop(ir_img, top, left, height, width)
            if bool(torch.rand(()) < 0.5):
                vis_img = TF.hflip(vis_img)
                ir_img = TF.hflip(ir_img)
        else:
            vis_img = TF.resize(vis_img, [224, 224])
            ir_img = TF.resize(ir_img, [224, 224])

        return (
            self.vis_normalize(TF.to_tensor(vis_img)),
            self.ir_normalize(TF.to_tensor(ir_img)),
        )


class PairedMultiModalDomainDataset(MultiModalDomainDataset):
    """Match modalities by filename stem and apply paired transformations."""

    def __init__(self, *args, **kwargs):
        root_dir = args[0] if args else kwargs["root_dir"]
        domain_type = kwargs.get(
            "domain_type",
            args[1] if len(args) > 1 else "source",
        )
        phase = kwargs.get("phase", args[2] if len(args) > 2 else "train")
        super().__init__(*args, **kwargs)
        self.samples, self.pairing_stats = self._build_consistent_pairs(
            root_dir=root_dir,
            domain_type=domain_type,
            phase=phase,
        )
        if not self.samples:
            raise ValueError(
                f"No exact VIS/IR filename pairs found under {root_dir!r}/{phase}."
            )
        self.labels = [sample["label"] for sample in self.samples]
        self.unique_labels = sorted(np.unique(self.labels))
        self.num_classes = len(self.label_map)
        transform_phase = "val" if self.deterministic_transform else self.phase
        transform_val_augment = (
            False if self.deterministic_transform else self.val_augment
        )
        self.paired_transform = PairedModalTransform(
            phase=transform_phase,
            val_augment=transform_val_augment,
        )

    @staticmethod
    def _image_map(directory):
        valid_extensions = (".jpg", ".jpeg", ".png")
        return {
            os.path.splitext(filename)[0]: os.path.join(directory, filename)
            for filename in sorted(os.listdir(directory))
            if filename.lower().endswith(valid_extensions)
        }

    @staticmethod
    def _modal_dir(root_dir, domain_type, phase, class_name, modality):
        if domain_type == "source":
            return os.path.join(root_dir, phase, modality, class_name)
        return os.path.join(root_dir, phase, class_name, modality)

    def _build_consistent_pairs(self, root_dir, domain_type, phase):
        """Recover pairs from the train/val union, then split pairs together.

        The supplied dataset split assigned VIS and IR files independently.
        Consequently, positional pairing mixed observations and the two
        modalities of one vessel could land on opposite sides of train/val.
        We reconstruct exact filename-stem pairs from the union and use a
        stable hash to choose the original per-class validation count.
        """
        if phase not in {"train", "val"}:
            raise ValueError(f"Unsupported phase for paired split: {phase!r}")

        class_names = set()
        for split in ("train", "val"):
            if domain_type == "source":
                class_root = os.path.join(root_dir, split, "可见光")
            else:
                class_root = os.path.join(root_dir, split)
            if os.path.isdir(class_root):
                class_names.update(
                    name
                    for name in os.listdir(class_root)
                    if os.path.isdir(os.path.join(class_root, name))
                )

        samples = []
        stats = {
            "union_vis_files": 0,
            "union_ir_files": 0,
            "matched_pairs_all": 0,
            "selected_phase_pairs": 0,
            "dropped_unpaired_vis": 0,
            "dropped_unpaired_ir": 0,
        }
        for class_name in sorted(class_names):
            vis_by_stem = {}
            ir_by_stem = {}
            raw_val_vis = {}
            raw_val_ir = {}
            for split in ("train", "val"):
                vis_dir = self._modal_dir(
                    root_dir, domain_type, split, class_name, "可见光"
                )
                ir_dir = self._modal_dir(
                    root_dir, domain_type, split, class_name, "红外"
                )
                split_vis = self._image_map(vis_dir) if os.path.isdir(vis_dir) else {}
                split_ir = self._image_map(ir_dir) if os.path.isdir(ir_dir) else {}
                vis_by_stem.update(split_vis)
                ir_by_stem.update(split_ir)
                if split == "val":
                    raw_val_vis = split_vis
                    raw_val_ir = split_ir

            common_stems = set(vis_by_stem) & set(ir_by_stem)
            desired_val_count = min(
                len(raw_val_vis),
                len(raw_val_ir),
                len(common_stems),
            )
            ranked_stems = sorted(
                common_stems,
                key=lambda stem: hashlib.sha256(
                    f"{os.path.basename(root_dir)}:{class_name}:{stem}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            selected_stems = (
                ranked_stems[:desired_val_count]
                if phase == "val"
                else ranked_stems[desired_val_count:]
            )
            stats["union_vis_files"] += len(vis_by_stem)
            stats["union_ir_files"] += len(ir_by_stem)
            stats["matched_pairs_all"] += len(common_stems)
            stats["dropped_unpaired_vis"] += len(vis_by_stem) - len(common_stems)
            stats["dropped_unpaired_ir"] += len(ir_by_stem) - len(common_stems)
            stats["selected_phase_pairs"] += len(selected_stems)
            samples.extend(
                {
                    "vis": vis_by_stem[stem],
                    "ir": ir_by_stem[stem],
                    "label": int(class_name),
                }
                for stem in selected_stems
            )
        return samples, stats

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            vis_img = Image.open(sample["vis"]).convert("RGB")
            ir_img = Image.open(sample["ir"]).convert("L").convert("RGB")
        except (OSError, ValueError):
            if len(self) <= 1:
                raise
            return self.__getitem__((idx + 1) % len(self))

        vis_tensor, ir_tensor = self.paired_transform(vis_img, ir_img)
        label_id = self.label_map.get(sample["label"], 0)
        return {
            "vis": vis_tensor,
            "ir": ir_tensor,
            "label": torch.tensor(label_id, dtype=torch.long),
            "domain_label": self.domain_label,
        }


__all__ = ["PairedModalTransform", "PairedMultiModalDomainDataset"]
