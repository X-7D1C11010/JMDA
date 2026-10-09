"""Leakage-safe VDS2Raw manifest and NPZ dataset implementation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ("Cargo", "Fishing", "Sailing & Pleasure")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
PARTITIONS = ("source_train", "target_train", "target_val", "target_test")
TRAIN_PARTITIONS = ("source_train", "target_train")
IMAGE_BAND_ORDER = ("B4", "B3", "B2", "B8")
REQUIRED_COLUMNS = {
    "sample_id",
    "partition",
    "label",
    "label_id",
    "npz_relpath",
    "scene",
    "day",
    "mmsi",
}


@dataclass(frozen=True)
class VDSRecord:
    sample_id: str
    partition: str
    label: str
    label_id: int
    npz_relpath: str
    scene: str
    day: str
    mmsi: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VDS2RawManifest:
    """Read and validate the fixed four-way VDS2Raw protocol."""

    def __init__(
        self,
        dataset_root: Union[Path, str],
        manifest_name: str = "dataset_manifest.csv",
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.manifest_path = (self.dataset_root / manifest_name).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"VDS2Raw manifest not found: {self.manifest_path}")
        self.records = self._read_records()
        self.by_partition: Dict[str, List[VDSRecord]] = {
            partition: [r for r in self.records if r.partition == partition]
            for partition in PARTITIONS
        }

    def _read_records(self) -> List[VDSRecord]:
        records: List[VDSRecord] = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(f"Manifest is missing required columns: {missing}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    label_id = int(row["label_id"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label_id at manifest row {row_number}: {row.get('label_id')!r}"
                    ) from exc
                records.append(
                    VDSRecord(
                        sample_id=row["sample_id"].strip(),
                        partition=row["partition"].strip(),
                        label=row["label"].strip(),
                        label_id=label_id,
                        npz_relpath=row["npz_relpath"].strip(),
                        scene=row["scene"].strip(),
                        day=row["day"].strip().lower(),
                        mmsi=row["mmsi"].strip(),
                    )
                )
        if not records:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        return records

    def resolve_npz(self, record: VDSRecord) -> Path:
        path = (self.dataset_root / record.npz_relpath).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"NPZ path escapes dataset root for sample {record.sample_id}: {path}"
            ) from exc
        return path

    def audit(self, validate_npz: bool = True) -> Dict[str, object]:
        errors: List[str] = []
        unknown_partitions = sorted({r.partition for r in self.records} - set(PARTITIONS))
        if unknown_partitions:
            errors.append(f"unknown partitions: {unknown_partitions}")

        sample_ids = [r.sample_id for r in self.records]
        npz_relpaths = [r.npz_relpath for r in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            errors.append("sample_id values are not unique")
        if len(npz_relpaths) != len(set(npz_relpaths)):
            errors.append("npz_relpath values are not unique")

        partition_counts: Dict[str, int] = {}
        class_counts: Dict[str, Dict[str, int]] = {}
        for partition in PARTITIONS:
            records = self.by_partition[partition]
            partition_counts[partition] = len(records)
            class_counts[partition] = {
                class_name: sum(r.label == class_name for r in records)
                for class_name in CLASS_NAMES
            }
            if not records:
                errors.append(f"partition {partition!r} is empty")
            missing_classes = [name for name, count in class_counts[partition].items() if count == 0]
            if missing_classes:
                errors.append(f"partition {partition!r} misses classes {missing_classes}")

        for record in self.records:
            expected_id = CLASS_TO_ID.get(record.label)
            if expected_id is None:
                errors.append(f"unknown label {record.label!r} for {record.sample_id}")
            elif record.label_id != expected_id:
                errors.append(
                    f"label mismatch for {record.sample_id}: {record.label!r} "
                    f"must map to {expected_id}, got {record.label_id}"
                )
            if record.partition == "source_train" and record.day != "day3":
                errors.append(f"source sample is not day3: {record.sample_id} ({record.day})")
            if record.partition != "source_train" and record.day == "day3":
                errors.append(f"target sample unexpectedly uses day3: {record.sample_id}")

        leakage: Dict[str, Dict[str, int]] = {"scene": {}, "mmsi": {}}
        for left, right in combinations(PARTITIONS, 2):
            key = f"{left}__{right}"
            for field in ("scene", "mmsi"):
                left_values = {
                    getattr(record, field)
                    for record in self.by_partition[left]
                    if getattr(record, field)
                }
                right_values = {
                    getattr(record, field)
                    for record in self.by_partition[right]
                    if getattr(record, field)
                }
                overlap = left_values & right_values
                leakage[field][key] = len(overlap)
                if overlap:
                    errors.append(f"{field} leakage in {key}: {sorted(overlap)[:5]}")

        schema_mismatches = 0
        missing_npz = 0
        if validate_npz:
            for record in self.records:
                path = self.resolve_npz(record)
                if not path.is_file():
                    missing_npz += 1
                    errors.append(f"missing NPZ: {path}")
                    continue
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        keys = set(payload.files)
                        if keys != {"rgb", "nir", "ais", "ais_mask", "label"}:
                            raise ValueError(f"unexpected keys {sorted(keys)}")
                        rgb = payload["rgb"]
                        nir = payload["nir"]
                        ais = payload["ais"]
                        ais_mask = payload["ais_mask"]
                        label = payload["label"]
                        if rgb.shape != (3, 96, 96) or rgb.dtype != np.uint16:
                            raise ValueError(f"rgb shape/dtype={rgb.shape}/{rgb.dtype}")
                        if nir.shape != (1, 96, 96) or nir.dtype != np.uint16:
                            raise ValueError(f"nir shape/dtype={nir.shape}/{nir.dtype}")
                        if ais.shape != (13,) or ais.dtype != np.float32:
                            raise ValueError(f"ais shape/dtype={ais.shape}/{ais.dtype}")
                        if ais_mask.shape != (13,) or ais_mask.dtype != np.float32:
                            raise ValueError(
                                f"ais_mask shape/dtype={ais_mask.shape}/{ais_mask.dtype}"
                            )
                        if not np.isfinite(ais).all():
                            raise ValueError("AIS contains non-finite values")
                        if not np.isin(ais_mask, (0.0, 1.0)).all():
                            raise ValueError("AIS mask is not binary")
                        if np.asarray(label).shape != () or int(label) != record.label_id:
                            raise ValueError(
                                f"NPZ label {np.asarray(label)!r} != manifest {record.label_id}"
                            )
                except (OSError, ValueError) as exc:
                    schema_mismatches += 1
                    errors.append(f"invalid NPZ for {record.sample_id}: {exc}")

        report: Dict[str, object] = {
            "status": "PASS" if not errors else "FAIL",
            "dataset_root": str(self.dataset_root),
            "manifest": str(self.manifest_path),
            "manifest_sha256": _sha256(self.manifest_path),
            "sample_count": len(self.records),
            "partition_counts": partition_counts,
            "class_counts_by_partition": class_counts,
            "scene_overlap_counts": leakage["scene"],
            "mmsi_overlap_counts": leakage["mmsi"],
            "missing_npz": missing_npz,
            "npz_schema_mismatches": schema_mismatches,
            "errors": errors,
        }
        if errors:
            raise ValueError("VDS2Raw audit failed:\n- " + "\n- ".join(errors[:30]))
        return report


def compute_train_image_stats(manifest: VDS2RawManifest) -> Dict[str, object]:
    """Fit per-band mean/std using training partitions and valid pixels only."""

    sums = np.zeros(4, dtype=np.float64)
    square_sums = np.zeros(4, dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    padded_pixels = 0
    total_pixels = 0
    records: List[VDSRecord] = []
    for partition in TRAIN_PARTITIONS:
        records.extend(manifest.by_partition[partition])
    if not records:
        raise ValueError("No source_train/target_train records available for image statistics.")

    for record in records:
        with np.load(manifest.resolve_npz(record), allow_pickle=False) as payload:
            image = np.concatenate([payload["rgb"], payload["nir"]], axis=0)
        valid = np.any(image != 0, axis=0)
        padded_pixels += int((~valid).sum())
        total_pixels += int(valid.size)
        for channel in range(4):
            values = image[channel][valid].astype(np.float64, copy=False)
            sums[channel] += values.sum(dtype=np.float64)
            square_sums[channel] += np.square(values).sum(dtype=np.float64)
            counts[channel] += values.size

    if np.any(counts == 0):
        raise ValueError(f"No valid pixels for one or more bands: {counts.tolist()}")
    means = sums / counts
    variances = np.maximum(square_sums / counts - np.square(means), 0.0)
    stds = np.sqrt(variances)
    if np.any(stds <= 0) or not np.isfinite(stds).all():
        raise ValueError(f"Invalid image standard deviations: {stds.tolist()}")
    return {
        "fit_partitions": list(TRAIN_PARTITIONS),
        "band_order": list(IMAGE_BAND_ORDER),
        "mean": means.tolist(),
        "std": stds.tolist(),
        "valid_pixel_count": counts.tolist(),
        "all_zero_padding_pixels": padded_pixels,
        "all_zero_padding_fraction": padded_pixels / max(total_pixels, 1),
        "padding_policy": "exclude from fit; restore normalized values to zero",
    }


class VDS2RawDataset(Dataset):
    """Load one immutable VDS2Raw protocol partition."""

    def __init__(
        self,
        manifest: VDS2RawManifest,
        partition: str,
        image_stats: Mapping[str, Sequence[float]],
    ):
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition {partition!r}; expected one of {PARTITIONS}")
        if tuple(image_stats.get("fit_partitions", ())) != TRAIN_PARTITIONS:
            raise ValueError(
                "Image statistics must be fitted on exactly source_train + target_train."
            )
        self.manifest = manifest
        self.partition = partition
        self.records = list(manifest.by_partition[partition])
        self.labels = [record.label_id for record in self.records]
        self.num_classes = len(CLASS_NAMES)
        self.rgb_mean = np.asarray(image_stats["mean"][:3], dtype=np.float32)[:, None, None]
        self.rgb_std = np.asarray(image_stats["std"][:3], dtype=np.float32)[:, None, None]
        self.nir_mean = np.asarray(image_stats["mean"][3:], dtype=np.float32)[:, None, None]
        self.nir_std = np.asarray(image_stats["std"][3:], dtype=np.float32)[:, None, None]
        if self.rgb_mean.shape != (3, 1, 1) or self.nir_mean.shape != (1, 1, 1):
            raise ValueError("Image statistics must contain exactly four bands in B4/B3/B2/B8 order.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        with np.load(self.manifest.resolve_npz(record), allow_pickle=False) as payload:
            rgb_raw = payload["rgb"].astype(np.float32)
            nir_raw = payload["nir"].astype(np.float32)
            ais = payload["ais"].astype(np.float32)
            ais_mask = payload["ais_mask"].astype(np.float32)
            label = int(payload["label"])
        if label != record.label_id:
            raise ValueError(
                f"Runtime label mismatch for {record.sample_id}: NPZ={label}, manifest={record.label_id}"
            )
        padding_mask = np.all(np.concatenate([rgb_raw, nir_raw], axis=0) == 0, axis=0)
        rgb = (rgb_raw - self.rgb_mean) / self.rgb_std
        nir = (nir_raw - self.nir_mean) / self.nir_std
        rgb[:, padding_mask] = 0.0
        nir[:, padding_mask] = 0.0
        ais_input = np.concatenate([ais, ais_mask], axis=0)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
            "nir": torch.from_numpy(np.ascontiguousarray(nir)),
            "ais": torch.from_numpy(np.ascontiguousarray(ais)),
            "ais_mask": torch.from_numpy(np.ascontiguousarray(ais_mask)),
            "ais_input": torch.from_numpy(np.ascontiguousarray(ais_input)),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "sample_id": record.sample_id,
            "partition": record.partition,
        }


def write_json(path: Union[Path, str], payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "PARTITIONS",
    "TRAIN_PARTITIONS",
    "IMAGE_BAND_ORDER",
    "VDSRecord",
    "VDS2RawManifest",
    "VDS2RawDataset",
    "compute_train_image_stats",
    "write_json",
]
