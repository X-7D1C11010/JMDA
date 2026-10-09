"""Training and evaluation engine for the dedicated VDS2Raw JMDA-Net path."""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Discriminator import (  # noqa: E402
    DomainDiscriminator,
    GradientReversal,
    compute_discriminator_loss,
    compute_generator_loss,
)
from Generator import NeuralOptimalTransportGenerator  # noqa: E402
from Tensor import TensorBasedAlignmentStable  # noqa: E402
from epoch_svd import format_svd_update, update_epoch_projections  # noqa: E402

try:  # noqa: E402
    from .models import (
        AISFeatureExtractor,
        BinaryDomainDiscriminator,
        Classifier,
        FEATURE_DIM,
        NIRFeatureExtractor,
        RGBFeatureExtractor,
    )
    from .paired_sampler import SameClassPairedBatcher
    from .training_utils import (
        binary_domain_loss,
        compute_joint_ot_scale,
        seed_everything,
        set_requires_grad,
        transport_projected_feature_basis,
    )
    from .vds2raw_dataset import CLASS_NAMES, VDS2RawDataset, write_json
except ImportError:
    from models import (  # type: ignore
        AISFeatureExtractor,
        BinaryDomainDiscriminator,
        Classifier,
        FEATURE_DIM,
        NIRFeatureExtractor,
        RGBFeatureExtractor,
    )
    from paired_sampler import SameClassPairedBatcher  # type: ignore
    from training_utils import (  # type: ignore
        binary_domain_loss,
        compute_joint_ot_scale,
        seed_everything,
        set_requires_grad,
        transport_projected_feature_basis,
    )
    from vds2raw_dataset import CLASS_NAMES, VDS2RawDataset, write_json  # type: ignore


ABLATION_SPECS = {
    "full": {"use_tensor": True, "use_ot": True},
    "no_tensor_no_ot": {"use_tensor": False, "use_ot": False},
    "no_tensor_with_ot": {"use_tensor": False, "use_ot": True},
    "with_tensor_no_ot": {"use_tensor": True, "use_ot": False},
}


@dataclass
class ExperimentModules:
    rgb: RGBFeatureExtractor
    nir: NIRFeatureExtractor
    ais: AISFeatureExtractor
    classifier: Classifier
    discriminator: nn.Module
    tensor: Optional[TensorBasedAlignmentStable]
    generator: Optional[NeuralOptimalTransportGenerator]
    fused_dim: int

    def encoders(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        return self.rgb, self.nir, self.ais

    def state_dicts(self) -> Dict[str, Mapping[str, torch.Tensor]]:
        states = {
            "rgb": self.rgb.state_dict(),
            "nir": self.nir.state_dict(),
            "ais": self.ais.state_dict(),
            "classifier": self.classifier.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }
        if self.tensor is not None:
            states["tensor"] = self.tensor.state_dict()
        if self.generator is not None:
            states["generator"] = self.generator.state_dict()
        return states

    def load_state_dicts(self, states: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.rgb.load_state_dict(states["rgb"])
        self.nir.load_state_dict(states["nir"])
        self.ais.load_state_dict(states["ais"])
        self.classifier.load_state_dict(states["classifier"])
        self.discriminator.load_state_dict(states["discriminator"])
        if self.tensor is not None:
            self.tensor.load_state_dict(states["tensor"])
        if self.generator is not None:
            self.generator.load_state_dict(states["generator"])

    def train(self) -> None:
        for module in self.encoders():
            module.train()
        self.classifier.train()
        self.discriminator.train()
        if self.tensor is not None:
            self.tensor.train()
        if self.generator is not None:
            self.generator.train()

    def eval(self) -> None:
        for module in self.encoders():
            module.eval()
        self.classifier.eval()
        self.discriminator.eval()
        if self.tensor is not None:
            self.tensor.eval()
        if self.generator is not None:
            self.generator.eval()


def build_modules(args, spec: Mapping[str, bool], device: torch.device) -> ExperimentModules:
    rgb = RGBFeatureExtractor(
        output_dim=FEATURE_DIM,
        pretrained=args.pretrained,
        pretrained_weights_path=args.pretrained_weights,
    )
    if args.freeze_rgb_early:
        rgb.freeze_early_stages()
    nir = NIRFeatureExtractor(output_dim=FEATURE_DIM)
    ais = AISFeatureExtractor(output_dim=FEATURE_DIM)

    if spec["use_tensor"]:
        tensor = TensorBasedAlignmentStable(
            input_dims=[FEATURE_DIM, FEATURE_DIM, FEATURE_DIM],
            output_dims=[args.projection_dim] * 3,
            num_modalities=3,
            max_svd_sweeps=args.svd_max_sweeps,
            svd_tolerance=args.svd_tolerance,
        )
        fused_dim = args.projection_dim * 3
    else:
        tensor = None
        fused_dim = FEATURE_DIM * 3

    classifier = Classifier(fused_dim, len(CLASS_NAMES))
    if spec["use_ot"]:
        generator = NeuralOptimalTransportGenerator(
            feature_dim=fused_dim,
            transport_mode=args.transport_mode,
            epsilon=args.ot_epsilon,
            sinkhorn_iterations=args.ot_sinkhorn_iterations,
            correction_scale=args.ot_correction_scale,
        )
        discriminator = DomainDiscriminator(feature_dim=fused_dim)
    else:
        generator = None
        discriminator = BinaryDomainDiscriminator(feature_dim=fused_dim)

    modules = ExperimentModules(
        rgb=rgb,
        nir=nir,
        ais=ais,
        classifier=classifier,
        discriminator=discriminator,
        tensor=tensor,
        generator=generator,
        fused_dim=fused_dim,
    )
    for module in (
        modules.rgb,
        modules.nir,
        modules.ais,
        modules.classifier,
        modules.discriminator,
        modules.tensor,
        modules.generator,
    ):
        if module is not None:
            module.to(device)
    return modules


def encode_batch(
    modules: ExperimentModules,
    batch: Mapping[str, object],
    device: torch.device,
) -> List[torch.Tensor]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    nir = batch["nir"].to(device, non_blocking=True)
    ais_input = batch["ais_input"].to(device, non_blocking=True)
    return [modules.rgb(rgb), modules.nir(nir), modules.ais(ais_input)]


def fuse_training_pair(
    modules: ExperimentModules,
    source_modalities: Sequence[torch.Tensor],
    target_modalities: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if modules.tensor is None:
        zero = source_modalities[0].new_zeros(())
        return torch.cat(source_modalities, dim=1), torch.cat(target_modalities, dim=1), zero
    projected_source, projected_target, alignment_loss = modules.tensor(
        list(source_modalities), list(target_modalities)
    )
    return (
        torch.cat(projected_source, dim=1),
        torch.cat(projected_target, dim=1),
        alignment_loss,
    )


def fuse_target(
    modules: ExperimentModules,
    modalities: Sequence[torch.Tensor],
) -> torch.Tensor:
    if modules.tensor is None:
        return torch.cat(list(modalities), dim=1)
    projected = [
        feature.matmul(projection)
        for feature, projection in zip(modalities, modules.tensor.V_matrices)
    ]
    return torch.cat(projected, dim=1)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    class_ids = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )
    accuracy = float((labels == predictions).mean()) if labels.size else 0.0
    return {
        "accuracy": accuracy,
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "sample_count": int(labels.size),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


@torch.no_grad()
def evaluate(
    modules: ExperimentModules,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    modules.eval()
    labels: List[int] = []
    predictions: List[int] = []
    rows: List[Dict[str, object]] = []
    loss_sum = 0.0
    count = 0
    for batch in loader:
        target = batch["label"].to(device, non_blocking=True)
        features = fuse_target(modules, encode_batch(modules, batch, device))
        logits = modules.classifier(features)
        loss = criterion(logits, target)
        predicted = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)
        loss_sum += float(loss) * target.shape[0]
        count += target.shape[0]
        batch_labels = target.cpu().tolist()
        batch_predictions = predicted.cpu().tolist()
        batch_probabilities = probabilities.cpu().tolist()
        sample_ids = list(batch["sample_id"])
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)
        for sample_id, label, prediction, probability in zip(
            sample_ids, batch_labels, batch_predictions, batch_probabilities
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "label_id": int(label),
                    "label": CLASS_NAMES[int(label)],
                    "predicted_id": int(prediction),
                    "predicted": CLASS_NAMES[int(prediction)],
                    **{
                        f"prob_{name}": float(probability[index])
                        for index, name in enumerate(CLASS_NAMES)
                    },
                }
            )
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions))
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics, rows


def _make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"vds2raw_jmda_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_epoch_record(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _make_optimizers(args, modules: ExperimentModules):
    rgb_parameters = [p for p in modules.rgb.parameters() if p.requires_grad]
    other_parameters = [
        p
        for module in (modules.nir, modules.ais, modules.classifier, modules.generator)
        if module is not None
        for p in module.parameters()
        if p.requires_grad
    ]
    optimizer_g = torch.optim.AdamW(
        [
            {"params": rgb_parameters, "lr": args.lr_rgb},
            {"params": other_parameters, "lr": args.lr_other},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_d = torch.optim.AdamW(
        modules.discriminator.parameters(),
        lr=args.lr_other,
        weight_decay=args.weight_decay,
    )

    def cosine_multiplier(epoch_index: int) -> float:
        progress = min(max(epoch_index / max(args.epochs, 1), 0.0), 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer_g, lr_lambda=cosine_multiplier)
    return optimizer_g, optimizer_d, scheduler


def _checkpoint_payload(
    args,
    modules: ExperimentModules,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    seed: int,
    ablation_name: str,
    val_metrics: Mapping[str, object],
    manifest_sha256: str,
) -> Dict[str, object]:
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "seed": int(seed),
        "ablation_name": ablation_name,
        "manifest_sha256": manifest_sha256,
        "config": dict(vars(args)),
        "val_metrics": dict(val_metrics),
        "models": modules.state_dicts(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _load_model_checkpoint(
    path: Path,
    modules: ExperimentModules,
    device: torch.device,
    expected_manifest_sha256: str,
) -> Mapping[str, object]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Checkpoint manifest hash does not match the current dataset")
    modules.load_state_dicts(checkpoint["models"])
    return checkpoint


def _update_tensor_projections(
    args,
    modules: ExperimentModules,
    source_dataset: VDS2RawDataset,
    target_dataset: VDS2RawDataset,
    device: torch.device,
    seed: int,
    epoch: int,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
) -> Mapping[str, object]:
    if modules.tensor is None:
        return {}
    old_source = [projection.detach().clone() for projection in modules.tensor.U_matrices]
    old_target = [projection.detach().clone() for projection in modules.tensor.V_matrices]
    information = update_epoch_projections(
        tal_module=modules.tensor,
        encoders=list(modules.encoders()),
        modality_keys=["rgb", "nir", "ais_input"],
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        batch_size=args.svd_stat_batch_size,
        device=device,
        seed=seed * 1000,
        class_paired=True,
        balance_classes=False,
    )
    if epoch > 0:
        information.update(
            transport_projected_feature_basis(
                classifier=modules.classifier,
                discriminator=modules.discriminator,
                generator=modules.generator,
                old_source_projections=old_source,
                old_target_projections=old_target,
                new_source_projections=modules.tensor.U_matrices,
                new_target_projections=modules.tensor.V_matrices,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
            )
        )
    return information


def _train_epoch(
    args,
    modules: ExperimentModules,
    batcher: SameClassPairedBatcher,
    device: torch.device,
    criterion: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    use_tensor: bool,
    use_ot: bool,
) -> Dict[str, float]:
    modules.train()
    joint_ot_scale = compute_joint_ot_scale(
        epoch=epoch,
        warmup_epochs=args.ot_warmup_epochs,
        ramp_epochs=args.ot_ramp_epochs,
        joint_tensor_ot=use_tensor and use_ot,
    )
    sums = {
        "loss": 0.0,
        "loss_cls": 0.0,
        "loss_tensor": 0.0,
        "loss_adv": 0.0,
        "loss_d": 0.0,
        "loss_ot_reg": 0.0,
        "ot_objective": 0.0,
        "ot_marginal_residual": 0.0,
        "ot_off_class_mass": 0.0,
    }
    source_correct = 0
    target_correct = 0
    observation_count = 0
    steps = 0
    ot_steps = 0
    total_steps = len(batcher)

    for source_batch, target_batch in batcher.iter_epoch(epoch):
        if args.max_train_batches > 0 and steps >= args.max_train_batches:
            break
        source_labels = source_batch["label"].to(device, non_blocking=True)
        target_labels = target_batch["label"].to(device, non_blocking=True)
        if not torch.equal(source_labels, target_labels):
            raise RuntimeError("Same-class batch pairing invariant was violated")

        source_modalities = encode_batch(modules, source_batch, device)
        target_modalities = encode_batch(modules, target_batch, device)
        source_features, target_features, tensor_loss = fuse_training_pair(
            modules, source_modalities, target_modalities
        )

        transport_details = None
        intermediate_features = None
        if use_ot and joint_ot_scale > 0.0:
            intermediate_features, transport_details = modules.generator(
                source_features,
                target_features,
                return_details=True,
                source_labels=source_labels if args.class_conditional_ot else None,
                target_labels=target_labels if args.class_conditional_ot else None,
            )

        discriminator_loss = source_features.new_zeros(())
        if (not use_ot) or intermediate_features is not None:
            optimizer_d.zero_grad(set_to_none=True)
            if use_ot:
                discriminator_loss, _ = compute_discriminator_loss(
                    modules.discriminator(source_features.detach()),
                    modules.discriminator(target_features.detach()),
                    modules.discriminator(intermediate_features.detach()),
                )
            else:
                discriminator_loss = binary_domain_loss(
                    modules.discriminator(source_features.detach()),
                    modules.discriminator(target_features.detach()),
                )
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(modules.discriminator.parameters(), 1.0)
            optimizer_d.step()

        optimizer_g.zero_grad(set_to_none=True)
        source_logits = modules.classifier(source_features)
        target_logits = modules.classifier(target_features)
        source_classification = criterion(source_logits, source_labels)
        target_classification = criterion(target_logits, target_labels)
        classification_numerator = source_classification + target_classification
        classification_denominator = 2.0

        if intermediate_features is not None:
            intermediate_logits = modules.classifier(intermediate_features)
            classification_numerator = classification_numerator + (
                joint_ot_scale * criterion(intermediate_logits, target_labels)
            )
            classification_denominator += joint_ot_scale
        classification_loss = classification_numerator / classification_denominator

        progress = (epoch * total_steps + steps) / max(args.epochs * total_steps, 1)
        alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        set_requires_grad(modules.discriminator, False)
        if use_ot:
            if intermediate_features is None:
                adversarial_loss = source_features.new_zeros(())
                correction_regularization = source_features.new_zeros(())
            else:
                adversarial_loss = compute_generator_loss(
                    modules.discriminator(intermediate_features), "kl_uniform"
                )
                correction_regularization = transport_details["correction_regularization"]
        else:
            reversed_source = GradientReversal.apply(source_features, alpha)
            reversed_target = GradientReversal.apply(target_features, alpha)
            adversarial_loss = binary_domain_loss(
                modules.discriminator(reversed_source),
                modules.discriminator(reversed_target),
            )
            correction_regularization = source_features.new_zeros(())

        total_loss = (
            classification_loss
            + (args.tensor_loss_weight * tensor_loss if use_tensor else 0.0)
            + args.adv_loss_weight * joint_ot_scale * adversarial_loss
            + args.ot_correction_reg_weight * joint_ot_scale * correction_regularization
        )
        total_loss.backward()
        set_requires_grad(modules.discriminator, True)
        for encoder in modules.encoders():
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(modules.classifier.parameters(), 1.0)
        if modules.generator is not None:
            torch.nn.utils.clip_grad_norm_(modules.generator.parameters(), 1.0)
        optimizer_g.step()

        batch_count = target_labels.shape[0]
        observation_count += batch_count
        source_correct += int((source_logits.argmax(1) == source_labels).sum())
        target_correct += int((target_logits.argmax(1) == target_labels).sum())
        sums["loss"] += float(total_loss.detach())
        sums["loss_cls"] += float(classification_loss.detach())
        sums["loss_tensor"] += float(tensor_loss.detach())
        sums["loss_adv"] += float(adversarial_loss.detach())
        sums["loss_d"] += float(discriminator_loss.detach())
        sums["loss_ot_reg"] += float(correction_regularization.detach())
        if transport_details is not None:
            ot_steps += 1
            sums["ot_objective"] += float(
                transport_details["regularized_ot_objective"].detach()
            )
            sums["ot_marginal_residual"] += float(
                transport_details["marginal_residual"].detach()
            )
            sums["ot_off_class_mass"] += float(
                transport_details["off_class_mass"].detach()
            )
        steps += 1

    if steps == 0:
        raise RuntimeError("Training epoch produced no batches")
    result = {key: value / steps for key, value in sums.items()}
    if ot_steps:
        for key in ("ot_objective", "ot_marginal_residual", "ot_off_class_mass"):
            result[key] = sums[key] / ot_steps
    result.update(
        {
            "source_train_accuracy": source_correct / max(observation_count, 1),
            "target_train_accuracy": target_correct / max(observation_count, 1),
            "train_observations": float(observation_count),
            "train_steps": float(steps),
            "ot_scale": float(joint_ot_scale),
        }
    )
    return result


def run_experiment(
    args,
    manifest,
    image_stats: Mapping[str, object],
    audit_report: Mapping[str, object],
    ablation_name: str,
    seed: int,
    suite_dir: Path,
) -> Dict[str, object]:
    """Run one seed and evaluate target_test only after validation selection."""

    if ablation_name not in ABLATION_SPECS:
        raise ValueError(f"Unknown ablation {ablation_name!r}")
    spec = ABLATION_SPECS[ablation_name]
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    seed_everything(seed, deterministic=args.deterministic)

    run_dir = suite_dir / f"{ablation_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _make_logger(run_dir)
    config = json.loads(json.dumps(vars(args), default=str))
    config.update(
        {
            "seed": seed,
            "ablation_name": ablation_name,
            "use_tensor": spec["use_tensor"],
            "use_ot": spec["use_ot"],
            "device_resolved": str(device),
        }
    )
    write_json(run_dir / "config.json", config)

    source_train = VDS2RawDataset(manifest, "source_train", image_stats)
    target_train = VDS2RawDataset(manifest, "target_train", image_stats)
    target_val = VDS2RawDataset(manifest, "target_val", image_stats)
    target_test = VDS2RawDataset(manifest, "target_test", image_stats)
    if spec["use_tensor"] and args.projection_dim > len(target_train) - 1:
        raise ValueError(
            f"projection_dim={args.projection_dim} exceeds centered SVD rank "
            f"{len(target_train) - 1} for the unbalanced target pairing"
        )
    batcher = SameClassPairedBatcher(source_train, target_train, args.batch_size, seed)
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(target_val, **loader_options)
    test_loader = DataLoader(target_test, **loader_options) if args.evaluate_test else None

    modules = build_modules(args, spec, device)
    optimizer_g, optimizer_d, scheduler = _make_optimizers(args, modules)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_checkpoint = run_dir / "best.pt"
    last_checkpoint = run_dir / "last.pt"
    metrics_path = run_dir / "metrics.csv"
    history: List[Dict[str, object]] = []
    best_key = (-float("inf"), -float("inf"))
    best_epoch = -1
    manifest_sha256 = str(audit_report["manifest_sha256"])

    logger.info("VDS2Raw supervised domain adaptation")
    logger.info(
        "ablation=%s seed=%d device=%s pretrained=%s tensor=%s ot=%s",
        ablation_name,
        seed,
        device,
        args.pretrained,
        spec["use_tensor"],
        spec["use_ot"],
    )
    logger.info(
        "partitions source_train=%d target_train=%d target_val=%d target_test=%d",
        len(source_train),
        len(target_train),
        len(target_val),
        len(target_test),
    )
    logger.info(
        "modalities RGB=3x96x96 NIR=1x96x96 AIS=13+13mask; fused_dim=%d",
        modules.fused_dim,
    )
    logger.info(
        "target_train coverage=%d samples/epoch, batches=%d, class balancing=disabled",
        batcher.epoch_sample_count,
        len(batcher),
    )

    for epoch in range(args.epochs):
        if modules.tensor is not None:
            svd_information = _update_tensor_projections(
                args,
                modules,
                source_train,
                target_train,
                device,
                seed,
                epoch,
                optimizer_g,
                optimizer_d,
            )
            logger.info(
                "epoch %d/%d | %s",
                epoch + 1,
                args.epochs,
                format_svd_update(dict(svd_information)),
            )
            projection_update_count = int(modules.tensor.projection_update_count.item())
        else:
            projection_update_count = -1

        train_metrics = _train_epoch(
            args,
            modules,
            batcher,
            device,
            criterion,
            optimizer_g,
            optimizer_d,
            epoch,
            use_tensor=bool(spec["use_tensor"]),
            use_ot=bool(spec["use_ot"]),
        )
        if modules.tensor is not None:
            current_count = int(modules.tensor.projection_update_count.item())
            if current_count != projection_update_count:
                raise RuntimeError("Tensor projection changed inside the mini-batch loop")

        val_metrics, _ = evaluate(modules, val_loader, device, criterion)
        scheduler.step()
        record: Dict[str, object] = {
            "epoch": epoch + 1,
            **train_metrics,
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision_macro": float(val_metrics["precision_macro"]),
            "val_recall_macro": float(val_metrics["recall_macro"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "lr_rgb": optimizer_g.param_groups[0]["lr"],
            "lr_other": optimizer_g.param_groups[1]["lr"],
        }
        history.append(record)
        _write_csv(metrics_path, [_flatten_epoch_record(row) for row in history])
        logger.info(
            "epoch %d/%d | loss=%.4f cls=%.4f tensor=%.4f adv=%.4f "
            "src_acc=%.4f tgt_train_acc=%.4f val_acc=%.4f val_f1=%.4f ot_scale=%.2f",
            epoch + 1,
            args.epochs,
            record["loss"],
            record["loss_cls"],
            record["loss_tensor"],
            record["loss_adv"],
            record["source_train_accuracy"],
            record["target_train_accuracy"],
            record["val_accuracy"],
            record["val_f1_macro"],
            record["ot_scale"],
        )

        checkpoint = _checkpoint_payload(
            args,
            modules,
            optimizer_g,
            optimizer_d,
            scheduler,
            epoch + 1,
            seed,
            ablation_name,
            val_metrics,
            manifest_sha256,
        )
        _save_checkpoint(last_checkpoint, checkpoint)
        selection_key = (float(val_metrics["accuracy"]), float(val_metrics["f1_macro"]))
        if selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch + 1
            _save_checkpoint(best_checkpoint, checkpoint)
            logger.info(
                "new best checkpoint: epoch=%d val_accuracy=%.4f val_f1=%.4f",
                best_epoch,
                best_key[0],
                best_key[1],
            )

    selected = _load_model_checkpoint(
        best_checkpoint,
        modules,
        device,
        expected_manifest_sha256=manifest_sha256,
    )
    if args.evaluate_test:
        test_metrics, test_rows = evaluate(modules, test_loader, device, criterion)
        _write_csv(run_dir / "target_test_predictions.csv", test_rows)
        write_json(run_dir / "target_test_metrics.json", test_metrics)
        test_evaluations = 1
    else:
        test_metrics = None
        test_evaluations = 0
    summary = {
        "ablation_name": ablation_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "selection_rule": "maximum target_val accuracy; macro-F1 tie break",
        "best_val_metrics": selected["val_metrics"],
        "target_test_metrics": test_metrics,
        "target_test_evaluations": test_evaluations,
        "manifest_sha256": manifest_sha256,
        "completed_epochs": args.epochs,
        "partial_training_epoch": args.max_train_batches > 0,
    }
    write_json(run_dir / "summary.json", summary)
    if test_metrics is not None:
        logger.info(
            "final target_test (best val epoch %d only): accuracy=%.4f macro_f1=%.4f",
            best_epoch,
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )
    else:
        logger.info("target_test skipped for this diagnostic run")
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    return summary


__all__ = [
    "ABLATION_SPECS",
    "ExperimentModules",
    "build_modules",
    "encode_batch",
    "fuse_training_pair",
    "fuse_target",
    "classification_metrics",
    "evaluate",
    "run_experiment",
]
