"""Reproducible model-complexity and runtime measurements.

FLOPs are collected with ``torch.profiler(with_flops=True)``.  PyTorch only
assigns FLOP counts to supported operators (principally convolution and
matrix multiplication), so the saved report states that scope explicitly
instead of presenting the estimate as an exact hardware instruction count.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _component_modules(modules) -> Dict[str, Optional[nn.Module]]:
    return {
        "rgb_encoder": modules.rgb,
        "nir_encoder": modules.nir,
        "ais_encoder": modules.ais,
        "tensor_alignment": modules.tensor,
        "classifier": modules.classifier,
        "ot_generator": modules.generator,
        "domain_discriminator": modules.discriminator,
    }


def parameter_report(modules) -> Dict[str, object]:
    components = {}
    for name, module in _component_modules(modules).items():
        if module is None:
            components[name] = {
                "enabled": False,
                "total_parameters": 0,
                "trainable_parameters": 0,
                "buffer_elements": 0,
            }
            continue
        components[name] = {
            "enabled": True,
            "total_parameters": int(sum(p.numel() for p in module.parameters())),
            "trainable_parameters": int(
                sum(p.numel() for p in module.parameters() if p.requires_grad)
            ),
            "buffer_elements": int(sum(b.numel() for b in module.buffers())),
        }

    inference_names = (
        "rgb_encoder",
        "nir_encoder",
        "ais_encoder",
        "tensor_alignment",
        "classifier",
    )
    return {
        "components": components,
        "total_parameters": int(
            sum(item["total_parameters"] for item in components.values())
        ),
        "trainable_parameters": int(
            sum(item["trainable_parameters"] for item in components.values())
        ),
        "inference_parameters": int(
            sum(components[name]["total_parameters"] for name in inference_names)
        ),
        "total_buffer_elements": int(
            sum(item["buffer_elements"] for item in components.values())
        ),
        "notes": {
            "total_parameters": "Includes encoders, classifier, and training-only OT/discriminator modules.",
            "inference_parameters": "Target inference uses encoders, optional Tensor projections, and classifier only.",
            "tensor_projections": "Tensor/SVD projection matrices are checkpointed buffers, not optimizer parameters.",
        },
    }


def _profile_flops(
    callback: Callable[[], object],
    device: torch.device,
) -> Dict[str, object]:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        synchronize(device)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
        ) as profiler:
            callback()
            synchronize(device)
        events = profiler.key_averages()
        counted = [event for event in events if int(getattr(event, "flops", 0) or 0) > 0]
        total = int(sum(int(event.flops) for event in counted))
        return {
            "status": "PASS" if total > 0 else "NO_SUPPORTED_OPS_COUNTED",
            "flops": total,
            "gflops": total / 1e9,
            "counted_operator_count": len(counted),
            "counted_operators": [
                {"name": event.key, "flops": int(event.flops)}
                for event in sorted(counted, key=lambda item: int(item.flops), reverse=True)
            ],
            "method": "torch.profiler.profile(with_flops=True)",
            "scope": "Forward-pass FLOPs reported only for operators supported by the installed PyTorch profiler; backward, optimizer, SVD update, data loading, and unsupported elementwise/reduction operations are excluded.",
        }
    except Exception as exc:  # Keep a long experiment alive if profiling is unavailable.
        return {
            "status": "ERROR",
            "flops": None,
            "gflops": None,
            "counted_operator_count": 0,
            "counted_operators": [],
            "method": "torch.profiler.profile(with_flops=True)",
            "scope": "FLOP profiling failed; other complexity measurements remain valid.",
            "error": f"{type(exc).__name__}: {exc}",
        }


def flop_report(
    modules,
    device: torch.device,
    source_batch: Mapping[str, object],
    target_batch: Mapping[str, object],
    encode_batch: Callable,
    fuse_training_pair: Callable,
    fuse_target: Callable,
    class_conditional_ot: bool,
) -> Dict[str, object]:
    modules.eval()
    target_count = int(target_batch["label"].shape[0])
    paired_count = min(
        int(source_batch["label"].shape[0]),
        int(target_batch["label"].shape[0]),
    )

    def inference_forward():
        with torch.no_grad():
            features = fuse_target(modules, encode_batch(modules, target_batch, device))
            return modules.classifier(features)

    def paired_training_forward():
        with torch.no_grad():
            source_modalities = encode_batch(modules, source_batch, device)
            target_modalities = encode_batch(modules, target_batch, device)
            source_features, target_features, _ = fuse_training_pair(
                modules, source_modalities, target_modalities
            )
            modules.classifier(source_features)
            modules.classifier(target_features)
            if modules.generator is not None:
                source_labels = source_batch["label"].to(device)
                target_labels = target_batch["label"].to(device)
                intermediate, _ = modules.generator(
                    source_features,
                    target_features,
                    return_details=True,
                    source_labels=source_labels if class_conditional_ot else None,
                    target_labels=target_labels if class_conditional_ot else None,
                )
                modules.classifier(intermediate)
                modules.discriminator(source_features)
                modules.discriminator(target_features)
                modules.discriminator(intermediate)
            else:
                modules.discriminator(source_features)
                modules.discriminator(target_features)

    # Warm up lazy backend paths outside the measured profiler window.
    inference_forward()
    paired_training_forward()
    synchronize(device)
    inference = _profile_flops(inference_forward, device)
    training_forward = _profile_flops(paired_training_forward, device)
    inference["batch_size"] = target_count
    training_forward["paired_batch_size"] = paired_count
    if inference["flops"] is not None:
        inference["flops_per_sample"] = inference["flops"] / max(target_count, 1)
        inference["gflops_per_sample"] = inference["flops_per_sample"] / 1e9
    if training_forward["flops"] is not None:
        training_forward["flops_per_source_target_pair"] = (
            training_forward["flops"] / max(paired_count, 1)
        )
        training_forward["gflops_per_source_target_pair"] = (
            training_forward["flops_per_source_target_pair"] / 1e9
        )
    return {
        "target_inference_forward": inference,
        "paired_training_forward": training_forward,
    }


def benchmark_inference(
    modules,
    batch: Mapping[str, object],
    device: torch.device,
    encode_batch: Callable,
    fuse_target: Callable,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, object]:
    modules.eval()
    sample_count = int(batch["label"].shape[0])

    def forward():
        features = fuse_target(modules, encode_batch(modules, batch, device))
        return modules.classifier(features)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            forward()
        synchronize(device)
        if device.type == "cuda":
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = None
            baseline_reserved = None
        started = time.perf_counter()
        for _ in range(measured_runs):
            forward()
        synchronize(device)
        elapsed = time.perf_counter() - started

    total_samples = sample_count * measured_runs
    result: Dict[str, object] = {
        "source": "first target_val batch preloaded on the selected device",
        "batch_size": sample_count,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": 1000.0 * elapsed / measured_runs,
        "latency_ms_per_sample": 1000.0 * elapsed / max(total_samples, 1),
        "throughput_samples_per_second": total_samples / max(elapsed, 1e-12),
        "includes": "RGB/NIR/AIS encoders, optional Tensor target projection, and classifier; excludes data loading, OT generator, and discriminator.",
    }
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        result["gpu_memory"] = {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": max(
                peak_allocated - int(baseline_allocated), 0
            ),
            "incremental_peak_reserved_bytes": max(
                peak_reserved - int(baseline_reserved), 0
            ),
        }
    else:
        result["gpu_memory"] = None
    return result


def environment_report(device: torch.device) -> Dict[str, object]:
    report: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_resolved": str(device),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        report["gpu"] = {
            "visible_device_index": int(index),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    else:
        report["gpu"] = None
    return report


def flatten_complexity(report: Mapping[str, object]) -> Dict[str, object]:
    parameters = report["parameters"]
    inference = report["flops"]["target_inference_forward"]
    training_forward = report["flops"]["paired_training_forward"]
    timing = report["timing"]
    benchmark = report["inference_benchmark"]
    training_memory = report.get("training_gpu_memory") or {}
    return {
        "total_parameters": parameters["total_parameters"],
        "trainable_parameters": parameters["trainable_parameters"],
        "inference_parameters": parameters["inference_parameters"],
        "inference_gflops_per_sample": inference.get("gflops_per_sample"),
        "training_forward_gflops_per_pair": training_forward.get(
            "gflops_per_source_target_pair"
        ),
        "training_loop_seconds": timing["training_loop_seconds"],
        "total_run_seconds": timing["total_run_seconds"],
        "mean_epoch_seconds": timing["mean_epoch_seconds"],
        "inference_latency_ms_per_sample": benchmark["latency_ms_per_sample"],
        "inference_throughput_samples_per_second": benchmark[
            "throughput_samples_per_second"
        ],
        "training_peak_allocated_bytes": training_memory.get(
            "peak_allocated_bytes"
        ),
        "training_peak_reserved_bytes": training_memory.get("peak_reserved_bytes"),
    }


__all__ = [
    "benchmark_inference",
    "environment_report",
    "flatten_complexity",
    "flop_report",
    "parameter_report",
    "synchronize",
]
