# VDS2Raw v2 JMDA-Net

This directory is the isolated training path for the prepared three-class
VDS2Raw v2 dataset. It does not reuse or modify the historical weather
VIS/IR data loader or its split rules.

## Fixed protocol

- Classes: `Cargo=0`, `Fishing=1`, `Sailing & Pleasure=2`.
- Source domain: `source_train`, all from day3.
- Target domain: `target_train`, `target_val`, and `target_test`, excluding day3.
- Only `source_train + target_train` can update model parameters, Tensor/SVD
  projections, AIS scalers, or image normalization statistics.
- `target_val` selects the checkpoint by accuracy with macro-F1 as tie break.
- `target_test` is evaluated once after loading that checkpoint. It is never
  used for model selection.
- Every target training sample occurs exactly once per epoch. A same-class
  source sample is selected without repeating target samples for balancing.
- RGB is B4/B3/B2, NIR is Sentinel-2 B8, and AIS is the 13-value vector plus
  its 13-value validity mask. MMSI and Ship type are not model inputs.

## Model

- RGB: ImageNet-pretrained ResNet18 at native 96x96 resolution. The stem and
  layer1 are frozen by default; layer2--layer4 and the projection are trained.
- NIR: independent one-channel convolutional encoder.
- AIS: MLP over the 26-dimensional value/mask input.
- Each encoder emits 512 features.
- Tensor mode uses three 64-dimensional epoch-SVD projections and concatenates
  them into 192 features.
- The full model applies class-conditional TransNet + Softmax-kernel Sinkhorn
  transport after a Tensor-only warm-up.

No geometric image augmentation is used because a flip/rotation would make
the absolute AIS heading inconsistent with the image.

## Windows commands

Use the requested environment:

```powershell
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\module_ablation.py --audit_only
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\test_vds2raw_pipeline.py
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\module_ablation.py --smoke_test
```

The smoke test runs one optimization batch but still builds SVD statistics
from the complete training pairing. It deliberately skips `target_test`.

Run a short complete-epoch test without touching `target_test`:

```powershell
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\module_ablation.py `
  --epochs 5 --seeds 42 --no-evaluate_test
```

Run the full model locally:

```powershell
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\module_ablation.py `
  --ablation_mode full --epochs 100 --seeds 42
```

Run all four full/ablation configurations with multiple seeds:

```powershell
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\module_ablation.py `
  --ablation_mode all --epochs 100 --seeds 42,43,44,45,46
```

## Linux server

Set `VDS2RAW_ROOT` to the copied `threeclass_ready` directory, then run the
same entry point. The manifest uses relative NPZ paths, so it is portable.
Copy the official cached `resnet18-f37072fd.pth` file as well and pass its
path explicitly. The training entry never downloads weights silently.

```bash
export VDS2RAW_ROOT=/data/VDS2Raw/threeclass_ready
python -B VDS2Raw_JMDA/module_ablation.py \
  --pretrained_weights /models/resnet18-f37072fd.pth \
  --ablation_mode all --epochs 100 --seeds 42,43,44,45,46
```

## Outputs

Each invocation creates one timestamped directory under `VDS2Raw_JMDA/runs`:

- immutable dataset audit and train-only image statistics;
- invocation and per-run configuration;
- raw epoch metrics CSV and UTF-8 log;
- `best.pt` and `last.pt` checkpoints;
- target-test metrics and per-sample probabilities for non-smoke final runs;
- per-seed summaries and aggregate mean/population-standard-deviation values.

The `runs` directory is ignored by Git but remains available locally.
