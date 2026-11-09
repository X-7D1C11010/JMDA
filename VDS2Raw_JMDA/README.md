# VDS2Raw v2 JMDA-Net

This directory is the isolated training path for the prepared three-class
VDS2Raw v2 dataset. It does not reuse or modify the historical weather
VIS/IR data loader or its split rules.

## Single-modality ResNet / ViT / ConvNeXt baselines

`single_modal_baselines.py` runs the requested 45 independent experiments:

- ResNet-50, ViT-B/16, and ConvNeXt-Base;
- RGB (B4/B3/B2), NIR (B8), and structured AIS as separate inputs;
- seeds 42, 43, 44, 45, and 46 for every model/modality pair.

RGB and NIR are evaluated at their native 96x96 resolution. Training resizes
them to 112x112 and takes a random 96x96 crop before horizontal flipping. AIS
uses only `heading`; length and width are deliberately removed because their
class-dependent distributions can make the task trivial. The prepared heading
is represented by `heading_sin`, `heading_cos`, and one validity indicator.
No length, width, navigation status, MMSI, ship type, name, coordinates, date,
scene, or other metadata enters an AIS model. The three AIS classifiers are
architecture-family adaptations: a residual MLP, a field-token Transformer,
and a one-dimensional ConvNeXt.

All models are initialized without downloaded pretrained weights. Training is
supervised on `source_train + target_train`; image statistics and class weights
also use only those partitions. The checkpoint with maximum `target_val` ACC
is selected, with macro F1 as the tie break. Only after restoring that
checkpoint is `target_test` constructed and evaluated once. Reported
Precision, Recall, and F1 are macro averages across all three classes.
Image models default to AdamW with peak learning rate `3e-4`, five warm-up
epochs, cosine decay, 100 epochs maximum, patience 20, and label smoothing
0.05. AIS keeps a 50-epoch maximum, patience 12, no warm-up, and no label
smoothing.

Local audit and smoke test:

```powershell
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\single_modal_baselines.py --audit-only
D:\Anaconda\envs\pytorch\python.exe -B .\VDS2Raw_JMDA\single_modal_baselines.py `
  --smoke-test --models resnet,vit,convnext --modalities rgb,nir,ais --seeds 42
```

Run all 45 experiments locally:

```powershell
D:\Anaconda\envs\pytorch\python.exe -u -B .\VDS2Raw_JMDA\single_modal_baselines.py `
  --dataset-root D:\Downloads\VDS2Raw\threeclass_ready `
  --run-name vds2raw_heading_only_tuned_images_45runs `
  --epochs 100 --patience 20 --warmup-epochs 5 `
  --lr-image 3e-4 --label-smoothing 0.05 --train-resize-size 112 `
  --ais-epochs 50 --ais-patience 12 --batch-size 32 --num-workers 0
```

Run all 45 experiments on the server:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -B VDS2Raw_JMDA/single_modal_baselines.py \
  --dataset-root /home/lixiang/lx/VDS2Raw/threeclass_ready \
  --run-name vds2raw_heading_only_tuned_images_45runs \
  --epochs 100 --patience 20 --warmup-epochs 5 \
  --lr-image 3e-4 --label-smoothing 0.05 --train-resize-size 112 \
  --ais-epochs 50 --ais-patience 12 --batch-size 32 --num-workers 4
```

Each seed directory contains the immutable configuration, epoch curve, log,
validation-selected `best.pt`, target-test metrics, and per-sample target-test
probabilities. The run root contains `results.csv`, `summary.csv`,
`summary.json`, the complete dataset/protocol audit, and train-only image
statistics. Reusing the same `--run-name` skips seed runs whose `result.json`
already has status `complete`, allowing an interrupted grid to resume.

For a 224x224 image-only experiment without random cropping, set
`--modalities rgb,nir --image-size 224 --train-resize-size 224`. Keeping both
sizes equal makes the spatial crop deterministic (the full resized image),
while the optional horizontal flip remains enabled. Use
`--label-smoothing 0` to remove smoothing for the small-data underfitting
diagnostic.

## Fixed protocol

- Classes: `Cargo=0`, `Fishing=1`, `Sailing & Pleasure=2`.
- Source domain: `source_train`, all from day3.
- Target domain: `target_train`, `target_val`, and `target_test`, excluding day3.
- Only `source_train + target_train` can update model parameters, Tensor/SVD
  projections, AIS scalers, or image normalization statistics.
- `target_val` selects the checkpoint by accuracy with macro-F1 as tie break.
  For the Full model, epochs before the OT warm-up/ramp is complete are not
  eligible for checkpoint selection.
- `target_test` is evaluated once after loading that checkpoint. It is never
  used for model selection.
- Every target training sample occurs at least once per epoch. Target classes
  are repeated to the largest target-class count and paired with same-class
  source observations, matching the historical weather-dataset sampler.
- RGB is B4/B3/B2, NIR is Sentinel-2 B8, and AIS is the 13-value vector plus
  its 13-value validity mask. MMSI and Ship type are not model inputs. The
  loader always zeros every `nav_*` value and mask, retaining only heading,
  length, and width so navigation-status label proxies cannot reach the model.

## Model

- RGB: the historical resize-256/random-crop-224/horizontal-flip train
  transform followed by the historical
  ImageNet-pretrained ResNet18 VIS encoder. The complete ResNet feature stack
  is frozen and only its 512-dimensional projection is trained, matching the
  old experiment's effective freeze behavior.
- NIR: the same synchronized image transform, repeated from one B8 channel to three
  encoder channels, followed by the historical four-stage U-Net-style IR
  encoder (64/128/256/512 channels).
- AIS: MLP over the 26-dimensional value/mask input.
- Each encoder emits 512 features.
- Tensor mode uses 64/32/32 dimensional RGB/NIR/AIS epoch-SVD projections and
  concatenates them into 128 features. Repeated single-band B8 and structured
  AIS use narrower projections consistent with their lower empirical ranks.
- The full model applies class-conditional TransNet + Softmax-kernel Sinkhorn
  transport after a Tensor-only warm-up.
- By default, classification supervision contains source CE and, when OT is
  active, transported-sample CE with source labels. Target CE is disabled.
  Target labels are still retained for same-class batching, class-balanced
  epoch-SVD statistics, class-conditional OT, and diagnostic accuracy. The
  unsupervised protocol excludes target CE; the supervised protocol includes
  it. The discriminator updates every two mini-batches.

SVD, validation, and test use deterministic 224x224 resize. During a training
horizontal flip, `heading_sin` is negated while `heading_cos` is preserved, so
the structured AIS direction remains consistent with the image. Rotation is
not used.

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
  --epochs 5 --seeds 42 --ot_warmup_epochs 1 --ot_ramp_epochs 1 --no-evaluate_test
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

`--seeds 42` is one independent iteration. A comma-separated seed list runs
multiple independent iterations and reports the mean, population/sample
variance, and population/sample standard deviation for accuracy, macro
precision, macro recall, and macro F1.

## Linux server

Set `VDS2RAW_ROOT` to the copied `threeclass_ready` directory, then run the
same entry point. The manifest uses relative NPZ paths, so it is portable.
Copy the official cached `resnet18-f37072fd.pth` file as well and pass its
path explicitly. The training entry never downloads weights silently.

Run `run_all_when_gpu_free.sh` once to execute only the three supervised
Tensor/OT ablations (Full runs are intentionally omitted). Before each
experiment it waits until the configured physical GPU has at least 10 GiB
free; combined output is written to `train.log`. All modes use the same
64/32/32 modality dimensions (128 after concatenation) and become eligible for
checkpoint selection only after the configured warm-up plus ramp (15 epochs
with the script defaults).

```bash
export VDS2RAW_ROOT=/data/VDS2Raw/threeclass_ready
python -B VDS2Raw_JMDA/module_ablation.py \
  --pretrained_weights /models/resnet18-f37072fd.pth \
  --ablation_mode all --epochs 100 --seeds 42,43,44,45,46
```

One command runs the complete model first and then the three required
Tensor/OT ablations:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u -B VDS2Raw_JMDA/module_ablation.py \
  --dataset_root /home/lixiang/lx/VDS2Raw/threeclass_ready \
  --output_root /home/lixiang/lx/JMDA/VDS2Raw_JMDA/runs \
  --ablation_mode all \
  --seeds 42,43,44,45,46 \
  --epochs 100 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --transport_mode sinkhorn \
  --projection_dim 64 \
  --nir_projection_dim 32 \
  --ais_projection_dim 32 \
  --svd_max_sweeps 10 \
  --svd_tolerance 1e-3 \
  --svd_stat_batch_size 128 \
  --lr_rgb 5e-6 \
  --lr_other 1e-4 \
  --lr_discriminator 5e-5 \
  --discriminator_update_interval 2 \
  --freeze_rgb_backbone \
  --image_augmentation \
  --ot_warmup_epochs 10 \
  --ot_ramp_epochs 5 \
  > /home/lixiang/lx/JMDA/VDS2Raw_JMDA/train.log 2>&1 &
```

In both protocols target labels remain visible to same-class pairing,
class-balanced epoch-SVD, class-conditional OT, target-train diagnostics, and
target-val checkpoint selection. They contribute to classification CE only in
the supervised protocol.
All AIS navigation-status fields are always removed in the loader, retaining
only heading, length, and width. This is a leakage safeguard, not an additional
ablation configuration, and applies to Full and all existing Tensor/OT modes.

## Outputs

Each invocation creates one timestamped directory under `VDS2Raw_JMDA/runs`:

- immutable dataset audit and train-only image statistics;
- `protocol_audit.json`, enumerating the fixed target-label visibility,
  checkpoint-selection rule, and unconditionally removed AIS leakage fields;
- invocation and per-run configuration;
- raw epoch metrics CSV and UTF-8 log;
- temporary `best.pt` and `last.pt` checkpoints during training; both are
  deleted automatically after final evaluation and report generation;
- target-test metrics and per-sample probabilities for non-smoke final runs;
- a clearly marked diagnostic best target-test checkpoint across the
  validation-selected seed checkpoints. This oracle maximum is not an
  unbiased final metric and must not be used for model selection;
- per-seed summaries and aggregate mean/population-standard-deviation values.
- separate `<ablation>/seed_<seed>/` raw-data directories for every run;
- per-epoch ACC, macro Precision, macro Recall, and macro F1;
- the validation-selected metric set plus independent maxima and epochs for
  each of those four metrics;
- per-seed and per-ablation CSV/JSON summaries with means, population/sample
  variances, and population/sample standard deviations;
- parameter counts by component, profiler-supported forward FLOPs, total and
  per-epoch training time, inference latency/throughput, hardware metadata,
  and CUDA allocated/reserved peak memory.

FLOPs are estimates from operators supported by
`torch.profiler(with_flops=True)`. The report deliberately excludes backward,
optimizer, SVD-update, data-loading, and unsupported elementwise/reduction
operations. Inference speed is model-only and uses a preloaded target-validation
batch; it therefore does not include storage or DataLoader latency.

The `runs` directory is ignored by Git but remains available locally.
