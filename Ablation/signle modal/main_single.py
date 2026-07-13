import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

from DataLoad_single import SingleModalityDataset, PairedSingleModalitySampler
from Discriminator import DomainDiscriminator, compute_discriminator_loss, compute_generator_loss
from Generator import NeuralOptimalTransportGenerator
from Models import VisualFeatureExtractor, IRFeatureExtractor, Classifier
from Models_AIS import create_ais_feature_extractor


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(
            log_preds, target, reduction=self.reduction)


def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad


def build_target_labeled_indices(dataset, ratio, seed):
    """Build a fixed, stratified target-label subset for one repeated run."""
    ratio = float(max(0.0, min(1.0, ratio)))
    if ratio <= 0.0 or len(dataset) == 0:
        return set()

    from collections import defaultdict

    rng = random.Random(seed + 2027)
    by_class = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        by_class[int(label)].append(idx)

    selected = set()
    for indices in by_class.values():
        selected.add(rng.choice(indices))

    target_count = max(len(selected), int(round(len(dataset) * ratio)))
    target_count = min(target_count, len(dataset))
    remaining = [idx for idx in range(len(dataset)) if idx not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[:max(0, target_count - len(selected))])
    return selected


def sampled_target_classification_loss(
    criterion,
    logits,
    labels,
    ratio,
    target_indices=None,
    labeled_indices=None,
):
    """Apply target classification only to the fixed labeled target subset."""
    ratio = float(max(0.0, min(1.0, ratio)))
    if ratio <= 0.0 or logits.size(0) == 0:
        return logits.new_tensor(0.0), 0

    if target_indices is not None and labeled_indices is not None:
        positions = [
            pos for pos, sample_idx in enumerate(target_indices.detach().cpu().tolist())
            if int(sample_idx) in labeled_indices
        ]
        if not positions:
            return logits.new_tensor(0.0), 0
        selected = torch.tensor(positions, dtype=torch.long, device=logits.device)
        return criterion(logits[selected], labels[selected]), len(positions)

    n_labeled = max(1, int(round(logits.size(0) * ratio)))
    n_labeled = min(n_labeled, logits.size(0))
    selected = torch.randperm(logits.size(0), device=logits.device)[:n_labeled]
    return criterion(logits[selected], labels[selected]), n_labeled


def select_report_metrics(metric_history, strategy='last_window', window=10):
    """Select stable validation metrics for one run without chasing a single peak."""
    if not metric_history:
        return None
    if strategy == 'best':
        return max(metric_history, key=lambda m: m['accuracy'])
    if strategy == 'last':
        return metric_history[-1]

    window = max(1, int(window))
    selected = metric_history[-window:]
    report = {}
    for key in selected[-1].keys():
        values = [m[key] for m in selected if isinstance(m.get(key), (int, float, np.floating))]
        report[key] = float(np.mean(values)) if values else selected[-1][key]
    return report


class UnpairedSingleModalitySampler:
    """Randomly pair source and target batches without using target labels."""

    def __init__(self, src_ds, tgt_ds, batch_size):
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = batch_size
        self.n_batches = min(len(src_ds), len(tgt_ds)) // batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            src_indices = random.choices(range(len(self.src_ds)), k=self.batch_size)
            tgt_indices = random.choices(range(len(self.tgt_ds)), k=self.batch_size)
            src_batch = torch.utils.data.default_collate([self.src_ds[i] for i in src_indices])
            tgt_batch = torch.utils.data.default_collate([self.tgt_ds[i] for i in tgt_indices])
            tgt_batch['sample_index'] = torch.tensor(tgt_indices, dtype=torch.long)
            yield src_batch, tgt_batch

    def __len__(self):
        return self.n_batches


def evaluate(feature_extractor, classifier, dataloader, device, label_map):
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    total = 0

    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data['data'].to(device)
            labels = data['label'].to(device)

            features = feature_extractor(inputs)
            outputs = classifier(features)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0

    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)

    all_class_labels = sorted(list(label_map.values()))
    present_class_labels = sorted(np.unique(all_labels).tolist())
    macro_labels = present_class_labels or all_class_labels

    precision_macro = precision_score(all_labels, all_predicted, average='macro',
                                     zero_division=0, labels=macro_labels)
    precision_micro = precision_score(all_labels, all_predicted, average='micro',
                                     zero_division=0, labels=all_class_labels)
    recall_macro = recall_score(all_labels, all_predicted, average='macro',
                                zero_division=0, labels=macro_labels)
    recall_micro = recall_score(all_labels, all_predicted, average='micro',
                                zero_division=0, labels=all_class_labels)
    f1_macro = f1_score(all_labels, all_predicted, average='macro',
                       zero_division=0, labels=macro_labels)
    f1_micro = f1_score(all_labels, all_predicted, average='micro',
                       zero_division=0, labels=all_class_labels)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        # Macro metrics are computed over classes that actually appear in the
        # target validation set. Keep coverage so incomplete target domains are
        # visible in logs and summaries.
        'val_class_coverage': len(present_class_labels) / len(all_class_labels) if all_class_labels else 0.0,
        'val_present_classes': len(present_class_labels),
        'val_total_classes': len(all_class_labels),
        'val_samples': len(all_labels),
    }

    return metrics


def run_single_iteration(args, seed, logger):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODALITY = args.modality
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    FEATURE_DIM = args.feature_dim

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    SOURCE_ROOT = args.source_root
    TARGET_ROOT = args.target_root
    AIS_DATA_PATH = args.ais_data_path
    target_weather = os.path.basename(TARGET_ROOT)

    if MODALITY == 'ais':
        src_train_ds = SingleModalityDataset(
            root_dir=None,
            modality='ais',
            domain_type='source',
            phase='train',
            ais_data_path=AIS_DATA_PATH
        )
        global_map = src_train_ds.get_label_map()

        tgt_train_ds = SingleModalityDataset(
            root_dir=None,
            modality='ais',
            domain_type='target',
            phase='train',
            global_label_map=global_map,
            ais_data_path=AIS_DATA_PATH
        )

        tgt_val_ds = SingleModalityDataset(
            root_dir=None,
            modality='ais',
            domain_type='target',
            phase='val',
            global_label_map=global_map,
            ais_data_path=AIS_DATA_PATH
        )
    else:
        src_train_ds = SingleModalityDataset(
            SOURCE_ROOT,
            modality=MODALITY,
            domain_type='source',
            phase='train'
        )
        global_map = src_train_ds.get_label_map()

        tgt_train_ds = SingleModalityDataset(
            TARGET_ROOT,
            modality=MODALITY,
            domain_type='target',
            phase='train',
            global_label_map=global_map
        )

        tgt_val_ds = SingleModalityDataset(
            TARGET_ROOT,
            modality=MODALITY,
            domain_type='target',
            phase='val',
            global_label_map=global_map
        )

    target_label_ratio = 1.0 if args.use_target_labels else args.target_label_ratio
    target_cls_weight = 1.0 if args.use_target_labels else args.target_cls_weight
    target_label_ratio = max(0.0, min(1.0, target_label_ratio))
    target_cls_weight = max(0.0, target_cls_weight)
    target_labeled_indices = None
    if not args.use_target_labels and target_label_ratio > 0.0 and target_cls_weight > 0.0:
        target_labeled_indices = build_target_labeled_indices(tgt_train_ds, target_label_ratio, seed)

    if args.use_target_labels:
        paired_loader = PairedSingleModalitySampler(src_train_ds, tgt_train_ds, BATCH_SIZE)
        logger.info("Training mode: full target-label supervision (class-paired batches).")
    else:
        paired_loader = UnpairedSingleModalitySampler(src_train_ds, tgt_train_ds, BATCH_SIZE)
        logger.info(
            "Training mode: unpaired target adaptation with controlled target-label loss "
            f"(ratio={target_label_ratio:.2f}, weight={target_cls_weight:.2f})."
        )
        if target_labeled_indices is not None:
            logger.info(
                f"Fixed labeled target subset: {len(target_labeled_indices)}/{len(tgt_train_ds)} samples."
            )
    val_loader = DataLoader(tgt_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           drop_last=False, num_workers=0)

    if MODALITY == 'vis':
        feature_extractor = VisualFeatureExtractor(output_dim=FEATURE_DIM).to(DEVICE)
        set_requires_grad(feature_extractor, False)
        for name, param in feature_extractor.named_parameters():
            if "layer2" in name or "layer3" in name or "layer4" in name or "proj" in name:
                param.requires_grad = True
    elif MODALITY == 'ir':
        feature_extractor = IRFeatureExtractor(output_dim=FEATURE_DIM).to(DEVICE)
        set_requires_grad(feature_extractor, True)
    else:
        sample_data = src_train_ds[0]['data']
        ais_input_dim = sample_data.shape[0] if len(sample_data.shape) == 1 else sample_data.shape[-1]
        feature_extractor = create_ais_feature_extractor(
            ais_data_shape=ais_input_dim,
            output_dim=FEATURE_DIM,
            architecture=args.ais_architecture
        ).to(DEVICE)
        set_requires_grad(feature_extractor, True)

    classifier = Classifier(input_dim=FEATURE_DIM, num_classes=len(global_map)).to(DEVICE)

    if args.use_domain_adaptation:
        generator = NeuralOptimalTransportGenerator(feature_dim=FEATURE_DIM).to(DEVICE)
        discriminator = DomainDiscriminator(feature_dim=FEATURE_DIM).to(DEVICE)
    else:
        generator = None
        discriminator = None

    feature_params = [p for p in feature_extractor.parameters() if p.requires_grad]

    if args.use_domain_adaptation:
        optimizer_params = [
            {'params': feature_params, 'lr': args.lr_feature},
            {'params': list(generator.parameters()) + list(classifier.parameters()),
             'lr': args.lr_other}
        ]
    else:
        optimizer_params = [
            {'params': feature_params, 'lr': args.lr_feature},
            {'params': classifier.parameters(), 'lr': args.lr_other}
        ]

    optimizer = optim.AdamW(optimizer_params, weight_decay=args.weight_decay)

    if args.use_domain_adaptation:
        optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr_other,
                                 weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    criterion_cls = LabelSmoothingCrossEntropy(eps=0.1)

    best_val_acc = 0.0
    best_metrics = None
    metric_history = []

    for epoch in range(EPOCHS):
        feature_extractor.train()
        classifier.train()
        if args.use_domain_adaptation:
            generator.train()
            discriminator.train()

        loss_accum = 0.0
        loss_cls_accum = 0.0
        loss_adv_accum = 0.0
        train_correct = 0
        train_total = 0
        steps = 0

        for src_data, tgt_data in paired_loader:
            steps += 1

            s_input = src_data['data'].to(DEVICE)
            t_input = tgt_data['data'].to(DEVICE)
            s_label = src_data['label'].to(DEVICE)
            t_label = tgt_data['label'].to(DEVICE)

            feat_src = feature_extractor(s_input)
            feat_tgt = feature_extractor(t_input)

            if args.use_domain_adaptation:
                feat_mid = generator(feat_src, feat_tgt)

                if steps % 2 == 0:
                    optimizer_d.zero_grad()
                    loss_d, _ = compute_discriminator_loss(
                        discriminator(feat_src.detach()),
                        discriminator(feat_tgt.detach()),
                        discriminator(feat_mid.detach())
                    )
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()

                optimizer.zero_grad()

                pred_src = classifier(feat_src)
                pred_tgt = classifier(feat_tgt)
                pred_mid = classifier(feat_mid)

                loss_cls_src = criterion_cls(pred_src, s_label)
                loss_cls_mid = criterion_cls(pred_mid, s_label)

                loss_cls_total = loss_cls_src + loss_cls_mid
                loss_cls_tgt, target_labeled_count = sampled_target_classification_loss(
                    criterion_cls,
                    pred_tgt,
                    t_label,
                    target_label_ratio,
                    tgt_data.get('sample_index'),
                    target_labeled_indices,
                )
                if target_labeled_count > 0 and target_cls_weight > 0.0:
                    loss_cls_total = loss_cls_total + target_cls_weight * loss_cls_tgt

                alpha = min(2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0, 1.0)
                loss_adv = compute_generator_loss(
                    discriminator(feat_mid, use_grl=True, alpha=alpha),
                    'kl_uniform'
                )

                loss_total = loss_cls_total + args.adv_loss_weight * loss_adv
                loss_adv_accum += loss_adv.item()

            else:
                optimizer.zero_grad()

                pred_src = classifier(feat_src)
                pred_tgt = classifier(feat_tgt)

                loss_cls_src = criterion_cls(pred_src, s_label)

                loss_cls_total = loss_cls_src
                loss_cls_tgt, target_labeled_count = sampled_target_classification_loss(
                    criterion_cls,
                    pred_tgt,
                    t_label,
                    target_label_ratio,
                    tgt_data.get('sample_index'),
                    target_labeled_indices,
                )
                if target_labeled_count > 0 and target_cls_weight > 0.0:
                    loss_cls_total = loss_cls_total + target_cls_weight * loss_cls_tgt
                loss_total = loss_cls_total

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            if args.use_domain_adaptation:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()

            loss_accum += loss_total.item()
            loss_cls_accum += loss_cls_total.item()

            train_logits = pred_tgt if args.use_target_labels else pred_src
            train_labels = t_label if args.use_target_labels else s_label
            _, predicted = torch.max(train_logits.data, 1)
            train_correct += (predicted == train_labels).sum().item()
            train_total += train_labels.size(0)

        val_metrics = evaluate(feature_extractor, classifier, val_loader, DEVICE, global_map)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_metrics['accuracy']

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        avg_loss = loss_accum / steps if steps > 0 else 0.0
        avg_loss_cls = loss_cls_accum / steps if steps > 0 else 0.0
        avg_loss_adv = loss_adv_accum / steps if steps > 0 else 0.0

        if args.use_domain_adaptation:
            log_msg = (f"Epoch [{epoch + 1}/{EPOCHS}] | "
                      f"Loss: {avg_loss:.4f} (Cls: {avg_loss_cls:.4f}, Adv: {avg_loss_adv:.4f}) | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Val P/R/F1 (Macro): {val_metrics['precision_macro']:.4f}/"
                      f"{val_metrics['recall_macro']:.4f}/"
                      f"{val_metrics['f1_macro']:.4f} | "
                      f"LR: {current_lr:.6f}")
        else:
            log_msg = (f"Epoch [{epoch + 1}/{EPOCHS}] | "
                      f"Loss: {avg_loss:.4f} (Cls: {avg_loss_cls:.4f}) | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Val P/R/F1 (Macro): {val_metrics['precision_macro']:.4f}/"
                      f"{val_metrics['recall_macro']:.4f}/"
                      f"{val_metrics['f1_macro']:.4f} | "
                      f"LR: {current_lr:.6f}")

        logger.info(log_msg)

        metric_history.append(dict(val_metrics))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics
            logger.info(f"  >>> New Best Val Acc: {best_val_acc:.4f}")

    report_metrics = select_report_metrics(
        metric_history,
        strategy=args.report_strategy,
        window=args.report_window,
    )
    logger.info(
        f"Report strategy: {args.report_strategy}, window={args.report_window}, "
        f"reported_acc={report_metrics['accuracy']:.4f}, best_acc={best_val_acc:.4f}"
    )
    return report_metrics


def run_single_experiment(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODALITY = args.modality
    NUM_ITERATIONS = args.num_iterations

    SOURCE_ROOT = args.source_root
    TARGET_ROOT = args.target_root
    target_weather = os.path.basename(TARGET_ROOT)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs_single")
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"{MODALITY}_{target_weather}_{timestamp}.log")

    logger = logging.getLogger(f'SingleModality_{MODALITY}_{target_weather}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info(f"单模态消融实验 - {MODALITY.upper()}")
    logger.info(f"目标域天气: {target_weather}")
    logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"模态: {MODALITY}")
    if MODALITY != 'ais':
        logger.info(f"源域: {SOURCE_ROOT}")
        logger.info(f"目标域: {TARGET_ROOT}")
    else:
        logger.info(f"AIS数据路径: {args.ais_data_path}")
        logger.info(f"AIS架构: {args.ais_architecture}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"总轮数: {args.epochs}")
    logger.info(f"迭代次数: {NUM_ITERATIONS}")
    logger.info(f"特征维度: {args.feature_dim}")
    logger.info(f"域适应: {'启用' if args.use_domain_adaptation else '禁用'}")
    logger.info(f"报告策略: {args.report_strategy}, 窗口: {args.report_window}")
    logger.info(f"对抗损失权重: {args.adv_loss_weight}")
    logger.info("=" * 80)

    all_iteration_results = []

    for iteration in range(NUM_ITERATIONS):
        seed = 42 + iteration
        logger.info(f"\n{'='*80}")
        logger.info(f"迭代 {iteration + 1}/{NUM_ITERATIONS}")
        logger.info(f"随机种子: {seed}")
        logger.info(f"{'='*80}")

        best_metrics = run_single_iteration(args, seed, logger)
        
        if best_metrics:
            all_iteration_results.append(best_metrics)
            logger.info(f"\n迭代 {iteration + 1} 报告指标:")
            logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            logger.info(f"  Precision (Macro): {best_metrics['precision_macro']:.4f}")
            logger.info(f"  Recall (Macro): {best_metrics['recall_macro']:.4f}")
            logger.info(f"  F1 (Macro): {best_metrics['f1_macro']:.4f}")
            logger.info(f"  Precision (Micro): {best_metrics['precision_micro']:.4f}")
            logger.info(f"  Recall (Micro): {best_metrics['recall_micro']:.4f}")
            logger.info(f"  F1 (Micro): {best_metrics['f1_micro']:.4f}")
            logger.info(
                "  Val class coverage: "
                f"{best_metrics['val_present_classes']:.0f}/"
                f"{best_metrics['val_total_classes']:.0f} "
                f"({best_metrics['val_class_coverage']:.4f}), "
                f"samples={best_metrics['val_samples']:.0f}"
            )

    logger.info("\n" + "=" * 80)
    logger.info(f"{NUM_ITERATIONS}次迭代统计结果")
    logger.info("=" * 80)

    metrics_to_report = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                        'precision_micro', 'recall_micro', 'f1_micro',
                        'val_class_coverage', 'val_present_classes',
                        'val_total_classes', 'val_samples']

    summary_results = {}
    for metric_name in metrics_to_report:
        values = np.array([r[metric_name] for r in all_iteration_results])
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        summary_results[metric_name] = {
            'values': values,
            'mean': mean_val,
            'std': std_val
        }
        
        logger.info(f"\n{metric_name}:")
        logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
        logger.info(f"  均值 (Mean): {mean_val:.4f}")
        logger.info(f"  标准差 (Std): {std_val:.4f}")

    logger.info(f"\n训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    return summary_results


def main():
    parser = argparse.ArgumentParser(description='单模态消融实验')
    parser.add_argument('--modality', type=str, default='all',
                       choices=['vis', 'ir', 'ais', 'all'],
                       help='模态类型: vis(可见光), ir(红外), ais(AIS信号), all(全部)')
    parser.add_argument('--source_root', type=str, default=r"D:\Code\JMDA-Net\Data\晴天",
                       help='源域数据路径')
    parser.add_argument('--target_root', type=str, default='all',
                       help='目标域数据路径，或填 all 自动遍历 source_root 同级目录下其他天气文件夹')
    parser.add_argument('--ais_data_path', type=str,
                       default=r"D:\Code\JMDA-Net\Data\AIS\balanced_AIS-dataset_16classes_100persample.mat",
                       help='AIS数据路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--num_iterations', type=int, default=5, help='迭代次数')
    parser.add_argument('--use_target_labels', action='store_true',
                       help='使用目标域训练标签进行半监督训练；默认不使用目标域标签')
    parser.add_argument('--target_label_ratio', type=float, default=0.35,
                       help='fraction of target batch labels used for controlled semi-supervision when --use_target_labels is off')
    parser.add_argument('--target_cls_weight', type=float, default=0.50,
                       help='weight for the controlled target classification loss when --use_target_labels is off')
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--lr_feature', type=float, default=1e-5, help='特征提取器学习率')
    parser.add_argument('--lr_other', type=float, default=5e-4, help='其他模块学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--adv_loss_weight', type=float, default=0.08,
                       help='domain adversarial loss weight for single-modality experiments')
    parser.add_argument('--report_strategy', type=str, default='last_window',
                       choices=['best', 'last', 'last_window'],
                       help='which epoch metrics to report for each iteration')
    parser.add_argument('--report_window', type=int, default=10,
                       help='number of final epochs averaged when report_strategy=last_window')
    parser.add_argument('--ais_architecture', type=str, default='mlp',
                       choices=['mlp', 'deep_mlp', 'cnn1d'],
                       help='AIS特征提取器架构')
    parser.add_argument('--use_domain_adaptation', action='store_true', default=True,
                       help='是否使用域适应')
    parser.add_argument('--no_domain_adaptation', dest='use_domain_adaptation',
                       action='store_false', help='禁用域适应')

    args = parser.parse_args()

    # 可见光单模态结果已稳定，默认 all 只继续跑需要跟踪的 IR/AIS。
    modalities = ['ir', 'ais'] if args.modality == 'all' else [args.modality]

    def check(d):
        return d == "雾天" or d == "雨天" or d == "逆光" or d == "黑天"

    if args.target_root == 'all':
        data_root = os.path.dirname(args.source_root)
        src_name = os.path.basename(args.source_root)
        target_roots = []
        if os.path.isdir(data_root):
            for d in sorted(os.listdir(data_root)):
                if check(d):
                    full_path = os.path.join(data_root, d)
                    if os.path.isdir(full_path) and d != src_name:
                        target_roots.append(full_path)
        if not target_roots:
            raise ValueError(f"未找到可用目标域目录，请检查 source_root 同级目录: {data_root}")
    else:
        target_roots = [args.target_root]

    exp_results = []
    total_exps = len(modalities) * len(target_roots)
    exp_idx = 0

    for modality in modalities:
        for target_root in target_roots:
            exp_idx += 1
            print(f"\n[Batch {exp_idx}/{total_exps}] modality={modality}, target={target_root}")
            exp_args = argparse.Namespace(**vars(args))
            exp_args.modality = modality
            exp_args.target_root = target_root

            summary = run_single_experiment(exp_args)
            exp_results.append({
                'modality': modality,
                'target_root': target_root,
                'summary': summary
            })

    print("\n" + "=" * 80)
    print("批量实验完成，结果汇总:")
    print("=" * 80)
    for i, r in enumerate(exp_results, 1):
        target_name = os.path.basename(r['target_root'])
        acc_mean = r['summary']['accuracy']['mean']
        acc_std = r['summary']['accuracy']['std']
        print(f"{i:02d}. modality={r['modality']:<3} | target={target_name} | "
              f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
