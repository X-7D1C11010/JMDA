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
import torch.nn.functional as F

from DataLoad import MultiModalDomainDataset
from main import PairedClassSampler
from Models import VisualFeatureExtractor, IRFeatureExtractor, Classifier
from Tensor import TensorBasedAlignmentStable
from Generator import NeuralOptimalTransportGenerator
from Discriminator import (
    DomainDiscriminator,
    GradientReversal,
    compute_discriminator_loss,
    compute_generator_loss,
)


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


class ChannelConcatenation(nn.Module):
    """Ablation replacement for Tensor alignment.

    This module deliberately has no trainable parameters. It keeps each
    modality feature in its native feature space and lets the caller fuse them
    by channel concatenation, so the fused dimension is sum(input_dims).
    """

    def __init__(self, expected_dims=None):
        super(ChannelConcatenation, self).__init__()
        self.expected_dims = expected_dims

    def _check_modalities(self, source_modalities, target_modalities):
        if len(source_modalities) != len(target_modalities):
            raise ValueError("Source and target must contain the same number of modalities.")
        if self.expected_dims is not None and len(source_modalities) != len(self.expected_dims):
            raise ValueError(
                f"Expected {len(self.expected_dims)} modalities, got {len(source_modalities)}."
            )
        for idx, (src_feat, tgt_feat) in enumerate(zip(source_modalities, target_modalities)):
            if src_feat.shape[1] != tgt_feat.shape[1]:
                raise ValueError(
                    f"Modality {idx} source/target dims mismatch: "
                    f"{src_feat.shape[1]} vs {tgt_feat.shape[1]}"
                )
            if self.expected_dims is not None and src_feat.shape[1] != self.expected_dims[idx]:
                raise ValueError(
                    f"Modality {idx} expected dim {self.expected_dims[idx]}, "
                    f"got {src_feat.shape[1]}."
                )

    def forward(self, source_modalities, target_modalities):
        self._check_modalities(source_modalities, target_modalities)
        zero_loss = source_modalities[0].new_tensor(0.0)
        return source_modalities, target_modalities, zero_loss


class BinaryDomainDiscriminator(nn.Module):
    """Standard DANN-style source/target discriminator for w/o OT.

    OT ablation must not introduce an intermediate-domain generator. This
    discriminator only separates source (0) from target (1), and GRL provides
    the adversarial signal to the feature extractors/fusion module.
    """

    def __init__(self, feature_dim, hidden_dims=None, dropout=0.3):
        super(BinaryDomainDiscriminator, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        input_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2))
        self.discriminator = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features, alpha=1.0, use_grl=False):
        if use_grl:
            features = GradientReversal.apply(features, alpha)
        return self.discriminator(features)


def compute_binary_domain_loss(src_logits, tgt_logits):
    batch_size = src_logits.size(0)
    device = src_logits.device
    src_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    tgt_labels = torch.ones(batch_size, dtype=torch.long, device=device)
    return 0.5 * (F.cross_entropy(src_logits, src_labels) + F.cross_entropy(tgt_logits, tgt_labels))


class UnpairedDomainSampler:
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
            yield src_batch, tgt_batch

    def __len__(self):
        return self.n_batches


def evaluate(net_vis, net_ir, tal_module, classifier, dataloader, device, 
             label_map, use_tensor_module):
    net_vis.eval()
    net_ir.eval()
    tal_module.eval()
    classifier.eval()

    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            vis_inputs = data['vis'].to(device)
            ir_inputs = data['ir'].to(device)
            labels = data['label'].to(device)

            f_vis = net_vis(vis_inputs)
            f_ir = net_ir(ir_inputs)

            if use_tensor_module:
                # Evaluation has only target-domain features. Use the target
                # projection matrices directly, matching the full-model path.
                p_vis = torch.mm(f_vis, tal_module.V_matrices[0])
                p_ir = torch.mm(f_ir, tal_module.V_matrices[1])
            else:
                # w/o Tensor: pure channel concatenation of native features.
                p_vis, p_ir = f_vis, f_ir

            features = torch.cat([p_vis, p_ir], dim=1)

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
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    SOURCE_ROOT = args.source_root
    TARGET_ROOT = args.target_root
    target_weather = os.path.basename(TARGET_ROOT)

    src_train_ds = MultiModalDomainDataset(SOURCE_ROOT, domain_type='source', phase='train')
    global_map = src_train_ds.get_label_map()

    tgt_train_ds = MultiModalDomainDataset(TARGET_ROOT, domain_type='target', phase='train',
                                           global_label_map=global_map)

    tgt_val_ds = MultiModalDomainDataset(TARGET_ROOT, domain_type='target', phase='val',
                                         global_label_map=global_map, val_augment=False)

    if args.use_target_labels:
        paired_loader = PairedClassSampler(src_train_ds, tgt_train_ds, BATCH_SIZE)
        logger.info("Training mode: supervised target labels enabled (class-paired batches).")
    else:
        paired_loader = UnpairedDomainSampler(src_train_ds, tgt_train_ds, BATCH_SIZE)
        logger.info("Training mode: unsupervised target adaptation (target labels ignored in training).")
    val_loader = DataLoader(tgt_val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                           drop_last=False, num_workers=0)

    VIS_DIM = 512
    IR_DIM = 512
    PROJ_DIM = 128
    TENSOR_LOSS_WEIGHT = 0.3 if args.use_tensor_module else 0.0
    ADV_LOSS_WEIGHT = 0.15
    DISCRIMINATOR_UPDATE_INTERVAL = 2

    if args.use_tensor_module:
        fused_dim = PROJ_DIM * 2
        fusion_desc = f"Tensor alignment ({VIS_DIM}+{IR_DIM} -> {PROJ_DIM}+{PROJ_DIM})"
    else:
        fused_dim = VIS_DIM + IR_DIM
        fusion_desc = f"Channel concatenation ({VIS_DIM}+{IR_DIM})"

    net_vis = VisualFeatureExtractor(output_dim=VIS_DIM).to(DEVICE)
    net_ir = IRFeatureExtractor(output_dim=IR_DIM).to(DEVICE)

    set_requires_grad(net_ir, True)
    set_requires_grad(net_vis, False)
    for name, param in net_vis.named_parameters():
        if "layer2" in name or "layer3" in name or "layer4" in name or "proj" in name:
            param.requires_grad = True

    if args.use_tensor_module:
        tal_module = TensorBasedAlignmentStable(
            input_dims=[VIS_DIM, IR_DIM], output_dims=[PROJ_DIM, PROJ_DIM], num_modalities=2
        ).to(DEVICE)
    else:
        tal_module = ChannelConcatenation(expected_dims=[VIS_DIM, IR_DIM]).to(DEVICE)

    if args.use_ot_module:
        generator = NeuralOptimalTransportGenerator(feature_dim=fused_dim).to(DEVICE)
        discriminator = DomainDiscriminator(feature_dim=fused_dim).to(DEVICE)
    else:
        generator = None
        discriminator = BinaryDomainDiscriminator(feature_dim=fused_dim).to(DEVICE)

    logger.info(f"Feature fusion: {fusion_desc}; fused_dim={fused_dim}")
    logger.info(
        "Domain adaptation: "
        + ("Neural OT generator with 3-domain discriminator" if args.use_ot_module
           else "standard source-target adversarial training, no generator")
    )
    logger.info(f"Loss weights: tensor={TENSOR_LOSS_WEIGHT:.2f}, adv={ADV_LOSS_WEIGHT:.2f}")

    classifier = Classifier(input_dim=fused_dim, num_classes=len(global_map)).to(DEVICE)

    vis_params = [p for p in net_vis.parameters() if p.requires_grad]
    rest_params = [p for p in net_ir.parameters() if p.requires_grad] + \
                  list(tal_module.parameters()) + \
                  list(classifier.parameters())
    if generator is not None:
        rest_params += list(generator.parameters())

    optimizer_g = optim.AdamW([
        {'params': vis_params, 'lr': 1e-5},
        {'params': rest_params, 'lr': 5e-4}
    ], weight_decay=1e-4)

    optimizer_d = optim.AdamW(discriminator.parameters(), lr=5e-4, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer_g, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    criterion_cls = LabelSmoothingCrossEntropy(eps=0.1)

    best_val_acc = 0.0
    best_metrics = None

    for epoch in range(EPOCHS):
        net_vis.train()
        net_ir.train()
        tal_module.train()
        if generator is not None:
            generator.train()
        classifier.train()
        discriminator.train()

        loss_accum = 0.0
        loss_cls_accum = 0.0
        loss_tal_accum = 0.0
        loss_adv_accum = 0.0
        train_correct = 0
        train_total = 0
        steps = 0

        for src_data, tgt_data in paired_loader:
            steps += 1

            s_vis = src_data['vis'].to(DEVICE)
            s_ir = src_data['ir'].to(DEVICE)
            s_label = src_data['label'].to(DEVICE)
            t_vis = tgt_data['vis'].to(DEVICE)
            t_ir = tgt_data['ir'].to(DEVICE)
            t_label = tgt_data['label'].to(DEVICE)

            f_s_vis, f_s_ir = net_vis(s_vis), net_ir(s_ir)
            f_t_vis, f_t_ir = net_vis(t_vis), net_ir(t_ir)

            # Feature fusion ablation:
            # - with Tensor: project each modality through TAL.
            # - w/o Tensor: keep native modality features and concatenate channels.
            (p_s_vis, p_s_ir), (p_t_vis, p_t_ir), loss_tal = tal_module(
                [f_s_vis, f_s_ir],
                [f_t_vis, f_t_ir],
            )

            feat_src = torch.cat([p_s_vis, p_s_ir], dim=1)
            feat_tgt = torch.cat([p_t_vis, p_t_ir], dim=1)

            feat_mid = generator(feat_src, feat_tgt) if args.use_ot_module else None

            if steps % DISCRIMINATOR_UPDATE_INTERVAL == 0:
                optimizer_d.zero_grad()
                if args.use_ot_module:
                    loss_d, _ = compute_discriminator_loss(
                        discriminator(feat_src.detach()),
                        discriminator(feat_tgt.detach()),
                        discriminator(feat_mid.detach())
                    )
                else:
                    loss_d = compute_binary_domain_loss(
                        discriminator(feat_src.detach()),
                        discriminator(feat_tgt.detach())
                    )
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()

            optimizer_g.zero_grad()

            pred_src = classifier(feat_src)
            pred_tgt = classifier(feat_tgt)

            loss_cls_src = criterion_cls(pred_src, s_label)
            loss_cls_total = loss_cls_src
            if args.use_target_labels:
                loss_cls_tgt = criterion_cls(pred_tgt, t_label)
                loss_cls_total = loss_cls_total + loss_cls_tgt
            if args.use_ot_module:
                pred_mid = classifier(feat_mid)
                loss_cls_mid = criterion_cls(pred_mid, s_label)
                loss_cls_total = loss_cls_total + loss_cls_mid

            alpha = min(2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0, 1.0)

            # Domain adaptation ablation:
            # - with OT: generator produces an intermediate domain and is trained
            #   to confuse the 3-domain discriminator.
            # - w/o OT: no generator or intermediate feature is created; GRL
            #   applies standard source-target adversarial training.
            set_requires_grad(discriminator, False)
            if args.use_ot_module:
                loss_adv = compute_generator_loss(
                    discriminator(feat_mid, use_grl=True, alpha=alpha),
                    'kl_uniform'
                )
            else:
                loss_adv = compute_binary_domain_loss(
                    discriminator(feat_src, use_grl=True, alpha=alpha),
                    discriminator(feat_tgt, use_grl=True, alpha=alpha)
                )

            loss_total = (
                loss_cls_total
                + TENSOR_LOSS_WEIGHT * loss_tal
                + ADV_LOSS_WEIGHT * loss_adv
            )

            loss_total.backward()
            set_requires_grad(discriminator, True)
            torch.nn.utils.clip_grad_norm_(net_vis.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(net_ir.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(tal_module.parameters(), max_norm=1.0)
            if generator is not None:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer_g.step()
            if args.use_tensor_module:
                tal_module.apply_orthogonal_projection()

            loss_accum += loss_total.item()
            loss_cls_accum += loss_cls_total.item()
            loss_tal_accum += loss_tal.item()
            loss_adv_accum += loss_adv.item()

            train_logits = pred_tgt if args.use_target_labels else pred_src
            train_labels = t_label if args.use_target_labels else s_label
            _, predicted = torch.max(train_logits.data, 1)
            train_correct += (predicted == train_labels).sum().item()
            train_total += train_labels.size(0)

        val_metrics = evaluate(net_vis, net_ir, tal_module, classifier, 
                              val_loader, DEVICE, global_map, args.use_tensor_module)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_metrics['accuracy']

        scheduler.step(val_acc)
        current_lr = optimizer_g.param_groups[0]['lr']

        avg_loss = loss_accum / steps if steps > 0 else 0.0
        avg_loss_cls = loss_cls_accum / steps if steps > 0 else 0.0
        avg_loss_tal = loss_tal_accum / steps if steps > 0 else 0.0
        avg_loss_adv = loss_adv_accum / steps if steps > 0 else 0.0

        log_msg = (f"Epoch [{epoch + 1}/{EPOCHS}] | "
                   f"Loss: {avg_loss:.4f} (Cls: {avg_loss_cls:.4f}, TAL: {avg_loss_tal:.4f}, Adv: {avg_loss_adv:.4f}) | "
                   f"Train Acc: {train_acc:.4f} | "
                   f"Val Acc: {val_acc:.4f} | "
                   f"Val P/R/F1 (Macro): {val_metrics['precision_macro']:.4f}/"
                   f"{val_metrics['recall_macro']:.4f}/"
                   f"{val_metrics['f1_macro']:.4f} | "
                   f"LR: {current_lr:.6f}")

        logger.info(log_msg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics
            logger.info(f"  >>> New Best Val Acc: {best_val_acc:.4f}")

    return best_metrics


def run_ablation_experiment(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_ITERATIONS = args.num_iterations

    SOURCE_ROOT = args.source_root
    TARGET_ROOT = args.target_root
    target_weather = os.path.basename(TARGET_ROOT)

    tensor_flag = "with_tensor" if args.use_tensor_module else "without_tensor"
    ot_flag = "with_ot" if args.use_ot_module else "without_ot"
    ablation_name = f"{tensor_flag}_{ot_flag}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs_module")
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"{ablation_name}_{target_weather}_{timestamp}.log")

    logger = logging.getLogger(f'ModuleAblation_{ablation_name}_{target_weather}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info(f"模块消融实验 - {ablation_name}")
    logger.info(f"目标域天气: {target_weather}")
    logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"源域: {SOURCE_ROOT}")
    logger.info(f"目标域: {TARGET_ROOT}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"总轮数: {args.epochs}")
    logger.info(f"迭代次数: {NUM_ITERATIONS}")
    logger.info(f"使用Tensor模块: {'是' if args.use_tensor_module else '否'}")
    logger.info(f"使用OT模块: {'是' if args.use_ot_module else '否'}")
    logger.info(f"特征融合方式: {'Tensor对齐' if args.use_tensor_module else '通道拼接'}")
    logger.info(f"域对齐方式: {'最优传输' if args.use_ot_module else '常规对抗'}")
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
            logger.info(f"\n迭代 {iteration + 1} 最佳指标:")
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
    parser = argparse.ArgumentParser(description='模块消融实验')
    parser.add_argument('--source_root', type=str, default=r"/home/lixiang/lx/Data/晴天",
                       help='源域数据路径')
    parser.add_argument('--target_root', type=str, default='all',
                       help='目标域数据路径，或填 all 自动遍历 source_root 同级目录下其他天气文件夹')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--num_iterations', type=int, default=5, help='迭代次数')
    parser.add_argument('--use_target_labels', action='store_true',
                       help='使用目标域训练标签进行半监督训练；默认不使用目标域标签')
    parser.add_argument('--use_tensor_module', action='store_true', help='使用Tensor模块')
    parser.add_argument('--no_tensor_module', dest='use_tensor_module', action='store_false',
                       help='不使用Tensor模块（使用通道拼接）')
    parser.add_argument('--use_ot_module', action='store_true', help='使用OT模块')
    parser.add_argument('--no_ot_module', dest='use_ot_module', action='store_false',
                       help='不使用OT模块（使用常规对抗训练）')
    parser.add_argument('--ablation_mode', type=str, default='all',
                       choices=['all', 'full', 'with_tensor_with_ot',
                                'no_tensor_no_ot', 'no_tensor_with_ot', 'with_tensor_no_ot'],
                       help='消融模式')

    args = parser.parse_args()

    ablation_combinations = []
    if False and args.ablation_mode == 'all':
        ablation_combinations = [
            {'use_tensor_module': False, 'use_ot_module': False, 'name': '组合1: 无Tensor+无OT'},
            {'use_tensor_module': False, 'use_ot_module': True, 'name': '组合2: 无Tensor+有OT'},
            {'use_tensor_module': True, 'use_ot_module': False, 'name': '组合3: 有Tensor+无OT'},
        ]
    elif False and args.ablation_mode == 'no_tensor_no_ot':
        ablation_combinations = [{'use_tensor_module': False, 'use_ot_module': False, 'name': '组合1: 无Tensor+无OT'}]
    elif False and args.ablation_mode == 'no_tensor_with_ot':
        ablation_combinations = [{'use_tensor_module': False, 'use_ot_module': True, 'name': '组合2: 无Tensor+有OT'}]
    elif False and args.ablation_mode == 'with_tensor_no_ot':
        ablation_combinations = [{'use_tensor_module': True, 'use_ot_module': False, 'name': '组合3: 有Tensor+无OT'}]

    if args.ablation_mode == 'all':
        # "all" means all ablation settings only. The complete model
        # (Tensor+OT) is not repeated here; run --ablation_mode full explicitly
        # if a full-model reference is needed.
        ablation_combinations = [
            {'use_tensor_module': False, 'use_ot_module': False, 'name': 'w/o Tensor + w/o OT'},
            {'use_tensor_module': False, 'use_ot_module': True, 'name': 'w/o Tensor + with OT'},
            {'use_tensor_module': True, 'use_ot_module': False, 'name': 'with Tensor + w/o OT'},
        ]
    elif args.ablation_mode in ('full', 'with_tensor_with_ot'):
        ablation_combinations = [
            {'use_tensor_module': True, 'use_ot_module': True, 'name': 'Full: Tensor+OT'}
        ]
    elif args.ablation_mode == 'no_tensor_no_ot':
        ablation_combinations = [
            {'use_tensor_module': False, 'use_ot_module': False, 'name': 'w/o Tensor + w/o OT'}
        ]
    elif args.ablation_mode == 'no_tensor_with_ot':
        ablation_combinations = [
            {'use_tensor_module': False, 'use_ot_module': True, 'name': 'w/o Tensor + with OT'}
        ]
    elif args.ablation_mode == 'with_tensor_no_ot':
        ablation_combinations = [
            {'use_tensor_module': True, 'use_ot_module': False, 'name': 'with Tensor + w/o OT'}
        ]

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
    total_exps = len(ablation_combinations) * len(target_roots)
    exp_idx = 0

    for combo in ablation_combinations:
        for target_root in target_roots:
            exp_idx += 1
            target_name = os.path.basename(target_root)
            print(f"\n[Batch {exp_idx}/{total_exps}] {combo['name']}, target={target_name}")
            
            exp_args = argparse.Namespace(**vars(args))
            exp_args.use_tensor_module = combo['use_tensor_module']
            exp_args.use_ot_module = combo['use_ot_module']
            exp_args.target_root = target_root

            summary = run_ablation_experiment(exp_args)
            exp_results.append({
                'ablation_name': combo['name'],
                'target_root': target_root,
                'summary': summary
            })

    print("\n" + "=" * 80)
    print("模块消融实验完成，结果汇总:")
    print("=" * 80)
    for i, r in enumerate(exp_results, 1):
        target_name = os.path.basename(r['target_root'])
        acc_mean = r['summary']['accuracy']['mean']
        acc_std = r['summary']['accuracy']['std']
        print(f"{i:02d}. {r['ablation_name']} | target={target_name} | "
              f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
