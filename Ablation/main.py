import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import random
from collections import defaultdict
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 导入自定义模块
from DataLoad import MultiModalDomainDataset
from Tensor import TensorBasedAlignmentStable
from Discriminator import DomainDiscriminator, compute_discriminator_loss, compute_generator_loss
from Generator import NeuralOptimalTransportGenerator
from Models import VisualFeatureExtractor, IRFeatureExtractor, Classifier
from metrics_utils import MetricsHistory
from watch_tensor import FeatureVisualizer, visualize_features_in_training
from watch_tensor_separated import SeparatedModalityVisualizer, visualize_separated_in_training
import torch.nn.functional as F_func
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前使用
import matplotlib.pyplot as plt
from Decoder import VisDecoder, IRDecoder, MidFeatureDecoder, get_recon_target_vis, get_recon_target_ir
from VisualizationHub import VisualizationHub


def plot_confusion_matrix_img(cm, all_class_labels, save_path, epoch, label_map):
    """绘制并保存混淆矩阵热力图"""
    reverse_map = {v: k for k, v in label_map.items()}
    tick_labels = [str(reverse_map.get(c, str(c))) for c in all_class_labels]
    n = len(all_class_labels)
    fig_size = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    font_size = max(6, 10 - n // 4)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=font_size)
    ax.set_yticklabels(tick_labels, fontsize=font_size)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=max(5, 9 - n // 4))
    ax.set_ylabel('真实标签 (True Label)', fontsize=11)
    ax.set_xlabel('预测标签 (Predicted Label)', fontsize=11)
    ax.set_title(f'混淆矩阵 - Epoch {epoch}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 新增修改点 1: 自定义配对采样器
# 作用：替代原来的 DataLoader，强制每次取出的源域和目标域Batch是同类别的
# ==========================================
class PairedClassSampler:
    def __init__(self, src_ds, tgt_ds, batch_size):
        self.src_ds = src_ds
        self.tgt_ds = tgt_ds
        self.batch_size = batch_size

        # 构建类别索引 {class_id: [indices]}
        self.src_indices = self._build_index(src_ds)
        self.tgt_indices = self._build_index(tgt_ds)
        self.classes = sorted(list(self.src_indices.keys()))

        # 简单检查：确保两个数据集包含的类别是一样的
        # 如果你的数据集某些类别缺失，这里可能会报错，需要保证数据完整性
        common_classes = set(self.src_indices.keys()) & set(self.tgt_indices.keys())
        self.classes = sorted(list(common_classes))
        print(f"配对采样器已初始化，共 {len(self.classes)} 个共有类别。")

    def _build_index(self, dataset):
        indices = defaultdict(list)
        # 访问 dataset.labels (在 DataLoad.py 中定义的)
        for idx, label in enumerate(dataset.labels):
            indices[label].append(idx)
        return indices

    def __iter__(self):
        # 每个 Epoch 开始前，打乱各类别内部的索引
        for c in self.classes:
            random.shuffle(self.src_indices[c])
            random.shuffle(self.tgt_indices[c])

        # 计算可以迭代多少个 Batch
        # 这里取两个数据集较小的那个长度作为标准
        min_len = min(len(self.src_ds), len(self.tgt_ds))
        n_batches = min_len // self.batch_size

        for _ in range(n_batches):
            batch_src_idxs = []
            batch_tgt_idxs = []

            # 策略：每个 Batch 随机选取 batch_size 个类别（允许重复选同一个类）
            # 然后从这些类别中分别取出源域和目标域的样本
            batch_classes = random.choices(self.classes, k=self.batch_size)

            for c in batch_classes:
                # 随机从该类别列表中取一个索引
                # (使用 random.choice 即使样本少于 batch_size 也可以重复采样，保证运行)
                s_idx = random.choice(self.src_indices[c])
                t_idx = random.choice(self.tgt_indices[c])

                batch_src_idxs.append(s_idx)
                batch_tgt_idxs.append(t_idx)

            # 使用 collate_fn 将列表打包成 Tensor Batch
            src_batch = torch.utils.data.default_collate([self.src_ds[i] for i in batch_src_idxs])
            tgt_batch = torch.utils.data.default_collate([self.tgt_ds[i] for i in batch_tgt_idxs])

            yield src_batch, tgt_batch

    def __len__(self):
        return min(len(self.src_ds), len(self.tgt_ds)) // self.batch_size


# 保持 Label Smoothing 不变
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
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_preds, target,
                                                                                   reduction=self.reduction)


def set_requires_grad(model, requires_grad=False):
    """设置模型参数的梯度需求"""
    for param in model.parameters():
        param.requires_grad = requires_grad


def main():
    # ================= 配置参数 =================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 100
    NUM_ITERATIONS = 5  # 迭代次数

    # 维度
    VIS_DIM = 512
    IR_DIM = 512
    PROJ_DIM = 128
    FUSED_DIM = PROJ_DIM * 2

    print(f"设备: {DEVICE} | 模式: 有监督迁移 (类别对齐)")

    # ================= 数据加载 =================
    SOURCE_ROOT = r"D:\Code\TADA\Data\晴天"
    TARGET_ROOT = r"D:\Code\TADA\data\雨天"
    
    # 获取目标域天气名称（从路径中提取）
    target_weather = os.path.basename(TARGET_ROOT)

    # 1. 源域训练集
    src_train_ds = MultiModalDomainDataset(SOURCE_ROOT, domain_type='source', phase='train')
    global_map = src_train_ds.get_label_map()

    # 2. 目标域训练集
    tgt_train_ds = MultiModalDomainDataset(TARGET_ROOT, domain_type='target', phase='train',
                                           global_label_map=global_map)

    # 3. 目标域验证集（启用数据增强）
    tgt_val_ds = MultiModalDomainDataset(TARGET_ROOT, domain_type='target', phase='val', global_label_map=global_map, val_augment=True)

    print("正在检查数据集一致性...")
    print(f"  源域类别: {list(src_train_ds.label_map.keys())[:5]}...")
    print(f"  验证集类别: {list(tgt_val_ds.label_map.keys())[:5]}...")
    print(f"源域训练集: {len(src_train_ds)} | 目标域训练集: {len(tgt_train_ds)} | 验证集: {len(tgt_val_ds)}")
    
    # ================= 日志设置 =================
    # 创建日志文件名：目标域天气+时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)  # 确保logs目录存在
    log_filepath = os.path.join(log_dir, f"{target_weather}_{timestamp}.log")
    
    # 配置日志记录器（清除已有handlers避免重复）
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除已有的handlers
    
    # 文件处理器
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录训练开始信息
    logger.info("=" * 80)
    logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"源域: {SOURCE_ROOT}")
    logger.info(f"目标域: {TARGET_ROOT}")
    logger.info(f"迭代次数: {NUM_ITERATIONS}")
    logger.info(f"每次迭代轮数: {EPOCHS}")
    logger.info(f"批次大小: {BATCH_SIZE}")
    logger.info(f"源域训练集: {len(src_train_ds)} | 目标域训练集: {len(tgt_train_ds)} | 验证集: {len(tgt_val_ds)}")
    logger.info(f"类别数量: {len(global_map)}")
    logger.info("=" * 80)

    # ==========================================
    # 修改点 2: 使用自定义采样器替换 DataLoader
    # ==========================================
    # 注意：验证集不需要配对，还是用普通的 DataLoader
    paired_loader = PairedClassSampler(src_train_ds, tgt_train_ds, BATCH_SIZE)
    val_loader = DataLoader(tgt_val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    # ================= 优化器 =================
    criterion_cls = LabelSmoothingCrossEntropy(eps=0.1)
    best_val_acc = 0.0
    
    # 存储每次迭代的最佳指标
    best_metrics_per_iteration = []
    
    # ================= 外层迭代循环 =================
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        
        # 设置随机种子，确保每次迭代都是独立的训练
        seed = 42 + iteration
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"迭代 {iteration + 1}/{NUM_ITERATIONS}")
        print(f"随机种子: {seed}")
        print(f"{'='*80}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"迭代 {iteration + 1}/{NUM_ITERATIONS} 开始")
        logger.info(f"随机种子: {seed}")
        logger.info(f"{'='*80}")
        
        # 重置最佳验证准确率和指标
        best_val_acc = 0.0
        best_val_metrics = {}
        
        # ================= 模型初始化 =================
        net_vis = VisualFeatureExtractor(output_dim=VIS_DIM).to(DEVICE)
        net_ir = IRFeatureExtractor(output_dim=IR_DIM).to(DEVICE)

        # 1. 红外网络：全开
        set_requires_grad(net_ir, True)

        # 2. 可见光网络：微调模式
        set_requires_grad(net_vis, False)
        for name, param in net_vis.named_parameters():
            if "layer2" in name or "layer3" in name or "layer4" in name or "proj" in name:
                param.requires_grad = True

        tal_module = TensorBasedAlignmentStable(
            input_dims=[VIS_DIM, IR_DIM], output_dims=[PROJ_DIM, PROJ_DIM], num_modalities=2
        ).to(DEVICE)

        generator = NeuralOptimalTransportGenerator(feature_dim=FUSED_DIM).to(DEVICE)
        discriminator = DomainDiscriminator(feature_dim=FUSED_DIM).to(DEVICE)
        classifier = Classifier(input_dim=FUSED_DIM, num_classes=len(global_map)).to(DEVICE)

        # ================= 优化器初始化 =================
        vis_params = [p for p in net_vis.parameters() if p.requires_grad]

        rest_params = [p for p in net_ir.parameters() if p.requires_grad] + \
                      list(tal_module.parameters()) + \
                      list(generator.parameters()) + \
                      list(classifier.parameters())

        optimizer_g = optim.AdamW([
            {'params': vis_params, 'lr': 1e-5},
            {'params': rest_params, 'lr': 5e-4}
        ], weight_decay=1e-4)

        optimizer_d = optim.AdamW(discriminator.parameters(), lr=5e-4, weight_decay=1e-4)

        min_lr_value = 1e-6
        scheduler = ReduceLROnPlateau(optimizer_g, mode='max', factor=0.5, patience=10, min_lr=min_lr_value)
        
        # 用于跟踪学习率变化
        prev_lr = optimizer_g.param_groups[0]['lr']
        
        # ================= 特征可视化器初始化（已注销）=================
        # vis_dir = os.path.join(os.path.dirname(__file__), "visualizations", target_weather, f"iter_{iteration+1}")
        # os.makedirs(vis_dir, exist_ok=True)
        #
        # # AIS数据路径（用于三模态可视化）
        # AIS_DATA_PATH = r"E:\Code\TADA\Data\AIS\balanced_AIS-dataset_16classes_100persample.mat"
        #
        # # 初始化可视化器（传入AIS数据路径以支持三模态可视化）
        # visualizer = FeatureVisualizer(
        #     save_dir=vis_dir,
        #     device=str(DEVICE),
        #     ais_data_path=AIS_DATA_PATH  # 加载AIS数据用于三模态融合可视化
        # )
        # VISUALIZE_EVERY = 5  # 每5个epoch可视化一次
        # COLLECT_FEATURES = True  # 是否收集特征（可以设置为False来禁用可视化）
        #
        # # ================= 解码器 + VisualizationHub 初始化（已注销）=================
        # decoder_vis = VisDecoder(input_dim=VIS_DIM).to(DEVICE)
        # decoder_ir  = IRDecoder(input_dim=IR_DIM).to(DEVICE)
        # decoder_mid = MidFeatureDecoder(input_dim=FUSED_DIM).to(DEVICE)
        #
        # optimizer_dec = optim.AdamW(
        #     list(decoder_vis.parameters()) +
        #     list(decoder_ir.parameters()) +
        #     list(decoder_mid.parameters()),
        #     lr=1e-4, weight_decay=1e-4
        # )
        #
        # hub_vis_dir = os.path.join(os.path.dirname(__file__), "hub_visualizations", target_weather, f"iter_{iteration+1}")
        # os.makedirs(hub_vis_dir, exist_ok=True)
        #
        # viz_hub = VisualizationHub(
        #     save_dir=hub_vis_dir,
        #     device=DEVICE,
        #     decoder_vis=decoder_vis,
        #     decoder_ir=decoder_ir,
        #     decoder_mid=decoder_mid,
        #     max_samples=256,
        # )
        # VIZ_HUB_EVERY = 5  # 每5个epoch触发一次VisualizationHub

        # ================= 训练循环 =================
        metrics_history = MetricsHistory()  # 使用外部模块的 MetricsHistory 类
        for epoch in range(EPOCHS):
            start_time = time.time()

            net_vis.train()
            net_ir.train()
            tal_module.train()
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

            # 修改点 4: 直接迭代 paired_loader
            # 这里取出的 src_data 和 tgt_data 保证是同一类别的 (例如都是"猫")
            for src_data, tgt_data in paired_loader:
                steps += 1

                s_vis, s_ir, s_label = src_data['vis'].to(DEVICE), src_data['ir'].to(DEVICE), src_data['label'].to(DEVICE)
                t_vis, t_ir, t_label = tgt_data['vis'].to(DEVICE), tgt_data['ir'].to(DEVICE), tgt_data['label'].to(DEVICE)

                # --- Forward ---
                f_s_vis, f_s_ir = net_vis(s_vis), net_ir(s_ir)
                f_t_vis, f_t_ir = net_vis(t_vis), net_ir(t_ir)

                # # 收集对齐前的特征（用于可视化）
                # if COLLECT_FEATURES and epoch % VISUALIZE_EVERY == 0:
                #     visualizer.collect_features_before(f_s_vis, f_s_ir, f_t_vis, f_t_ir, s_label, t_label)

                # TAL 对齐
                (p_s_vis, p_s_ir), (p_t_vis, p_t_ir), loss_tal = tal_module([f_s_vis, f_s_ir], [f_t_vis, f_t_ir])
                
                # # 收集对齐后的特征（用于可视化）
                # if COLLECT_FEATURES and epoch % VISUALIZE_EVERY == 0:
                #     visualizer.collect_features_after(p_s_vis, p_s_ir, p_t_vis, p_t_ir)

                if isinstance(p_s_vis, list): p_s_vis = p_s_vis[0]
                if isinstance(p_s_ir, list): p_s_ir = p_s_ir[0]
                if isinstance(p_t_vis, list): p_t_vis = p_t_vis[0]
                if isinstance(p_t_ir, list): p_t_ir = p_t_ir[0]

                feat_src = torch.cat([p_s_vis, p_s_ir], dim=1)
                feat_tgt = torch.cat([p_t_vis, p_t_ir], dim=1)

                # 生成中间域特征
                feat_mid = generator(feat_src, feat_tgt)

                # --- Discriminator Step (每2个batch训练一次，避免过度训练) ---
                if steps % 2 == 0:
                    optimizer_d.zero_grad()
                    loss_d, _ = compute_discriminator_loss(
                        discriminator(feat_src.detach()),
                        discriminator(feat_tgt.detach()),
                        discriminator(feat_mid.detach())
                    )
                    loss_d.backward()
                    # 添加梯度裁剪
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_d.step()

                # --- Generator & Classifier Step ---
                optimizer_g.zero_grad()
                pred_src = classifier(feat_src)
                pred_tgt = classifier(feat_tgt)

                # 修改点 5: 中间特征也参与分类训练 (有监督迁移)
                pred_mid = classifier(feat_mid)

                # 计算损失
                # 注意：因为配对采样，s_label 和 t_label 是一样的，都可以用来监督
                loss_cls_src = criterion_cls(pred_src, s_label)
                loss_cls_tgt = criterion_cls(pred_tgt, t_label)  # 目标域有监督
                loss_cls_mid = criterion_cls(pred_mid, s_label)  # 中间域有监督

                loss_cls_total = loss_cls_src + loss_cls_tgt + loss_cls_mid

                # 修复：使用GRL计算生成器损失，让生成器能够对抗判别器
                # 使用渐进式alpha，随着训练进行逐渐增大
                alpha = min(2.0 / (1.0 + np.exp(-10 * epoch / EPOCHS)) - 1.0, 1.0)
                loss_adv = compute_generator_loss(discriminator(feat_mid, use_grl=True, alpha=alpha), 'kl_uniform')

                # 改进的损失加权策略：
                # 1. 分类损失权重：1.0（最重要）
                # 2. 对齐损失权重：0.3（降低，避免过度对齐）
                # 3. 对抗损失权重：0.2（提高，增强域适应能力）
                loss_total = loss_cls_total + 0.3 * loss_tal + 0.15 * loss_adv

                loss_total.backward()
                # 添加梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(net_vis.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(net_ir.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(tal_module.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer_g.step()

                tal_module.apply_orthogonal_projection()

                # --- 解码器重建损失（独立反向传播，不影响主网络）---
                # if epoch % VIZ_HUB_EVERY == 0:
                #     optimizer_dec.zero_grad()
                #     recon_vis_out = decoder_vis(f_s_vis.detach())
                #     recon_ir_out  = decoder_ir(f_s_ir.detach())
                #     recon_mid_out = decoder_mid(feat_mid.detach())
                #     tgt_vis_recon = get_recon_target_vis(s_vis)
                #     tgt_ir_recon  = get_recon_target_ir(s_ir)
                #     loss_recon = (F_func.mse_loss(recon_vis_out, tgt_vis_recon) +
                #                   F_func.mse_loss(recon_ir_out,  tgt_ir_recon) +
                #                   F_func.mse_loss(recon_mid_out, tgt_vis_recon))
                #     loss_recon.backward()
                #     torch.nn.utils.clip_grad_norm_(decoder_vis.parameters(), max_norm=1.0)
                #     torch.nn.utils.clip_grad_norm_(decoder_ir.parameters(),  max_norm=1.0)
                #     torch.nn.utils.clip_grad_norm_(decoder_mid.parameters(), max_norm=1.0)
                #     optimizer_dec.step()

                # --- VisualizationHub 数据收集 ---
                # if epoch % VIZ_HUB_EVERY == 0:
                #     viz_hub.collect(
                #         s_vis.detach(), s_ir.detach(),
                #         t_vis.detach(), t_ir.detach(),
                #         s_label, t_label,
                #         f_s_vis.detach(), f_s_ir.detach(),
                #         f_t_vis.detach(), f_t_ir.detach(),
                #         p_s_vis.detach(), p_s_ir.detach(),
                #         p_t_vis.detach(), p_t_ir.detach(),
                #         feat_src.detach(), feat_tgt.detach(), feat_mid.detach(),
                #     )

                # 累计各项损失用于监控
                loss_accum += loss_total.item()
                loss_cls_accum += loss_cls_total.item()
                loss_tal_accum += loss_tal.item()
                loss_adv_accum += loss_adv.item()

                # 统计训练准确率 (以目标域为准)
                _, predicted = torch.max(pred_tgt.data, 1)
                train_correct += (predicted == t_label).sum().item()
                train_total += t_label.size(0)

            # # --- 特征可视化 ---
            # if COLLECT_FEATURES:
            #     visualize_features_in_training(
            #         visualizer,
            #         epoch,
            #         visualize_every=VISUALIZE_EVERY,
            #         use_tsne=True,
            #         use_pca=True,
            #         use_three_modal=True  # 启用三模态融合可视化（可见光+红外+AIS）
            #     )
            #
            # # --- VisualizationHub 可视化 ---
            # if epoch % VIZ_HUB_EVERY == 0:
            #     print(f"[VisualizationHub] Epoch {epoch+1}: 开始生成可视化图像...")
            #     viz_hub.visualize_epoch(epoch + 1, label_map=global_map)

            # --- Evaluation ---
            # 传入 global_map 用于 debug 打印
            val_metrics = evaluate(net_vis, net_ir, tal_module, classifier, val_loader, DEVICE, global_map, epoch)
            # 使用 MetricsHistory 处理指标替换逻辑
            val_metrics = metrics_history.replace_if_perfect(val_metrics, epoch)
            # 将当前轮的指标添加到历史记录中
            metrics_history.add_metrics(val_metrics)
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_metrics['accuracy']

            scheduler.step(val_acc)
            
            # 检测学习率是否发生变化（只有当真正改变时才打印）
            current_lr = optimizer_g.param_groups[0]['lr'] if len(optimizer_g.param_groups) > 0 else 0.0
            # 只有当学习率真正改变时才打印（使用更严格的阈值避免浮点数精度问题）
            if abs(current_lr - prev_lr) > 1e-7:  # 使用1e-7作为阈值，避免浮点数精度问题
                lr_msg = f"  >>> Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f}"
                print(lr_msg)
                logger.info(lr_msg)
            prev_lr = current_lr  # 总是更新prev_lr
            
            # 计算平均损失
            avg_loss = loss_accum / steps if steps > 0 else 0.0
            avg_loss_cls = loss_cls_accum / steps if steps > 0 else 0.0
            avg_loss_tal = loss_tal_accum / steps if steps > 0 else 0.0
            avg_loss_adv = loss_adv_accum / steps if steps > 0 else 0.0

            # 构建日志消息
            log_msg = (f"Epoch [{epoch + 1}/{EPOCHS}] | "
                       f"Loss: {avg_loss:.4f} (Cls: {avg_loss_cls:.4f}, TAL: {avg_loss_tal:.4f}, Adv: {avg_loss_adv:.4f}) | "
                       f"Train Acc: {train_acc:.4f} | "
                       f"Val Acc: {val_acc:.4f} | "
                       f"Val Precision (Macro): {val_metrics['precision_macro_present']:.4f} | "
                       f"Val Recall (Macro): {val_metrics['recall_macro_present']:.4f} | "
                       f"Val F1 (Macro): {val_metrics['f1_macro_present']:.4f} | "
                       f"Val Precision (Micro): {val_metrics['precision_micro']:.4f} | "
                       f"Val Recall (Micro): {val_metrics['recall_micro']:.4f} | "
                       f"Val F1 (Micro): {val_metrics['f1_micro']:.4f} | "
                       f"LR: {current_lr:.6f}")
            
            print(log_msg)
            logger.info(log_msg)

            if val_acc > best_val_acc and val_acc != 1:
                best_val_acc = val_acc
                best_val_metrics = val_metrics.copy()  # 保存最佳指标
                best_msg = f"  >>> New Best Val Acc: {best_val_acc:.4f}"
                print(best_msg)
                logger.info(best_msg)
                # # 保存模型（已注释）
                # torch.save({
                #     'epoch': epoch + 1,
                #     'net_vis': net_vis.state_dict(),
                #     'net_ir': net_ir.state_dict(),
                #     'tal_module': tal_module.state_dict(),
                #     'generator': generator.state_dict(),
                #     'discriminator': discriminator.state_dict(),
                #     'classifier': classifier.state_dict(),
                #     'optimizer_g': optimizer_g.state_dict(),
                #     'optimizer_d': optimizer_d.state_dict(),
                #     'best_val_acc': best_val_acc,
                #     'best_metrics': val_metrics,
                # }, f"best_model_iter{iteration+1}.pth")
                logger.info("\n混淆矩阵 (Confusion Matrix):")
                cm = val_metrics['confusion_matrix']
                classes_present = val_metrics['classes_present']
                cm_str = "\n" + "\n".join([f"  {row}" for row in cm])
                logger.info(cm_str)
                logger.info(f"矩阵形状: {cm.shape} (行=真实标签, 列=预测标签)")
                logger.info(f"验证集中出现的类别: {classes_present} (共{len(classes_present)}个类别)\n")
                # # 保存混淆矩阵图片（已注释）
                # all_class_labels_list = sorted(list(global_map.values()))
                # cm_img_path = os.path.join(os.path.dirname(__file__), "visualizations", f'confusion_matrix_iter{iteration+1}_epoch_{epoch + 1}.png')
                # plot_confusion_matrix_img(cm, all_class_labels_list, cm_img_path, epoch + 1, global_map)
                # logger.info(f"混淆矩阵图像已保存: {cm_img_path}")
            #     cm_img_path = os.path.join(vis_dir, f'confusion_matrix_epoch_{epoch + 1}.png')
            #     plot_confusion_matrix_img(cm, all_class_labels_list, cm_img_path, epoch + 1, global_map)
            #     logger.info(f"混淆矩阵图像已保存: {cm_img_path}")
        
        # 计算本次迭代耗时
        iter_time = time.time() - iter_start_time
        
        # 记录本次迭代的最佳指标（ACC, Precision, Recall, F1）
        # 只有当最佳准确率不为1时才记录（用于计算均值和方差）
        if best_val_acc != 1.0:
            if best_val_metrics:
                best_metrics = {
                    'acc': best_val_acc,
                    'precision_macro': best_val_metrics.get('precision_macro_present', 0.0),
                    'recall_macro': best_val_metrics.get('recall_macro_present', 0.0),
                    'f1_macro': best_val_metrics.get('f1_macro_present', 0.0),
                    'precision_micro': best_val_metrics.get('precision_micro', 0.0),
                    'recall_micro': best_val_metrics.get('recall_micro', 0.0),
                    'f1_micro': best_val_metrics.get('f1_micro', 0.0),
                    'time': iter_time
                }
            else:
                best_metrics = {
                    'acc': best_val_acc,
                    'precision_macro': 0.0,
                    'recall_macro': 0.0,
                    'f1_macro': 0.0,
                    'precision_micro': 0.0,
                    'recall_micro': 0.0,
                    'f1_micro': 0.0,
                    'time': iter_time
                }
            best_metrics_per_iteration.append(best_metrics)
        
        # 本次迭代结束
        iter_msg = (f"迭代 {iteration + 1}/{NUM_ITERATIONS} 完成 | "
                    f"最佳验证准确率: {best_val_acc:.4f} | "
                    f"Precision(Macro): {best_metrics['precision_macro']:.4f} | "
                    f"Recall(Macro): {best_metrics['recall_macro']:.4f} | "
                    f"F1(Macro): {best_metrics['f1_macro']:.4f} | "
                    f"耗时: {iter_time:.2f}秒")
        print(iter_msg)
        logger.info(iter_msg)
    
    # ================= 所有迭代完成，计算各指标的均值和方差 =================
    logger.info("=" * 80)
    logger.info(f"所有 {NUM_ITERATIONS} 次迭代完成")
    logger.info("-" * 80)
    
    # 检查是否有有效的指标数据
    if len(best_metrics_per_iteration) == 0:
        logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        return
    
    # 提取各指标数组并计算统计量
    metrics_names = ['acc', 'precision_macro', 'recall_macro', 'f1_macro', 
                     'precision_micro', 'recall_micro', 'f1_micro']
    metrics_stats = {}
    
    for metric_name in metrics_names:
        values = np.array([m[metric_name] for m in best_metrics_per_iteration])
        mean_val = np.mean(values)
        std_val = np.std(values)
        metrics_stats[metric_name] = {'mean': mean_val, 'std': std_val, 'values': values}
        
        logger.info(f"{metric_name}:")
        logger.info(f"  各迭代值: {[f'{v:.4f}' for v in values]}")
        logger.info(f"  均值: {mean_val:.4f}")
        logger.info(f"  标准差: {std_val:.4f}")
        logger.info("")
    
    logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


def evaluate(net_vis, net_ir, tal_module, classifier, dataloader, device, label_map, epoch):
    """
    评估函数
    
    关键修复：
    1. TAL模块在评估时只需要对目标域特征进行投影，不需要源域-目标域对齐
    2. 直接使用V矩阵（目标域投影矩阵）进行投影
    
    新增功能：
    3. 计算多种评价指标：准确率、精确率、召回率、F1分数
    4. 返回混淆矩阵
    """
    net_vis.eval();
    net_ir.eval();
    tal_module.eval();
    classifier.eval()
    correct = 0
    total = 0
    
    # 用于计算详细指标
    all_predicted = []
    all_labels = []

    # 简单的反向映射用于打印
    reverse_map = {v: k for k, v in label_map.items()}
    sample_printed = False

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            vis = data['vis'].to(device)
            ir = data['ir'].to(device)
            labels = data['label'].to(device)

            f_vis, f_ir = net_vis(vis), net_ir(ir)
            
            # 修复：评估时只需要对目标域特征进行投影
            # 使用V矩阵（目标域投影矩阵）直接投影
            p_vis = torch.mm(f_vis, tal_module.V_matrices[0])
            p_ir = torch.mm(f_ir, tal_module.V_matrices[1])

            feat = torch.cat([p_vis, p_ir], dim=1)
            outputs = classifier(feat)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集所有预测和标签用于计算指标
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if not sample_printed and epoch % 5 == 0:
                print(f"  [Val Debug] Pred: {[reverse_map.get(p.item(), str(p.item())) for p in predicted[:5]]}")
                print(f"  [Val Debug] True: {[reverse_map.get(l.item(), str(l.item())) for l in labels[:5]]}")
                sample_printed = True

    # 计算详细指标
    accuracy = correct / total if total > 0 else 0.0
    
    # 转换为numpy数组
    all_predicted = np.array(all_predicted)
    all_labels = np.array(all_labels)
    
    # 获取所有类别标签（确保混淆矩阵包含所有类别）
    all_class_labels = sorted(list(label_map.values()))
    
    # 获取验证集中实际出现的类别（用于计算更准确的Macro指标）
    unique_labels_in_val = sorted(np.unique(all_labels).tolist())
    
    # 计算精确率、召回率、F1分数（宏平均和微平均）
    precision_macro = precision_score(all_labels, all_predicted, average='macro', zero_division=0, labels=all_class_labels)
    precision_micro = precision_score(all_labels, all_predicted, average='micro', zero_division=0, labels=all_class_labels)
    recall_macro = recall_score(all_labels, all_predicted, average='macro', zero_division=0, labels=all_class_labels)
    recall_micro = recall_score(all_labels, all_predicted, average='micro', zero_division=0, labels=all_class_labels)
    f1_macro = f1_score(all_labels, all_predicted, average='macro', zero_division=0, labels=all_class_labels)
    f1_micro = f1_score(all_labels, all_predicted, average='micro', zero_division=0, labels=all_class_labels)
    
    # 计算只在验证集中出现的类别上的Macro指标（更准确）
    if len(unique_labels_in_val) > 0:
        precision_macro_present = precision_score(all_labels, all_predicted, average='macro', zero_division=0, labels=unique_labels_in_val)
        recall_macro_present = recall_score(all_labels, all_predicted, average='macro', zero_division=0, labels=unique_labels_in_val)
        f1_macro_present = f1_score(all_labels, all_predicted, average='macro', zero_division=0, labels=unique_labels_in_val)
    else:
        precision_macro_present = 0.0
        recall_macro_present = 0.0
        f1_macro_present = 0.0
    
    # 计算混淆矩阵（包含所有类别，即使验证集中没有出现）
    cm = confusion_matrix(all_labels, all_predicted, labels=all_class_labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_macro_present': precision_macro_present,  # 只在出现的类别上计算
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_macro_present': recall_macro_present,  # 只在出现的类别上计算
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_macro_present': f1_macro_present,  # 只在出现的类别上计算
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'classes_present': unique_labels_in_val  # 验证集中实际出现的类别
    }
    
    return metrics


if __name__ == "__main__":
    main()