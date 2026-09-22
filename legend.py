import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # 【新增】引入手动构造图例对象的模块
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')  # 忽略 sklearn 的 FutureWarning

# ==========================================
# 全局设置：严格保持新罗马字体和粗体要求
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

SAVE_DIR = "visualizations_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

# 显式定义你指定的 8 个目标类别
TARGET_CLASSES = [0, 1, 4, 5, 6, 8, 10, 11]


# ==========================================
# 终极修改版单图绘制函数（手动硬编码强制图例）
# ==========================================

def _plot_tsne_single_fixed(ax, feat_src: np.ndarray, feat_tgt: np.ndarray,
                            labels_src: np.ndarray, labels_tgt: np.ndarray,
                            title: str, perplexity: int, n_iter: int):
    all_features = np.vstack([feat_src, feat_tgt])
    all_labels = np.concatenate([labels_src, labels_tgt])
    domain_labels = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42, verbose=0)
    features_2d = tsne.fit_transform(all_features)

    colors = plt.cm.tab20(np.linspace(0, 1, len(TARGET_CLASSES)))

    # 实际画散点图（不再依赖这里的 label 给图例用）
    for i, target_lbl in enumerate(TARGET_CLASSES):
        mask = (all_labels == target_lbl) | (all_labels == i)

        src_mask = mask & (domain_labels == 0)
        if np.any(src_mask):
            ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                       c=[colors[i]], marker='o', s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

        tgt_mask = mask & (domain_labels == 1)
        if np.any(tgt_mask):
            ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                       c=[colors[i]], marker='s', s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

    # 【核心：手动硬编码图例】
    custom_handles = []
    custom_labels = []
    for i, target_lbl in enumerate(TARGET_CLASSES):
        # 强制捏造带有指定颜色和样式的原型句柄
        handle = mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None',
                               markersize=8, alpha=0.6, markeredgecolor='black', markeredgewidth=0.5)
        custom_handles.append(handle)
        custom_labels.append(f'Class {target_lbl}')

    # 直接塞给 ax.legend，这样绝不可能丢弃任何一个类
    ax.legend(custom_handles, custom_labels,
              loc='lower right', bbox_to_anchor=(1.0, 1.03),
              ncol=2, fontsize=10, title='Categories', title_fontsize=11)
    ax.grid(True, alpha=0.3)


def _plot_pca_single_fixed(ax, feat_src: np.ndarray, feat_tgt: np.ndarray,
                           labels_src: np.ndarray, labels_tgt: np.ndarray,
                           title: str, n_components: int):
    all_features = np.vstack([feat_src, feat_tgt])
    all_labels = np.concatenate([labels_src, labels_tgt])
    domain_labels = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])

    pca = PCA(n_components=n_components, random_state=42)
    features_2d = pca.fit_transform(all_features)
    explained_var = pca.explained_variance_ratio_
    var_str = f' (EVR: {explained_var[0]:.2%}, {explained_var[1]:.2%})'

    colors = plt.cm.tab20(np.linspace(0, 1, len(TARGET_CLASSES)))

    for i, target_lbl in enumerate(TARGET_CLASSES):
        mask = (all_labels == target_lbl) | (all_labels == i)

        src_mask = mask & (domain_labels == 0)
        if np.any(src_mask):
            ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                       c=[colors[i]], marker='o', s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

        tgt_mask = mask & (domain_labels == 1)
        if np.any(tgt_mask):
            ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                       c=[colors[i]], marker='s', s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

    ax.set_title(title + var_str, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained_var[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.2%})', fontsize=12)

    # 【核心：手动硬编码图例】
    custom_handles = []
    custom_labels = []
    for i, target_lbl in enumerate(TARGET_CLASSES):
        handle = mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None',
                               markersize=8, alpha=0.6, markeredgecolor='black', markeredgewidth=0.5)
        custom_handles.append(handle)
        custom_labels.append(f'Class {target_lbl}')

    ax.legend(custom_handles, custom_labels,
              loc='lower right', bbox_to_anchor=(1.0, 1.03),
              ncol=2, fontsize=10, title='Categories', title_fontsize=11)
    ax.grid(True, alpha=0.3)


def _plot_three_modal_single_fixed(ax, feat_src: np.ndarray, feat_tgt: np.ndarray,
                                   labels_src: np.ndarray, labels_tgt: np.ndarray,
                                   title: str, perplexity: int, n_iter: int):
    all_features = np.vstack([feat_src, feat_tgt])
    all_labels = np.concatenate([labels_src, labels_tgt])
    domain_labels = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])

    effective_perplexity = min(perplexity, len(all_features) - 1)
    tsne = TSNE(n_components=2, perplexity=effective_perplexity, max_iter=n_iter, random_state=42, verbose=0)
    features_2d = tsne.fit_transform(all_features)

    src_center = np.mean(features_2d[domain_labels == 0], axis=0)
    tgt_center = np.mean(features_2d[domain_labels == 1], axis=0)
    domain_dist = np.linalg.norm(src_center - tgt_center)

    colors = plt.cm.tab20(np.linspace(0, 1, len(TARGET_CLASSES)))

    for i, target_lbl in enumerate(TARGET_CLASSES):
        mask = (all_labels == target_lbl) | (all_labels == i)

        src_mask = mask & (domain_labels == 0)
        if np.any(src_mask):
            ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                       c=[colors[i]], marker='o', s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

        tgt_mask = mask & (domain_labels == 1)
        if np.any(tgt_mask):
            ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                       c=[colors[i]], marker='^', s=50, alpha=0.6,
                       edgecolors='black', linewidths=0.5)

    # 域中心
    ax.scatter(src_center[0], src_center[1], c='blue', marker='*', s=200,
               edgecolors='white', linewidths=2, zorder=10)
    ax.scatter(tgt_center[0], tgt_center[1], c='red', marker='*', s=200,
               edgecolors='white', linewidths=2, zorder=10)
    ax.plot([src_center[0], tgt_center[0]], [src_center[1], tgt_center[1]], 'k--', linewidth=2, alpha=0.5)

    ax.set_title(f'{title}\nDomain Distance: {domain_dist:.2f}', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

    # 【核心：手动硬编码三模态图例】
    domain_handles = [
        mlines.Line2D([], [], color='white', markerfacecolor='blue', marker='*', linestyle='None', markersize=14,
                      markeredgecolor='white', markeredgewidth=1, label='Source Domain Center'),
        mlines.Line2D([], [], color='white', markerfacecolor='red', marker='*', linestyle='None', markersize=14,
                      markeredgecolor='white', markeredgewidth=1, label='Target Domain Center'),
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='Source Samples'),
        mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=8, label='Target Samples')
    ]
    domain_labels_list = [h.get_label() for h in domain_handles]

    class_handles = []
    class_labels = []
    for i, target_lbl in enumerate(TARGET_CLASSES):
        h = mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None',
                          markersize=8, alpha=0.6, markeredgecolor='black', markeredgewidth=0.5)
        class_handles.append(h)
        class_labels.append(f'Class {target_lbl}')

    all_handles = domain_handles + class_handles
    all_labels_legend = domain_labels_list + class_labels

    ax.legend(all_handles, all_labels_legend,
              loc='lower right', bbox_to_anchor=(1.0, 1.03),
              ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)


# ==========================================
# 测试运行模块
# ==========================================
if __name__ == "__main__":
    print(f"正在生成模拟数据，目标锁定为 8 个类: {TARGET_CLASSES}...")
    num_samples = 300
    feat_dim = 128

    # 哪怕真实数据中只有极少数标签（比如这里模拟只有 2 个标签 0和1），图例也会强制打印出全部 8 个
    mock_labels_src = np.random.randint(0, 2, size=num_samples)
    mock_labels_tgt = np.random.randint(0, 2, size=num_samples)

    mock_feat_src = np.random.randn(num_samples, feat_dim)
    mock_feat_tgt = np.random.randn(num_samples, feat_dim) + 0.5

    # 1. 测试双模态图例 (t-SNE)
    fig, axes = plt.subplots(1, 2, figsize=(20, 11.5))
    fig.suptitle('Epoch 5 - Feature Comparison of Tensor-based Alignment Module (t-SNE)',
                 fontsize=16, fontweight='bold', y=0.99)

    print("正在绘制双模态对比图...")
    _plot_tsne_single_fixed(axes[0], mock_feat_src, mock_feat_tgt, mock_labels_src, mock_labels_tgt, 'Before Alignment',
                            30, 250)
    _plot_tsne_single_fixed(axes[1], mock_feat_src, mock_feat_tgt, mock_labels_src, mock_labels_tgt, 'After Alignment',
                            30, 250)

    plt.tight_layout()
    tsne_save_path = os.path.join(SAVE_DIR, 'tsne_fixed_labels.svg')
    plt.savefig(tsne_save_path, bbox_inches='tight')
    plt.close()
    print(f">> t-SNE 矢量图已保存: {tsne_save_path}")

    # 2. 测试三模态图例
    fig, axes = plt.subplots(1, 2, figsize=(20, 11.5))
    fig.suptitle(
        'Epoch 5 - Tri-modal Feature Fusion Comparison (Optical + IR + AIS)\nEffectiveness of Tensor-based Alignment Module',
        fontsize=16, fontweight='bold', y=0.99)

    print("正在绘制三模态对比图...")
    _plot_three_modal_single_fixed(axes[0], mock_feat_src, mock_feat_tgt, mock_labels_src, mock_labels_tgt,
                                   'Before Alignment', 30, 250)
    _plot_three_modal_single_fixed(axes[1], mock_feat_src, mock_feat_tgt, mock_labels_src, mock_labels_tgt,
                                   'After Alignment', 30, 250)

    plt.tight_layout()
    tri_save_path = os.path.join(SAVE_DIR, 'three_modal_fixed_labels.svg')
    plt.savefig(tri_save_path, bbox_inches='tight')
    plt.close()
    print(f">> 三模态 矢量图已保存: {tri_save_path}")