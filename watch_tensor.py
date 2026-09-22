"""
Feature Visualization Module - Used to visualize features before and after the tensor-based alignment module.

Functions:
1. Capture features before and after feature extraction and alignment.
2. Diminishing dimensionality for visualization using t-SNE and PCA.
3. Save feature statistics and visualization images.
4. Support tri-modal fusion visualization (Optical + IR + AIS).

Usage:
    Mode 1 (Fused, Recommended): Optical + IR fusion, showing comparison before and after alignment.
    Mode 2 (Tri-modal Fusion): Optical + IR + AIS all fused together for display.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import scipy.io

# 全局设置：新罗马字体，粗体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


def load_ais_data(mat_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load AIS dataset (supports MATLAB v7.3 and older formats)
    """
    features = None
    labels = None

    try:
        data = scipy.io.loadmat(mat_path)
        for key in data.keys():
            if key.startswith('__'):
                continue
            value = data[key]
            if hasattr(value, 'shape'):
                if len(value.shape) == 2 and value.shape[0] > 1 and value.shape[1] > 1:
                    if features is None or value.size > features.size:
                        features = value
                elif len(value.shape) <= 2 and min(value.shape if len(value.shape) == 2 else (value.shape[0], 1)) == 1:
                    if labels is None:
                        labels = value.flatten()

        if features is not None and features.shape[0] < features.shape[1]:
            features = features.T
        return features, labels

    except NotImplementedError:
        pass
    except Exception as e:
        print(f"scipy.io.loadmat failed: {e}")

    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            print(f"Reading MATLAB v7.3 file using h5py...")
            for key in f.keys():
                if key.startswith('#') or key.startswith('_'):
                    continue
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    value = np.array(dataset)
                    if len(value.shape) == 2 and value.shape[0] > 1 and value.shape[1] > 1:
                        if features is None or value.size > features.size:
                            features = value
                    elif len(value.shape) <= 2:
                        if value.shape[0] == 1 or (len(value.shape) == 2 and value.shape[1] == 1):
                            if labels is None:
                                labels = value.flatten()
                        elif len(value.shape) == 1:
                            if labels is None:
                                labels = value

            if features is not None:
                if features.shape[0] < features.shape[1]:
                    features = features.T
                print(f"AIS features final shape: {features.shape}")
            if labels is not None:
                labels = labels.flatten()
                print(f"AIS labels final shape: {labels.shape}")

        return features, labels

    except ImportError:
        print("Error: h5py library is required to read MATLAB v7.3 files.")
        return None, None
    except Exception as e:
        print(f"h5py read failed: {e}")
        return None, None


class FeatureVisualizer:
    """Feature Visualizer - Supports Bi-modal and Tri-modal fusion visualization"""

    def __init__(self, save_dir: str = "visualizations", device: str = "cpu",
                 ais_data_path: str = None):
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

        self.features_before = {
            'src_vis': [], 'src_ir': [], 'src_ais': [],
            'tgt_vis': [], 'tgt_ir': [], 'tgt_ais': []
        }
        self.features_after = {
            'src_vis': [], 'src_ir': [], 'src_ais': [],
            'tgt_vis': [], 'tgt_ir': [], 'tgt_ais': []
        }
        self.labels = {'src': [], 'tgt': []}

        self.ais_features = None
        self.ais_labels = None
        self.ais_by_class = {}

        if ais_data_path and os.path.exists(ais_data_path):
            self._load_ais_data(ais_data_path)

        self.stats = {}

    def _load_ais_data(self, mat_path: str):
        self.ais_features, self.ais_labels = load_ais_data(mat_path)
        if self.ais_features is not None and self.ais_labels is not None:
            for idx, label in enumerate(self.ais_labels):
                label = int(label)
                if label not in self.ais_by_class:
                    self.ais_by_class[label] = []
                self.ais_by_class[label].append(self.ais_features[idx])
            print(f"AIS data loaded: {len(self.ais_features)} samples, {len(self.ais_by_class)} classes")
        else:
            print("Warning: Failed to load AIS data")

    def get_ais_features_by_labels(self, labels: np.ndarray) -> Optional[np.ndarray]:
        if self.ais_features is None or len(self.ais_features) == 0:
            return None

        ais_feats = []
        ais_class_keys = list(self.ais_by_class.keys())

        for label in labels:
            label = int(label)
            if label in self.ais_by_class and len(self.ais_by_class[label]) > 0:
                idx = np.random.randint(0, len(self.ais_by_class[label]))
                ais_feats.append(self.ais_by_class[label][idx])
            elif ais_class_keys:
                mapped_label = label % len(ais_class_keys)
                actual_label = ais_class_keys[mapped_label]
                idx = np.random.randint(0, len(self.ais_by_class[actual_label]))
                ais_feats.append(self.ais_by_class[actual_label][idx])
            else:
                feat_dim = self.ais_features.shape[1] if self.ais_features is not None else 128
                ais_feats.append(np.zeros(feat_dim))

        return np.array(ais_feats)

    def collect_features_before(self, f_s_vis: torch.Tensor, f_s_ir: torch.Tensor,
                                f_t_vis: torch.Tensor, f_t_ir: torch.Tensor,
                                s_label: torch.Tensor, t_label: torch.Tensor):
        s_labels_np = s_label.detach().cpu().numpy()
        t_labels_np = t_label.detach().cpu().numpy()

        self.features_before['src_vis'].append(f_s_vis.detach().cpu().numpy())
        self.features_before['src_ir'].append(f_s_ir.detach().cpu().numpy())
        self.features_before['tgt_vis'].append(f_t_vis.detach().cpu().numpy())
        self.features_before['tgt_ir'].append(f_t_ir.detach().cpu().numpy())
        self.labels['src'].append(s_labels_np)
        self.labels['tgt'].append(t_labels_np)

        if self.ais_features is not None:
            src_ais = self.get_ais_features_by_labels(s_labels_np)
            tgt_ais = self.get_ais_features_by_labels(t_labels_np)
            if src_ais is not None and len(src_ais) > 0:
                self.features_before['src_ais'].append(src_ais)
            if tgt_ais is not None and len(tgt_ais) > 0:
                self.features_before['tgt_ais'].append(tgt_ais)

    def collect_features_after(self, p_s_vis: torch.Tensor, p_s_ir: torch.Tensor,
                               p_t_vis: torch.Tensor, p_t_ir: torch.Tensor):
        if isinstance(p_s_vis, list): p_s_vis = p_s_vis[0]
        if isinstance(p_s_ir, list): p_s_ir = p_s_ir[0]
        if isinstance(p_t_vis, list): p_t_vis = p_t_vis[0]
        if isinstance(p_t_ir, list): p_t_ir = p_t_ir[0]

        self.features_after['src_vis'].append(p_s_vis.detach().cpu().numpy())
        self.features_after['src_ir'].append(p_s_ir.detach().cpu().numpy())
        self.features_after['tgt_vis'].append(p_t_vis.detach().cpu().numpy())
        self.features_after['tgt_ir'].append(p_t_ir.detach().cpu().numpy())

        if self.features_before['src_ais']:
            self.features_after['src_ais'].append(self.features_before['src_ais'][-1])
        if self.features_before['tgt_ais']:
            self.features_after['tgt_ais'].append(self.features_before['tgt_ais'][-1])

    def clear_features(self):
        self.features_before = {k: [] for k in self.features_before.keys()}
        self.features_after = {k: [] for k in self.features_after.keys()}
        self.labels = {'src': [], 'tgt': []}

    def _concatenate_features(self, feature_dict: Dict[str, List]) -> Dict[str, np.ndarray]:
        result = {}
        for key, value_list in feature_dict.items():
            result[key] = np.concatenate(value_list, axis=0) if value_list else np.array([])
        return result

    def visualize_with_tsne(self, epoch: int, max_samples: int = 1000, perplexity: int = 30, n_iter: int = 1000):
        """t-SNE Visualization (Bi-modal Fusion)"""
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)
        labels_src = np.concatenate(self.labels['src'], axis=0) if self.labels['src'] else np.array([])
        labels_tgt = np.concatenate(self.labels['tgt'], axis=0) if self.labels['tgt'] else np.array([])

        if len(feat_before['src_vis']) == 0:
            print("Warning: No features collected, skipping visualization.")
            return

        src_feat_before = np.concatenate([feat_before['src_vis'], feat_before['src_ir']], axis=1)
        tgt_feat_before = np.concatenate([feat_before['tgt_vis'], feat_before['tgt_ir']], axis=1)
        src_feat_after = np.concatenate([feat_after['src_vis'], feat_after['src_ir']], axis=1)
        tgt_feat_after = np.concatenate([feat_after['tgt_vis'], feat_after['tgt_ir']], axis=1)

        total_samples = len(src_feat_before)
        if total_samples > max_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
            src_feat_before, tgt_feat_before = src_feat_before[indices], tgt_feat_before[indices]
            src_feat_after, tgt_feat_after = src_feat_after[indices], tgt_feat_after[indices]
            labels_src, labels_tgt = labels_src[indices], labels_tgt[indices]

        # 适当增大 figsize 的高度 (8 -> 9)，给顶部留出空间放多列图例
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle(f'Epoch {epoch} - Feature Comparison of Tensor-based Alignment Module (t-SNE)',
                     fontsize=16, fontweight='bold', y=0.98)

        self._plot_tsne_single(axes[0], src_feat_before, tgt_feat_before, labels_src, labels_tgt, 'Before Alignment', perplexity, n_iter)
        self._plot_tsne_single(axes[1], src_feat_after, tgt_feat_after, labels_src, labels_tgt, 'After Alignment', perplexity, n_iter)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'tsne_epoch_{epoch}.svg') # 保存为 SVG
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"t-SNE visualization saved: {save_path}")

    def _plot_tsne_single(self, ax, feat_src: np.ndarray, feat_tgt: np.ndarray,
                          labels_src: np.ndarray, labels_tgt: np.ndarray,
                          title: str, perplexity: int, n_iter: int):
        all_features = np.vstack([feat_src, feat_tgt])
        all_labels = np.concatenate([labels_src, labels_tgt])
        domain_labels = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=0)
        features_2d = tsne.fit_transform(all_features)

        unique_labels = np.unique(all_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = all_labels == label
            src_mask = mask & (domain_labels == 0)
            if np.any(src_mask):
                ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                          c=[colors[i]], marker='o', s=50, alpha=0.6,
                          edgecolors='black', linewidths=0.5, label=f'Class {int(label)}')
            tgt_mask = mask & (domain_labels == 1)
            if np.any(tgt_mask):
                ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                          c=[colors[i]], marker='s', s=50, alpha=0.6,
                          edgecolors='black', linewidths=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

        # 提取唯一的类别图例
        handles, labels_legend = ax.get_legend_handles_labels()
        simplified_handles, simplified_labels = [], []
        seen_labels = set()
        for handle, label in zip(handles, labels_legend):
            if label not in seen_labels:
                simplified_handles.append(handle)
                simplified_labels.append(label)
                seen_labels.add(label)

        # 排序方便阅读
        sort_idx = np.argsort([int(l.split(' ')[-1]) for l in simplified_labels])
        simplified_handles = [simplified_handles[idx] for idx in sort_idx]
        simplified_labels = [simplified_labels[idx] for idx in sort_idx]

        # 每 4 个一组排成一列 (ncol=4)，利用 bbox_to_anchor 摆在上方避免遮挡点
        ax.legend(simplified_handles, simplified_labels,
                  loc='lower right', bbox_to_anchor=(1.0, 1.02),
                  ncol=4, fontsize=10, title='Categories', title_fontsize=11)
        ax.grid(True, alpha=0.3)

    def visualize_with_pca(self, epoch: int, n_components: int = 2, max_samples: int = 1000):
        """PCA Visualization (Bi-modal Fusion)"""
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)
        labels_src = np.concatenate(self.labels['src'], axis=0) if self.labels['src'] else np.array([])
        labels_tgt = np.concatenate(self.labels['tgt'], axis=0) if self.labels['tgt'] else np.array([])

        if len(feat_before['src_vis']) == 0:
            print("Warning: No features collected, skipping visualization.")
            return

        src_feat_before = np.concatenate([feat_before['src_vis'], feat_before['src_ir']], axis=1)
        tgt_feat_before = np.concatenate([feat_before['tgt_vis'], feat_before['tgt_ir']], axis=1)
        src_feat_after = np.concatenate([feat_after['src_vis'], feat_after['src_ir']], axis=1)
        tgt_feat_after = np.concatenate([feat_after['tgt_vis'], feat_after['tgt_ir']], axis=1)

        total_samples = len(src_feat_before)
        if total_samples > max_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
            src_feat_before, tgt_feat_before = src_feat_before[indices], tgt_feat_before[indices]
            src_feat_after, tgt_feat_after = src_feat_after[indices], tgt_feat_after[indices]
            labels_src, labels_tgt = labels_src[indices], labels_tgt[indices]

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle(f'Epoch {epoch} - Feature Comparison of Tensor-based Alignment Module (PCA)',
                     fontsize=16, fontweight='bold', y=0.98)

        self._plot_pca_single(axes[0], src_feat_before, tgt_feat_before, labels_src, labels_tgt, 'Before Alignment - Fused (Optical+IR)', n_components)
        self._plot_pca_single(axes[1], src_feat_after, tgt_feat_after, labels_src, labels_tgt, 'After Alignment - Fused (Optical+IR)', n_components)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'pca_epoch_{epoch}.svg') # 保存为 SVG
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"PCA visualization saved: {save_path}")

    def _plot_pca_single(self, ax, feat_src: np.ndarray, feat_tgt: np.ndarray,
                        labels_src: np.ndarray, labels_tgt: np.ndarray,
                        title: str, n_components: int):
        all_features = np.vstack([feat_src, feat_tgt])
        all_labels = np.concatenate([labels_src, labels_tgt])
        domain_labels = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])

        pca = PCA(n_components=n_components, random_state=42)
        features_2d = pca.fit_transform(all_features)

        explained_var = pca.explained_variance_ratio_
        var_str = f' (EVR: {explained_var[0]:.2%}, {explained_var[1]:.2%})'

        unique_labels = np.unique(all_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = all_labels == label
            src_mask = mask & (domain_labels == 0)
            if np.any(src_mask):
                ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                          c=[colors[i]], marker='o', s=50, alpha=0.6,
                          edgecolors='black', linewidths=0.5, label=f'Class {int(label)}')
            tgt_mask = mask & (domain_labels == 1)
            if np.any(tgt_mask):
                ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                          c=[colors[i]], marker='s', s=50, alpha=0.6,
                          edgecolors='black', linewidths=0.5)

        ax.set_title(title + var_str, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%})', fontsize=12)

        handles, labels_legend = ax.get_legend_handles_labels()
        simplified_handles, simplified_labels = [], []
        seen_labels = set()
        for handle, label in zip(handles, labels_legend):
            if label not in seen_labels:
                simplified_handles.append(handle)
                simplified_labels.append(label)
                seen_labels.add(label)

        sort_idx = np.argsort([int(l.split(' ')[-1]) for l in simplified_labels])
        simplified_handles = [simplified_handles[idx] for idx in sort_idx]
        simplified_labels = [simplified_labels[idx] for idx in sort_idx]

        ax.legend(simplified_handles, simplified_labels,
                  loc='lower right', bbox_to_anchor=(1.0, 1.02),
                  ncol=4, fontsize=10, title='Categories', title_fontsize=11)
        ax.grid(True, alpha=0.3)

    def visualize_three_modalities(self, epoch: int, max_samples: int = 500, perplexity: int = 30, n_iter: int = 1000):
        """Tri-modal Fusion Visualization (Optical + IR + AIS)"""
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)
        labels_src = np.concatenate(self.labels['src'], axis=0) if self.labels['src'] else np.array([])
        labels_tgt = np.concatenate(self.labels['tgt'], axis=0) if self.labels['tgt'] else np.array([])

        if len(feat_before.get('src_vis', [])) == 0:
            print("Warning: No features collected, skipping tri-modal visualization.")
            return

        src_ais_data = feat_before.get('src_ais', np.array([]))
        has_ais = (isinstance(src_ais_data, np.ndarray) and src_ais_data.size > 0 and len(src_ais_data.shape) >= 1)

        if not has_ais:
            print("Notice: AIS features not detected, falling back to bi-modal visualization.")
            self.visualize_with_tsne(epoch, max_samples, perplexity, n_iter)
            return

        print(f"Generating tri-modal fusion visualization...")

        src_vis_before, src_ir_before, src_ais_before = feat_before['src_vis'], feat_before['src_ir'], feat_before['src_ais']
        tgt_vis_before, tgt_ir_before, tgt_ais_before = feat_before['tgt_vis'], feat_before['tgt_ir'], feat_before['tgt_ais']
        src_vis_after, src_ir_after = feat_after['src_vis'], feat_after['src_ir']
        tgt_vis_after, tgt_ir_after = feat_after['tgt_vis'], feat_after['tgt_ir']

        min_samples = min(len(src_vis_before), len(src_ir_before), len(src_ais_before),
                          len(tgt_vis_before), len(tgt_ir_before), len(tgt_ais_before),
                          len(labels_src), len(labels_tgt))

        if min_samples == 0:
            print("Warning: Zero samples, skipping visualization.")
            return

        indices = np.random.choice(min_samples, max_samples, replace=False) if min_samples > max_samples else np.arange(min_samples)

        src_vis_before, src_ir_before, src_ais_before = src_vis_before[indices], src_ir_before[indices], src_ais_before[indices]
        tgt_vis_before, tgt_ir_before, tgt_ais_before = tgt_vis_before[indices], tgt_ir_before[indices], tgt_ais_before[indices]
        src_vis_after = src_vis_after[indices] if len(src_vis_after) > 0 else src_vis_before
        src_ir_after = src_ir_after[indices] if len(src_ir_after) > 0 else src_ir_before
        tgt_vis_after = tgt_vis_after[indices] if len(tgt_vis_after) > 0 else tgt_vis_before
        tgt_ir_after = tgt_ir_after[indices] if len(tgt_ir_after) > 0 else tgt_ir_before
        labels_src_sampled, labels_tgt_sampled = labels_src[indices], labels_tgt[indices]

        ais_dim = src_ais_before.shape[1]
        target_dim = min(src_vis_before.shape[1], ais_dim)

        if ais_dim > target_dim:
            pca_ais = PCA(n_components=target_dim, random_state=42)
            all_ais = np.vstack([src_ais_before, tgt_ais_before])
            all_ais_reduced = pca_ais.fit_transform(all_ais)
            src_ais_proj = all_ais_reduced[:len(indices)]
            tgt_ais_proj = all_ais_reduced[len(indices):]
        else:
            src_ais_proj, tgt_ais_proj = src_ais_before, tgt_ais_before

        src_fused_before = np.concatenate([src_vis_before, src_ir_before, src_ais_proj], axis=1)
        tgt_fused_before = np.concatenate([tgt_vis_before, tgt_ir_before, tgt_ais_proj], axis=1)
        src_fused_after = np.concatenate([src_vis_after, src_ir_after, src_ais_proj], axis=1)
        tgt_fused_after = np.concatenate([tgt_vis_after, tgt_ir_after, tgt_ais_proj], axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle(f'Epoch {epoch} - Tri-modal Feature Fusion Comparison (Optical + IR + AIS)\n'
                     f'Effectiveness of Tensor-based Alignment Module on Modality & Domain Alignment',
                     fontsize=16, fontweight='bold', y=0.98)

        self._plot_three_modal_single(axes[0], src_fused_before, tgt_fused_before, labels_src_sampled, labels_tgt_sampled, 'Before Alignment - Tri-modal Features', perplexity, n_iter)
        self._plot_three_modal_single(axes[1], src_fused_after, tgt_fused_after, labels_src_sampled, labels_tgt_sampled, 'After Alignment - Tri-modal Features', perplexity, n_iter)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'three_modal_epoch_{epoch}.svg') # 保存为 SVG
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Tri-modal visualization saved: {save_path}")

    def _plot_three_modal_single(self, ax, feat_src: np.ndarray, feat_tgt: np.ndarray,
                                  labels_src: np.ndarray, labels_tgt: np.ndarray,
                                  title: str, perplexity: int, n_iter: int):
        all_features = np.vstack([feat_src, feat_tgt])
        all_labels = np.concatenate([labels_src, labels_tgt])
        domain_labels = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])

        effective_perplexity = min(perplexity, len(all_features) - 1)
        tsne = TSNE(n_components=2, perplexity=effective_perplexity, n_iter=n_iter, random_state=42, verbose=0)
        features_2d = tsne.fit_transform(all_features)

        src_center = np.mean(features_2d[domain_labels == 0], axis=0)
        tgt_center = np.mean(features_2d[domain_labels == 1], axis=0)
        domain_dist = np.linalg.norm(src_center - tgt_center)

        unique_labels = np.unique(all_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_labels), 20)))

        for i, label in enumerate(unique_labels):
            color_idx = i % len(colors)
            mask = all_labels == label

            src_mask = mask & (domain_labels == 0)
            if np.any(src_mask):
                ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                          c=[colors[color_idx]], marker='o', s=50, alpha=0.6,
                          edgecolors='black', linewidths=0.5, label=f'Class {int(label)}')

            tgt_mask = mask & (domain_labels == 1)
            if np.any(tgt_mask):
                ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                          c=[colors[color_idx]], marker='^', s=50, alpha=0.6,
                          edgecolors='black', linewidths=0.5)

        # 域中心
        ax.scatter(src_center[0], src_center[1], c='blue', marker='*', s=200,
                  edgecolors='white', linewidths=2, label='Source Domain Center', zorder=10)
        ax.scatter(tgt_center[0], tgt_center[1], c='red', marker='*', s=200,
                  edgecolors='white', linewidths=2, label='Target Domain Center', zorder=10)

        ax.plot([src_center[0], tgt_center[0]], [src_center[1], tgt_center[1]], 'k--', linewidth=2, alpha=0.5)

        ax.set_title(f'{title}\nDomain Distance: {domain_dist:.2f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

        # 增加虚拟标记以清晰区分源和目标样本的符号
        ax.scatter([], [], c='gray', marker='o', s=50, label='Source Samples')
        ax.scatter([], [], c='gray', marker='^', s=50, label='Target Samples')

        # 提取类别和中心的图例
        handles, labels_legend = ax.get_legend_handles_labels()
        class_handles, class_labels = [], []
        domain_handles, domain_labels_list = [], []

        for h, l in zip(handles, labels_legend):
            if 'Class ' in l:
                if l not in class_labels:
                    class_handles.append(h)
                    class_labels.append(l)
            else:
                domain_handles.append(h)
                domain_labels_list.append(l)

        # 类别排序
        sort_idx = np.argsort([int(cl.split(' ')[-1]) for cl in class_labels])
        class_handles = [class_handles[idx] for idx in sort_idx]
        class_labels = [class_labels[idx] for idx in sort_idx]

        # 将网格中心标记加在最前，后面跟每行4列的类别
        all_handles = domain_handles + class_handles
        all_labels_legend = domain_labels_list + class_labels

        # 同样放置在右上方外部，通过 ncol=4 进行 4列排版
        ax.legend(all_handles, all_labels_legend,
                  loc='lower right', bbox_to_anchor=(1.0, 1.02),
                  ncol=4, fontsize=10)
        ax.grid(True, alpha=0.3)

    def compute_statistics(self, epoch: int) -> Dict:
        """
        Compute feature statistics (output logs remain in txt format)
        """
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)

        stats = {'epoch': epoch, 'before': {}, 'after': {}}

        for key, feat in feat_before.items():
            if len(feat) > 0:
                stats['before'][key] = {
                    'mean': float(np.mean(feat)), 'std': float(np.std(feat)),
                    'min': float(np.min(feat)), 'max': float(np.max(feat)), 'shape': feat.shape
                }
        for key, feat in feat_after.items():
            if len(feat) > 0:
                stats['after'][key] = {
                    'mean': float(np.mean(feat)), 'std': float(np.std(feat)),
                    'min': float(np.min(feat)), 'max': float(np.max(feat)), 'shape': feat.shape
                }

        stats_path = os.path.join(self.save_dir, f'stats_epoch_{epoch}.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Epoch {epoch} Feature Statistics\n")
            f.write("=" * 80 + "\n\n")
            f.write("Before Alignment:\n")
            for key, stat in stats['before'].items():
                f.write(f"  {key}:\n    Mean: {stat['mean']:.6f}\n    Std: {stat['std']:.6f}\n    Shape: {stat['shape']}\n")
            f.write("\nAfter Alignment:\n")
            for key, stat in stats['after'].items():
                f.write(f"  {key}:\n    Mean: {stat['mean']:.6f}\n    Std: {stat['std']:.6f}\n    Shape: {stat['shape']}\n")

        print(f"Statistics saved: {stats_path}")
        return stats


def visualize_features_in_training(visualizer: FeatureVisualizer, epoch: int, visualize_every: int = 5,
                                   use_tsne: bool = True, use_pca: bool = True, use_three_modal: bool = True):
    if epoch % visualize_every == 0:
        if use_three_modal:
            visualizer.visualize_three_modalities(epoch)
        elif use_tsne:
            visualizer.visualize_with_tsne(epoch)

        if use_pca:
            visualizer.visualize_with_pca(epoch)

        visualizer.compute_statistics(epoch)
        visualizer.clear_features()