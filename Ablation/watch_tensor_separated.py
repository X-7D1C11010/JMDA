"""
特征可视化模块 - 分离模态版本
用于可视化基于张量的对齐模块前后的特征，分开展示各模态

功能：
1. 分开展示三种模态：可见光、红外、AIS信号
2. 对比对齐前后的特征分布
3. 体现模态和领域对齐效果
4. 支持按类别着色和按域着色

注意：本文件是 watch_tensor.py 的补充版本，主文件已实现融合展示
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SeparatedModalityVisualizer:
    """
    分离模态特征可视化器

    与 watch_tensor.py 中的 FeatureVisualizer 不同，本类：
    1. 分别展示每种模态的特征
    2. 支持三种模态：可见光(VIS)、红外(IR)、AIS信号
    3. 更详细地展示模态对齐和领域对齐效果
    """

    def __init__(self, save_dir: str = "visualizations_separated", device: str = "cpu"):
        """
        初始化分离模态可视化器

        Args:
            save_dir: 保存可视化结果的目录
            device: 设备（cpu或cuda）
        """
        self.save_dir = save_dir
        self.device = device
        os.makedirs(save_dir, exist_ok=True)

        # 存储特征 - 三模态版本
        self.features_before = {
            'src_vis': [],   # 源域可见光
            'src_ir': [],    # 源域红外
            'src_ais': [],   # 源域AIS
            'tgt_vis': [],   # 目标域可见光
            'tgt_ir': [],    # 目标域红外
            'tgt_ais': []    # 目标域AIS
        }
        self.features_after = {
            'src_vis': [],
            'src_ir': [],
            'src_ais': [],
            'tgt_vis': [],
            'tgt_ir': [],
            'tgt_ais': []
        }
        self.labels = {
            'src': [],
            'tgt': []
        }

        # 颜色映射
        self.domain_colors = {
            'source': '#1f77b4',  # 蓝色 - 源域
            'target': '#ff7f0e'   # 橙色 - 目标域
        }
        self.modality_colors = {
            'vis': '#2ecc71',     # 绿色 - 可见光
            'ir': '#e74c3c',      # 红色 - 红外
            'ais': '#9b59b6'      # 紫色 - AIS
        }

    def collect_features_before_3modal(self,
                                        f_s_vis: torch.Tensor,
                                        f_s_ir: torch.Tensor,
                                        f_s_ais: torch.Tensor,
                                        f_t_vis: torch.Tensor,
                                        f_t_ir: torch.Tensor,
                                        f_t_ais: torch.Tensor,
                                        s_label: torch.Tensor,
                                        t_label: torch.Tensor):
        """
        收集对齐前的三模态特征

        Args:
            f_s_vis: 源域可见光特征 [B, D]
            f_s_ir: 源域红外特征 [B, D]
            f_s_ais: 源域AIS特征 [B, D]
            f_t_vis: 目标域可见光特征 [B, D]
            f_t_ir: 目标域红外特征 [B, D]
            f_t_ais: 目标域AIS特征 [B, D]
            s_label: 源域标签 [B]
            t_label: 目标域标签 [B]
        """
        self.features_before['src_vis'].append(f_s_vis.detach().cpu().numpy())
        self.features_before['src_ir'].append(f_s_ir.detach().cpu().numpy())
        self.features_before['src_ais'].append(f_s_ais.detach().cpu().numpy())
        self.features_before['tgt_vis'].append(f_t_vis.detach().cpu().numpy())
        self.features_before['tgt_ir'].append(f_t_ir.detach().cpu().numpy())
        self.features_before['tgt_ais'].append(f_t_ais.detach().cpu().numpy())
        self.labels['src'].append(s_label.detach().cpu().numpy())
        self.labels['tgt'].append(t_label.detach().cpu().numpy())

    def collect_features_before_2modal(self,
                                        f_s_vis: torch.Tensor,
                                        f_s_ir: torch.Tensor,
                                        f_t_vis: torch.Tensor,
                                        f_t_ir: torch.Tensor,
                                        s_label: torch.Tensor,
                                        t_label: torch.Tensor):
        """
        收集对齐前的双模态特征（无AIS时使用）

        Args:
            f_s_vis: 源域可见光特征 [B, D]
            f_s_ir: 源域红外特征 [B, D]
            f_t_vis: 目标域可见光特征 [B, D]
            f_t_ir: 目标域红外特征 [B, D]
            s_label: 源域标签 [B]
            t_label: 目标域标签 [B]
        """
        self.features_before['src_vis'].append(f_s_vis.detach().cpu().numpy())
        self.features_before['src_ir'].append(f_s_ir.detach().cpu().numpy())
        self.features_before['tgt_vis'].append(f_t_vis.detach().cpu().numpy())
        self.features_before['tgt_ir'].append(f_t_ir.detach().cpu().numpy())
        self.labels['src'].append(s_label.detach().cpu().numpy())
        self.labels['tgt'].append(t_label.detach().cpu().numpy())

    def collect_features_after_3modal(self,
                                       p_s_vis: torch.Tensor,
                                       p_s_ir: torch.Tensor,
                                       p_s_ais: torch.Tensor,
                                       p_t_vis: torch.Tensor,
                                       p_t_ir: torch.Tensor,
                                       p_t_ais: torch.Tensor):
        """收集对齐后的三模态特征"""
        # 处理可能的列表格式
        if isinstance(p_s_vis, list): p_s_vis = p_s_vis[0]
        if isinstance(p_s_ir, list): p_s_ir = p_s_ir[0]
        if isinstance(p_s_ais, list): p_s_ais = p_s_ais[0]
        if isinstance(p_t_vis, list): p_t_vis = p_t_vis[0]
        if isinstance(p_t_ir, list): p_t_ir = p_t_ir[0]
        if isinstance(p_t_ais, list): p_t_ais = p_t_ais[0]

        self.features_after['src_vis'].append(p_s_vis.detach().cpu().numpy())
        self.features_after['src_ir'].append(p_s_ir.detach().cpu().numpy())
        self.features_after['src_ais'].append(p_s_ais.detach().cpu().numpy())
        self.features_after['tgt_vis'].append(p_t_vis.detach().cpu().numpy())
        self.features_after['tgt_ir'].append(p_t_ir.detach().cpu().numpy())
        self.features_after['tgt_ais'].append(p_t_ais.detach().cpu().numpy())

    def collect_features_after_2modal(self,
                                       p_s_vis: torch.Tensor,
                                       p_s_ir: torch.Tensor,
                                       p_t_vis: torch.Tensor,
                                       p_t_ir: torch.Tensor):
        """收集对齐后的双模态特征（无AIS时使用）"""
        if isinstance(p_s_vis, list): p_s_vis = p_s_vis[0]
        if isinstance(p_s_ir, list): p_s_ir = p_s_ir[0]
        if isinstance(p_t_vis, list): p_t_vis = p_t_vis[0]
        if isinstance(p_t_ir, list): p_t_ir = p_t_ir[0]

        self.features_after['src_vis'].append(p_s_vis.detach().cpu().numpy())
        self.features_after['src_ir'].append(p_s_ir.detach().cpu().numpy())
        self.features_after['tgt_vis'].append(p_t_vis.detach().cpu().numpy())
        self.features_after['tgt_ir'].append(p_t_ir.detach().cpu().numpy())

    def clear_features(self):
        """清空已收集的特征"""
        for key in self.features_before:
            self.features_before[key] = []
        for key in self.features_after:
            self.features_after[key] = []
        self.labels['src'] = []
        self.labels['tgt'] = []

    def _concatenate_features(self, feature_dict: Dict[str, List]) -> Dict[str, np.ndarray]:
        """将收集的特征列表拼接成数组"""
        result = {}
        for key, value_list in feature_dict.items():
            if value_list:
                result[key] = np.concatenate(value_list, axis=0)
            else:
                result[key] = np.array([])
        return result

    def _has_ais_features(self, feat_dict: Dict[str, np.ndarray]) -> bool:
        """检查是否有AIS特征"""
        return len(feat_dict.get('src_ais', [])) > 0 and feat_dict['src_ais'].size > 0

    def visualize_separated_modalities_tsne(self,
                                            epoch: int,
                                            max_samples: int = 500,
                                            perplexity: int = 30,
                                            n_iter: int = 1000):
        """
        使用t-SNE分别可视化每种模态的特征（对齐前后对比）

        这种可视化方式能够：
        1. 展示每种模态内部的领域对齐效果（源域和目标域是否靠近）
        2. 展示每种模态的类别分离效果

        Args:
            epoch: 当前epoch
            max_samples: 最大采样数量
            perplexity: t-SNE困惑度
            n_iter: t-SNE迭代次数
        """
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)
        labels_src = np.concatenate(self.labels['src'], axis=0) if self.labels['src'] else np.array([])
        labels_tgt = np.concatenate(self.labels['tgt'], axis=0) if self.labels['tgt'] else np.array([])

        if len(feat_before.get('src_vis', [])) == 0:
            print("警告：没有收集到特征，跳过分离模态可视化")
            return

        has_ais = self._has_ais_features(feat_before)
        n_modalities = 3 if has_ais else 2
        modality_names = ['可见光 (VIS)', '红外 (IR)', 'AIS信号'] if has_ais else ['可见光 (VIS)', '红外 (IR)']
        modality_keys = ['vis', 'ir', 'ais'] if has_ais else ['vis', 'ir']

        # 采样
        total_samples = len(feat_before['src_vis'])
        if total_samples > max_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
        else:
            indices = np.arange(total_samples)

        # 创建图形：每行一个模态，每列分别是对齐前和对齐后
        fig, axes = plt.subplots(n_modalities, 2, figsize=(16, 6 * n_modalities))
        fig.suptitle(f'Epoch {epoch} - 分离模态特征可视化 (t-SNE)\n'
                     f'体现基于张量的对齐模块对模态和领域对齐的效果',
                     fontsize=16, fontweight='bold', y=1.02)

        for m_idx, (m_name, m_key) in enumerate(zip(modality_names, modality_keys)):
            # 获取该模态的特征
            src_key = f'src_{m_key}'
            tgt_key = f'tgt_{m_key}'

            if src_key not in feat_before or len(feat_before[src_key]) == 0:
                continue

            src_feat_before = feat_before[src_key][indices]
            tgt_feat_before = feat_before[tgt_key][indices]
            src_feat_after = feat_after[src_key][indices] if src_key in feat_after and len(feat_after[src_key]) > 0 else None
            tgt_feat_after = feat_after[tgt_key][indices] if tgt_key in feat_after and len(feat_after[tgt_key]) > 0 else None

            sampled_labels_src = labels_src[indices]
            sampled_labels_tgt = labels_tgt[indices]

            # 对齐前
            ax_before = axes[m_idx, 0] if n_modalities > 1 else axes[0]
            self._plot_single_modality_tsne(
                ax_before,
                src_feat_before,
                tgt_feat_before,
                sampled_labels_src,
                sampled_labels_tgt,
                f'{m_name} - 对齐前',
                perplexity,
                n_iter
            )

            # 对齐后
            ax_after = axes[m_idx, 1] if n_modalities > 1 else axes[1]
            if src_feat_after is not None and tgt_feat_after is not None:
                self._plot_single_modality_tsne(
                    ax_after,
                    src_feat_after,
                    tgt_feat_after,
                    sampled_labels_src,
                    sampled_labels_tgt,
                    f'{m_name} - 对齐后',
                    perplexity,
                    n_iter
                )
            else:
                ax_after.text(0.5, 0.5, '无对齐后数据', ha='center', va='center', fontsize=14)
                ax_after.set_title(f'{m_name} - 对齐后')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'separated_tsne_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"分离模态t-SNE可视化已保存: {save_path}")

    def _plot_single_modality_tsne(self,
                                    ax,
                                    feat_src: np.ndarray,
                                    feat_tgt: np.ndarray,
                                    labels_src: np.ndarray,
                                    labels_tgt: np.ndarray,
                                    title: str,
                                    perplexity: int,
                                    n_iter: int):
        """绘制单个模态的t-SNE图"""
        # 合并特征
        all_features = np.vstack([feat_src, feat_tgt])
        all_labels = np.concatenate([labels_src, labels_tgt])
        domain_labels = np.concatenate([
            np.zeros(len(feat_src)),  # 0 = 源域
            np.ones(len(feat_tgt))    # 1 = 目标域
        ])

        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(all_features) - 1),
                    n_iter=n_iter, random_state=42, verbose=0)
        features_2d = tsne.fit_transform(all_features)

        # 按类别和域绘制
        unique_labels = np.unique(all_labels)
        n_classes = len(unique_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))

        for i, label in enumerate(unique_labels):
            color_idx = i % len(colors)
            mask = all_labels == label

            # 源域 - 圆形，实心
            src_mask = mask & (domain_labels == 0)
            if np.any(src_mask):
                ax.scatter(features_2d[src_mask, 0], features_2d[src_mask, 1],
                          c=[colors[color_idx]], marker='o', s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5)

            # 目标域 - 三角形，实心
            tgt_mask = mask & (domain_labels == 1)
            if np.any(tgt_mask):
                ax.scatter(features_2d[tgt_mask, 0], features_2d[tgt_mask, 1],
                          c=[colors[color_idx]], marker='^', s=60, alpha=0.7,
                          edgecolors='black', linewidths=0.5)

        # 添加图例说明
        ax.scatter([], [], c='gray', marker='o', s=60, label='源域', edgecolors='black')
        ax.scatter([], [], c='gray', marker='^', s=60, label='目标域', edgecolors='black')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE维度1', fontsize=11)
        ax.set_ylabel('t-SNE维度2', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    def visualize_domain_alignment_comparison(self,
                                               epoch: int,
                                               max_samples: int = 500):
        """
        专门可视化领域对齐效果的对比图

        重点展示：对齐前后源域和目标域的分布变化
        使用颜色区分域，形状区分类别

        Args:
            epoch: 当前epoch
            max_samples: 最大采样数量
        """
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)
        labels_src = np.concatenate(self.labels['src'], axis=0) if self.labels['src'] else np.array([])
        labels_tgt = np.concatenate(self.labels['tgt'], axis=0) if self.labels['tgt'] else np.array([])

        if len(feat_before.get('src_vis', [])) == 0:
            print("警告：没有收集到特征，跳过领域对齐可视化")
            return

        has_ais = self._has_ais_features(feat_before)
        n_modalities = 3 if has_ais else 2
        modality_names = ['可见光', '红外', 'AIS'] if has_ais else ['可见光', '红外']
        modality_keys = ['vis', 'ir', 'ais'] if has_ais else ['vis', 'ir']

        # 采样
        total_samples = len(feat_before['src_vis'])
        if total_samples > max_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
        else:
            indices = np.arange(total_samples)

        # 创建大图：展示领域对齐效果
        fig, axes = plt.subplots(2, n_modalities, figsize=(7 * n_modalities, 12))
        fig.suptitle(f'Epoch {epoch} - 领域对齐效果对比\n'
                     f'蓝色=源域, 橙色=目标域 | 圆形=源域, 三角=目标域',
                     fontsize=16, fontweight='bold', y=1.02)

        for m_idx, (m_name, m_key) in enumerate(zip(modality_names, modality_keys)):
            src_key = f'src_{m_key}'
            tgt_key = f'tgt_{m_key}'

            if src_key not in feat_before or len(feat_before[src_key]) == 0:
                continue

            src_before = feat_before[src_key][indices]
            tgt_before = feat_before[tgt_key][indices]
            src_after = feat_after.get(src_key, np.array([]))
            tgt_after = feat_after.get(tgt_key, np.array([]))

            if len(src_after) > 0:
                src_after = src_after[indices]
            if len(tgt_after) > 0:
                tgt_after = tgt_after[indices]

            # 对齐前
            ax_before = axes[0, m_idx] if n_modalities > 1 else axes[0]
            self._plot_domain_alignment(
                ax_before,
                src_before,
                tgt_before,
                f'{m_name} - 对齐前'
            )

            # 对齐后
            ax_after = axes[1, m_idx] if n_modalities > 1 else axes[1]
            if len(src_after) > 0 and len(tgt_after) > 0:
                self._plot_domain_alignment(
                    ax_after,
                    src_after,
                    tgt_after,
                    f'{m_name} - 对齐后'
                )
            else:
                ax_after.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax_after.set_title(f'{m_name} - 对齐后')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'domain_alignment_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"领域对齐对比图已保存: {save_path}")

    def _plot_domain_alignment(self,
                                ax,
                                feat_src: np.ndarray,
                                feat_tgt: np.ndarray,
                                title: str):
        """绘制领域对齐图（按域着色）"""
        # 合并特征
        all_features = np.vstack([feat_src, feat_tgt])

        # PCA降维（比t-SNE快，适合快速对比）
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(all_features)

        n_src = len(feat_src)
        src_2d = features_2d[:n_src]
        tgt_2d = features_2d[n_src:]

        # 绘制源域和目标域
        ax.scatter(src_2d[:, 0], src_2d[:, 1],
                  c=self.domain_colors['source'], marker='o', s=50, alpha=0.6,
                  label='源域', edgecolors='white', linewidths=0.3)
        ax.scatter(tgt_2d[:, 0], tgt_2d[:, 1],
                  c=self.domain_colors['target'], marker='^', s=50, alpha=0.6,
                  label='目标域', edgecolors='white', linewidths=0.3)

        # 计算并显示域间距离
        src_center = np.mean(src_2d, axis=0)
        tgt_center = np.mean(tgt_2d, axis=0)
        domain_dist = np.linalg.norm(src_center - tgt_center)

        ax.set_title(f'{title}\n域间距离: {domain_dist:.2f}', fontsize=13, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    def visualize_modality_alignment(self,
                                      epoch: int,
                                      max_samples: int = 500):
        """
        可视化模态对齐效果

        将所有模态的特征投影到同一空间，展示模态间的对齐程度

        Args:
            epoch: 当前epoch
            max_samples: 最大采样数量
        """
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)

        if len(feat_before.get('src_vis', [])) == 0:
            print("警告：没有收集到特征，跳过模态对齐可视化")
            return

        has_ais = self._has_ais_features(feat_before)

        # 采样
        total_samples = len(feat_before['src_vis'])
        if total_samples > max_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
        else:
            indices = np.arange(total_samples)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Epoch {epoch} - 模态对齐效果\n'
                     f'不同颜色代表不同模态，展示模态特征空间的对齐程度',
                     fontsize=16, fontweight='bold', y=1.02)

        # 对齐前
        self._plot_modality_space(
            axes[0],
            feat_before,
            indices,
            has_ais,
            '对齐前 - 模态特征空间'
        )

        # 对齐后
        if len(feat_after.get('src_vis', [])) > 0:
            self._plot_modality_space(
                axes[1],
                feat_after,
                indices,
                has_ais,
                '对齐后 - 模态特征空间'
            )
        else:
            axes[1].text(0.5, 0.5, '无对齐后数据', ha='center', va='center')
            axes[1].set_title('对齐后 - 模态特征空间')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'modality_alignment_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"模态对齐可视化已保存: {save_path}")

    def _plot_modality_space(self,
                              ax,
                              feat_dict: Dict[str, np.ndarray],
                              indices: np.ndarray,
                              has_ais: bool,
                              title: str):
        """绘制模态空间图"""
        # 收集所有模态的特征
        modalities_data = []
        modality_labels = []

        modality_keys = ['vis', 'ir', 'ais'] if has_ais else ['vis', 'ir']
        modality_names = ['可见光', '红外', 'AIS'] if has_ais else ['可见光', '红外']

        for m_key, m_name in zip(modality_keys, modality_names):
            src_key = f'src_{m_key}'
            tgt_key = f'tgt_{m_key}'

            if src_key in feat_dict and len(feat_dict[src_key]) > 0:
                src_feat = feat_dict[src_key][indices]
                tgt_feat = feat_dict[tgt_key][indices]

                modalities_data.append(src_feat)
                modality_labels.extend([f'{m_name}-源域'] * len(src_feat))

                modalities_data.append(tgt_feat)
                modality_labels.extend([f'{m_name}-目标域'] * len(tgt_feat))

        if not modalities_data:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            ax.set_title(title)
            return

        all_features = np.vstack(modalities_data)

        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=min(30, len(all_features) - 1),
                    n_iter=1000, random_state=42, verbose=0)
        features_2d = tsne.fit_transform(all_features)

        # 为每种模态-域组合分配颜色
        unique_groups = list(set(modality_labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))
        color_map = {g: colors[i] for i, g in enumerate(unique_groups)}

        # 绘制
        for group in unique_groups:
            mask = np.array(modality_labels) == group
            marker = 'o' if '源域' in group else '^'
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[color_map[group]], marker=marker, s=50, alpha=0.6,
                      label=group, edgecolors='white', linewidths=0.3)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE维度1', fontsize=11)
        ax.set_ylabel('t-SNE维度2', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    def compute_alignment_metrics(self, epoch: int) -> Dict:
        """
        计算对齐效果的量化指标

        指标包括：
        1. 域间距离（对齐前后）
        2. 类内距离
        3. 类间距离
        4. 模态间对齐度

        Returns:
            包含各种指标的字典
        """
        feat_before = self._concatenate_features(self.features_before)
        feat_after = self._concatenate_features(self.features_after)
        labels_src = np.concatenate(self.labels['src'], axis=0) if self.labels['src'] else np.array([])
        labels_tgt = np.concatenate(self.labels['tgt'], axis=0) if self.labels['tgt'] else np.array([])

        metrics = {'epoch': epoch, 'before': {}, 'after': {}}

        has_ais = self._has_ais_features(feat_before)
        modality_keys = ['vis', 'ir', 'ais'] if has_ais else ['vis', 'ir']

        for m_key in modality_keys:
            src_key = f'src_{m_key}'
            tgt_key = f'tgt_{m_key}'

            if src_key not in feat_before or len(feat_before[src_key]) == 0:
                continue

            # 对齐前指标
            src_before = feat_before[src_key]
            tgt_before = feat_before[tgt_key]
            metrics['before'][m_key] = {
                'domain_distance': self._compute_domain_distance(src_before, tgt_before),
                'src_variance': float(np.var(src_before)),
                'tgt_variance': float(np.var(tgt_before))
            }

            # 对齐后指标
            if src_key in feat_after and len(feat_after[src_key]) > 0:
                src_after = feat_after[src_key]
                tgt_after = feat_after[tgt_key]
                metrics['after'][m_key] = {
                    'domain_distance': self._compute_domain_distance(src_after, tgt_after),
                    'src_variance': float(np.var(src_after)),
                    'tgt_variance': float(np.var(tgt_after))
                }

        # 保存指标
        metrics_path = os.path.join(self.save_dir, f'alignment_metrics_epoch_{epoch}.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Epoch {epoch} 对齐效果量化指标\n")
            f.write("=" * 80 + "\n\n")

            for m_key in modality_keys:
                if m_key in metrics['before']:
                    f.write(f"【{m_key.upper()}模态】\n")
                    f.write(f"  对齐前:\n")
                    f.write(f"    域间距离: {metrics['before'][m_key]['domain_distance']:.4f}\n")
                    f.write(f"    源域方差: {metrics['before'][m_key]['src_variance']:.4f}\n")
                    f.write(f"    目标域方差: {metrics['before'][m_key]['tgt_variance']:.4f}\n")

                    if m_key in metrics['after']:
                        f.write(f"  对齐后:\n")
                        f.write(f"    域间距离: {metrics['after'][m_key]['domain_distance']:.4f}\n")
                        f.write(f"    源域方差: {metrics['after'][m_key]['src_variance']:.4f}\n")
                        f.write(f"    目标域方差: {metrics['after'][m_key]['tgt_variance']:.4f}\n")

                        # 计算改善率
                        dist_before = metrics['before'][m_key]['domain_distance']
                        dist_after = metrics['after'][m_key]['domain_distance']
                        if dist_before > 0:
                            improvement = (dist_before - dist_after) / dist_before * 100
                            f.write(f"    域间距离改善: {improvement:.2f}%\n")
                    f.write("\n")

        print(f"对齐指标已保存: {metrics_path}")
        return metrics

    def _compute_domain_distance(self, feat_src: np.ndarray, feat_tgt: np.ndarray) -> float:
        """计算源域和目标域特征中心的距离"""
        src_center = np.mean(feat_src, axis=0)
        tgt_center = np.mean(feat_tgt, axis=0)
        return float(np.linalg.norm(src_center - tgt_center))


def visualize_separated_in_training(visualizer: SeparatedModalityVisualizer,
                                     epoch: int,
                                     visualize_every: int = 5,
                                     use_tsne: bool = True,
                                     use_domain_alignment: bool = True,
                                     use_modality_alignment: bool = True):
    """
    在训练过程中进行分离模态可视化的辅助函数

    Args:
        visualizer: SeparatedModalityVisualizer实例
        epoch: 当前epoch
        visualize_every: 每N个epoch可视化一次
        use_tsne: 是否使用t-SNE分离模态可视化
        use_domain_alignment: 是否可视化领域对齐效果
        use_modality_alignment: 是否可视化模态对齐效果
    """
    if epoch % visualize_every == 0:
        if use_tsne:
            visualizer.visualize_separated_modalities_tsne(epoch)
        if use_domain_alignment:
            visualizer.visualize_domain_alignment_comparison(epoch)
        if use_modality_alignment:
            visualizer.visualize_modality_alignment(epoch)
        visualizer.compute_alignment_metrics(epoch)
        visualizer.clear_features()


# ==================== AIS数据加载工具 ====================

def load_ais_data(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载AIS数据集

    Args:
        mat_path: .mat文件路径

    Returns:
        features: AIS特征数据 [N, D]
        labels: 类别标签 [N]
    """
    try:
        data = scipy.io.loadmat(mat_path)

        # 尝试常见的键名
        feature_keys = ['X', 'data', 'features', 'ais_data', 'signal']
        label_keys = ['Y', 'labels', 'label', 'y', 'class']

        features = None
        labels = None

        for key in data.keys():
            if key.startswith('__'):
                continue
            if key.lower() in [k.lower() for k in feature_keys] or 'data' in key.lower() or 'feature' in key.lower():
                if features is None:
                    features = data[key]
            if key.lower() in [k.lower() for k in label_keys] or 'label' in key.lower() or 'class' in key.lower():
                if labels is None:
                    labels = data[key]

        # 如果没找到，尝试获取最大的数组作为特征
        if features is None:
            max_size = 0
            for key, value in data.items():
                if not key.startswith('__') and hasattr(value, 'shape'):
                    if value.size > max_size:
                        max_size = value.size
                        features = value

        if features is not None:
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(-1, 1)

        if labels is not None:
            labels = np.array(labels).flatten()

        return features, labels

    except Exception as e:
        print(f"加载AIS数据时出错: {e}")
        return None, None


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    使用示例：展示如何在训练中使用分离模态可视化器

    在main.py中集成时，需要：
    1. 导入本模块
    2. 创建SeparatedModalityVisualizer实例
    3. 在训练循环中收集特征
    4. 调用可视化函数
    """

    print("分离模态可视化器使用示例")
    print("=" * 60)

    # 创建可视化器
    visualizer = SeparatedModalityVisualizer(
        save_dir="visualizations_separated_demo",
        device="cpu"
    )

    # 模拟特征数据（实际使用时从模型获取）
    batch_size = 32
    feature_dim = 512
    projected_dim = 128

    # 模拟对齐前的特征
    f_s_vis = torch.randn(batch_size, feature_dim)
    f_s_ir = torch.randn(batch_size, feature_dim)
    f_t_vis = torch.randn(batch_size, feature_dim) + 2  # 目标域有偏移
    f_t_ir = torch.randn(batch_size, feature_dim) + 2

    # 模拟标签
    s_label = torch.randint(0, 10, (batch_size,))
    t_label = s_label.clone()  # 配对采样，标签相同

    # 收集对齐前特征
    visualizer.collect_features_before_2modal(
        f_s_vis, f_s_ir, f_t_vis, f_t_ir, s_label, t_label
    )

    # 模拟对齐后的特征（域偏移减小）
    p_s_vis = torch.randn(batch_size, projected_dim)
    p_s_ir = torch.randn(batch_size, projected_dim)
    p_t_vis = torch.randn(batch_size, projected_dim) + 0.5  # 偏移减小
    p_t_ir = torch.randn(batch_size, projected_dim) + 0.5

    # 收集对齐后特征
    visualizer.collect_features_after_2modal(
        p_s_vis, p_s_ir, p_t_vis, p_t_ir
    )

    # 执行可视化
    print("\n正在生成可视化...")
    visualizer.visualize_separated_modalities_tsne(epoch=0, max_samples=100)
    visualizer.visualize_domain_alignment_comparison(epoch=0, max_samples=100)
    visualizer.visualize_modality_alignment(epoch=0, max_samples=100)
    visualizer.compute_alignment_metrics(epoch=0)

    print("\n可视化完成！请查看 visualizations_separated_demo 目录")
    print("\n" + "=" * 60)
    print("集成到main.py的示例代码：")
    print("""
    # 在main.py中添加以下代码：

    from watch_tensor_separated import SeparatedModalityVisualizer, visualize_separated_in_training

    # 初始化分离模态可视化器
    sep_visualizer = SeparatedModalityVisualizer(
        save_dir=os.path.join(vis_dir, "separated"),
        device=str(DEVICE)
    )

    # 在训练循环中收集特征（TAL对齐前后）
    if COLLECT_FEATURES and epoch % VISUALIZE_EVERY == 0:
        sep_visualizer.collect_features_before_2modal(
            f_s_vis, f_s_ir, f_t_vis, f_t_ir, s_label, t_label
        )
        # ... TAL对齐 ...
        sep_visualizer.collect_features_after_2modal(
            p_s_vis, p_s_ir, p_t_vis, p_t_ir
        )

    # 在epoch结束时可视化
    visualize_separated_in_training(sep_visualizer, epoch, visualize_every=VISUALIZE_EVERY)
    """)
