import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy as np
import torch
import torch.nn.functional as F

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'

# ── Module-level helper functions ──────────────────────────────────────────────

def denorm_vis_np(tensor):
    """[B,3,H,W] ImageNet-normalized tensor -> [B,H,W,3] numpy in [0,1]"""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    out = tensor.cpu() * std + mean
    out = out.clamp(0.0, 1.0)
    return out.permute(0, 2, 3, 1).numpy()


def denorm_ir_np(tensor):
    """[B,3,H,W] IR-normalized tensor (mean=0.5, std=0.5) -> [B,H,W,3] numpy in [0,1]"""
    out = tensor.cpu() * 0.5 + 0.5
    out = out.clamp(0.0, 1.0)
    return out.permute(0, 2, 3, 1).numpy()


def decoded_to_np(tensor):
    """Decoder Tanh output [B,3,H,W] in [-1,1] -> [B,H,W,3] numpy in [0,1]"""
    out = (tensor.cpu() + 1.0) / 2.0
    out = out.clamp(0.0, 1.0)
    return out.permute(0, 2, 3, 1).numpy()


# ── VisualizationHub ───────────────────────────────────────────────────────────

class VisualizationHub:
    """Collects features and images during training, generates 5 visualization
    types per epoch, and saves them to organised subfolders as SVG vectors."""

    def __init__(self, save_dir, device,
                 decoder_vis=None, decoder_ir=None, decoder_mid=None,
                 max_samples=256):
        self.save_dir     = save_dir
        self.device       = device
        self.decoder_vis  = decoder_vis
        self.decoder_ir   = decoder_ir
        self.decoder_mid  = decoder_mid
        self.max_samples  = max_samples
        self._data        = {}
        self.reset()

    # ── data management ──────────────────────────────────────────────────────

    def reset(self):
        """Clear all collected data."""
        self._data = {
            'src_vis_img': [],
            'src_ir_img':  [],
            'tgt_vis_img': [],
            'tgt_ir_img':  [],
            'src_label':   [],
            'tgt_label':   [],
            'f_s_vis':     [],
            'f_s_ir':      [],
            'f_t_vis':     [],
            'f_t_ir':      [],
            'p_s_vis':     [],
            'p_s_ir':      [],
            'p_t_vis':     [],
            'p_t_ir':      [],
            'feat_src':    [],
            'feat_tgt':    [],
            'feat_mid':    [],
        }
        self._n_collected = 0

    def collect(self,
                src_vis_img, src_ir_img, tgt_vis_img, tgt_ir_img,
                src_label, tgt_label,
                f_s_vis, f_s_ir, f_t_vis, f_t_ir,
                p_s_vis, p_s_ir, p_t_vis, p_t_ir,
                feat_src, feat_tgt, feat_mid):
        """Collect one batch. Stores up to max_samples total; skips if full."""
        if self._n_collected >= self.max_samples:
            return

        def _cpu(t):
            return t.detach().cpu() if isinstance(t, torch.Tensor) else torch.tensor(t)

        batch_size = src_vis_img.size(0)

        self._data['src_vis_img'].append(_cpu(src_vis_img))
        self._data['src_ir_img'].append(_cpu(src_ir_img))
        self._data['tgt_vis_img'].append(_cpu(tgt_vis_img))
        self._data['tgt_ir_img'].append(_cpu(tgt_ir_img))
        self._data['src_label'].append(_cpu(src_label))
        self._data['tgt_label'].append(_cpu(tgt_label))
        self._data['f_s_vis'].append(_cpu(f_s_vis))
        self._data['f_s_ir'].append(_cpu(f_s_ir))
        self._data['f_t_vis'].append(_cpu(f_t_vis))
        self._data['f_t_ir'].append(_cpu(f_t_ir))
        self._data['p_s_vis'].append(_cpu(p_s_vis))
        self._data['p_s_ir'].append(_cpu(p_s_ir))
        self._data['p_t_vis'].append(_cpu(p_t_vis))
        self._data['p_t_ir'].append(_cpu(p_t_ir))
        self._data['feat_src'].append(_cpu(feat_src))
        self._data['feat_tgt'].append(_cpu(feat_tgt))
        self._data['feat_mid'].append(_cpu(feat_mid))

        self._n_collected += batch_size

    def _concat_data(self):
        """Concatenate all stored batch lists and truncate to max_samples."""
        out = {}
        for key, batches in self._data.items():
            if not batches:
                out[key] = None
            else:
                out[key] = torch.cat(batches, dim=0)[:self.max_samples]
        return out

    # ── epoch entry point ────────────────────────────────────────────────────

    def visualize_epoch(self, epoch, label_map=None):
        """Run all 5 visualizations for this epoch, then reset collected data."""
        data = self._concat_data()

        # Build epoch subfolder tree
        epoch_dir = os.path.join(self.save_dir, f'epoch_{epoch:03d}')
        subdirs = [
            '1_reconstructed',
            '2_nearest_neighbor',
            '3_tal_comparison',
            '4_ot_transport',
            '5_tsne_thumbnails',
        ]
        for sub in subdirs:
            os.makedirs(os.path.join(epoch_dir, sub), exist_ok=True)

        self._visualize_reconstructed(epoch, epoch_dir, data)
        self._visualize_nearest_neighbor(epoch, epoch_dir, data, label_map)
        self._visualize_tal_comparison(epoch, epoch_dir, data, label_map)
        self._visualize_ot_transport(epoch, epoch_dir, data, label_map)
        self._visualize_tsne_thumbnails(epoch, epoch_dir, data)

        self.reset()

    # ── visualization 1: reconstruction ──────────────────────────────────────

    def _visualize_reconstructed(self, epoch, epoch_dir, data):
        if any(data[k] is None for k in ('src_vis_img', 'src_ir_img',
                                          'tgt_vis_img', 'feat_mid')):
            print("[Visualization] Reconstruction: Insufficient data, skipping.")
            return

        decoders_ready = (self.decoder_vis is not None and
                          self.decoder_ir  is not None and
                          self.decoder_mid is not None)

        n_available = data['src_vis_img'].size(0)
        n_show = min(8, n_available)

        src_vis = data['src_vis_img'][:n_show]
        src_ir  = data['src_ir_img'][:n_show]
        tgt_vis = data['tgt_vis_img'][:n_show]
        f_s_vis = data['f_s_vis'][:n_show]
        f_s_ir  = data['f_s_ir'][:n_show]
        feat_mid = data['feat_mid'][:n_show]

        with torch.no_grad():
            if decoders_ready:
                dev = self.device
                recon_vis = self.decoder_vis(f_s_vis.to(dev)).cpu()
                recon_ir  = self.decoder_ir(f_s_ir.to(dev)).cpu()
                recon_mid = self.decoder_mid(feat_mid.to(dev)).cpu()
            else:
                recon_vis = torch.zeros_like(src_vis)
                recon_ir  = torch.zeros_like(src_ir)
                recon_mid = torch.zeros(n_show, 3, 224, 224)

        src_vis_np  = denorm_vis_np(src_vis)
        src_ir_np   = denorm_ir_np(src_ir)
        tgt_vis_np  = denorm_vis_np(tgt_vis)
        recon_vis_np = decoded_to_np(recon_vis)
        recon_ir_np  = decoded_to_np(recon_ir)
        recon_mid_np = decoded_to_np(recon_mid)

        col_titles = [
            'Original Vis', 'Recon Vis',
            'Original IR',   'Recon IR',
            'Target Vis (Orig)', 'Mid Domain (Recon)',
        ]
        n_cols = len(col_titles)

        fig, axes = plt.subplots(n_show, n_cols,
                                 figsize=(n_cols * 2.5, n_show * 2.5))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=10)

        arrays = [src_vis_np, recon_vis_np, src_ir_np,
                  recon_ir_np, tgt_vis_np, recon_mid_np]
        for row in range(n_show):
            for col, arr in enumerate(arrays):
                axes[row, col].imshow(arr[row].clip(0, 1))
                axes[row, col].axis('off')

        plt.tight_layout()
        save_path = os.path.join(epoch_dir, '1_reconstructed',
                                 f'reconstruction_epoch_{epoch:03d}.svg')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualization] Reconstruction figure saved: {save_path}")

    # ── visualization 2: nearest neighbor ────────────────────────────────────

    def _visualize_nearest_neighbor(self, epoch, epoch_dir, data, label_map=None):
        if any(data[k] is None for k in ('feat_src', 'feat_tgt',
                                          'src_vis_img', 'tgt_vis_img',
                                          'src_label', 'tgt_label')):
            print("[Visualization] Nearest Neighbor: Insufficient data, skipping.")
            return

        feat_src  = data['feat_src'].float()
        feat_tgt  = data['feat_tgt'].float()
        src_imgs  = data['src_vis_img']
        tgt_imgs  = data['tgt_vis_img']
        src_labels = data['src_label']
        tgt_labels = data['tgt_label']

        n_src = feat_src.size(0)
        n_tgt = feat_tgt.size(0)
        n_query = min(6, n_src)
        n_neighbors = min(4, n_tgt)

        if n_query == 0 or n_neighbors == 0:
            print("[Visualization] Nearest Neighbor: Insufficient sample count, skipping.")
            return

        src_norm = F.normalize(feat_src, dim=1)
        tgt_norm = F.normalize(feat_tgt, dim=1)
        sim_matrix = torch.mm(src_norm, tgt_norm.t())  # [n_src, n_tgt]

        src_vis_np = denorm_vis_np(src_imgs)
        tgt_vis_np = denorm_vis_np(tgt_imgs)

        n_cols = 1 + n_neighbors
        fig, axes = plt.subplots(n_query, n_cols,
                                 figsize=(n_cols * 2.2, n_query * 2.2))
        if n_query == 1:
            axes = axes[np.newaxis, :]

        for row in range(n_query):
            axes[row, 0].imshow(src_vis_np[row].clip(0, 1))
            axes[row, 0].axis('off')
            if row == 0:
                axes[row, 0].set_title('Query', fontsize=10)

            top_k_idx = sim_matrix[row].topk(n_neighbors).indices

            for col_offset, nb_idx in enumerate(top_k_idx):
                nb_idx = nb_idx.item()
                col = col_offset + 1
                axes[row, col].imshow(tgt_vis_np[nb_idx].clip(0, 1))
                axes[row, col].axis('off')

                sim_score = sim_matrix[row, nb_idx].item()
                match = (src_labels[row].item() == tgt_labels[nb_idx].item())
                color = 'green' if match else 'red'

                axes[row, col].set_title(f'{sim_score:.3f}',
                                         fontsize=10, color=color)
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
                axes[row, col].tick_params(left=False, bottom=False)

        plt.tight_layout()
        save_path = os.path.join(epoch_dir, '2_nearest_neighbor',
                                 f'nearest_neighbor_epoch_{epoch:03d}.svg')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualization] Nearest Neighbor figure saved: {save_path}")

    # ── visualization 3: TAL comparison t-SNE ────────────────────────────────

    def _visualize_tal_comparison(self, epoch, epoch_dir, data, label_map=None):
        required = ('f_s_vis', 'f_s_ir', 'f_t_vis', 'f_t_ir',
                    'p_s_vis', 'p_s_ir', 'p_t_vis', 'p_t_ir',
                    'src_label', 'tgt_label')
        if any(data[k] is None for k in required):
            print("[Visualization] TAL Comparison: Insufficient data, skipping.")
            return

        from sklearn.manifold import TSNE

        f_s_vis = data['f_s_vis'].float().numpy()
        f_s_ir  = data['f_s_ir'].float().numpy()
        f_t_vis = data['f_t_vis'].float().numpy()
        f_t_ir  = data['f_t_ir'].float().numpy()
        p_s_vis = data['p_s_vis'].float().numpy()
        p_s_ir  = data['p_s_ir'].float().numpy()
        p_t_vis = data['p_t_vis'].float().numpy()
        p_t_ir  = data['p_t_ir'].float().numpy()

        src_labels = data['src_label'].numpy()
        tgt_labels = data['tgt_label'].numpy()

        n = f_s_vis.shape[0]
        labels_all = np.concatenate([src_labels, src_labels,
                                     tgt_labels, tgt_labels])

        pre_feats  = np.concatenate([f_s_vis, f_s_ir, f_t_vis, f_t_ir], axis=0)
        post_feats = np.concatenate([p_s_vis, p_s_ir, p_t_vis, p_t_ir], axis=0)

        n_total = pre_feats.shape[0]
        perp = max(5, min(30, n_total // 4 - 1))

        print(f"[Visualization] TAL Comparison: Computing t-SNE (pre-TAL, n={n_total})...")
        tsne_pre = TSNE(n_components=2, perplexity=perp,
                        random_state=42, n_iter=500)
        xy_pre = tsne_pre.fit_transform(pre_feats)

        print(f"[Visualization] TAL Comparison: Computing t-SNE (post-TAL, n={n_total})...")
        tsne_post = TSNE(n_components=2, perplexity=perp,
                         random_state=42, n_iter=500)
        xy_post = tsne_post.fit_transform(post_feats)

        xy_pre_sv,  xy_pre_si,  xy_pre_tv,  xy_pre_ti  = np.split(xy_pre,  4)
        xy_post_sv, xy_post_si, xy_post_tv, xy_post_ti = np.split(xy_post, 4)

        domain_colors  = ['blue', 'cyan', 'red', 'orange']
        domain_markers = ['o',    's',    '^',   'D']
        domain_labels  = ['src-vis', 'src-ir', 'tgt-vis', 'tgt-ir']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'TAL Comparison (Domain) - Epoch {epoch:03d}', fontsize=10)

        for ax, (xy_sv, xy_si, xy_tv, xy_ti), title in zip(
                axes,
                [(xy_pre_sv,  xy_pre_si,  xy_pre_tv,  xy_pre_ti),
                 (xy_post_sv, xy_post_si, xy_post_tv, xy_post_ti)],
                ['Before TAL (512-dim)', 'After TAL (128-dim)']):

            for xys, color, marker, lbl in zip(
                    [xy_sv, xy_si, xy_tv, xy_ti],
                    domain_colors, domain_markers, domain_labels):
                ax.scatter(xys[:, 0], xys[:, 1], c=color, marker=marker,
                           s=20, alpha=0.6, label=lbl)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=10, markerscale=1.2)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        path_dom = os.path.join(epoch_dir, '3_tal_comparison',
                                f'tal_comparison_domain_epoch_{epoch:03d}.svg')
        fig.savefig(path_dom, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualization] TAL domain comparison figure saved: {path_dom}")

        all_classes = np.unique(labels_all)
        n_classes = len(all_classes)
        cmap = plt.cm.tab20
        color_vals = np.linspace(0, 1, n_classes)
        class_color_map = {cls: cmap(color_vals[i])
                           for i, cls in enumerate(all_classes)}

        if label_map is not None:
            reverse_map = {v: k for k, v in label_map.items()}
        else:
            reverse_map = {}

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'TAL Comparison (Class) - Epoch {epoch:03d}', fontsize=10)

        for ax, xy, title in zip(
                axes,
                [xy_pre, xy_post],
                ['Before TAL (512-dim)', 'After TAL (128-dim)']):

            for cls in all_classes:
                mask = (labels_all == cls)
                lbl_str = reverse_map.get(int(cls), str(int(cls)))
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           c=[class_color_map[cls]], s=20, alpha=0.6,
                           label=lbl_str)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=10, markerscale=1.2,
                      loc='upper right', ncol=max(1, n_classes // 10))
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        path_cls = os.path.join(epoch_dir, '3_tal_comparison',
                                f'tal_comparison_class_epoch_{epoch:03d}.svg')
        fig.savefig(path_cls, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualization] TAL class comparison figure saved: {path_cls}")

    # ── visualization 4: OT transport ────────────────────────────────────────

    def _visualize_ot_transport(self, epoch, epoch_dir, data, label_map=None):
        required = ('feat_src', 'feat_mid', 'feat_tgt', 'src_label', 'tgt_label')
        if any(data[k] is None for k in required):
            print("[Visualization] OT Transport: Insufficient data, skipping.")
            return

        from sklearn.manifold import TSNE

        feat_src = data['feat_src'].float().numpy()
        feat_mid = data['feat_mid'].float().numpy()
        feat_tgt = data['feat_tgt'].float().numpy()
        src_labels = data['src_label'].numpy()
        tgt_labels = data['tgt_label'].numpy()

        n = feat_src.shape[0]
        all_feats = np.concatenate([feat_src, feat_mid, feat_tgt], axis=0)
        n_total = all_feats.shape[0]
        perp = max(5, min(30, n_total // 4 - 1))

        print(f"[Visualization] OT Transport: Computing t-SNE (n={n_total})...")
        tsne = TSNE(n_components=2, perplexity=perp,
                    random_state=42, n_iter=500)
        xy_all = tsne.fit_transform(all_feats)

        src_2d = xy_all[:n]
        mid_2d = xy_all[n:2 * n]
        tgt_2d = xy_all[2 * n:]

        all_classes = np.unique(np.concatenate([src_labels, tgt_labels]))
        n_classes = len(all_classes)
        cmap = plt.cm.tab20
        color_vals = np.linspace(0, 1, n_classes)
        class_color_map = {cls: cmap(color_vals[i])
                           for i, cls in enumerate(all_classes)}

        src_colors = [class_color_map[c] for c in src_labels]
        tgt_colors = [class_color_map[c] for c in tgt_labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f'OT Transport Path - Epoch {epoch:03d}', fontsize=10)

        ax.scatter(src_2d[:, 0], src_2d[:, 1], c=src_colors,
                   marker='o', s=40, alpha=0.7, label='Source', zorder=3)
        ax.scatter(mid_2d[:, 0], mid_2d[:, 1], c=src_colors,
                   marker='*', s=60, alpha=0.7, label='Mid', zorder=3)
        ax.scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=tgt_colors,
                   marker='^', s=40, alpha=0.7, label='Target', zorder=3)

        n_arrows = min(20, n)
        arrow_idx = np.random.choice(n, size=n_arrows, replace=False)

        for idx in arrow_idx:
            ax.annotate('',
                        xy=mid_2d[idx], xytext=src_2d[idx],
                        arrowprops=dict(arrowstyle='->', color='steelblue',
                                        lw=1.0, connectionstyle='arc3,rad=0.05'),
                        zorder=2)
            ax.annotate('',
                        xy=tgt_2d[idx], xytext=mid_2d[idx],
                        arrowprops=dict(arrowstyle='->', color='tomato',
                                        lw=1.0, linestyle='dashed',
                                        connectionstyle='arc3,rad=0.05'),
                        zorder=2)

        legend_handles = [
            mpatches.Patch(color='grey', label='Circle = Source (o)'),
            mpatches.Patch(color='grey', label='Star = Mid (*)'),
            mpatches.Patch(color='grey', label='Triangle = Target (^)'),
        ]
        ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
        ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        save_path = os.path.join(epoch_dir, '4_ot_transport',
                                 f'ot_transport_epoch_{epoch:03d}.svg')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualization] OT Transport figure saved: {save_path}")

    # ── visualization 5: t-SNE thumbnails ────────────────────────────────────

    def _visualize_tsne_thumbnails(self, epoch, epoch_dir, data):
        required = ('feat_src', 'feat_tgt',
                    'src_vis_img', 'src_ir_img',
                    'tgt_vis_img', 'tgt_ir_img')
        if any(data[k] is None for k in required):
            print("[Visualization] t-SNE Thumbnails: Insufficient data, skipping.")
            return

        from sklearn.manifold import TSNE
        import torchvision.transforms.functional as TF

        n_max = 64
        feat_src = data['feat_src'].float()
        feat_tgt = data['feat_tgt'].float()

        n_src = min(n_max // 2, feat_src.size(0))
        n_tgt = min(n_max // 2, feat_tgt.size(0))
        n_used = n_src + n_tgt

        feat_src = feat_src[:n_src]
        feat_tgt = feat_tgt[:n_tgt]

        src_vis = data['src_vis_img'][:n_src]
        src_ir  = data['src_ir_img'][:n_src]
        tgt_vis = data['tgt_vis_img'][:n_tgt]
        tgt_ir  = data['tgt_ir_img'][:n_tgt]

        all_feats = torch.cat([feat_src, feat_tgt], dim=0).numpy()
        perp = max(5, min(30, n_used // 4 - 1))

        print(f"[Visualization] t-SNE Thumbnails: Computing t-SNE (n={n_used})...")
        tsne = TSNE(n_components=2, perplexity=perp,
                    random_state=42, n_iter=500)
        xy = tsne.fit_transform(all_feats)

        src_xy = xy[:n_src]
        tgt_xy = xy[n_src:]

        def _make_thumbs_np(tensor, denorm_fn):
            imgs_np = denorm_fn(tensor)
            thumbs = []
            for img in imgs_np:
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                t = F.interpolate(t.float(), size=(32, 32), mode='bilinear',
                                  align_corners=False).squeeze(0)
                thumbs.append(t.permute(1, 2, 0).numpy().clip(0, 1))
            return thumbs

        src_vis_thumbs = _make_thumbs_np(src_vis, denorm_vis_np)
        tgt_vis_thumbs = _make_thumbs_np(tgt_vis, denorm_vis_np)
        src_ir_thumbs  = _make_thumbs_np(src_ir,  denorm_ir_np)
        tgt_ir_thumbs  = _make_thumbs_np(tgt_ir,  denorm_ir_np)

        def _plot_tsne_with_images(src_thumbs, tgt_thumbs, title, save_path):
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_title(title, fontsize=10)
            ax.set_xlim(xy[:, 0].min() - 5, xy[:, 0].max() + 5)
            ax.set_ylim(xy[:, 1].min() - 5, xy[:, 1].max() + 5)
            ax.set_xticks([]); ax.set_yticks([])

            for i, (pos, thumb) in enumerate(zip(src_xy, src_thumbs)):
                bordered = _add_border(thumb, color='blue')
                oi = OffsetImage(bordered, zoom=0.5)
                ab = AnnotationBbox(oi, pos, frameon=False)
                ax.add_artist(ab)

            for i, (pos, thumb) in enumerate(zip(tgt_xy, tgt_thumbs)):
                bordered = _add_border(thumb, color='red')
                oi = OffsetImage(bordered, zoom=0.5)
                ab = AnnotationBbox(oi, pos, frameon=False)
                ax.add_artist(ab)

            src_patch = mpatches.Patch(color='blue', label='Source')
            tgt_patch = mpatches.Patch(color='red',  label='Target')
            ax.legend(handles=[src_patch, tgt_patch], fontsize=10,
                      loc='upper right')

            plt.tight_layout()
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        def _add_border(img_np, color, border=2):
            h, w, c = img_np.shape
            bordered = np.ones((h + 2 * border, w + 2 * border, c), dtype=np.float32)
            if color == 'blue':
                bordered[:, :] = [0.0, 0.0, 1.0]
            else:
                bordered[:, :] = [1.0, 0.0, 0.0]
            bordered[border:border + h, border:border + w, :] = img_np
            return bordered

        path_vis = os.path.join(epoch_dir, '5_tsne_thumbnails',
                                f'tsne_thumbnails_vis_epoch_{epoch:03d}.svg')
        path_ir  = os.path.join(epoch_dir, '5_tsne_thumbnails',
                                f'tsne_thumbnails_ir_epoch_{epoch:03d}.svg')

        _plot_tsne_with_images(
            src_vis_thumbs, tgt_vis_thumbs,
            f't-SNE Thumbnails (Visible) - Epoch {epoch:03d}',
            path_vis)
        print(f"[Visualization] t-SNE visible thumbnail figure saved: {path_vis}")

        _plot_tsne_with_images(
            src_ir_thumbs, tgt_ir_thumbs,
            f't-SNE Thumbnails (Infrared) - Epoch {epoch:03d}',
            path_ir)
        print(f"[Visualization] t-SNE infrared thumbnail figure saved: {path_ir}")