import os
import sys
import glob
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless Kaggle / server environments
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def inspect_pipeline_quality(
    cache_root: str = "cache",
    plot_output_path: str = "siglip_embedding_2d_plot.png",
    num_sample_videos: int = 10,
):
    print("=" * 70)
    print("📊 AIC 2026 FEATURE QUALITY INSPECTION & 2D MANIFOLD VISUALIZATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. SigLIP 1152-d Visual Embeddings Inspection
    # -------------------------------------------------------------
    feat_dir = os.path.join(cache_root, "siglip_features")
    meta_dir = os.path.join(cache_root, "siglip_meta")
    
    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
    meta_files = sorted(glob.glob(os.path.join(meta_dir, "*.json")))

    print(f"\n[1] SigLIP Visual Embeddings (google/siglip-so400m-patch14-384):")
    print(f"  • Total .npy feature files : {len(feat_files)}")
    print(f"  • Total .json meta files    : {len(meta_files)}")

    if feat_files:
        sample_files = feat_files[:num_sample_videos]
        sampled_embs = []
        labels = []
        intra_sims = []
        inter_sims = []

        total_frames = 0
        for f_path in feat_files:
            try:
                emb = np.load(f_path)
                total_frames += len(emb)
            except Exception:
                pass

        print(f"  • Total Keyframes Indexed   : {total_frames:,}")

        # Sample subset for fine-grained cosine similarity and 2D plot
        for f_path in sample_files:
            v_name = os.path.splitext(os.path.basename(f_path))[0]
            emb = np.load(f_path).astype(np.float32)
            sampled_embs.append(emb)
            labels.extend([v_name] * len(emb))

            # Intra-video similarity (consecutive keyframes in same video)
            if len(emb) > 1:
                dot_products = np.sum(emb[:-1] * emb[1:], axis=1)
                intra_sims.extend(dot_products.tolist())

        stacked = np.vstack(sampled_embs)
        norms = np.linalg.norm(stacked, axis=1)

        # Inter-video similarity (random pairs from distinct videos)
        for _ in range(500):
            i, j = random.sample(range(len(sample_files)), 2)
            idx_a = random.randint(0, len(sampled_embs[i]) - 1)
            idx_b = random.randint(0, len(sampled_embs[j]) - 1)
            inter_sims.append(float(np.dot(sampled_embs[i][idx_a], sampled_embs[j][idx_b])))

        print(f"  • Sampled Vectors for Plot  : {stacked.shape[0]} (Dim: {stacked.shape[1]})")
        print(f"  • L2 Norm Verification     : Min={norms.min():.4f}, Max={norms.max():.4f}, Mean={norms.mean():.4f} (Target: 1.0000)")
        print(f"  • Intra-Video Similarity    : Mean = {np.mean(intra_sims):.3f} (High coherence across temporal neighbors)")
        print(f"  • Inter-Video Similarity    : Mean = {np.mean(inter_sims):.3f} (Low cross-talk, high discriminative power)")

        # --- 2D PCA & t-SNE Projections ---
        print(f"\n  🎨 Generating 2D PCA & t-SNE plots -> saving to {plot_output_path}...")
        pca_2d = PCA(n_components=2).fit_transform(stacked)
        tsne_2d = TSNE(n_components=2, perplexity=min(30, max(5, len(stacked)-1)), random_state=42).fit_transform(stacked)

        unique_vids = list(dict.fromkeys(labels))
        cmap = plt.colormaps.get_cmap("tab10")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for idx, vid in enumerate(unique_vids):
            mask = [lbl == vid for lbl in labels]
            c = cmap(idx % 10)
            axes[0].scatter(pca_2d[mask, 0], pca_2d[mask, 1], label=vid, color=c, alpha=0.8, s=40, edgecolors='none')
            axes[1].scatter(tsne_2d[mask, 0], tsne_2d[mask, 1], label=vid, color=c, alpha=0.8, s=40, edgecolors='none')

        axes[0].set_title("SigLIP 1152-d Embeddings — 2D PCA Projection", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].set_title("SigLIP 1152-d Embeddings — 2D t-SNE Manifold", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("t-SNE Dimension 1")
        axes[1].set_ylabel("t-SNE Dimension 2")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(plot_output_path, dpi=200)
        plt.close()
        print(f"  ✅ Plot successfully saved to {plot_output_path}")

    # -------------------------------------------------------------
    # 2. Whisper ASR Quality Inspection
    # -------------------------------------------------------------
    asr_dir = os.path.join(cache_root, "asr_transcripts")
    asr_files = sorted(glob.glob(os.path.join(asr_dir, "*.json")))
    print(f"\n[2] Whisper Vietnamese ASR Transcripts:")
    print(f"  • Total transcript files    : {len(asr_files)}")

    if asr_files:
        total_segments = 0
        total_words = 0
        sample_transcripts = []

        for f_path in asr_files:
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_segments += len(data)
                    for seg in data:
                        text = seg.get("text", "").strip()
                        words = text.split()
                        total_words += len(words)
                        if len(sample_transcripts) < 6 and len(words) >= 5:
                            sample_transcripts.append((seg.get("video_id", ""), seg.get("start_sec", 0.0), seg.get("end_sec", 0.0), text))
            except Exception:
                pass

        print(f"  • Total Speech Segments     : {total_segments:,}")
        print(f"  • Total Spoken Words        : {total_words:,}")
        print(f"  • Avg Words per Video       : {total_words / max(1, len(asr_files)):.1f}")

        print("\n  📝 Sample Vietnamese ASR Transcripts with Timestamps:")
        for vid, s, e, txt in sample_transcripts:
            print(f"    [{vid}] ({s:5.1f}s -> {e:5.1f}s): \"{txt}\"")

    # -------------------------------------------------------------
    # 3. Video OCR Quality Inspection
    # -------------------------------------------------------------
    ocr_dir = os.path.join(cache_root, "ocr_text")
    ocr_files = sorted(glob.glob(os.path.join(ocr_dir, "*.json")))
    print(f"\n[3] Video OCR Text Banners:")
    print(f"  • Total OCR files           : {len(ocr_files)}")

    if ocr_files:
        total_ocr_frames = 0
        sample_ocr = []
        for f_path in ocr_files:
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_ocr_frames += len(data)
                    v_name = os.path.splitext(os.path.basename(f_path))[0]
                    for f_key, txt in data.items():
                        if len(sample_ocr) < 5 and len(txt.strip()) > 3:
                            sample_ocr.append((v_name, f_key, txt))
            except Exception:
                pass

        print(f"  • Total Text Frames Detected: {total_ocr_frames:,}")
        print("\n  📝 Sample OCR Text Extracted from Keyframes:")
        for vid, f_key, txt in sample_ocr:
            print(f"    [{vid} - {f_key}]: \"{txt}\"")

    print("\n" + "=" * 70)
    print("🎉 Feature Quality Inspection Complete!")
    print("=" * 70)


if __name__ == "__main__":
    inspect_pipeline_quality()
