import json
import os

def create_kaggle_notebook(output_path: str = "notebooks/AIC_2026_Kaggle_Pipeline.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 AIC 2026: End-to-End Dual-GPU (2× T4) Production Pipeline (Scene-Adaptive Sampling)\n",
                "\n",
                "This notebook implements the complete **AI Challenge (AIC) 2026** multi-modal retrieval pipeline:\n",
                "1. **GPU 0**: SigLIP-SO400M @ 384px **scene-adaptive shot sampling** (eliminates redundant static frames)\n",
                "2. **GPU 1**: Whisper large-v3 Vietnamese audio transcription & on-screen OCR\n",
                "3. **Unified Indexing**: Scalable FAISS FlatIP vector index (~120k–400k frames) + Multi-modal BM25 lexical index\n",
                "4. **Stage 2 Exact Localizer**: 30fps dense video decode around candidate timestamps (KIS & Q&A)\n",
                "5. **TRAKE Stage 1**: DP-aligned video retrieval via scene index (Stage 2 VLM localization = future work)\n",
                "6. **Submission Engine**: 100-rank portfolio optimization for competition metric $\\frac{1}{5}\\sum R@k$\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify Dual GPU Hardware (2x NVIDIA T4) & Setup Working Directory\n",
                "import torch, os, sys\n",
                "\n",
                "# Navigate into cloned repository if present\n",
                "REPO_DIR = '/kaggle/working/aic2026'\n",
                "if os.path.exists(REPO_DIR):\n",
                "    os.chdir(REPO_DIR)\n",
                "    print(f'Active working directory set to: {os.getcwd()}')\n",
                "\n",
                "if os.getcwd() not in sys.path:\n",
                "    sys.path.insert(0, os.getcwd())\n",
                "\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f'Detected {num_gpus} CUDA GPUs:')\n",
                "for i in range(num_gpus):\n",
                "    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)')\n",
                "\n",
                "assert num_gpus >= 1, 'Please enable GPU accelerator in Kaggle Settings (2x T4 recommended)!'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Install Required Dependencies\n",
                "!pip install -q open-clip-torch transformers faster-whisper openai-whisper faiss-cpu rank-bm25 deep-translator opencv-python easyocr fiftyone"
            ]

        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Symlink / Prepare Data Directories from Kaggle Input into aic2026/data\n",
                "import os, glob\n",
                "\n",
                "target_dir = os.path.join(os.getcwd(), 'data')\n",
                "os.makedirs(target_dir, exist_ok=True)\n",
                "\n",
                "if os.path.exists('/kaggle/input'):\n",
                "    print(f'Detected Kaggle environment. Linking input datasets into {target_dir}...')\n",
                "    for p in glob.glob('/kaggle/input/**/Videos_*', recursive=True):\n",
                "        dest = os.path.join(target_dir, os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "    for p in glob.glob('/kaggle/input/**/Keyframes_*', recursive=True):\n",
                "        dest = os.path.join(target_dir, os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "    for p in glob.glob('/kaggle/input/**/*media-info*', recursive=True):\n",
                "        dest = os.path.join(target_dir, os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "    for p in glob.glob('/kaggle/input/**/*map-keyframes*', recursive=True):\n",
                "        dest = os.path.join(target_dir, os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "    for p in glob.glob('/kaggle/input/**/*-aic25-b1', recursive=True):\n",
                "        dest = os.path.join(target_dir, os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "\n",
                "print(f'Linked data contents ({target_dir}):', sorted(os.listdir(target_dir)))"
            ]
        },

        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### ⚡ Step 4: Parallel Dual-GPU Feature Extraction (Scene-Adaptive Video + Whisper ASR)"
            ]
        },
        {
            "cell_type": "code",

            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, time\n",
                "import subprocess\n",
                "\n",
                "print('[Pipeline] Starting Dual-GPU Parallel Feature Extraction across all modalities...')\n",
                "t_start = time.time()\n",
                "\n",
                "repo_root = os.getcwd() if os.path.exists('src') else ('/kaggle/working/aic2026' if os.path.exists('/kaggle/working/aic2026') else '.')\n",
                "\n",
                "# Environment isolation for Dual-GPU concurrency with PYTHONPATH\n",
                "env_gpu0 = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': repo_root}\n",
                "env_gpu1 = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': repo_root}\n",
                "\n",
                "# --- Stage 1: Dual-GPU SigLIP Vision Extraction (Sharded 50/50 across GPU 0 and GPU 1) ---\n",
                "print('\\n🚀 Stage 1/3: Extracting SigLIP Vision Features in parallel on GPU 0 & GPU 1...')\n",
                "p1_0 = subprocess.Popen(['python', 'scripts/extract_siglip_features.py', '--device', 'cuda:0', '--batch-size', '256', '--num-shards', '2', '--shard-id', '0'], env=env_gpu0, cwd=repo_root)\n",
                "p1_1 = subprocess.Popen(['python', 'scripts/extract_siglip_features.py', '--device', 'cuda:0', '--batch-size', '256', '--num-shards', '2', '--shard-id', '1'], env=env_gpu1, cwd=repo_root)\n",
                "ret1_0 = p1_0.wait()\n",
                "ret1_1 = p1_1.wait()\n",
                "if ret1_0 != 0 or ret1_1 != 0:\n",
                "    raise RuntimeError(f'SigLIP extraction failed (GPU 0: {ret1_0}, GPU 1: {ret1_1})')\n",
                "print('✅ Stage 1 complete: SigLIP features extracted across all videos.')\n",
                "\n",
                "# --- Stage 2: Dual-GPU PhoWhisper-small ASR Extraction (Sharded 50/50 across GPU 0 and GPU 1) ---\n",
                "print('\\n🚀 Stage 2/3: Extracting PhoWhisper-small ASR in parallel on GPU 0 & GPU 1...')\n",
                "p2_0 = subprocess.Popen(['python', 'scripts/extract_whisper_asr.py', '--device', 'cuda:0', '--model-size', 'vinai/PhoWhisper-small', '--batch-size', '32', '--beam-size', '1', '--num-shards', '2', '--shard-id', '0'], env=env_gpu0, cwd=repo_root)\n",
                "p2_1 = subprocess.Popen(['python', 'scripts/extract_whisper_asr.py', '--device', 'cuda:0', '--model-size', 'vinai/PhoWhisper-small', '--batch-size', '32', '--beam-size', '1', '--num-shards', '2', '--shard-id', '1'], env=env_gpu1, cwd=repo_root)\n",
                "ret2_0 = p2_0.wait()\n",
                "ret2_1 = p2_1.wait()\n",
                "if ret2_0 != 0 or ret2_1 != 0:\n",
                "    raise RuntimeError(f'PhoWhisper ASR extraction failed (GPU 0: {ret2_0}, GPU 1: {ret2_1})')\n",
                "print('✅ Stage 2 complete: PhoWhisper transcripts extracted across all videos.')\n",
                "\n",
                "# --- Stage 3: Dual-GPU OCR Text Extraction (Sharded 50/50 across GPU 0 and GPU 1) ---\n",
                "print('\\n🚀 Stage 3/3: Extracting Video OCR in parallel on GPU 0 & GPU 1...')\n",
                "p3_0 = subprocess.Popen(['python', 'scripts/extract_ocr.py', '--device', 'cuda:0', '--num-shards', '2', '--shard-id', '0'], env=env_gpu0, cwd=repo_root)\n",
                "p3_1 = subprocess.Popen(['python', 'scripts/extract_ocr.py', '--device', 'cuda:0', '--num-shards', '2', '--shard-id', '1'], env=env_gpu1, cwd=repo_root)\n",
                "ret3_0 = p3_0.wait()\n",
                "ret3_1 = p3_1.wait()\n",
                "if ret3_0 != 0 or ret3_1 != 0:\n",
                "    raise RuntimeError(f'OCR extraction failed (GPU 0: {ret3_0}, GPU 1: {ret3_1})')\n",
                "\n",
                "elapsed = (time.time() - t_start) / 60\n",
                "print(f'🎉 All Dual-GPU feature extractions complete in {elapsed:.1f} minutes!')"
            ]
        },


        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🏗️ Step 5: Build Scalable FAISS & Multi-Modal BM25 Indices"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build production FAISS index & unified lexical BM25 index\n",
                "!python scripts/build_faiss_index.py\n",
                "\n",
                "from src.index.metadata_indexer import MetadataIndexer\n",
                "meta_idx = MetadataIndexer().build_and_cache(force=True)\n",
                "print('Unified indices successfully built!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 Step 6: Interactive Multi-Modal Retrieval with Stage 2 Dense Localization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob\n",
                "import numpy as np\n",
                "import torch\n",
                "from src.index.faiss_index import FAISSIndex\n",
                "from src.index.metadata_indexer import MetadataIndexer\n",
                "from src.index.object_indexer import ObjectIndexer\n",
                "from src.encoding.siglip_encoder import SigLIPEncoder\n",
                "from src.query.text_encoder import CLIPTextEncoder\n",
                "from src.query.translator import QueryTranslator\n",
                "from src.retrieval.video_decoder import VideoDecoder\n",
                "from src.evaluation.submission_generator import SubmissionGenerator\n",
                "\n",
                "# Load FAISS Index and Lexical Indices\n",
                "faiss_idx = FAISSIndex().load('cache/faiss_siglip')\n",
                "meta_idx = MetadataIndexer().build_and_cache()\n",
                "obj_idx = ObjectIndexer().build_and_cache()\n",
                "\n",
                "# Select matching text encoder based on index dimensions\n",
                "device = 'cuda:0' if torch.cuda.is_available() else 'cpu'\n",
                "if faiss_idx.dim == 1152:\n",
                "    print('Using SigLIP Text Encoder (1152-dim)...')\n",
                "    text_encoder = SigLIPEncoder(device=device)\n",
                "else:\n",
                "    print('Using CLIP ViT-B/32 Text Encoder (512-dim)...')\n",
                "    text_encoder = CLIPTextEncoder(device=device)\n",
                "\n",
                "translator = QueryTranslator(use_online=True)\n",
                "sub_gen = SubmissionGenerator(output_dir='submissions')\n",
                "video_decoder = VideoDecoder(encoder=text_encoder, device=device)\n",
                "\n",
                "def search_query(query_vi: str, top_k: int = 100, use_stage2_refinement: bool = True):\n",
                "    en_query = translator.translate(query_vi)\n",
                "    prompts = translator.generate_prompts(en_query)\n",
                "    q_vec = text_encoder.encode_text(prompts, ensemble=True)\n",
                "    \n",
                "    # 1. FAISS Stage 1 dense candidate retrieval\n",
                "    dense_results = faiss_idx.search(q_vec, top_k=top_k * 2)\n",
                "    \n",
                "    # 2. Hybrid re-ranking with Metadata BM25\n",
                "    meta_scores = meta_idx.query(f'{query_vi} {en_query}', top_k=50)\n",
                "    \n",
                "    reranked = []\n",
                "    for rec, score in dense_results:\n",
                "        vid = rec['video_id']\n",
                "        boost = 0.2 * (meta_scores.get(vid, 0.0) / max(1.0, max(meta_scores.values() or [1.0])))\n",
                "        reranked.append((rec, score + boost))\n",
                "        \n",
                "    reranked.sort(key=lambda x: x[1], reverse=True)\n",
                "    candidates = reranked[:top_k]\n",
                "    \n",
                "    # 3. Stage 2 Exact Localization for top-5 candidates (30fps continuous decode)\n",
                "    if use_stage2_refinement:\n",
                "        final_results = []\n",
                "        for i, (rec, score) in enumerate(candidates):\n",
                "            if i < 5:  # Refine top-5\n",
                "                vid_p = glob.glob(f'data/**/video/{rec[\"video_id\"]}.mp4', recursive=True)\n",
                "                if vid_p and os.path.exists(vid_p[0]):\n",
                "                    best_f, best_s, best_t = video_decoder.localize_exact_frame(\n",
                "                        video_path=vid_p[0],\n",
                "                        candidate_pts_time=rec['pts_time'],\n",
                "                        query_vec=q_vec,\n",
                "                        window_seconds=6.0\n",
                "                    )\n",
                "                    refined_rec = dict(rec)\n",
                "                    refined_rec['frame_idx'] = best_f\n",
                "                    refined_rec['pts_time'] = best_t\n",
                "                    final_results.append((refined_rec, max(score, best_s)))\n",
                "                    continue\n",
                "            final_results.append((rec, score))\n",
                "        return final_results\n",
                "        \n",
                "    return candidates\n",
                "\n",
                "# Test sample query\n",
                "sample_query = 'Người dẫn chương trình thời sự 60 giây trong trường quay'\n",
                "results = search_query(sample_query, top_k=100, use_stage2_refinement=False)\n",
                "print(f'Top 5 Results for \"{sample_query}\":')\n",
                "for i, (rec, score) in enumerate(results[:5], 1):\n",
                "    print(f'  [{i}] Video: {rec[\"video_id\"]}, Frame: {rec[\"frame_idx\"]}, Score: {score:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",

            "metadata": {},
            "source": [
                "### 📦 Step 7: Export Official Competition Submissions (Exact 100 Rows)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate official 100-row submission CSV and package bundle ZIP\n",
                "sample_preds = [{'video_id': r[0]['video_id'], 'frame_idx': r[0]['frame_idx']} for r in results]\n",
                "lines = sub_gen.format_kis_submission('query_01', sample_preds)\n",
                "sub_gen.save_submission_file('query_01', lines)\n",
                "zip_path = sub_gen.package_submission_zip('AIC2026_Submission_Bundle.zip')\n",
                "print(f'Official Submission Package Ready: {zip_path} (Contains {len(lines)} rows in query_01.csv)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📥 Step 8: Package & Download All Encoded Features (Visual, Audio ASR, OCR & FAISS Indices)\n",
                "\n",
                "Compresses all generated `.npy` visual embeddings, Whisper transcripts, OCR text, and FAISS indices into a single downloadable ZIP archive in `/kaggle/working/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, zipfile\n",
                "from IPython.display import FileLink\n",
                "\n",
                "zip_filename = 'aic2026_features_and_cache.zip'\n",
                "zip_path = os.path.join('/kaggle/working' if os.path.exists('/kaggle/working') else '.', zip_filename)\n",
                "\n",
                "print(f'[Export] Packaging cache directory into {zip_path}...')\n",
                "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
                "    for root, dirs, files in os.walk('cache'):\n",
                "        for file in files:\n",
                "            if file.endswith(('.npy', '.json', '.index', '.pkl')) and '.tmp.' not in file:\n",
                "                abs_path = os.path.join(root, file)\n",
                "                rel_path = os.path.relpath(abs_path, '.')\n",
                "                zf.write(abs_path, arcname=rel_path)\n",
                "\n",
                "zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)\n",
                "print(f'✅ All features successfully packaged! Total Archive Size: {zip_size_mb:.2f} MB')\n",
                "print(f'Artifact saved at: {zip_path}')\n",
                "FileLink(zip_filename)"
            ]
        }
    ]


    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)

    print(f"Generated updated Kaggle master notebook at {output_path}")

if __name__ == "__main__":
    create_kaggle_notebook()
