import json
import os

def create_kaggle_notebook(output_path: str = "notebooks/AIC_2026_Kaggle_Pipeline.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 AIC 2026: End-to-End Dual-GPU (2× T4) Production Pipeline (5 FPS Dense Sampling)\n",
                "\n",
                "This notebook implements the complete **AI Challenge (AIC) 2026** multi-modal retrieval pipeline:\n",
                "1. **GPU 0**: SigLIP-SO400M @ 384px **5 fps dense visual extraction** (every 6 frames / 0.20s interval)\n",
                "2. **GPU 1**: Whisper large-v3 Vietnamese audio transcription & on-screen OCR\n",
                "3. **Unified Indexing**: Scalable FAISS vector index + Multi-modal BM25 lexical index\n",
                "4. **Stage 2 Exact Localizer**: 30fps dense video decode around candidate timestamps\n",
                "5. **Submission Engine**: 100-rank portfolio optimization for competition metric $\\frac{1}{5}\\sum R@k$\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify Dual GPU Hardware (2x NVIDIA T4)\n",
                "import torch, os, sys\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f\"Detected {num_gpus} CUDA GPUs:\")\n",
                "for i in range(num_gpus):\n",
                "    print(f\"  GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)\")\n",
                "\n",
                "assert num_gpus >= 1, \"Please enable GPU accelerator in Kaggle Settings (2x T4 recommended)!\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Install Required Dependencies\n",
                "!pip install -q open-clip-torch transformers openai-whisper faiss-cpu rank-bm25 deep-translator opencv-python easyocr fiftyone"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Symlink / Prepare Data Directories from Kaggle Input\n",
                "import os, glob\n",
                "\n",
                "if os.path.exists('/kaggle/input'):\n",
                "    print('Detected Kaggle environment. Linking input directories...')\n",
                "    os.makedirs('data', exist_ok=True)\n",
                "    for p in glob.glob('/kaggle/input/**/Videos_*', recursive=True):\n",
                "        dest = os.path.join('data', os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "    for p in glob.glob('/kaggle/input/**/Keyframes_*', recursive=True):\n",
                "        dest = os.path.join('data', os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "    for p in glob.glob('/kaggle/input/**/*-aic25-b1', recursive=True):\n",
                "        dest = os.path.join('data', os.path.basename(p))\n",
                "        if not os.path.exists(dest):\n",
                "            os.symlink(p, dest)\n",
                "\n",
                "print('Data directory contents:', os.listdir('data') if os.path.exists('data') else 'None')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### ⚡ Step 4: Parallel Dual-GPU Feature Extraction (5 FPS Dense Video + Whisper ASR)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import multiprocessing\n",
                "\n",
                "def run_siglip():\n",
                "    print('[Process 1] Starting SigLIP-SO400M 5fps extraction on GPU 0...')\n",
                "    subprocess.run(['python', 'scripts/extract_siglip_features.py', '--device', 'cuda:0', '--fps', '5.0', '--batch-size', '128'])\n",
                "\n",
                "def run_whisper_and_ocr():\n",
                "    print('[Process 2] Starting Whisper ASR on GPU 1...')\n",
                "    subprocess.run(['python', 'scripts/extract_whisper_asr.py', '--device', 'cuda:1'])\n",
                "    print('[Process 2] Starting Keyframe OCR on GPU 1...')\n",
                "    subprocess.run(['python', 'scripts/extract_ocr.py', '--device', 'cuda:1'])\n",
                "\n",
                "# Launch both processes concurrently on separate GPUs\n",
                "p1 = multiprocessing.Process(target=run_siglip)\n",
                "p2 = multiprocessing.Process(target=run_whisper_and_ocr)\n",
                "\n",
                "p1.start()\n",
                "p2.start()\n",
                "\n",
                "p1.join()\n",
                "p2.join()\n",
                "print('Dual-GPU extraction complete!')"
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
                "### 🔍 Step 6: Interactive Multi-Modal Retrieval with Dimension-Aligned Text Encoders"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
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
                "\n",
                "def search_query(query_vi: str, top_k: int = 100):\n",
                "    en_query = translator.translate(query_vi)\n",
                "    prompts = translator.generate_prompts(en_query)\n",
                "    q_vec = text_encoder.encode_text(prompts, ensemble=True)\n",
                "    \n",
                "    # 1. FAISS dense candidate retrieval\n",
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
                "    return reranked[:top_k]\n",
                "\n",
                "# Test sample query\n",
                "sample_query = 'Người dẫn chương trình thời sự 60 giây trong trường quay'\n",
                "results = search_query(sample_query, top_k=100)\n",
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
