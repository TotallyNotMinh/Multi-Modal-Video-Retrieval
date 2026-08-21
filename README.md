# 🇻🇳 AIC 2026 - Multi-Modal Video Retrieval Studio

An end-to-end, high-throughput multi-modal video retrieval and analysis platform.

The system combines **SigLIP dense visual feature search**, **PhoWhisper ASR speech transcription**, **PaddleOCR on-screen text extraction**, **BM25 inverted lexical indexing**, and an interactive real-time web studio for **KIS (Known-Item Search)**, **Q&A**, and **TRAKE (Temporal Action Event Localization)**.

---

## 📂 Data & Directory Layout

```text
├── data/                                 # Video datasets and official challenge metadata
│   ├── Videos_L21_a/video/*.mp4          # Video batches (L21 to L30)
│   ├── Videos_L22_a/video/*.mp4
│   ├── ...
│   ├── map-keyframes-aic25-b1/           # Keyframe timestamp mapping CSVs (pts_time, fps, frame_idx)
│   ├── media-info-aic25-b1/              # Video duration, resolution, codecs
│   └── objects-aic25-b1/                 # Precomputed object detection annotations
│
├── cache/                                # Precomputed feature matrices and indexed metadata
│   ├── features_matrix.npy               # SigLIP visual embeddings matrix (177k keyframes × 1152-d)
│   ├── faiss_siglip_meta.pkl             # Global keyframe metadata records
│   ├── faiss_siglip.index                # Indexed vector index
│   ├── asr_transcripts/                  # 873 ASR speech transcript JSON files (timestamped)
│   ├── ocr_text/                         # Extracted on-screen OCR text JSON files
│   ├── objects_index.pkl                 # Spatial object index
│   └── thumbnails/                       # Extracted/cached frame previews
│
├── scripts/                              # Processing, extraction & benchmarking scripts
│   ├── extract_siglip_features.py        # Extract frame embeddings using SigLIP-SO400M
│   ├── extract_whisper_asr.py            # Batch audio extraction & PhoWhisper speech transcription
│   ├── evaluate_candidate_models.py      # LLM ASR transcript refinement arena (multi-GPU auto-sharded)
│   ├── run_dual_gpu_refinement.py        # Distributed dual-GPU refinement runner
│   ├── extract_ocr.py                    # On-screen text extraction via OCR
│   ├── convert_transcripts.py            # JSON <-> TXT batch transcript conversion
│   └── share_ngrok.py                    # Public tunnel helper for remote hosting
│
├── src/                                  # Core library modules
│   ├── encoding/                         # SigLIP vision encoder, PhoWhisper ASR
│   ├── index/                            # Frame mapper, metadata indexer
│   ├── query/                            # Query translator, prompt ensembling
│   ├── retrieval/                        # Fusion retrieval, video decoder
│   └── ui/                               # Search web app (HTTP server + Tailwind frontend)
│       └── search_app.py
│
└── kaggle_dual_gpu_refinement.ipynb      # Kaggle multi-GPU refinement notebook
```

---

## ⚡ Precomputed Cache Download

To run the search studio without having to re-extract all keyframes, visual embeddings, OCR, and ASR transcripts (~873 videos):

📥 **Google Drive Cache Folder:**  
🔗 [**Download Cache Directory (Google Drive)**](https://drive.google.com/drive/folders/1Xr6YYo7p13c8gIalcjV8634b1g7mZUbA?usp=sharing)

### Quick Setup:
1. Download the files or `cache.zip` from the link above.
2. Extract the contents directly into the project root `cache/` directory:
   ```bash
   unzip cache.zip -d cache/
   ```

---

## 🚀 Installation & Environment Setup

### 1. Requirements
- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- PyAV, OpenCV, Pillow, Transformers, Accelerate, FAISS

### 2. Environment Setup (Conda / Pip)
```bash
# Clone the repository
git clone https://github.com/TotallyNotMinh/aic2026.git
cd aic2026

# Create and activate environment
conda create -n aic2026 python=3.11 -y
conda activate aic2026

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers accelerate bitsandbytes sentencepiece protobuf \
            av opencv-python pillow numpy tqdm faiss-cpu
```

---

## 🖥️ Launching the Search Studio Web App

Start the multi-modal search engine server:

```bash
python src/ui/search_app.py
```

- **Local URL:** `http://localhost:8080`
- The web app provides:
  - **Instant Visual & Lexical Hybrid Search:** Weighted combination of SigLIP visual embeddings + BM25 Vietnamese speech dialogue & OCR.
  - **Live Subtitle Display:** Automatic temporal alignment showing speech dialogue below each video card.
  - **Interactive Modal Video Player:** Live timestamp tracking, subtitle sync, speed control, and one-click submission copying (`VideoID,FrameIdx`).
  - **Task Modes:**
    - **KIS (Known-Item Search):** Keyframe pinning, relevance feedback, negative refinement.
    - **Q&A Mode:** In-line answer input, character counters, and question package management.
    - **TRAKE Mode:** Temporal event sequence marker with multi-frame segment saving.

### Public Access via Ngrok (Optional)
```bash
python scripts/share_ngrok.py --port 8080
```

---

## 🛠️ Data Processing & Feature Extraction

### 1. Extract Speech Transcripts (ASR)
```bash
python scripts/extract_whisper_asr.py \
    --device cuda:0 \
    --model-size vinai/PhoWhisper-small \
    --batch-size 32
```

### 2. Benchmark Candidate LLMs for Vietnamese ASR Refinement
```bash
# Runs multi-GPU auto-sharding benchmark comparing Sailor2-8B, Qwen2.5-7B, SeaLLMs-v3-7B, Qwen2.5-14B (4-bit)
python scripts/evaluate_candidate_models.py
```

### 3. Extract SigLIP Visual Embeddings
```bash
python scripts/extract_siglip_features.py \
    --model-name google/siglip-so400m-patch14-384 \
    --batch-size 64 \
    --output cache/features_matrix.npy
```

---

## 📝 License & Competition Notes
Developed for the **Ho Chi Minh City AI Challenge (AIC) 2026**.
