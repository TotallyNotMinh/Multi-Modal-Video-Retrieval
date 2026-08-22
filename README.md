# Multi-Modal Video Retrieval Studio

An end-to-end, high-throughput multi-modal video retrieval and analysis platform.

The system combines **SigLIP dense visual feature search**, **LLM-refined PhoWhisper ASR speech transcription**, **intfloat/multilingual-e5-large dense semantic text search**, **PaddleOCR on-screen text extraction**, **BM25 inverted lexical indexing**, and an interactive real-time web studio for **KIS (Known-Item Search)**, **Q&A**, and **TRAKE (Temporal Action Event Localization)**.

---

## 🌟 Core Features & Architecture

```mermaid
flowchart TD
    %% Styling definitions
    classDef offline fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef online fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef stage2 fill:#18181b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef data fill:#334155,stroke:#94a3b8,stroke-width:1px,color:#f1f5f9;

    subgraph PHASE1 ["<b>PHASE 1: MULTI-MODAL EXTRACTION & INDEXING</b>"]
        direction TB
        V["<b>Raw Video Corpus (.mp4)</b><br>873 Videos / 120+ Hours"]:::data

        V -->|I-Frame Extraction| KF["<b>Keyframe Extraction</b><br>177,321 Keyframes"]:::data
        V -->|Silero VAD + PhoWhisper| ASR["<b>Speech ASR</b><br>16,660 Audio Segments"]:::data
        V -->|EasyOCR / PaddleOCR| OCR["<b>On-Screen OCR Banners</b><br>18,451 Text Detections"]:::data

        KF -->|SigLIP SO400M / CLIP| SIG_IDX["<b>Visual Feature Matrix</b><br><code>cache/features_matrix.npy</code> (177k × 512)"]:::offline
        ASR -->|LLM Context Correction| REF_ASR["<b>Refined Transcripts</b><br>Diacritics & Proper Nouns Fixed"]:::offline
        
        REF_ASR -->|E5-Large FP16| E5_IDX["<b>Dense Semantic Index</b><br>FAISS IndexFlatIP (16.6k × 1024)"]:::offline
        REF_ASR & OCR -->|Robertson-Spärck Jones IDF| BM25_IDX["<b>Inverted BM25 Index</b><br>35,111 Speech & OCR Documents"]:::offline
    end

    subgraph PHASE2 ["<b>PHASE 2: STAGE 1 HYBRID CANDIDATE RETRIEVAL (80–120ms)</b>"]
        direction TB
        Q["<b>User Natural Language Query (VI / EN)</b>"]:::data
        
        Q -->|Query Translator & Prompt Ensemble| Q_VIS["<b>Visual Concept Vector</b>"]:::online
        Q -->|Passage Formatter| Q_E5["<b>E5 Query Vector</b>"]:::online
        Q -->|Lexical Tokenizer| Q_BM25["<b>BM25 Query Tokens</b>"]:::online

        Q_VIS -->|Cosine Dot Product| SCORE_VIS["<b>Visual Dense Scores</b>"]:::online
        Q_E5 -->|FAISS Semantic Search| SCORE_E5["<b>Speech Semantic Scores</b>"]:::online
        Q_BM25 -->|BM25 Inverted Search| SCORE_BM25["<b>Speech + OCR Lexical Scores</b>"]:::online

        SCORE_VIS & SCORE_E5 & SCORE_BM25 --> FUSE["<b>Calibrated Tri-Modal Fusion & Temporal Smoothing</b><br>• Sigmoidal Soft-Saturation Calibration<br>• ±3.0s Temporal Window Aggregation<br>• Temporal NMS Shot Deduplication"]:::online
        
        FUSE --> TOP_CANDS["<b>Top 50 Ranked Video Clips & Timestamps</b>"]:::online
    end

    subgraph PHASE3 ["<b>PHASE 3: STAGE 2 MULTI-MODAL VERIFICATION & WEB STUDIO</b>"]
        direction TB
        TOP_CANDS --> CLIP_PICK["<b>Selected Candidate Video Clip</b><br>[start_sec, end_sec]"]:::stage2
        
        CLIP_PICK --> VLM["<b>Stage 2 Neural Cross-Encoder & VLM Assistant</b><br>• BGE-Reranker-v2-m3 Evidence Scoring (6.0ms)<br>• 6× 512px Clip Frames + ASR Window + OCR Context<br>• Adversarial Unanswerable Query Detection (TNR 69.1%, TPR 70.6%)"]:::stage2
        
        VLM --> OUT_ANS["<b>Verified Answer & Timestamp Grounding</b><br>or <b>'Information Not Present' Abstention</b>"]:::stage2
        OUT_ANS --> STUDIO["<b>Interactive Multi-Modal Retrieval Studio</b><br><code>http://localhost:8080</code> (Timeline Viewer, Keyframe Flooding, QA)"]:::stage2
    end

    PHASE1 --> PHASE2 --> PHASE3
```

### 1. 🖼️ Dense Visual Search (SigLIP)
- **Model:** `google/siglip-so400m-patch14-384` / OpenCLIP.
- **Index:** Matrix dot product over **177,321 keyframe vectors** with cosine normalization.
- **Prompt Ensembling & Query Translation:** Automatic Vietnamese $\leftrightarrow$ English query expansion and ensemble prompting for visual concepts.

### 2. 🗣️ Conservative LLM Transcript Refinement
- **100% Corpus Refinement:** All 873 video transcripts (16,660 segments) processed.
- **Segment Tagging (`<SEGMENT_i>`):** Strictly preserves temporal boundaries and video start/end timestamps.
- **Error Correction:** Fixes missing Vietnamese diacritics, broken words, phonetic homophones, and misheard proper nouns without hallucination or stylistic paraphrasing.

### 3. 🧠 Dense Semantic Speech Indexing (Multilingual-E5)
- **Model:** `intfloat/multilingual-e5-large` (1024-dim, FP16 GPU accelerated).
- **Index:** `FAISS IndexFlatIP` over all 16,660 refined transcript passages.
- **Conceptual Matching:** Discovers semantic intent even with zero keyword overlap (e.g., *"miền Tây"* $\leftrightarrow$ *"Đồng bằng sông Cửu Long / Tây Nam Bộ"*).

### 4. 🔤 Exact Lexical & OCR Search (BM25)
- **Inverted Index:** Fast inverted index with Robertson-Spärck Jones IDF over refined transcripts and on-screen OCR text.
- **Named Entity Precision:** Guarantees exact matches for proper nouns, acronyms, license plates, locations, and numbers.

### 5. 🎯 Tri-Modal Temporal Fusion & Smoothing
- **Temporal Alignment:** Maps segment-level transcript scores to video keyframe timelines with $\pm 3\text{s}$ window aggregation.
- **1D Gaussian Temporal Smoothing:** Smooths scores across continuous scene shots.
- **Shot Deduplication / NMS:** Deduplicates adjacent keyframes within $\pm 2\text{s}$ in the same video to maximize Recall@K diversity.

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
│   ├── features_matrix.npy               # SigLIP visual embeddings matrix (177k keyframes)
│   ├── faiss_siglip_meta.pkl             # Global keyframe metadata records
│   ├── faiss_siglip.index                # Indexed visual vector index
│   ├── asr_transcripts/                  # 873 raw ASR speech transcript JSON files
│   ├── asr_transcripts_refined/          # 873 LLM-refined ASR transcript JSON files
│   ├── transcript_embeddings.npy         # Dense E5-large transcript embeddings (16,660 x 1024)
│   ├── transcript_semantic.index         # FAISS dense vector index for transcripts
│   ├── transcript_semantic_meta.pkl      # Transcript segment metadata ledger
│   ├── ocr_text/                         # Extracted on-screen OCR text JSON files
│   └── thumbnails/                       # Extracted/cached frame previews
│
├── scripts/                              # Processing, extraction & indexing scripts
│   ├── build_transcript_index.py        # Build dense FAISS index with E5-Large over refined transcripts
│   ├── extract_siglip_features.py        # Extract frame embeddings using SigLIP-SO400M
│   ├── extract_whisper_asr.py            # Batch audio extraction & PhoWhisper speech transcription
│   ├── refine_transcripts_qwen.py        # Local Qwen2.5/Qwen3 LLM transcript refinement
│   ├── run_dual_gpu_refinement.py        # Distributed dual-GPU parallel refinement runner
│   ├── refine_transcripts_mimo.py        # API-based transcript refinement pipeline
│   ├── evaluate_candidate_models.py      # LLM ASR transcript refinement arena
│   ├── extract_ocr.py                    # On-screen text extraction via OCR
│   └── share_ngrok.py                    # Public tunnel helper for remote hosting
│
├── src/                                  # Core library modules
│   ├── encoding/                         # SigLIP vision encoder, E5 transcript encoder
│   ├── index/                            # Frame mapper, semantic indexer, metadata indexer
│   ├── query/                            # Query translator, prompt ensembling
│   ├── retrieval/                        # Tri-modal fusion, hybrid transcript engine, video decoder
│   └── ui/                               # Search web studio (HTTP server + Tailwind frontend)
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
- **Search Capabilities:**
  - **Tri-Modal Fusion:** Combined SigLIP visual embeddings + BM25 lexical search + E5 dense semantic matching on refined transcripts.
  - **Live Subtitle Display:** Automatic temporal alignment displaying speech dialogue below each video card.
  - **Interactive Modal Video Player:** Live timestamp tracking, subtitle sync, speed control, and one-click submission copying (`VideoID,FrameIdx`).
  - **Task Modes:**
    - **KIS (Known-Item Search):** Keyframe pinning, relevance feedback, negative refinement, neighbor frame flooding.
    - **Q&A Mode:** In-line answer input, character counters, and question package management.
    - **TRAKE Mode:** Temporal event sequence marker with multi-frame segment saving.

### Public Access via Ngrok (Optional)
```bash
python scripts/share_ngrok.py --port 8080
```

---

## 🛠️ Offline Indexing & Processing Commands

### 1. Build Dense Semantic Transcript Index (E5-Large)
```bash
python scripts/build_transcript_index.py \
    --refined-dir cache/asr_transcripts_refined \
    --raw-dir cache/asr_transcripts \
    --model intfloat/multilingual-e5-large \
    --batch-size 64 \
    --device cuda
```

### 2. Run Multi-GPU ASR Transcript Refinement
```bash
# Dual-GPU Parallel Runner (Kaggle or local 2x GPU node)
python scripts/run_dual_gpu_refinement.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --batch-size 12 \
    --num-gpus 2
```

### 3. Extract Speech Transcripts (PhoWhisper)
```bash
python scripts/extract_whisper_asr.py \
    --device cuda:0 \
    --model-size vinai/PhoWhisper-small \
    --batch-size 32
```

### 4. Extract SigLIP Visual Embeddings
```bash
python scripts/extract_siglip_features.py \
    --model-name google/siglip-so400m-patch14-384 \
    --batch-size 64 \
    --output cache/features_matrix.npy
```

---

## 📝 License & Competition Notes
Developed for the **Ho Chi Minh City AI Challenge (AIC) 2026**.
