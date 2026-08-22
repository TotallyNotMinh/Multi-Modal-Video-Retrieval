# Task 3 Report: OCR into Candidate Generation

**Date:** 2026-08-22  
**Dataset:** `eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl` (627 answerable queries)  
**Evaluator:** Complete 5-Configuration Ablation Suite with Pre-extracted EasyOCR Ingestion  

---

## 1. Executive Summary

We integrated **18,451 merged on-screen OCR text banner segments** (pre-extracted via EasyOCR across all 873 local MP4 videos) directly into `BM25Engine` for initial candidate generation in `SearchEngine`.

### Key Outcomes:
1. **Zero Runtime Overhead:** Ingesting and building the combined ASR + OCR inverted index (35,111 total text documents) takes only **0.26 seconds** at engine initialization.
2. **Pure Text/OCR Jump:** Moving OCR from display-only to candidate generation lifted Pure Speech/OCR performance across the board:
   - Overall Recall@1: **22.3% $\rightarrow$ 24.2%** ($\Delta = +1.9\%$)
   - Overall MRR: **0.2938 $\rightarrow$ 0.3124** ($\Delta = +0.0186$)
3. **Category Breakthroughs with OCR:**
   - `VISUAL_HYBRID` ($n=103$): Pure Speech/OCR R@1 jumped from **21.4% to 32.0%** ($+10.6\%$).
   - `LOW_OVERLAP` ($n=109$): Pure Speech/OCR R@1 broke through the 7.3% plateau to **10.1%** (MRR 0.1294).
   - `ENTITY_SEARCH` ($n=100$): Pure Speech/OCR R@1 moved to **14.0%**, while Pure Visual Dense remained at **0.0% R@1**, proving that visual embeddings alone cannot resolve text entities without OCR.
4. **Dominant Winner: ASR-Heavy Hybrid (0.30 Dense / 0.70 ASR+OCR):**
   - **Recall@1:** **28.7%**  
   - **Recall@5:** **50.2%**  
   - **Recall@10:** **58.9%**  
   - **MRR:** **0.3857**

---

## 2. Overall Performance Comparison (627 Answerable Queries)

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Pure Visual Dense (1.0 / 0.0)** | 3.3% | 8.1% | 10.0% | 0.0579 | Blind to names, numbers, titles, and speech. |
| **Dense-Heavy Hybrid (0.85 / 0.15)** | 11.2% | 25.2% | 32.2% | 0.1815 | Visual features dilute text semantics when over-weighted. |
| **Hybrid Baseline (0.70 / 0.30)** | 18.3% | 38.4% | 47.4% | 0.2774 | Strong top-10 recall, but suppresses exact transcript matches. |
| **Pure Speech/OCR (0.0 / 1.0)** | 24.2% | 37.5% | 46.7% | 0.3124 | Solid lexical & semantic matching on speech + on-screen banners. |
| **ASR-Heavy Hybrid (0.30 / 0.70)** | **28.7%** | **50.2%** | **58.9%** | **0.3857** | **Best overall configuration across every metric.** |

---

## 3. Category-by-Category Breakdown

| Category | n | Pure Dense (Visual) | Pure Speech / OCR | Hybrid Baseline (0.70 / 0.30) | ASR-Heavy Hybrid (0.30 / 0.70) | Dense-Heavy (0.85 / 0.15) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `DIRECT_INFO` | 105 | 3.8% (0.0629) | 25.7% (0.3752) | 28.6% (0.4179) | **43.8% (0.5535)** | 18.1% (0.2678) |
| `MULTI_SEGMENT` | 98 | 3.1% (0.0667) | 38.8% (0.4561) | 32.7% (0.4418) | **46.9% (0.5552)** | 16.3% (0.2766) |
| `VISUAL_HYBRID` | 103 | 5.8% (0.0836) | 32.0% (0.3719) | 14.6% (0.2560) | **32.0% (0.4377)** | 9.7% (0.1762) |
| `NUMERIC_TEMPORAL` | 46 | 2.2% (0.0526) | 19.6% (0.3177) | 19.6% (0.3375) | **28.3% (0.4601)** | 10.9% (0.2311) |
| `SEMANTIC_PARAPHRASE`| 66 | 7.6% (0.1033) | 30.3% (0.3819) | 18.2% (0.2844) | **27.3% (0.4001)** | 15.2% (0.2110) |
| `ENTITY_SEARCH` | 100 | 0.0% (0.0152) | **14.0% (0.1959)**| 8.0% (0.1464) | 12.0% (0.2056) | 3.0% (0.0690) |
| `LOW_OVERLAP` | 109 | 1.8% (0.0352) | 10.1% (0.1294) | 8.3% (0.1051) | **11.0% (0.1480)** | 6.4% (0.0820) |

*(Values in parentheses indicate MRR)*
