# Task 2 Report: Unanswerable Calibration & Threshold Analysis

**Date:** 2026-08-22  
**Dataset:** `eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl` (627 answerable, 68 hard unanswerable queries)  
**Evaluator:** Calibration Suite with Absolute Sigmoid Normalization  

---

## 1. Executive Summary

We evaluated absolute/sigmoidal score calibration across all search modalities (BM25 lexical, E5 semantic dense, BGE cross-encoder reranker, and CLIP visual dense) to assess unanswerable query abstention ($\tau$-thresholding).

### Key Findings:
1. **Mathematical Calibration Verification:**
   - **BM25 Soft Saturation:** $\text{norm\_bm25} = \frac{\text{BM25}}{25.0 + \text{BM25}} \in [0, 1)$
   - **E5 Semantic Cosine:** $\text{norm\_e5} = \text{clip}\left(\frac{\cos - 0.70}{0.20}, 0.0, 1.0\right)$
   - **BGE Reranker Logits:** $\text{norm\_bge} = \frac{1}{1 + e^{-\text{logit}}} \in (0, 1)$
   - **Visual CLIP Confidence:** $c_{\text{conf}} = \text{clip}\left(\frac{c_{\max} - 0.20}{0.13}, 0.15, 1.0\right)$
2. **Empirical Distribution on Frozen Benchmark:**
   - **Answerable Queries ($n=627$):** Top-1 score Mean = **0.7160**, Median = **0.7386**, Max = **0.8207**.
   - **Ground-Truth Correct Hits ($n=180$):** Top-1 score Mean = **0.7396**, Median = **0.7491**.
   - **Unanswerable Queries ($n=68$):** Top-1 score Mean = **0.7191**, Median = **0.7446**.
3. **The "Hard Distractor" Semantic Trap Phenomenon:**
   - Unanswerable queries in the benchmark are **adversarial in-domain questions** (e.g. asking for the exact cooking duration or year of closure of a topic discussed in the video, but where that specific number is never spoken).
   - As a result, candidate retrieval achieves high BM25 ($\mu = 33.16$) and high E5 cosine ($\mu = 0.8679$) on the target video topic.
   - **Architectural Conclusion:** Pre-retrieval embedding $\tau$-thresholding cannot separate topical relevance from question unanswerability. Abstention for fine-grained unanswerable questions belongs in the **Stage 2 VLM / Multimodal QA Verification Layer** ([`src/ui/search_app.py:591-608`](file:///home/totallynotminh/Documents/PyTorch-Learning/src/ui/search_app.py#L591-L608)), where the LLM/VLM reads the extracted transcript/frames and declares *"Information not present in clip"*.

---

## 2. Component Signal Diagnosis (Answerable vs. Unanswerable)

| Signal Modality | Answerable Queries ($n=30$) | Unanswerable Hard Negatives ($n=30$) | Structural Explanation |
| :--- | :---: | :---: | :--- |
| **BM25 Max Score** | $28.59$ (Median $27.43$) | $33.16$ (Median $30.56$) | Queries share keywords with the topic of the video. |
| **E5 Max Cosine** | $0.8741$ (Median $0.8750$) | $0.8679$ (Median $0.8697$) | Dense semantic embeddings capture overall domain topic. |
| **BGE Top Logit** | $0.6820$ (Median $0.8267$) | $0.6051$ (Median $0.6799$) | Cross-encoder matches the rich passage context. |
| **CLIP Max Dot** | $0.3314$ (Median $0.3328$) | $0.3364$ (Median $0.3365$) | Visual scene embeddings match general video setting. |

---

## 3. Threshold ($\tau$) Operating Curve

| Threshold ($\tau$) | Abstention Accuracy | False Positive Rate | Answerable Hit Retention (R@1) | Tradeoff Note |
| :---: | :---: | :---: | :---: | :--- |
| $\tau = 0.20$ | 0.0% | 100.0% | 100.0% | Default unconstrained retrieval |
| $\tau = 0.40$ | 0.0% | 100.0% | 100.0% | Baseline floor |
| $\tau = 0.55$ | 1.5% | 98.5% | 98.9% | Minor tail filtering |
| $\tau = 0.60$ | 5.9% | 94.1% | 98.3% | Initial separation |
| $\tau = 0.65$ | 17.6% | 82.4% | 93.9% | Rejection starts without harming top hits |
| $\tau = 0.70$ | 32.4% | 67.6% | 81.1% | Aggressive filtering begins |
| $\tau = 0.75$ | 61.8% | 38.2% | 48.9% | Significant loss of true answerable hits |

---

## 4. Production Recommendation

1. Set candidate retrieval $\tau = 0.55$ to reject completely out-of-domain / gibberish queries without hurting answerable recall (98.9% retention).
2. Rely on the **VLM Assistant** ([`src/ui/search_app.py`](file:///home/totallynotminh/Documents/PyTorch-Learning/src/ui/search_app.py)) with prompt instruction:
   > *"Nếu câu hỏi của người dùng yêu cầu thông tin chi tiết (ví dụ: thời gian cụ thể, năm, số tiền) mà không xuất hiện trong đoạn clip / lời thoại / OCR, hãy nêu rõ ràng rằng video không đề cập đến chi tiết này."*
