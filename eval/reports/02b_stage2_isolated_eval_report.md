# Isolated Stage 2 Evaluation Report: Specificity (TNR) & Sensitivity (TPR)

**Date:** 2026-08-23  
**Evaluation Set:** 68 Hard In-Domain Unanswerables vs. 68 Matched True Positives (in the same $[0.70, 0.78]$ Stage 1 score band)  
**Evaluator:** Stage 2 Cross-Encoder Context Verification (`BAAI/bge-reranker-v2-m3` on full ASR+OCR passage window)  

---

## 1. Executive Summary

In Stage 1 (bi-encoder + BM25 retrieval), in-domain unanswerable queries were completely indistinguishable from true hits ($\mu_{\text{neg}} = 0.7191$ vs $\mu_{\text{pos}} = 0.7207$, $\Delta = +0.0016$).

When evaluated under **Stage 2 Evidence-Grounded Cross-Encoder Verification** (joint all-to-all attention between query tokens and evidence transcript/OCR passages):
1. **Strong Logit Separation:**
   - **Hard Unanswerable Negatives ($n=68$):** Mean Logit = **-2.1000** (Median = **-1.5137**)
   - **Matched Answerable Positives ($n=68$):** Mean Logit = **+1.2987** (Median = **+2.1377**)
   - **Logit Shift ($\Delta$):** **+3.3987** separation between answerable and unanswerable distributions.
2. **Balanced Performance:**
   - **Specificity (TNR on Unanswerables):** **69.12%** (Wilson 95% CI: `[57.4%, 78.8%]`) — 47 / 68 hard adversarial distractors correctly rejected.
   - **Sensitivity (TPR on Matched Positives):** **70.59%** (Wilson 95% CI: `[58.9%, 80.1%]`) — 48 / 68 true positive answers retained.
   - **Balanced Accuracy:** **69.85%**
3. **Execution Latency:** **6.0 ms per verification call** on GPU (0.82s total for 136 samples), well within the 1.32s latency budget.

---

## 2. Confusion Matrix & Key Metrics

| Metric | Point Estimate | Wilson 95% Confidence Interval |
| :--- | :---: | :---: |
| **True Positive Rate (TPR / Sensitivity)** | **70.59%** (48 / 68) | `[58.9%, 80.1%]` |
| **True Negative Rate (TNR / Specificity)** | **69.12%** (47 / 68) | `[57.4%, 78.8%]` |
| **False Positive Rate (FPR / Hallucination)** | **30.88%** (21 / 68) | `[21.2%, 42.6%]` |
| **False Negative Rate (FNR / Rejection)** | **29.41%** (20 / 68) | `[19.9%, 41.1%]` |
| **Balanced Accuracy (BACC)** | **69.85%** | — |

---

## 3. Stage 2 Operating Curve Sweep

| Verifier Logit Threshold ($\theta$) | Specificity (TNR on Negatives) | Sensitivity (TPR on Positives) | Balanced Accuracy | Operational Role |
| :---: | :---: | :---: | :---: | :--- |
| $\theta < -2.00$ | 45.6% (31 / 68) | **88.2% (60 / 68)** | 66.9% | High-Recall Mode (Retain almost all true answers) |
| $\theta < -1.00$ | 54.4% (37 / 68) | **79.4% (54 / 68)** | 66.9% | Moderate Filtering |
| $\mathbf{\theta < 0.50}$ | **69.1% (47 / 68)** | **70.6% (48 / 68)** | **69.9%** | **Optimal Balanced Operating Point** |
| $\theta < 1.50$ | **79.4% (54 / 68)** | 57.4% (39 / 68) | 68.4% | Strict Precision Mode |
| $\theta < 2.00$ | **82.4% (56 / 68)** | 51.5% (35 / 68) | 66.9% | High-Confidence Abstention |
| $\theta < 3.00$ | **91.2% (62 / 68)** | 27.9% (19 / 68) | 59.6% | Rejection Floor |
