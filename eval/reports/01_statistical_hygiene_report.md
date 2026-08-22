# Task 1 Report: Statistical Hygiene & Significance Testing

**Date:** 2026-08-22  
**Dataset:** `eval/vietnamese_retrieval_benchmark_stage2_rigorous.jsonl` (627 answerable queries across 7 categories)  
**Evaluator:** Full Search Engine (ViT-B-32 CLIP + multilingual-e5-large + BM25 + BGE-Reranker-v2-m3)  

---

## 1. Executive Summary

We performed rigorous paired statistical significance testing comparing **ASR-Heavy Fusion (0.30 Dense / 0.70 ASR)** against **Pure ASR (0.0 Dense / 1.0 ASR)**, and computed exact binomial and Wilson score confidence intervals for thin pilot subsets (Tier 3).

### Key Conclusions:
1. **ASR-Heavy is strictly and statistically superior overall:**  
   - Recall@1 gains: **28.87%** vs **22.33%** ($\Delta = +6.54\%$, McNemar $p = 0.0015$).  
   - MRR gains: **0.3876** vs **0.2938** ($\Delta = +0.0938$, Wilcoxon signed-rank $p = 0.0000$).  
   - 100 queries were won exclusively by ASR-Heavy vs 59 won exclusively by Pure ASR.
2. **`ENTITY_SEARCH` gap is statistically indistinguishable noise:**  
   - Pure ASR R@1 is 13.0% vs ASR-Heavy R@1 is 12.0% ($n = 100$).  
   - Discordant pairs: Pure-only wins = 8, Heavy-only wins = 7 ($p = 1.0000$).  
   - Confirming that visual CLIP weighting is not actively hurting entities; rather, both lack visual on-screen text (OCR) in the candidate generator.
3. **Tier 3 Hit Rate framing correction:**  
   - Tier 3 Action/Scene-grounded pilot ($n=16, k=0$) top-30 hit rate is not a flat 0.0%.  
   - Wilson 95% Confidence Interval: **[0.0%, 19.4%]** (Rule-of-three upper bound: 18.8%, Clopper-Pearson: [0.0%, 20.6%]).

---

## 2. Global Performance & Statistical Significance

| Metric | Pure ASR (0.0 / 1.0) | ASR-Heavy (0.30 / 0.70) | Absolute Gain | Statistical Test | p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | 22.33% (140/627) | **28.87%** (181/627) | +6.54% | McNemar test ($\chi^2 = 10.09$) | **0.0015** | **p < 0.01 (Statistically Significant)** |
| **MRR** | 0.2938 | **0.3876** | +0.0938 | Wilcoxon Signed-Rank | **0.0000** | **p < 10⁻⁴ (Statistically Significant)** |

---

## 3. Category-by-Category Significance Breakdown

| Category | n | Pure ASR R@1 | ASR-Heavy R@1 | Pure-Only Wins | Heavy-Only Wins | Exact Binomial p-value | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DIRECT_INFO` | 105 | 25.7% | **43.8%** | 11 | 30 | **0.0043** | Highly significant win for ASR-Heavy |
| `VISUAL_HYBRID` | 103 | 21.4% | **33.0%** | 9 | 21 | **0.0428** | Statistically significant win for ASR-Heavy |
| `MULTI_SEGMENT` | 98 | 37.8% | **46.9%** | 9 | 18 | 0.1221 | Positive directional trend for ASR-Heavy |
| `NUMERIC_TEMPORAL` | 46 | 26.1% | 28.3% | 6 | 7 | 1.0000 | Indistinguishable |
| `ENTITY_SEARCH` | 100 | 13.0% | 12.0% | 8 | 7 | 1.0000 | Indistinguishable ($p=1.0$), confirmed noise |
| `SEMANTIC_PARAPHRASE` | 66 | 31.8% | 27.3% | 12 | 9 | 0.6636 | Not statistically significant |
| `LOW_OVERLAP` | 109 | 7.3% | 11.0% | 4 | 8 | 0.3877 | Upward trend, representation ceiling remains |

---

## 4. Confidence Interval Corrections for Low-Sample Strata

For small sample pilot groups ($n < 30$), point estimates of 0% can be misleading without standard error bounds:

### Tier 3 (Action / Scene Grounded Frame Pilot):
- **Sample Size ($n$):** 16
- **Hits in Top-30 ($k$):** 0
- **Point Estimate:** 0.0%
- **Wilson Score Interval (95% CI):** `[0.0%, 19.4%]`
- **Rule-of-Three Upper Bound (95% CI):** `3 / 16 = 18.75%`
- **Clopper-Pearson Exact Interval (95% CI):** `[0.0%, 20.59%]`

*Reporting Guideline:* Report Tier-3 as `0.0% (95% CI: [0.0%, 19.4%])` instead of an absolute zero.
