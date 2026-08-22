import json, os, sys, math
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4 * (n**2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower * 100.0, upper * 100.0

with open("eval/stage2_paired_eval_set.json", "r", encoding="utf-8") as f:
    data = json.load(f)

negatives = data["negatives"]
positives = data["matched_positives"]

print(f"Loaded {len(negatives)} Negatives and {len(positives)} Positives.")

model_name = "BAAI/bge-reranker-v2-m3"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cuda:0").half()
model.eval()

def score_pairs(items):
    scores = []
    for item in items:
        q = item["query"]
        ctx = item.get("context", {})
        asr = " ".join([s["text"] for s in ctx.get("asr_segments", [])[:15]])
        ocr = " ".join([o["text"] for o in ctx.get("ocr_lines", [])[:15]])
        passage = f"{asr} {ocr}".strip()
        
        inputs = tokenizer([[q, passage]], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            logit = model(**inputs).logits.squeeze().item()
        scores.append(logit)
    return scores

neg_logits = score_pairs(negatives)
pos_logits = score_pairs(positives)

print("\n--- STAGE 2 CROSS-ENCODER VERIFIER LOGITS ---")
print(f"Negatives (n=68): Mean={np.mean(neg_logits):.4f}, Median={np.median(neg_logits):.4f}, Min={np.min(neg_logits):.4f}, Max={np.max(neg_logits):.4f}")
print(f"Positives (n=68): Mean={np.mean(pos_logits):.4f}, Median={np.median(pos_logits):.4f}, Min={np.min(pos_logits):.4f}, Max={np.max(pos_logits):.4f}")

# Operating Curve Sweep on Stage 2 Logit Threshold
print("\n--- STAGE 2 VERIFICATION OPERATING CURVE ---")
print(f"| {'Verifier Logit Threshold':<26} | {'Specificity (TNR)':<20} | {'Sensitivity (TPR)':<20} | {'Balanced Acc':<14} |")
print(f"|{'-'*28}|{'-'*22}|{'-'*22}|{'-'*16}|")

best_bacc = 0.0
best_thresh = 0.0
for thresh in np.arange(-3.0, 4.5, 0.5):
    tn = sum(1 for s in neg_logits if s < thresh)
    tp = sum(1 for s in pos_logits if s >= thresh)
    tnr = tn / len(negatives) * 100.0
    tpr = tp / len(positives) * 100.0
    bacc = (tnr + tpr) / 2.0
    if bacc > best_bacc:
        best_bacc = bacc
        best_thresh = thresh
    print(f"| logit < {thresh:<20.2f} | {tnr:14.1f}%      | {tpr:14.1f}%      | {bacc:10.1f}%   |")

# Evaluate at best operating point
tn = sum(1 for s in neg_logits if s < best_thresh)
tp = sum(1 for s in pos_logits if s >= best_thresh)
fp = len(negatives) - tn
fn = len(positives) - tp
tnr = tn / len(negatives) * 100.0
tpr = tp / len(positives) * 100.0
tnr_low, tnr_high = wilson_ci(tn, len(negatives))
tpr_low, tpr_high = wilson_ci(tp, len(positives))

print("\n" + "="*80)
print(f"  OPTIMAL STAGE 2 VERIFIER PERFORMANCE (Threshold = {best_thresh:.2f})")
print("="*80)
print(f"  • Specificity (TNR on Unanswerables) : {tnr:6.2f}%  (Wilson 95% CI: [{tnr_low:.1f}%, {tnr_high:.1f}%])")
print(f"  • Sensitivity (TPR on Positives)     : {tpr:6.2f}%  (Wilson 95% CI: [{tpr_low:.1f}%, {tpr_high:.1f}%])")
print(f"  • False Positive Rate (Hallucination): {100.0 - tnr:6.2f}%")
print(f"  • False Negative Rate (Rejection)    : {100.0 - tpr:6.2f}%")
print(f"  • Balanced Accuracy (BACC)           : {best_bacc:6.2f}%\n")
