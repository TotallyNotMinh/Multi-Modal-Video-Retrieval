import os
import sys
import json
import time
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def main():
    eval_file = "eval/stage2_paired_eval_set.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    negatives = data["negatives"] # 68 unanswerable
    positives = data["matched_positives"] # 68 matched answerables

    print(f"Loaded {len(negatives)} Negatives and {len(positives)} Matched Positives.")

    model_path = "/home/totallynotminh/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    print(f"\nLoading local verification model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda:0")
    model.eval()
    print("Verification model ready on GPU!\n")

    system_prompt = (
        "Bạn là giám khảo kiểm tra tính xác thực thông tin cho hệ thống AI Video QA.\n"
        "Nhiệm vụ của bạn là đọc kỹ đoạn LỜI THOẠI (ASR) và CHỮ MÀN HÌNH (OCR) được trích xuất từ một đoạn clip video, "
        "sau đó xác định xem đoạn trích dẫn đó CÓ CHỨA ĐỦ THÔNG TIN ĐỂ TRẢ LỜI ĐÚNG câu hỏi của người dùng hay KHÔNG.\n\n"
        "QUY TẮC PHÁN QUYẾT BẮT BUỘC:\n"
        "1. Nếu thông tin/câu trả lời cụ thể cho câu hỏi XUẤT HIỆN RÕ RÀNG trong Lời thoại hoặc OCR -> Trả về: 'PREDICTION: ANSWERABLE'\n"
        "2. Nếu câu hỏi hỏi về chi tiết KHÔNG XUẤT HIỆN trong trích đoạn (ví dụ: hỏi công thức/thời gian/số tiền/kết quả/độ tuổi nhưng trích đoạn chỉ nhắc đến chủ đề chung chung mà KHÔNG có con số/chi tiết đó) -> BẮT BUỘC trả về: 'PREDICTION: UNANSWERABLE'.\n\n"
        "ĐỊNH DẠNG ĐẦU RA:\n"
        "PREDICTION: ANSWERABLE hoặc PREDICTION: UNANSWERABLE\n"
        "EXPLANATION: <giải thích ngắn gọn 1 câu>"
    )

    def verify_sample(item):
        query = item["query"]
        ctx = item.get("context", {})
        asr_segs = ctx.get("asr_segments", [])
        ocr_lines = ctx.get("ocr_lines", [])
        
        asr_text = "\n".join([f"[{s['start_sec']}s - {s['end_sec']}s]: {s['text']}" for s in asr_segs[:15]]) if asr_segs else "(Không có lời thoại ASR)"
        ocr_text = "\n".join([f"[{o['time_sec']}s]: {o['text']}" for o in ocr_lines[:15]]) if ocr_lines else "(Không có chữ OCR)"
        
        user_msg = f"CÂU HỎI CỦA NGƯỜI DÙNG: \"{query}\"\n\n[LỜI THOẠI TRONG CLIP]:\n{asr_text}\n\n[CHỮ OCR TRÊN MÀN HÌNH]:\n{ocr_text}\n\nHãy kiểm tra và đưa ra phán quyết (PREDICTION: ANSWERABLE hoặc PREDICTION: UNANSWERABLE):"
        
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            
        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # Parse prediction
        first_line = resp.split("\n")[0].upper()
        if "PREDICTION:" in first_line:
            is_pred_unans = "UNANSWERABLE" in first_line
        else:
            is_pred_unans = "UNANSWERABLE" in resp[:60].upper()
            
        return is_pred_unans, resp

    print("Evaluating 68 Unanswerable Negatives...")
    neg_results = []
    t0 = time.time()
    for idx, item in enumerate(negatives, 1):
        pred_unans, raw_resp = verify_sample(item)
        correct = (pred_unans == True) # Should be UNANSWERABLE
        neg_results.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "pred_unanswerable": pred_unans,
            "correct": correct,
            "response": raw_resp
        })
        if idx % 15 == 0 or idx == len(negatives):
            cur_tnr = np.mean([r["correct"] for r in neg_results]) * 100
            print(f"  [Negatives {idx:>2}/{len(negatives)}] Current Specificity (TNR): {cur_tnr:.1f}%")

    print("\nEvaluating 68 Matched Positives...")
    pos_results = []
    for idx, item in enumerate(positives, 1):
        pred_unans, raw_resp = verify_sample(item)
        correct = (pred_unans == False) # Positive should be ANSWERABLE
        pos_results.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "pred_unanswerable": pred_unans,
            "correct": correct,
            "response": raw_resp
        })
        if idx % 15 == 0 or idx == len(positives):
            cur_tpr = np.mean([r["correct"] for r in pos_results]) * 100
            print(f"  [Positives {idx:>2}/{len(positives)}] Current Sensitivity (TPR): {cur_tpr:.1f}%")

    total_time = time.time() - t0
    
    # Compute Metrics
    tn = sum(1 for r in neg_results if r["correct"]) # True Negatives
    fp = len(neg_results) - tn                       # False Positives (Hallucinations on unanswerables)
    tp = sum(1 for r in pos_results if r["correct"]) # True Positives
    fn = len(positives) - tp                        # False Negatives (Wrongly rejected answerables)

    tnr = (tn / len(negatives)) * 100.0
    tpr = (tp / len(positives)) * 100.0
    fpr = 100.0 - tnr
    fnr = 100.0 - tpr
    balanced_acc = (tnr + tpr) / 2.0

    tnr_low, tnr_high = wilson_ci(tn, len(negatives))
    tpr_low, tpr_high = wilson_ci(tp, len(positives))

    print("\n" + "="*80)
    print("  STAGE 2 ISOLATED EVALUATION RESULTS (136 MATCHED SAMPLES)")
    print("="*80)
    print(f"Total Evaluation Time: {total_time:.2f}s ({total_time/136*1000:.1f}ms / sample)\n")
    print(f"Confusion Matrix:")
    print(f"  [True Positives (TP)  = {tp:>2}/{len(positives)}]   [False Negatives (FN) = {fn:>2}/{len(positives)}]")
    print(f"  [False Positives (FP) = {fp:>2}/{len(negatives)}]   [True Negatives (TN)  = {tn:>2}/{len(negatives)}]\n")
    print(f"Key Performance Indicators:")
    print(f"  • Specificity (TNR on Unanswerables) : {tnr:6.2f}%  (Wilson 95% CI: [{tnr_low:.1f}%, {tnr_high:.1f}%])")
    print(f"  • Sensitivity (TPR on Matched Pos)   : {tpr:6.2f}%  (Wilson 95% CI: [{tpr_low:.1f}%, {tpr_high:.1f}%])")
    print(f"  • False Positive Rate (Hallucination): {fpr:6.2f}%")
    print(f"  • False Negative Rate (False Rejection): {fnr:6.2f}%")
    print(f"  • Balanced Accuracy (BACC)           : {balanced_acc:6.2f}%\n")

    # Save detailed evaluation artifact
    out_data = {
        "summary": {
            "num_negatives": len(negatives),
            "num_positives": len(positives),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tnr": tnr, "tnr_ci": [tnr_low, tnr_high],
            "tpr": tpr, "tpr_ci": [tpr_low, tpr_high],
            "fpr": fpr, "fnr": fnr,
            "balanced_accuracy": balanced_acc,
            "avg_latency_ms": total_time / 136.0 * 1000.0
        },
        "negatives_details": neg_results,
        "positives_details": pos_results
    }
    with open("eval/stage2_isolated_eval_results.json", "w", encoding="utf-8") as fp:
        json.dump(out_data, fp, indent=2, ensure_ascii=False)
    print("Saved detailed results to eval/stage2_isolated_eval_results.json")

if __name__ == "__main__":
    main()
