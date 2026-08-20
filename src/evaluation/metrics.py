import numpy as np
from typing import List, Dict, Tuple, Union, Optional

class AICMetrics:
    """
    Official AIC 2026 Evaluation Metrics:
    - Textual KIS: R-Score = I(v_i == GT_v and fid in [s, e])
    - TRAKE: R-Score = (1/N) * sum_{j=1}^N I(fid_j in [s_j, e_j]) if v_i == GT_v else 0.0
    - Q&A: R-Score = I(v_i == GT_v and fid in [s, e] and ans == GT_ans)
    - Top-k R-Score: R@k = max_{1 <= i <= k} R-Score(r_i) for k in {1, 5, 20, 50, 100}
    - Final Score: 1/5 * sum_{k in {1, 5, 20, 50, 100}} R@k
    """
    CUTOFFS = [1, 5, 20, 50, 100]

    @staticmethod
    def evaluate_kis_query(
        predictions: List[Dict],  # list of {"video_id": ..., "frame_idx": ...}
        ground_truth: Dict        # {"video_id": ..., "frame_start": ..., "frame_end": ...} or {"intervals": [...]}
    ) -> Dict[str, float]:
        gt_vid = ground_truth["video_id"]
        
        # Support single interval or multiple intervals
        if "intervals" in ground_truth:
            intervals = ground_truth["intervals"]
        else:
            intervals = [(ground_truth.get("frame_start", 0), ground_truth.get("frame_end", 0))]

        r_scores = []
        for pred in predictions[:100]:
            p_vid = pred["video_id"]
            p_fid = pred["frame_idx"]
            is_hit = 1.0 if (p_vid == gt_vid and any(s <= p_fid <= e for s, e in intervals)) else 0.0
            r_scores.append(is_hit)

        if not r_scores:
            r_scores = [0.0]

        r_at_k = {}
        for k in AICMetrics.CUTOFFS:
            subset = r_scores[:k]
            r_at_k[f"R@{k}"] = max(subset) if subset else 0.0

        final_score = sum(r_at_k.values()) / len(AICMetrics.CUTOFFS)
        return {
            **r_at_k,
            "Final_Score": final_score
        }

    @staticmethod
    def evaluate_trake_query(
        predictions: List[Dict],      # List of {"video_id": ..., "aligned_frames": [f1, ..., fN]}
        ground_truth: Dict            # {"video_id": ..., "event_intervals": [(s1, e1), ..., (sN, eN)]}
    ) -> Dict[str, float]:
        gt_vid = ground_truth["video_id"]
        intervals = ground_truth.get("event_intervals", [])
        N = len(intervals)
        
        if N == 0:
            return {f"R@{k}": 0.0 for k in AICMetrics.CUTOFFS} | {"Final_Score": 0.0}

        r_scores = []
        for pred in predictions[:100]:
            if pred["video_id"] != gt_vid:
                r_scores.append(0.0)
                continue
            pred_frames = pred.get("aligned_frames", [])
            if len(pred_frames) != N:
                r_scores.append(0.0)
                continue
            matches = sum(1.0 for f, (s, e) in zip(pred_frames, intervals) if s <= f <= e)
            r_scores.append(matches / float(N))

        if not r_scores:
            r_scores = [0.0]

        r_at_k = {}
        for k in AICMetrics.CUTOFFS:
            subset = r_scores[:k]
            r_at_k[f"R@{k}"] = max(subset) if subset else 0.0

        final_score = sum(r_at_k.values()) / len(AICMetrics.CUTOFFS)
        return {**r_at_k, "Final_Score": final_score}

    @staticmethod
    def evaluate_qa_query(
        predictions: List[Dict],      # List of {"video_id": ..., "frame_idx": ..., "answer": ...}
        ground_truth: Dict            # {"video_id": ..., "frame_start": ..., "frame_end": ..., "answers": [...]}
    ) -> Dict[str, float]:
        gt_vid = ground_truth["video_id"]
        gt_s = ground_truth.get("frame_start", 0)
        gt_e = ground_truth.get("frame_end", 0)
        
        raw_ans = ground_truth.get("answers", [ground_truth.get("answer", "")])
        gt_ans = [str(a).strip().lower() for a in raw_ans]

        r_scores = []
        for pred in predictions[:100]:
            p_vid = pred["video_id"]
            p_fid = pred["frame_idx"]
            p_ans = str(pred.get("answer", "")).strip().lower()
            is_hit = 1.0 if (p_vid == gt_vid and gt_s <= p_fid <= gt_e and p_ans in gt_ans) else 0.0
            r_scores.append(is_hit)

        if not r_scores:
            r_scores = [0.0]

        r_at_k = {}
        for k in AICMetrics.CUTOFFS:
            subset = r_scores[:k]
            r_at_k[f"R@{k}"] = max(subset) if subset else 0.0

        final_score = sum(r_at_k.values()) / len(AICMetrics.CUTOFFS)
        return {**r_at_k, "Final_Score": final_score}

    @staticmethod
    def evaluate_benchmark(
        all_predictions: List[List[Dict]],
        all_ground_truths: List[Dict]
    ) -> Dict[str, float]:
        summary = {f"R@{k}": 0.0 for k in AICMetrics.CUTOFFS}
        summary["Final_Score"] = 0.0

        N = len(all_ground_truths)
        if N == 0:
            return summary

        for preds, gt in zip(all_predictions, all_ground_truths):
            if "event_intervals" in gt:
                res = AICMetrics.evaluate_trake_query(preds, gt)
            elif "answer" in gt or "answers" in gt:
                res = AICMetrics.evaluate_qa_query(preds, gt)
            else:
                res = AICMetrics.evaluate_kis_query(preds, gt)
                
            for k, v in res.items():
                summary[k] += v / N

        return summary
