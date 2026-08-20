import os
import zipfile
from typing import List, Dict, Tuple, Optional

class SubmissionGenerator:
    """
    AIC 2026 Official Submission Formatter.
    Generates exactly 100-slot ranked submission files for Textual KIS, Q&A, and TRAKE queries.
    """
    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _pad_predictions(self, preds: List[Dict], default_item: Dict, target_len: int = 100) -> List[Dict]:
        """
        Guarantees exactly target_len items by padding with fallback frames if necessary.
        """
        if not preds:
            return [default_item] * target_len
        if len(preds) >= target_len:
            return preds[:target_len]
        last = preds[-1]
        return preds + [last] * (target_len - len(preds))

    def format_kis_submission(
        self,
        query_id: str,
        ranked_predictions: List[Dict]  # List of {"video_id": ..., "frame_idx": ...}
    ) -> List[str]:
        """
        Formats exactly 100 rows of <video_id>,<frame_idx>
        """
        default_pred = {"video_id": "L21_V001", "frame_idx": 0}
        padded = self._pad_predictions(ranked_predictions, default_pred, target_len=100)
        lines = []
        for pred in padded:
            vid = pred["video_id"]
            fid = pred["frame_idx"]
            lines.append(f"{vid},{fid}")
        return lines

    def format_qa_submission(
        self,
        query_id: str,
        ranked_predictions: List[Dict]  # List of {"video_id": ..., "frame_idx": ..., "answer": ...}
    ) -> List[str]:
        """
        Formats exactly 100 rows of <video_id>,<frame_idx>,<answer>
        """
        default_pred = {"video_id": "L21_V001", "frame_idx": 0, "answer": ""}
        padded = self._pad_predictions(ranked_predictions, default_pred, target_len=100)
        lines = []
        for pred in padded:
            vid = pred["video_id"]
            fid = pred["frame_idx"]
            ans_clean = str(pred.get("answer", "")).replace("\n", " ").strip()
            if "," in ans_clean and not (ans_clean.startswith('"') and ans_clean.endswith('"')):
                ans_clean = f'"{ans_clean}"'
            lines.append(f"{vid},{fid},{ans_clean}")
        return lines

    def format_trake_submission(
        self,
        query_id: str,
        aligned_predictions: List[Dict]  # List of {"video_id": ..., "aligned_frames": [f1, f2, ..., fN]}
    ) -> List[str]:
        """
        Formats exactly 100 rows of <video_id>,<frame_1>,...,<frame_N>
        """
        default_pred = {"video_id": "L21_V001", "aligned_frames": [0, 30, 60, 90]}
        padded = self._pad_predictions(aligned_predictions, default_pred, target_len=100)
        lines = []
        for pred in padded:
            vid = pred["video_id"]
            frames_str = ",".join(str(f) for f in pred["aligned_frames"])
            lines.append(f"{vid},{frames_str}")
        return lines

    def save_submission_file(self, query_id: str, lines: List[str]) -> str:
        filename = f"{query_id}.csv"
        sub_path = os.path.join(self.output_dir, filename)
        with open(sub_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{line}\n")
        return sub_path

    def package_submission_zip(
        self,
        zip_filename: str = "submission_aic2026.zip",
        query_ids: Optional[List[str]] = None
    ) -> str:
        """
        Compresses generated query CSVs into the official competition ZIP bundle.
        """
        zip_path = os.path.join(self.output_dir, zip_filename)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if query_ids:
                for qid in query_ids:
                    csv_f = f"{qid}.csv"
                    csv_p = os.path.join(self.output_dir, csv_f)
                    if os.path.exists(csv_p):
                        zf.write(csv_p, arcname=csv_f)
            else:
                for root, _, files in os.walk(self.output_dir):
                    for f in sorted(files):
                        if f.endswith(".csv") and not f.startswith("test_"):
                            abs_p = os.path.join(root, f)
                            zf.write(abs_p, arcname=f)

        print(f"[SubmissionGenerator] Created submission package: {zip_path}")
        return zip_path
