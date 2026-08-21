import os
import io
import csv
import zipfile
from typing import List, Dict, Tuple, Optional

class SubmissionGenerator:
    """
    AIC 2026 Official Submission Formatter and Validator.
    Generates and packages valid submission files for Textual KIS, Q&A, and TRAKE queries
    strictly conforming to rules.txt.
    """
    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _clean_video_id(video_id: str) -> str:
        """
        Removes file extensions (.mp4, .avi, etc.) and paths from video names.
        Example: 'path/to/L01_V028.mp4' -> 'L01_V028'
        """
        base = os.path.basename(str(video_id).strip())
        name, _ = os.path.splitext(base)
        return name

    @staticmethod
    def _escape_csv_field(val: str, max_len: int = 100) -> str:
        """
        Formats and escapes text according to RFC-4180 CSV specifications.
        Enforces maximum character length (rules.txt limit: 100 chars).
        """
        cleaned = str(val).replace("\r", " ").replace("\n", " ").strip()[:max_len]
        if "," in cleaned or '"' in cleaned:
            escaped = cleaned.replace('"', '""')
            return f'"{escaped}"'
        return cleaned

    def _pad_predictions(self, preds: List[Dict], default_item: Dict, target_len: int = 100) -> List[Dict]:
        """
        Guarantees up to target_len items (max 100 lines) padding with fallback frames if needed.
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
        Formats at most 100 rows of: <video_name>,<frame_idx>
        """
        default_pred = {"video_id": "L21_V001", "frame_idx": 0}
        padded = self._pad_predictions(ranked_predictions, default_pred, target_len=100)
        lines = []
        for pred in padded:
            vid = self._clean_video_id(pred["video_id"])
            fid = int(round(float(pred["frame_idx"])))
            lines.append(f"{vid},{fid}")
        return lines

    def format_qa_submission(
        self,
        query_id: str,
        ranked_predictions: List[Dict]  # List of {"video_id": ..., "frame_idx": ..., "answer": ...}
    ) -> List[str]:
        """
        Formats at most 100 rows of: <video_name>,<frame_idx>,<answer>
        """
        default_pred = {"video_id": "L21_V001", "frame_idx": 0, "answer": ""}
        padded = self._pad_predictions(ranked_predictions, default_pred, target_len=100)
        lines = []
        for pred in padded:
            vid = self._clean_video_id(pred["video_id"])
            fid = int(round(float(pred["frame_idx"])))
            ans = self._escape_csv_field(pred.get("answer", ""), max_len=100)
            lines.append(f"{vid},{fid},{ans}")
        return lines

    def format_trake_submission(
        self,
        query_id: str,
        aligned_predictions: List[Dict],  # List of {"video_id": ..., "aligned_frames": [f1, f2, ..., fN]}
        num_events: Optional[int] = None
    ) -> List[str]:
        """
        Formats at most 100 rows of: <video_name>,<frame_1>,...,<frame_N>
        """
        if aligned_predictions and "aligned_frames" in aligned_predictions[0]:
            N = len(aligned_predictions[0]["aligned_frames"])
        elif num_events is not None:
            N = num_events
        else:
            N = 4

        fallback_frames = [i * 30 for i in range(N)]
        default_pred = {"video_id": "L21_V001", "aligned_frames": fallback_frames}
        padded = self._pad_predictions(aligned_predictions, default_pred, target_len=100)
        lines = []
        for pred in padded:
            vid = self._clean_video_id(pred["video_id"])
            frames = [int(round(float(f))) for f in pred.get("aligned_frames", fallback_frames)]
            frames_str = ",".join(str(f) for f in frames)
            lines.append(f"{vid},{frames_str}")
        return lines

    def save_submission_file(self, query_id: str, lines: List[str]) -> str:
        filename = f"{query_id}.csv"
        sub_path = os.path.join(self.output_dir, filename)
        with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
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
        Enforces rules.txt requirement: all CSVs MUST reside under the 'submission/' root folder.
        """
        zip_path = os.path.join(self.output_dir, zip_filename)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if query_ids:
                for qid in query_ids:
                    csv_f = f"{qid}.csv" if not qid.endswith(".csv") else qid
                    csv_p = os.path.join(self.output_dir, csv_f)
                    if os.path.exists(csv_p):
                        zf.write(csv_p, arcname=os.path.join("submission", os.path.basename(csv_f)))
            else:
                for root, _, files in os.walk(self.output_dir):
                    for f in sorted(files):
                        if f.endswith(".csv") and not f.startswith("test_"):
                            abs_p = os.path.join(root, f)
                            zf.write(abs_p, arcname=os.path.join("submission", f))

        print(f"[SubmissionGenerator] Created official submission package: {zip_path}")
        return zip_path

    @staticmethod
    def validate_csv(csv_content_or_path: str, query_type: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validates a submission CSV against all rules.txt guidelines:
        - Max 100 lines
        - No headers
        - No .mp4 extensions in video_id
        - Integer frame numbers
        - Q&A answer length <= 100 chars
        - TRAKE frame count and monotonic order
        """
        errors = []
        if os.path.exists(csv_content_or_path):
            with open(csv_content_or_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = csv_content_or_path

        reader = list(csv.reader(io.StringIO(content.strip())))
        if not reader:
            return False, ["CSV is empty."]

        if len(reader) > 100:
            errors.append(f"Exceeded max allowed rows (found {len(reader)} rows, max is 100).")

        first_row = reader[0]
        if first_row and any(kw in str(first_row[0]).lower() for kw in ["video", "query", "id", "frame"]):
            errors.append("Header row detected. rules.txt requires direct data rows without headers.")

        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            vid = row[0].strip()
            if vid.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                errors.append(f"Row {row_idx}: Video name '{vid}' contains a file extension.")
            if "/" in vid or "\\" in vid:
                errors.append(f"Row {row_idx}: Video name '{vid}' contains directory separators.")

            # Type-specific validation
            if query_type == "kis" or (query_type is None and len(row) == 2):
                if len(row) != 2:
                    errors.append(f"Row {row_idx}: KIS requires exactly 2 columns (<video_id>, <frame_idx>), found {len(row)}.")
                elif not row[1].strip().isdigit():
                    errors.append(f"Row {row_idx}: Frame ID '{row[1]}' is not a valid integer.")

            elif query_type == "qa":
                if len(row) < 3:
                    errors.append(f"Row {row_idx}: Q&A requires at least 3 columns (<video_id>, <frame_idx>, <answer>).")
                else:
                    if not row[1].strip().isdigit():
                        errors.append(f"Row {row_idx}: Frame ID '{row[1]}' is not a valid integer.")
                    ans = row[2]
                    if len(ans) > 100:
                        errors.append(f"Row {row_idx}: Q&A answer length ({len(ans)} chars) exceeds 100 character limit.")

            elif query_type == "trake" or (query_type is None and len(row) > 3):
                if len(row) < 3:
                    errors.append(f"Row {row_idx}: TRAKE requires at least 2 event frames.")
                else:
                    try:
                        frames = [int(f.strip()) for f in row[1:]]
                        if any(frames[i] > frames[i+1] for i in range(len(frames)-1)):
                            errors.append(f"Row {row_idx}: TRAKE frames {frames} are not in chronological order.")
                    except ValueError:
                        errors.append(f"Row {row_idx}: One or more TRAKE frame IDs are not valid integers: {row[1:]}")

        return (len(errors) == 0), errors

    @classmethod
    def validate_submission_zip(cls, zip_path: str) -> Tuple[bool, List[str]]:
        """
        Validates the submission ZIP bundle against rules.txt requirements.
        """
        if not os.path.exists(zip_path):
            return False, [f"File {zip_path} does not exist."]
        if not zip_path.endswith(".zip"):
            return False, [f"File {zip_path} is not a .zip archive."]

        errors = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            csv_files = [n for n in namelist if n.endswith(".csv")]
            if not csv_files:
                errors.append("No CSV files found in ZIP archive.")

            for name in csv_files:
                if not name.startswith("submission/"):
                    errors.append(f"File '{name}' is not located inside the mandatory 'submission/' root folder.")
                
                # Determine query type from filename
                qtype = None
                basename = os.path.basename(name).lower()
                if "kis" in basename:
                    qtype = "kis"
                elif "qa" in basename:
                    qtype = "qa"
                elif "trake" in basename:
                    qtype = "trake"

                content = zf.read(name).decode("utf-8")
                valid, csv_errs = cls.validate_csv(content, query_type=qtype)
                if not valid:
                    errors.extend([f"[{name}] {e}" for e in csv_errs])

        return (len(errors) == 0), errors
