import unittest
import os
import io
import zipfile
import shutil
from src.evaluation.submission_generator import SubmissionGenerator

class TestBatchQueryWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = "submissions_batch_test"
        os.makedirs(self.test_dir, exist_ok=True)
        self.generator = SubmissionGenerator(output_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_batch_query_zip_lifecycle(self):
        # 1. Simulate user queries ZIP package
        query_package = [
            {
                "id": "query-1-kis",
                "mode": "kis",
                "prompt": "Mẩu tin giới thiệu về đàn hổ",
                "predictions": [
                    {"video_id": "L00_V000", "frame_idx": 1234},
                    {"video_id": "L00_V055", "frame_idx": 5555},
                    {"video_id": "L01_V028", "frame_idx": 25300}
                ]
            },
            {
                "id": "query-2-qa",
                "mode": "qa",
                "prompt": "Có bao nhiêu người trong cảnh?",
                "predictions": [
                    {"video_id": "L01_V028", "frame_idx": 3450, "answer": "5"},
                    {"video_id": "L02_V011", "frame_idx": 1200, "answer": "Năm người"},
                    {"video_id": "L03_V005", "frame_idx": 2800, "answer": "Màu đỏ, rất đẹp"}
                ]
            },
            {
                "id": "query-3-trake",
                "mode": "trake",
                "prompt": "Chuỗi sự kiện lễ hội",
                "predictions": [
                    {"video_id": "L10_V001", "aligned_frames": [1200, 1850, 2100, 2450]},
                    {"video_id": "L10_V001", "aligned_frames": [1180, 1820, 2080, 2420]}
                ]
            }
        ]

        # 2. Format individual CSVs
        for q in query_package:
            if q["mode"] == "kis":
                lines = self.generator.format_kis_submission(q["id"], q["predictions"])
            elif q["mode"] == "qa":
                lines = self.generator.format_qa_submission(q["id"], q["predictions"])
            elif q["mode"] == "trake":
                lines = self.generator.format_trake_submission(q["id"], q["predictions"], num_events=4)
            self.generator.save_submission_file(q["id"], lines)

        # 3. Package into official ZIP bundle
        zip_path = self.generator.package_submission_zip(zip_filename="submission_aic2026.zip")
        self.assertTrue(os.path.exists(zip_path))

        # 4. Validate ZIP against all rules.txt constraints
        valid, errors = SubmissionGenerator.validate_submission_zip(zip_path)
        self.assertTrue(valid, f"Validation errors: {errors}")

        # 5. Check ZIP internal structure
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            self.assertIn("submission/query-1-kis.csv", names)
            self.assertIn("submission/query-2-qa.csv", names)
            self.assertIn("submission/query-3-trake.csv", names)

if __name__ == "__main__":
    unittest.main()
