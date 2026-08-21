"""
Unit tests for Qwen ASR Transcript Refinement script.
"""

import os
import json
import tempfile
import unittest
from scripts.refine_transcripts_qwen import QwenRefiner, RefinementTracker, process_single_video


class TestQwenRefiner(unittest.TestCase):

    def test_parse_and_validate_tags_success(self):
        output_text = """<SEGMENT_0>
Chào mừng quý vị đến với chương trình 60 Giây của Đài Truyền hình Thành phố Hồ Chí Minh.
</SEGMENT_0>
<SEGMENT_1>
Đồng bằng sông Cửu Long với tình trạng sụt lún gấp gần 20 lần so với nước biển dâng.
</SEGMENT_1>"""

        orig_texts = [
            "Chào mừng quý vị đến với chương trình sống bây dây của Đại truyền hình thành phố Chị mình",
            "Đồng băng sống cửa Long với tình trạng sụp lũng gấp gần 20 lần so với nước biển nhân",
        ]

        ok, refined, err = QwenRefiner.parse_and_validate_tags(output_text, 2, orig_texts)
        self.assertTrue(ok)
        self.assertEqual(len(refined), 2)
        self.assertIn("60 Giây", refined[0])
        self.assertIn("sụt lún", refined[1])

    def test_parse_and_validate_count_mismatch(self):
        output_text = """<SEGMENT_0>
Chào mừng quý vị.
</SEGMENT_0>"""

        orig_texts = ["Chào mừng quý vị.", "Câu thứ hai."]
        ok, refined, err = QwenRefiner.parse_and_validate_tags(output_text, 2, orig_texts)
        self.assertFalse(ok)
        self.assertIn("segment_count_mismatch", err)

    def test_parse_and_validate_missing_index(self):
        output_text = """<SEGMENT_0>
Chào mừng quý vị.
</SEGMENT_0>
<SEGMENT_2>
Câu thứ hai.
</SEGMENT_2>"""

        orig_texts = ["Chào mừng quý vị.", "Câu thứ hai."]
        ok, refined, err = QwenRefiner.parse_and_validate_tags(output_text, 2, orig_texts)
        self.assertFalse(ok)
        self.assertIn("missing_segment_index", err)

    def test_tracker_and_atomic_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = os.path.join(tmpdir, "manifest.json")
            tracker = RefinementTracker(manifest_file, "Qwen/Qwen3-1.7B")
            self.assertFalse(tracker.is_completed("VID_001"))

            tracker.record_success(
                video_id="VID_001",
                total_segments=10,
                refined_segments=10,
                changed_segments=4,
                words_before=100,
                words_after=98,
                elapsed_sec=2.5,
            )
            self.assertTrue(tracker.is_completed("VID_001"))

            # Reload tracker from disk
            tracker2 = RefinementTracker(manifest_file, "Qwen/Qwen3-1.7B")
            self.assertTrue(tracker2.is_completed("VID_001"))
            self.assertEqual(tracker2.data["summary"]["completed"], 1)
            self.assertEqual(tracker2.data["summary"]["total_segments"], 10)
            self.assertEqual(tracker2.data["summary"]["changed_segments"], 4)

    def test_abort_on_validation_failure_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_data = [
                {"video_id": "TEST_001", "text": "Original text 1", "raw_text": "Original text 1"},
                {"video_id": "TEST_001", "text": "Original text 2", "raw_text": "Original text 2"},
            ]
            test_file = os.path.join(tmpdir, "TEST_001.json")
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(orig_data, f)

            manifest_file = os.path.join(tmpdir, "manifest.json")
            tracker = RefinementTracker(manifest_file, "Qwen/Qwen3-1.7B")

            # Mock refiner that fails validation (returns empty / bad count)
            class FailingRefiner(QwenRefiner):
                def __init__(self):
                    super().__init__(dry_run=True)
                def generate_refinements_chunk(self, raw_texts):
                    return False, [], "Simulated validation error: tag drop"

            bad_refiner = FailingRefiner()
            ok, msg = process_single_video(
                transcript_path=test_file,
                refiner=bad_refiner,
                tracker=tracker,
                output_dir=tmpdir,
                force=True,
            )

            self.assertFalse(ok)
            # Verify original file was NOT modified or corrupted
            with open(test_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data, orig_data)

            # Verify manifest recorded failure
            self.assertFalse(tracker.is_completed("TEST_001"))
            self.assertEqual(tracker.data["records"]["TEST_001"]["status"], "failed")
            self.assertEqual(tracker.data["records"]["TEST_001"]["failure_stage"], "validation")


if __name__ == "__main__":
    unittest.main()
