import unittest
import csv
import io
from src.evaluation.submission_generator import SubmissionGenerator

class TestTaskModes(unittest.TestCase):
    def setUp(self):
        self.generator = SubmissionGenerator(output_dir="submissions_test")

    def test_kis_formatting_and_validation(self):
        predictions = [
            {"video_id": "L00_V000", "frame_idx": 1234},
            {"video_id": "L00_V055", "frame_idx": 5555},
            {"video_id": "L01_V028", "frame_idx": 25300}
        ]
        lines = self.generator.format_kis_submission("query-1", predictions)
        self.assertEqual(lines[0], "L00_V000,1234")
        self.assertEqual(lines[1], "L00_V055,5555")
        self.assertEqual(lines[2], "L01_V028,25300")

        csv_content = "\n".join(lines)
        valid, errors = self.generator.validate_csv(csv_content, query_type="kis")
        self.assertTrue(valid, f"Validation errors: {errors}")

    def test_qa_formatting_and_escaping(self):
        predictions = [
            {"video_id": "L01_V028", "frame_idx": 3450, "answer": "5"},
            {"video_id": "L02_V011", "frame_idx": 1200, "answer": "Năm người"},
            {"video_id": "L03_V005", "frame_idx": 2800, "answer": "Màu đỏ, rất đẹp"},
            {"video_id": "L04_V012", "frame_idx": 4100, "answer": 'Anh ấy nói "Tuyệt vời"'}
        ]
        lines = self.generator.format_qa_submission("query-2", predictions)
        self.assertEqual(lines[0], "L01_V028,3450,5")
        self.assertEqual(lines[1], "L02_V011,1200,Năm người")
        self.assertEqual(lines[2], 'L03_V005,2800,"Màu đỏ, rất đẹp"')
        self.assertEqual(lines[3], 'L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""')

        csv_content = "\n".join(lines)
        valid, errors = self.generator.validate_csv(csv_content, query_type="qa")
        self.assertTrue(valid, f"Validation errors: {errors}")

    def test_trake_formatting_and_validation(self):
        predictions = [
            {"video_id": "L10_V001", "aligned_frames": [1200, 1850, 2100, 2450]},
            {"video_id": "L10_V001", "aligned_frames": [1180, 1820, 2080, 2420]},
            {"video_id": "L11_V003", "aligned_frames": [5100, 5700, 6200, 6800]}
        ]
        lines = self.generator.format_trake_submission("query-3", predictions, num_events=4)
        self.assertEqual(lines[0], "L10_V001,1200,1850,2100,2450")
        self.assertEqual(lines[1], "L10_V001,1180,1820,2080,2420")
        self.assertEqual(lines[2], "L11_V003,5100,5700,6200,6800")

        csv_content = "\n".join(lines)
        valid, errors = self.generator.validate_csv(csv_content, query_type="trake")
        self.assertTrue(valid, f"Validation errors: {errors}")

    def test_plan_txt_specimens_exact_validation(self):
        # Textual KIS
        kis_specimen = """L00_V000,1234
L00_V055,5555
L01_V028,25300"""
        v_kis, err_kis = self.generator.validate_csv(kis_specimen, query_type="kis")
        self.assertTrue(v_kis, f"KIS errors: {err_kis}")

        # Q&A specimen
        qa_specimen = '''L01_V028,3450,5
L02_V011,1200,Năm người
L03_V005,2800,"Màu đỏ, rất đẹp"
L04_V012,4100,"Anh ấy nói ""Tuyệt vời"""'''
        v_qa, err_qa = self.generator.validate_csv(qa_specimen, query_type="qa")
        self.assertTrue(v_qa, f"QA errors: {err_qa}")

        # TRAKE specimen
        trake_specimen = """L10_V001,1200,1850,2100,2450
L10_V001,1180,1820,2080,2420
L11_V003,5100,5700,6200,6800"""
        v_trake, err_trake = self.generator.validate_csv(trake_specimen, query_type="trake")
        self.assertTrue(v_trake, f"TRAKE errors: {err_trake}")

if __name__ == "__main__":
    unittest.main()
