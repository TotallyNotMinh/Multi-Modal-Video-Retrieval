import os
import sys
import unittest
import numpy as np
import tempfile
import json
import torch

# Ensure workspace is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.index.frame_mapper import FrameMapper
from src.index.matrix_builder import FeatureMatrixBuilder
from src.index.object_indexer import ObjectIndexer
from src.index.metadata_indexer import MetadataIndexer
from src.index.faiss_index import FAISSIndex
from src.query.translator import QueryTranslator
from src.query.text_encoder import CLIPTextEncoder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.trake_aligner import TRAKEAligner
from src.evaluation.metrics import AICMetrics
from src.evaluation.submission_generator import SubmissionGenerator

class TestAIC2026Pipeline(unittest.TestCase):

    def test_01_frame_mapper(self):
        mapper = FrameMapper()
        info = mapper.get_frame_info("L21_V001", 0)
        self.assertIn("frame_idx", info)
        self.assertIn("pts_time", info)
        self.assertEqual(info["n"], 1)

    def test_02_translator(self):
        trans = QueryTranslator(use_online=False)  # test dictionary fallback
        res = trans.translate("Tìm video về người đàn ông mặc áo đỏ phát biểu")
        self.assertTrue(len(res) > 0)
        prompts = trans.generate_prompts(res)
        self.assertEqual(len(prompts), 5)

    def test_03_faiss_index(self):
        dim = 512
        N = 100
        vecs = np.random.randn(N, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        records = [{"video_id": f"V_{i//10}", "frame_idx": i * 30, "global_idx": i} for i in range(N)]

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "test_faiss")
            idx = FAISSIndex(dim=dim, index_type="FlatIP")
            idx.build(vecs, records, save_path_prefix=prefix)

            # Test load and search
            loaded_idx = FAISSIndex().load(prefix)
            q_vec = vecs[5]
            hits = loaded_idx.search(q_vec, top_k=5)
            self.assertEqual(len(hits), 5)
            self.assertEqual(hits[0][0]["global_idx"], 5)
            self.assertAlmostEqual(hits[0][1], 1.0, places=4)

    def test_04_trake_aligner(self):
        dim = 512
        N = 50
        vecs = np.random.randn(N, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        records = [{"video_id": "L21_V001", "frame_idx": i * 30, "global_idx": i} for i in range(N)]

        dense = DenseRetriever(vecs, records)
        aligner = TRAKEAligner(dense)

        # 3 sequential event vectors
        events = [vecs[10], vecs[20], vecs[30]]
        res = aligner.align_sequence(events, top_k_videos=1)
        self.assertTrue(len(res) > 0)
        frames = res[0]["aligned_frames"]
        self.assertEqual(len(frames), 3)
        # Check strict monotonicity
        self.assertTrue(frames[0] < frames[1] < frames[2])

    def test_05_submission_generator_padding_and_escaping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = SubmissionGenerator(output_dir=tmpdir)
            
            # Test padding when only 10 candidates provided
            few_preds = [{"video_id": "L21_V001", "frame_idx": i * 30} for i in range(10)]
            lines = sub.format_kis_submission("q01", few_preds)
            self.assertEqual(len(lines), 100)  # Must be padded to exactly 100 rows
            sub.save_submission_file("q01", lines)

            # Test Q&A comma escaping
            qa_preds = [{"video_id": "L21_V001", "frame_idx": 100, "answer": "màu xanh, đậm\n"}]
            qa_lines = sub.format_qa_submission("qa01", qa_preds)
            self.assertEqual(len(qa_lines), 100)
            self.assertIn('"màu xanh, đậm"', qa_lines[0])

            # Test TRAKE submission
            trake_preds = [{"video_id": "L21_V001", "aligned_frames": [100, 200, 300, 400]}]
            t_lines = sub.format_trake_submission("trake01", trake_preds)
            self.assertEqual(len(t_lines), 100)
            self.assertEqual(t_lines[0], "L21_V001,100,200,300,400")

            zip_p = sub.package_submission_zip("test.zip", query_ids=["q01", "qa01", "trake01"])
            self.assertTrue(os.path.exists(zip_p))

    def test_06_metrics_kis_trake_qa(self):
        # 1. KIS Metric Hit at rank 3
        preds = [{"video_id": "L21_V001", "frame_idx": 100 + i * 10} for i in range(100)]
        preds[2] = {"video_id": "L21_V001", "frame_idx": 505}
        gt = {"video_id": "L21_V001", "frame_start": 500, "frame_end": 510}
        score = AICMetrics.evaluate_kis_query(preds, gt)
        self.assertEqual(score["R@1"], 0.0)
        self.assertEqual(score["R@5"], 1.0)
        self.assertAlmostEqual(score["Final_Score"], 0.80, places=4)

        # 2. TRAKE Partial Alignment Metric (3 out of 4 events hit)
        trake_preds = [
            {"video_id": "L10_V010", "aligned_frames": [101, 156, 203, 251]}  # event 2 is 156 outside [145, 155]
        ] * 100
        trake_gt = {
            "video_id": "L10_V010",
            "event_intervals": [(95, 105), (145, 155), (195, 205), (245, 255)]
        }
        trake_score = AICMetrics.evaluate_trake_query(trake_preds, trake_gt)
        self.assertAlmostEqual(trake_score["R@1"], 0.75, places=4)
        self.assertAlmostEqual(trake_score["Final_Score"], 0.75, places=4)

        # 3. QA Metric Hit with correct semantic answer
        qa_preds = [
            {"video_id": "L05_V005", "frame_idx": 850, "answer": "màu xanh"}
        ] * 100
        qa_gt = {
            "video_id": "L05_V005",
            "frame_start": 800,
            "frame_end": 900,
            "answers": ["màu xanh", "xanh"]
        }
        qa_score = AICMetrics.evaluate_qa_query(qa_preds, qa_gt)
        self.assertEqual(qa_score["R@1"], 1.0)
        self.assertEqual(qa_score["Final_Score"], 1.0)

    def test_07_dense_retriever_2d_shape(self):
        matrix = np.random.randn(20, 128).astype(np.float32)
        records = [{"video_id": f"V_{i}", "frame_idx": i} for i in range(20)]
        dense = DenseRetriever(matrix, records)

        # Query with 2D shape (1, 128)
        q_2d = np.random.randn(1, 128).astype(np.float32)
        hits = dense.search(q_2d, top_k=5)
        self.assertEqual(len(hits), 5)
        # Query with 1D shape (128,)
        q_1d = np.random.randn(128).astype(np.float32)
        hits_1d = dense.search(q_1d, top_k=5)
        self.assertEqual(len(hits_1d), 5)

    def test_08_scene_detector(self):
        """
        Tests SceneDetector on L21_V001.mp4 (must exist at data/Videos_L21_a/video/).
        Falls back gracefully if the video is not present (e.g. CI environment).
        """
        from src.encoding.scene_detector import SceneDetector

        detector = SceneDetector(threshold=0.35, min_shot_frames=3)

        video_path = "data/Videos_L21_a/video/L21_V001.mp4"
        if not os.path.exists(video_path):
            self.skipTest(f"Test video not found at {video_path} — skipping.")

        shots = detector.detect_shots(video_path)

        # Must detect at least one shot
        self.assertGreater(len(shots), 0, "Expected at least one shot to be detected.")

        # Every shot must have valid frame indices
        for shot in shots:
            self.assertIn("shot_id", shot)
            self.assertIn("start_frame", shot)
            self.assertIn("end_frame", shot)
            self.assertIn("duration_sec", shot)
            self.assertGreaterEqual(shot["end_frame"], shot["start_frame"])
            self.assertGreaterEqual(shot["duration_sec"], 0.0)

        # Test adaptive sampling policy per duration class
        fps = 30.0

        # Short shot (<1.5s): exactly 1 frame
        short_shot = {"shot_id": 0, "start_frame": 0, "end_frame": 30, "duration_sec": 1.0}
        short_frames = detector.get_sample_frames(short_shot, fps)
        self.assertEqual(len(short_frames), 1, "Short shot must yield exactly 1 frame.")

        # Medium shot (1.5–5s): exactly 2 frames
        medium_shot = {"shot_id": 1, "start_frame": 0, "end_frame": 90, "duration_sec": 3.0}
        medium_frames = detector.get_sample_frames(medium_shot, fps)
        self.assertEqual(len(medium_frames), 2, "Medium shot must yield exactly 2 frames.")

        # Long shot (>5s): at least 3 frames
        long_shot = {"shot_id": 2, "start_frame": 0, "end_frame": 600, "duration_sec": 20.0}
        long_frames = detector.get_sample_frames(long_shot, fps)
        self.assertGreaterEqual(len(long_frames), 3, "Long shot must yield at least 3 frames.")

        # All returned indices must be within [start_frame, end_frame]
        for shot_case, frames in [(short_shot, short_frames), (medium_shot, medium_frames), (long_shot, long_frames)]:
            for f in frames:
                self.assertGreaterEqual(f, shot_case["start_frame"])
                self.assertLessEqual(f, shot_case["end_frame"])

        # Verify sorted and non-empty
        for frames in [short_frames, medium_frames, long_frames]:
            self.assertEqual(frames, sorted(frames), "Sample frames must be in sorted order.")
            self.assertGreater(len(frames), 0)

    def test_09_faster_whisper_asr(self):
        """
        Tests WhisperASR with faster-whisper initialization, fallback handling, and schema validation.
        """
        from src.encoding.whisper_asr import WhisperASR

        # Initialize with CPU-friendly settings for fast unit testing
        asr = WhisperASR(model_size="tiny", device="cpu", language="vi", initial_batch_size=16)
        self.assertEqual(asr.model_size, "tiny")
        self.assertEqual(asr.language, "vi")
        self.assertEqual(asr._batch_size, 16)

        # Test on non-existent video path (graceful empty return)
        res_empty = asr.transcribe_video("non_existent_video.mp4")
        self.assertEqual(res_empty, [])

        # Test dummy fallback mode
        asr_dummy = WhisperASR(model_size="dummy_test", device="cpu")
        asr_dummy.model = "dummy"
        res_dummy = asr_dummy.transcribe_video("any_video.mp4")
        self.assertEqual(res_dummy, [])


if __name__ == "__main__":
    unittest.main()

