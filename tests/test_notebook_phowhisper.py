import unittest
import os
import sys
import json
import queue
import tempfile
import shutil

class TestPhoWhisperNotebookPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.kaggle_dir = os.path.join(self.test_dir, "kaggle", "working", "cache", "asr_transcripts")
        self.output_dir = os.path.join(self.test_dir, "cache", "asr_transcripts")
        self.audio_dir = os.path.join(self.test_dir, "cache", "audio_extracted")
        os.makedirs(self.kaggle_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_notebook_json_validity(self):
        self.assertTrue(os.path.exists("training_notebook.ipynb"), "training_notebook.ipynb does not exist")
        with open("training_notebook.ipynb", "r", encoding="utf-8") as f:
            nb = json.load(f)
        self.assertIn("cells", nb)
        self.assertGreaterEqual(len(nb["cells"]), 15)
        
        # Verify Cell 10 contains the PhoWhisper code
        cell_10_src = "".join(nb["cells"][10]["source"])
        self.assertIn("run_phowhisper_worker", cell_10_src)
        self.assertIn("min_silence_duration_ms=500", cell_10_src)
        self.assertIn("no_speech_threshold=0.3", cell_10_src)
        self.assertIn("log_prob_threshold=-1.0", cell_10_src)
        self.assertIn("faster_whisper", cell_10_src)

    def test_audio_extraction_fallback(self):
        def extract_single_audio(vid_path, audio_dir, kaggle_audio_dir):
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            out_wav = os.path.join(audio_dir, f"{vid_name}.wav")
            return vid_name, vid_path

        vid_path = "data/Videos_L21_a/video/L21_V001.mp4"
        v_name, target_path = extract_single_audio(vid_path, self.audio_dir, None)
        self.assertEqual(v_name, "L21_V001")
        self.assertEqual(target_path, vid_path)

    def test_transcription_persistence_and_schema(self):
        vid_name = "L21_V001"
        out_json = os.path.join(self.output_dir, f"{vid_name}.json")
        kaggle_json = os.path.join(self.kaggle_dir, f"{vid_name}.json")

        mock_segments = [
            {
                "video_id": vid_name,
                "start_sec": 0.0,
                "end_sec": 4.5,
                "start_frame": 0,
                "end_frame": 112,
                "text": "Chào mừng quý vị khán giả đến với chương trình.",
            },
            {
                "video_id": vid_name,
                "start_sec": 4.8,
                "end_sec": 9.2,
                "start_frame": 120,
                "end_frame": 230,
                "text": "Hôm nay chúng ta sẽ tìm hiểu về đàn hổ tại miền Nam.",
            }
        ]

        tmp_json = f"{out_json}.tmp.test"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(mock_segments, f, indent=2, ensure_ascii=False)
        os.replace(tmp_json, out_json)

        shutil.copy2(out_json, kaggle_json)

        self.assertTrue(os.path.exists(out_json))
        self.assertTrue(os.path.exists(kaggle_json))

        with open(kaggle_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["video_id"], "L21_V001")
        self.assertIn("start_frame", loaded[0])
        self.assertIn("text", loaded[0])

    def test_dynamic_oom_fallback_logic(self):
        current_batch_size = 32
        calls = []

        def mock_transcribe(audio_path, batch_size):
            calls.append(batch_size)
            if batch_size > 8:
                raise RuntimeError("CUDA out of memory. Tried to allocate...")
            return [{"text": "Mock speech output", "timestamp": (0.0, 5.0)}]

        bs = current_batch_size
        while bs >= 2:
            try:
                res = mock_transcribe("dummy.wav", bs)
                break
            except Exception as e:
                if "out of memory" in str(e).lower():
                    bs = max(2, bs // 2)

        self.assertEqual(bs, 8)
        self.assertEqual(calls, [32, 16, 8])

if __name__ == "__main__":
    unittest.main()
