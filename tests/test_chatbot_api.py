import os
import sys
import unittest
import json
import threading
import time
import urllib.request
import urllib.parse
from http.server import ThreadingHTTPServer

# Ensure workspace is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.ui.search_app import SearchEngine, RequestHandler, launch_server
import src.ui.search_app as search_app_module

class TestAIChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = SearchEngine()
        search_app_module.ENGINE = cls.engine
        cls.port = 8899
        cls.server = ThreadingHTTPServer(("127.0.0.1", cls.port), RequestHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_01_get_video_context(self):
        # Direct method call
        ctx = self.engine.get_video_context("L21_V001", pts_time=10.0, window_sec=30.0)
        self.assertIn("video_id", ctx)
        self.assertEqual(ctx["video_id"], "L21_V001")
        self.assertIn("asr_segments", ctx)
        self.assertIn("ocr_lines", ctx)
        self.assertIsInstance(ctx["asr_segments"], list)
        self.assertIsInstance(ctx["ocr_lines"], list)

    def test_02_extract_clip_frames(self):
        # Extract 4 frames from L21_V001 between 2.0s and 6.0s
        frames = self.engine.extract_clip_frames("L21_V001", start_sec=2.0, end_sec=6.0, num_frames=4)
        self.assertIsInstance(frames, list)
        self.assertEqual(len(frames), 4)
        for f in frames:
            self.assertTrue(f.startswith("data:image/jpeg;base64,"))

    def test_03_clip_range_validation(self):
        # When start_sec or end_sec is missing
        res_missing = self.engine.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            video_id="L21_V001",
            start_sec=None,
            end_sec=None
        )
        self.assertIn("error", res_missing)
        self.assertIn("Vui lòng đánh dấu", res_missing["error"])

        # When start_sec >= end_sec
        res_invalid = self.engine.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            video_id="L21_V001",
            start_sec=10.0,
            end_sec=5.0
        )
        self.assertIn("error", res_invalid)
        self.assertIn("không hợp lệ", res_invalid["error"])

    def test_04_http_get_video_context(self):
        # HTTP GET /api/video_context endpoint
        url = f"http://127.0.0.1:{self.port}/api/video_context?video_id=L21_V001&pts_time=10.0"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["video_id"], "L21_V001")
            self.assertIn("asr_segments", data)
            self.assertIn("ocr_lines", data)

    def test_05_http_post_chat_missing_range(self):
        # HTTP POST /api/chat without start_sec and end_sec
        url = f"http://127.0.0.1:{self.port}/api/chat"
        payload = {
            "video_id": "L21_V001",
            "frame_idx": 150,
            "pts_time": 6.0,
            "query": "đàn hổ con",
            "messages": [{"role": "user", "content": "Tóm tắt cảnh này"}],
            "api_key": "dummy_key",
            "provider": "openrouter"
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("error", data)
            self.assertIn("Vui lòng đánh dấu", data["error"])

    def test_06_http_post_chat_missing_key(self):
        # HTTP POST /api/chat with valid range but empty key
        old_openrouter = os.environ.pop("OPENROUTER_API_KEY", None)
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        try:
            url = f"http://127.0.0.1:{self.port}/api/chat"
            payload = {
                "video_id": "L21_V001",
                "frame_idx": 150,
                "pts_time": 6.0,
                "start_sec": 2.0,
                "end_sec": 6.0,
                "query": "đàn hổ con",
                "messages": [{"role": "user", "content": "Tóm tắt cảnh này"}],
                "api_key": "",
                "provider": "openrouter"
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("error", data)
                self.assertIn("API Key", data["error"])
        finally:
            if old_openrouter:
                os.environ["OPENROUTER_API_KEY"] = old_openrouter
            if old_openai:
                os.environ["OPENAI_API_KEY"] = old_openai

    def test_07_non_existent_video_context(self):
        ctx = self.engine.get_video_context("NON_EXISTENT_VIDEO_999", pts_time=0.0)
        self.assertEqual(ctx["video_id"], "NON_EXISTENT_VIDEO_999")
        self.assertEqual(len(ctx["asr_segments"]), 0)
        self.assertEqual(len(ctx["ocr_lines"]), 0)


if __name__ == "__main__":
    unittest.main()
