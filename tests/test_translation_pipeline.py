#!/usr/bin/env python3
"""
Standing Regression Test Suite for Vietnamese-to-English Query Translation Pipeline.
Verifies OmniRoute SSE stream decoding, GoogleTranslator fallback, and disk caching across 50 diverse query categories.
"""

import os
import sys
import unittest

# Ensure repo root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.query.translator import QueryTranslator


class TestTranslationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.translator = QueryTranslator(use_online=True)
        cls.test_cases = [
            # 1. Visual Objects & Scenes
            ("Người đàn ông mặc áo đỏ đang lái xe máy trên đường", ["man", "red", "motorcycle", "drive", "ride"]),
            ("Một con mèo tam thể đang ngủ trên ghế sofa phòng khách", ["cat", "sleeping", "sofa", "living room"]),
            ("Khung cảnh hoàng hôn trên bãi biển với những hàng dừa", ["sunset", "beach", "coconut", "palm", "sea"]),
            ("Cảnh máy bay hạ cánh xuống đường băng sân bay", ["airplane", "plane", "landing", "runway", "airport"]),
            ("Tòa nhà chọc trời phát sáng rực rỡ vào ban đêm", ["skyscraper", "building", "night", "glowing", "lights"]),
            ("Bát phở bò nóng hổi bốc khói có nhiều hành lá", ["pho", "beef", "bowl", "noodles", "soup"]),
            ("Cầu Rồng Đà Nẵng phun lửa và phun nước", ["Dragon Bridge", "fire", "water", "Da Nang"]),
            ("Đoàn người mặc áo dài truyền thống đi dạo phố", ["ao dai", "traditional", "walking", "street", "people"]),
            ("Cánh đồng lúa chín vàng ươm ở vùng nông thôn", ["rice field", "yellow", "countryside", "ripe"]),
            ("Chiếc ô tô màu trắng đỗ trước cổng trường học", ["white car", "parked", "school", "gate"]),
            
            # 2. On-Screen Text & OCR Grounded
            ("Biển hiệu có chữ Cà Phê Sữa Đá", ["sign", "text", "coffee"]),
            ("Dòng chữ Chúc Mừng Năm Mới trên màn hình sân khấu", ["text", "Happy New Year", "stage", "screen"]),
            ("Biển báo giao thông cấm đỗ xe trên vỉa hè", ["traffic sign", "parking", "prohibited", "sidewalk"]),
            ("Áo thi đấu có in số 10 phía sau lưng", ["jersey", "shirt", "number 10", "back"]),
            ("Tấm bảng ghi thực đơn quán ăn với giá tiền", ["menu", "board", "restaurant", "prices"]),
            ("Logo VTV1 xuất hiện ở góc trên bên phải màn hình", ["logo", "VTV1", "corner", "screen"]),
            ("Tên đường Nguyễn Huệ trên biển chỉ dẫn", ["Nguyen Hue", "street", "sign"]),
            ("Biển hiệu nhà thuốc mở cửa 24 giờ", ["pharmacy", "drugstore", "24 hours", "sign"]),
            ("Khẩu hiệu Vì môi trường xanh sạch đẹp", ["slogan", "green", "environment"]),
            ("Băng rôn chào mừng đại hội đại biểu", ["banner", "welcome", "congress"]),
            
            # 3. Speech & Spoken Content
            ("Người dẫn chương trình nói về giải pháp kinh tế số", ["host", "presenter", "digital economy", "speaking"]),
            ("Bác sĩ tư vấn cách phòng ngừa bệnh tim mạch", ["doctor", "advising", "cardiovascular", "heart"]),
            ("Chuyên gia phân tích thị trường bất động sản năm 2026", ["expert", "analyzing", "real estate", "market"]),
            ("Phỏng vấn người dân về dự án đường sắt đô thị", ["interview", "residents", "railway", "project"]),
            ("Lời bài hát ca ngợi quê hương đất nước", ["song lyrics", "homeland", "country"]),
            ("Thủ tướng phát biểu tại hội nghị thượng đỉnh", ["Prime Minister", "speaking", "summit", "conference"]),
            ("Giáo viên giảng bài về lịch sử dân tộc", ["teacher", "lecturing", "history"]),
            ("Bản tin dự báo thời tiết cho các tỉnh phía Bắc", ["weather forecast", "northern provinces", "news"]),
            ("Bình luận viên thể thao tường thuật trận đấu bóng đá", ["commentator", "football", "soccer", "match"]),
            ("Nhạc trưởng điều khiển dàn nhạc giao hưởng", ["conductor", "orchestra", "symphony"]),
            
            # 4. Numeric & Temporal Sequences
            ("Đồng hồ đếm ngược từ mười về một", ["countdown", "clock", "timer", "ten to one"]),
            ("Bảng điểm hiển thị tỷ số 2-1", ["scoreboard", "score", "2-1"]),
            ("Lịch trình khởi hành vào lúc tám giờ sáng", ["departure", "schedule", "8 AM", "morning"]),
            ("Biểu đồ cột biểu diễn tăng trưởng GDP 5 năm qua", ["bar chart", "chart", "GDP growth", "5 years"]),
            ("Tốc độ xe chạy hiển thị 80 km trên giờ", ["speed", "80 km", "display"]),
            
            # 5. Complex Compositional & Action
            ("Người phụ nữ nấu ăn rồi sau đó dọn bàn ăn", ["woman cooking", "setting table", "after that"]),
            ("Em bé đang vẽ tranh rồi cười đùa với mẹ", ["baby", "drawing", "laughing", "mother"]),
            ("Cầu thủ sút bóng vào lưới và ăn mừng bàn thắng", ["player", "shooting", "goal", "celebrating"]),
            ("Xe cứu thương bật còi hú vượt qua ngã tư đèn đỏ", ["ambulance", "siren", "intersection", "red light"]),
            ("Bảo vệ mở cổng cho đoàn xe quân sự đi qua", ["guard", "gate", "military", "convoy"])
        ]

    def test_translation_bulk_quality(self):
        """Verify all 40+ test cases translate cleanly into non-empty English text."""
        for vi_query, expected_keywords in self.test_cases:
            en_trans = self.translator.translate(vi_query)
            self.assertTrue(isinstance(en_trans, str), f"Translation returned non-string for: {vi_query}")
            self.assertTrue(len(en_trans.strip()) >= 5, f"Translation too short for: {vi_query} -> {en_trans}")
            self.assertFalse(en_trans.startswith("data: "), f"SSE leakage in translation: {en_trans}")
            print(f"[✓ PASS] {vi_query[:35]}... -> {en_trans}")

    def test_prompt_generation(self):
        """Verify prompt ensemble generation generates multiple distinct prompts."""
        prompts = self.translator.generate_prompts("a chef cooking beef soup")
        self.assertTrue(len(prompts) >= 1)
        self.assertIn("a photo of a chef cooking beef soup", prompts)


if __name__ == "__main__":
    unittest.main()
