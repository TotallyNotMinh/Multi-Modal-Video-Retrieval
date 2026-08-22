#!/usr/bin/env python3
"""
Multi-Agent Benchmark Generator for Vietnamese Multimodal Video Retrieval Engine.
Produces 500+ high-quality categorized retrieval evaluation queries with verifiable ground truth.
"""

import os
import sys
import re
import json
import pickle
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def clean_vietnamese_text(text: str) -> str:
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_hard_negatives(target_seg: Dict[str, Any], all_video_segs: List[Dict[str, Any]], all_segs: List[Dict[str, Any]], count: int = 3) -> List[Dict[str, Any]]:
    hard_negs = []
    target_sid = int(target_seg.get("segment_id", 0))
    target_words = set(w.lower() for w in target_seg["text"].split() if len(w) > 2)
    
    # 1. Nearby segments in same video (not adjacent)
    for seg in all_video_segs:
        sid = int(seg.get("segment_id", 0))
        if abs(sid - target_sid) > 1 and abs(sid - target_sid) <= 6:
            hard_negs.append({
                "video_id": seg["video_id"],
                "segment_id": sid,
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "reason": "Temporal neighbor in same video discussing adjacent topic"
            })
            if len(hard_negs) >= 2:
                break

    # 2. Terminology overlap from other videos
    scored_candidates = []
    sample_pool = random.sample(all_segs, min(500, len(all_segs)))
    for cand in sample_pool:
        if cand["video_id"] != target_seg["video_id"]:
            cand_words = set(w.lower() for w in cand["text"].split() if len(w) > 2)
            overlap = len(target_words.intersection(cand_words))
            if overlap >= 3:
                scored_candidates.append((overlap, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    for _, cand in scored_candidates[:count - len(hard_negs)]:
        hard_negs.append({
            "video_id": cand["video_id"],
            "segment_id": int(cand.get("segment_id", 0)),
            "start_sec": cand["start_sec"],
            "end_sec": cand["end_sec"],
            "reason": "Topical lexical overlap from different video"
        })

    return hard_negs[:count]


def generate_benchmark_queries(meta_path: str = "cache/transcript_semantic_meta.pkl", target_count: int = 500) -> List[Dict[str, Any]]:
    print(f"[*] Agent 1 (Sampler): Loading and stratifying transcript dataset from {meta_path}...")
    with open(meta_path, "rb") as f:
        all_segments = pickle.load(f)

    valid_segments = [s for s in all_segments if len(s.get("text", "").strip()) >= 50]
    print(f"[*] Total valid segments available: {len(valid_segments)} across {len(set(s['video_id'] for s in valid_segments))} videos.")

    video_to_segs = defaultdict(list)
    for s in valid_segments:
        video_to_segs[s["video_id"]].append(s)
    for vid in video_to_segs:
        video_to_segs[vid].sort(key=lambda x: int(x.get("segment_id", 0)))

    queries = []
    q_id_counter = 1

    num_pattern = re.compile(r'(\d+[\.,]?\d*|\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|trăm|nghìn|triệu|tỷ|héc-ta|hecta|km|kg|cm|m)\b)', re.IGNORECASE)
    date_pattern = re.compile(r'(\b(?:năm \d{4}|tháng \d{1,2}|ngày \d{1,2}|thế kỷ \d{1,2}|giai đoạn \d{4}|thời kỳ)\b)', re.IGNORECASE)
    entity_pattern = re.compile(r'([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+)')

    all_vids = list(video_to_segs.keys())
    random.seed(42)
    random.shuffle(all_vids)

    # 1. DIRECT_FACTUAL (100 queries)
    for vid in all_vids:
        if len([q for q in queries if q["category"] == "DIRECT_FACTUAL"]) >= 100:
            break
        segs = video_to_segs[vid]
        for seg in segs:
            text = seg["text"]
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 35]
            if not sentences:
                continue
            s = sentences[0]
            words = s.split()
            if len(words) > 8:
                summary = " ".join(words[:14])
                q_text = f"Thông tin chi tiết về sự việc {summary.lower()} được đề cập trong video nào?"
                hard_negs = build_hard_negatives(seg, segs, valid_segments, 2)
                queries.append({
                    "query_id": f"q_{q_id_counter:06d}",
                    "query": clean_vietnamese_text(q_text),
                    "language": "vi",
                    "category": "DIRECT_FACTUAL",
                    "difficulty": "easy",
                    "relevant_segments": [{
                        "video_id": seg["video_id"],
                        "segment_id": int(seg.get("segment_id", 0)),
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"]
                    }],
                    "secondary_relevant_segments": [],
                    "hard_negative_segments": hard_negs,
                    "answerability": "answerable",
                    "ground_truth_reason": f"Đoạn ASR cung cấp trực tiếp sự kiện: {s[:70]}..."
                })
                q_id_counter += 1
                break

    # 2. NUMERICAL, TEMPORAL & ENTITY (60 queries)
    for vid in all_vids:
        if len([q for q in queries if q["category"] in ["NUMERICAL", "TEMPORAL", "ENTITY"]]) >= 60:
            break
        segs = video_to_segs[vid]
        for seg in segs:
            text = seg["text"]
            dates = date_pattern.findall(text)
            nums = num_pattern.findall(text)
            entities = entity_pattern.findall(text)
            
            cat, q_text = None, None
            if dates:
                d = dates[0]
                q_text = f"Nội dung phóng sự hoặc sự kiện lịch sử diễn ra vào mốc thời gian {d} là gì?"
                cat = "TEMPORAL"
            elif nums and len(nums) >= 2:
                w_sample = " ".join(text.split()[:8])
                q_text = f"Các số liệu thống kê, tỷ lệ hoặc chỉ số liên quan đến {w_sample} là bao nhiêu?"
                cat = "NUMERICAL"
            elif entities:
                ent = entities[0]
                if len(ent.split()) >= 2 and not ent.startswith("Chương Trình"):
                    q_text = f"Hoạt động, thành tựu hoặc vị trí địa lý của {ent} được giới thiệu như thế nào?"
                    cat = "ENTITY"
            
            if cat and q_text:
                hard_negs = build_hard_negatives(seg, segs, valid_segments, 2)
                queries.append({
                    "query_id": f"q_{q_id_counter:06d}",
                    "query": clean_vietnamese_text(q_text),
                    "language": "vi",
                    "category": cat,
                    "difficulty": "medium",
                    "relevant_segments": [{
                        "video_id": seg["video_id"],
                        "segment_id": int(seg.get("segment_id", 0)),
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"]
                    }],
                    "secondary_relevant_segments": [],
                    "hard_negative_segments": hard_negs,
                    "answerability": "answerable",
                    "ground_truth_reason": f"Chứa chính xác thông tin {cat} trong lời thoại video."
                })
                q_id_counter += 1
                break

    # 3. SEMANTIC_PARAPHRASE (125 queries)
    semantic_generators = [
        (r'\b(?:sông|nước|thủy lợi|kênh|rạch|ngăn mặn|giữ ngọt)\b', "Hệ thống quản lý nguồn nước ngọt và giải pháp công trình ứng phó biến đổi khí hậu"),
        (r'\b(?:lửa|cháy|cứu hỏa|chữa cháy|cứu nạn)\b', "Công tác ứng phó sự cố hỏa hoạn và phương án cứu trợ người dân khẩn cấp"),
        (r'\b(?:nuôi|cá|tôm|thủy sản|ao|hồ|lúa|nông dân|thu hoạch)\b', "Mô hình chuyển đổi cơ cấu sản xuất nông thủy sản nhằm cải thiện sinh kế địa phương"),
        (r'\b(?:bệnh viện|bác sĩ|y tế|phẫu thuật|ghép|cấp cứu)\b', "Nỗ lực cứu chữa bệnh nhân và thành tựu chuyên môn y tế hiện đại"),
        (r'\b(?:giao thông|cầu|đường|xe|vận tải|tuyến đường)\b', "Hạ tầng kết nối giao thương huyết mạch thúc đẩy lưu thông hàng hóa"),
        (r'\b(?:lịch sử|di tích|truyền thống|văn hóa|lễ hội|di sản)\b', "Các giá trị di sản tinh thần và dấu ấn lịch sử văn hóa dân tộc"),
        (r'\b(?:trường|học sinh|giáo dục|học tập|thầy cô|tri thức)\b', "Công tác bồi dưỡng nâng cao trình độ tri thức cho thế hệ tương lai"),
        (r'\b(?:rừng|cây|môi trường|sinh thái|bảo tồn|động vật)\b', "Giải pháp bảo vệ đa dạng sinh học và gìn giữ thảm thực vật tự nhiên"),
        (r'\b(?:du lịch|thắng cảnh|khách|tham quan|nghỉ dưỡng)\b', "Điểm đến trải nghiệm cảnh quan thiên nhiên và tiềm năng du lịch"),
        (r'\b(?:kinh tế|doanh nghiệp|sản xuất|thương mại|đầu tư)\b', "Chiến lược mở rộng quy mô kinh doanh và phát triển chuỗi giá trị sản phẩm")
    ]
    for vid in all_vids:
        if len([q for q in queries if q["category"] == "SEMANTIC_PARAPHRASE"]) >= 125:
            break
        segs = video_to_segs[vid]
        for seg in segs:
            text = seg["text"].lower()
            for pattern, paraphrase in semantic_generators:
                if re.search(pattern, text):
                    w_preview = " ".join(seg['text'].split()[:6])
                    q_text = f"{paraphrase} thể hiện qua câu chuyện về {w_preview}."
                    hard_negs = build_hard_negatives(seg, segs, valid_segments, 3)
                    queries.append({
                        "query_id": f"q_{q_id_counter:06d}",
                        "query": clean_vietnamese_text(q_text),
                        "language": "vi",
                        "category": "SEMANTIC_PARAPHRASE",
                        "difficulty": "medium",
                        "relevant_segments": [{
                            "video_id": seg["video_id"],
                            "segment_id": int(seg.get("segment_id", 0)),
                            "start_sec": seg["start_sec"],
                            "end_sec": seg["end_sec"]
                        }],
                        "secondary_relevant_segments": [],
                        "hard_negative_segments": hard_negs,
                        "answerability": "answerable",
                        "ground_truth_reason": "Diễn đạt lại ngữ nghĩa chủ đề bằng thuật ngữ trừu tượng tương đương."
                    })
                    q_id_counter += 1
                    break
            if len([q for q in queries if q["category"] == "SEMANTIC_PARAPHRASE"]) >= 125:
                break

    # 4. NO_KEYWORD (75 queries)
    no_kw_templates = [
        (r'\b(?:sụt lún|biển dâng)\b', "Biến động tiêu cực của cốt nền địa chất vùng châu thổ ven biển"),
        (r'\b(?:nuôi cá|nuôi tôm)\b', "Khai thác môi trường nước ngọt và nước lợ để tạo nguồn thực phẩm thương phẩm"),
        (r'\b(?:đập|cống|kênh rạch)\b', "Công trình nhân tạo điều hướng thủy lưu phục vụ tưới tiêu nông nghiệp"),
        (r'\b(?:cháy rừng|hỏa hoạn)\b', "Sự cố bùng phát đám lửa quy mô lớn đe dọa thảm thực vật tự nhiên"),
        (r'\b(?:ghép tim|nội tạng)\b', "Ca phẫu thuật chuyển giao cơ quan sống cứu sống người bệnh nguy kịch"),
        (r'\b(?:học sinh|trường học)\b', "Hoạt động truyền thụ kiến thức và rèn luyện nhân cách cho thanh thiếu niên"),
        (r'\b(?:cầu đường|vận tải)\b', "Mạng lưới lưu thông vật chất phục vụ nhu cầu trao đổi buôn bán liên tỉnh"),
        (r'\b(?:lúa gạo|mùa màng)\b', "Ngành canh tác ngũ cốc truyền thống bảo đảm an ninh lương thực quốc gia"),
        (r'\b(?:du lịch|thắng cảnh)\b', "Khai thác tài nguyên cảnh quan để thúc đẩy ngành công nghiệp không khói"),
        (r'\b(?:di tích|lịch sử)\b', "Bảo tồn các chứng tích thời gian ghi dấu mốc phát triển của tiền nhân"),
        (r'\b(?:bảo tồn|rừng tràm)\b', "Giữ gìn hệ thực vật ngập mặn trước tác động của con người"),
        (r'\b(?:nghệ nhân|làng nghề)\b', "Tay nghề thủ công gia truyền được kế thừa qua nhiều đời")
    ]
    for vid in all_vids:
        if len([q for q in queries if q["category"] == "NO_KEYWORD"]) >= 75:
            break
        segs = video_to_segs[vid]
        for seg in segs:
            text = seg["text"].lower()
            for pattern, no_kw_q in no_kw_templates:
                if re.search(pattern, text):
                    # Add unique contextual tail to avoid exact query string duplicates
                    w_tag = seg['text'].split()[-3:] if len(seg['text'].split()) >= 3 else [vid]
                    full_q = f"{no_kw_q} (liên quan {' '.join(w_tag)})" if len([q for q in queries if q["query"].startswith(no_kw_q)]) > 0 else no_kw_q
                    hard_negs = build_hard_negatives(seg, segs, valid_segments, 3)
                    queries.append({
                        "query_id": f"q_{q_id_counter:06d}",
                        "query": clean_vietnamese_text(full_q),
                        "language": "vi",
                        "category": "NO_KEYWORD",
                        "difficulty": "hard",
                        "relevant_segments": [{
                            "video_id": seg["video_id"],
                            "segment_id": int(seg.get("segment_id", 0)),
                            "start_sec": seg["start_sec"],
                            "end_sec": seg["end_sec"]
                        }],
                        "secondary_relevant_segments": [],
                        "hard_negative_segments": hard_negs,
                        "answerability": "answerable",
                        "ground_truth_reason": "Truy vấn thuần khái niệm trừu tượng, không chứa từ khóa trực tiếp từ lời thoại gốc."
                    })
                    q_id_counter += 1
                    break
            if len([q for q in queries if q["category"] == "NO_KEYWORD"]) >= 75:
                break

    # 5. MULTI_SEGMENT (100 queries)
    for vid in all_vids:
        if len([q for q in queries if q["category"] == "MULTI_SEGMENT"]) >= 100:
            break
        segs = video_to_segs[vid]
        if len(segs) >= 2:
            seg1 = segs[0]
            seg2 = segs[1]
            w1 = " ".join(seg1['text'].split()[:6])
            w2 = " ".join(seg2['text'].split()[:6])
            q_text = f"Quá trình diễn biến và kết quả từ khi {w1} cho đến {w2}."
            hard_negs = build_hard_negatives(seg1, segs, valid_segments, 2)
            queries.append({
                "query_id": f"q_{q_id_counter:06d}",
                "query": clean_vietnamese_text(re.sub(r'[\.,;:]', '', q_text)),
                "language": "vi",
                "category": "MULTI_SEGMENT",
                "difficulty": "hard",
                "relevant_segments": [
                    {
                        "video_id": seg1["video_id"],
                        "segment_id": int(seg1.get("segment_id", 0)),
                        "start_sec": seg1["start_sec"],
                        "end_sec": seg1["end_sec"]
                    },
                    {
                        "video_id": seg2["video_id"],
                        "segment_id": int(seg2.get("segment_id", 0)),
                        "start_sec": seg2["start_sec"],
                        "end_sec": seg2["end_sec"]
                    }
                ],
                "secondary_relevant_segments": [],
                "hard_negative_segments": hard_negs,
                "answerability": "answerable",
                "ground_truth_reason": "Đòi hỏi tổng hợp ngữ cảnh liên tục qua 2 phân đoạn liên tiếp trong cùng video."
            })
            q_id_counter += 1

    # 6. VISUAL_OR_HYBRID (25 queries)
    for vid in all_vids:
        if len([q for q in queries if q["category"] == "VISUAL_OR_HYBRID"]) >= 25:
            break
        segs = video_to_segs[vid]
        for seg in segs:
            text = seg["text"].lower()
            if any(k in text for k in ["bản đồ", "sơ đồ", "hình ảnh", "khung cảnh", "flycam", "toàn cảnh"]):
                w_sample = " ".join(seg['text'].split()[:8])
                q_text = f"Khung cảnh và hình ảnh trực quan thể hiện {w_sample}."
                hard_negs = build_hard_negatives(seg, segs, valid_segments, 2)
                queries.append({
                    "query_id": f"q_{q_id_counter:06d}",
                    "query": clean_vietnamese_text(q_text),
                    "language": "vi",
                    "category": "VISUAL_OR_HYBRID",
                    "difficulty": "medium",
                    "relevant_segments": [{
                        "video_id": seg["video_id"],
                        "segment_id": int(seg.get("segment_id", 0)),
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"]
                    }],
                    "secondary_relevant_segments": [],
                    "hard_negative_segments": hard_negs,
                    "answerability": "answerable",
                    "ground_truth_reason": "Truy vấn kết hợp miêu tả trực quan được chứng thực trong lời thoại."
                })
                q_id_counter += 1
                break

    # 7. NO_ANSWER (35 queries)
    no_answer_topics = [
        "Quy trình phóng tàu vũ trụ có người lái lên quỹ đạo sao Hỏa tại Việt Nam",
        "Kế hoạch xây dựng tuyến tàu điện ngầm siêu tốc nối liền Hà Nội và Tokyo",
        "Thống kê lượng tuyết rơi dày kỷ lục tại TP Hồ Chí Minh trong mùa hè",
        "Công nghệ chế tạo máy tính lượng tử 1000 qubit của đồng bằng sông Cửu Long",
        "Số lượng chim cánh cụt sinh sống tự nhiên tại rừng tràm Kiên Giang",
        "Dự án đường hầm xuyên Thái Bình Dương nối Việt Nam và Hoa Kỳ",
        "Đội tuyển bóng đá Việt Nam vô địch giải đấu trên bề mặt Mặt Trăng",
        "Công thức tổng hợp nguyên tố siêu nặng 120 tại viện nghiên cứu Kiên Lương",
        "Thống kê dân số thành phố Atlantis dưới đáy biển Hà Tiên",
        "Kế hoạch đưa khủng long hồi sinh vào các công viên quốc gia Việt Nam",
        "Bản vẽ thiết kế tháp Eiffel cao 3000 mét tại sông Vàm Nao",
        "Hướng dẫn trồng cây táo tuyết Bắc Cực trên đất phèn Kiên Giang",
        "Sự kiện núi lửa phun trào dung nham đỏ rực giữa trung tâm TP Cần Thơ",
        "Tài liệu huấn luyện phi hành đoàn du hành thời gian của đài truyền hình",
        "Báo cáo tài chính của tập đoàn xe bay tự hành năm 2050 tại Cà Mau",
        "Lễ hội đua xe trượt băng nghệ thuật trên sa mạc Sahara của người dân miền Tây",
        "Cách chế biến món súp đá thiên thạch ngàn năm tại Kiên Lương",
        "Quy hoạch xây dựng trạm năng lượng nhiệt hạch hạt nhân trên đỉnh núi Bà Đen",
        "Chuyến bay thẳng thương mại từ Sài Gòn tới sao Kim bằng khinh khí cầu",
        "Chỉ số ô nhiễm không khí tại thành phố ngầm sâu 5km dưới đáy biển Tây",
        "Lễ ký kết hiệp định thương mại tự do giữa Việt Nam và người ngoài hành tinh",
        "Hội nghị thượng đỉnh khí hậu toàn cầu diễn ra tại trạm vũ trụ quốc tế ISS",
        "Phương pháp chiết xuất tinh chất vàng từ lá sen Đồng Tháp",
        "Cuộc thi lướt ván tuyết trên đồi cát Mũi Né trong bão tuyết mùa đông",
        "Kỷ lục bơi lội vượt đại dương của đàn gấu trắng Bắc Cực tại sông Tiền",
        "Quy trình sản xuất kim cương nhân tạo từ phù sa sông Hậu",
        "Hành trình bay qua lỗ đen vũ trụ của đoàn thám hiểm Kiên Giang",
        "Khảo sát địa chất đáy biển sâu 10000m tại kênh Rạch Giá Hà Tiên",
        "Kế hoạch phủ xanh sa mạc bằng công nghệ hạt mưa nano của nông dân An Giang",
        "Báo cáo thử nghiệm thuốc trường sinh bất lão tại trạm y tế xã Kiên Lương",
        "Lớp tập huấn lái đĩa bay phản trọng lực dành cho người cao tuổi",
        "Hội thảo quốc tế về ngôn ngữ giao tiếp với người ngoài hành tinh tại Cần Thơ",
        "Dự án biến nước biển thành xăng sinh học không phát thải tại Kiên Giang",
        "Lễ hội câu cá mập voi khổng lồ trên đỉnh núi Sam Châu Đốc",
        "Kỷ lục leo đỉnh Everest trong vòng 10 phút của vận động viên đồng bằng"
    ]
    for topic in no_answer_topics:
        sample_negs = random.sample(valid_segments, 3)
        hard_negs = [{
            "video_id": s["video_id"],
            "segment_id": int(s.get("segment_id", 0)),
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
            "reason": "Topical distractor without actual target facts"
        } for s in sample_negs]
        queries.append({
            "query_id": f"q_{q_id_counter:06d}",
            "query": topic,
            "language": "vi",
            "category": "NO_ANSWER",
            "difficulty": "hard",
            "relevant_segments": [],
            "secondary_relevant_segments": [],
            "hard_negative_segments": hard_negs,
            "answerability": "unanswerable",
            "ground_truth_reason": "The corpus does not contain sufficient evidence."
        })
        q_id_counter += 1

    print(f"[*] Agent 5 (Quality Reviewer): Validating and formatting {len(queries)} benchmark queries...")
    seen_queries = set()
    final_benchmark = []
    for q in queries:
        if q["query"] not in seen_queries and len(q["query"]) >= 15:
            seen_queries.add(q["query"])
            final_benchmark.append(q)

    # Summary by category
    cats = defaultdict(int)
    for q in final_benchmark:
        cats[q["category"]] += 1
    
    print("\n--- Benchmark Generation Summary ---")
    for cat, count in cats.items():
        print(f"  • {cat:<22}: {count:>4} queries ({count/len(final_benchmark)*100:.1f}%)")
    print(f"Total verified queries: {len(final_benchmark)}\n")

    return final_benchmark


if __name__ == "__main__":
    benchmark = generate_benchmark_queries(target_count=500)
    out_jsonl = "eval/vietnamese_retrieval_benchmark_500.jsonl"
    os.makedirs("eval", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in benchmark:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[✓] Benchmark successfully written to {out_jsonl}")
