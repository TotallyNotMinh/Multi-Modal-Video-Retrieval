#!/usr/bin/env python3
"""
Evaluate and Benchmark Top Candidate LLMs for Vietnamese ASR Transcript Refinement.

Compares:
  1. sail/Sailor2-8B-Chat (Southeast Asian & Vietnamese Specialized)
  2. Qwen/Qwen2.5-7B-Instruct (Global SOTA Multilingual 7B)
  3. SeaLLMs/SeaLLMs-v3-7B-Chat (Southeast Asian Specialized)
  4. Qwen/Qwen3-14B (4-bit Quantized)

Measures inference time, throughput (tokens/sec), tag fidelity, and Vietnamese error correction quality.
"""

import os
import sys
import time
import json
import gc
import re
import argparse
from typing import List, Dict, Any

# Bypass broken torchvision op registrations in Kaggle/custom environments
for _mod in ["torchvision", "torchvision.io", "torchvision.ops", "torchvision._meta_registrations"]:
    sys.modules[_mod] = None

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BENCHMARK_SEGMENTS = [
    (
        "Chào mừng quý vị đến với chương trình 60 giây của đài truyền hình Thành phố Hồ Chí Minh "
        "Chương trình xét này có những thông tin nổi bật sau đây Đông băng sông Cổ Long với tình trạng "
        "sụt lúng gấp gần 20 lần so với nước biển nhân Vận chuyển cáp tốc trái tim từ Hà Nội về Huế ghép "
        "cho bệnh nhân Châu Âu chúng dọi với nhiệt độ nóng như thiêu đốt cùng đám cháy rừng Sụt lúng "
        "đang là vấn đề cấp bách với đông băng sông Cổ Long khi có nơi sụt lúng trung bình lên tới 5,7cm "
        "hột năm Tức là gấp gần 20 lần so với nước biển nhân"
    ),
    (
        "dự báo phần lớn diện tích có thể sẽ nằm dưới nước biển trung bình vào cuối thế kỷ 21. "
        "Chiều 31-7, Hà Đội Cơ quan chuyên môn đã báo cáo Lãnh đạo Bộ Nông nghiệp và Phát triển Nông thôn "
        "về đề án tổng thể phòng trống sụt lúng đất, sạt lở bờ sông, bờ biển, ngập úm, hạn hán, sát nhập mặn "
        "tại đồng bàn sông Cổ Long, đồng thời lấy ý kiến của các nhà khoa học góp ý cho đề án này."
    ),
    (
        "Tại thành phố Cần Thơ, 7 tháng đầu năm 2024, trên địa bàn sẽ ra 24 vụ sạt lở bờ sông tại các quận huyện "
        "Bình Thủy, Ômô, Thốt Nốt, Phong Điền, Cờ Đỏ và Cái Răng, gây thiết hại hơn 14,5 tỷ đồng. "
        "Tổng chức gia bị ảnh hưởng do sạt lở hơn 830 mét, gây thiết hại nặng nề về nhà cửa và tài sản của người dân. "
        "Có 13 căn nhà bị sạt hoàn toàn, 1 nhà kho bị sụp lúng, 34 căn nhà bị sạt một phần hoặc bị ảnh hưởng."
    ),
    (
        "Vẫn chuyện cấp tốc trái tim từ Hà Nội về với ghép cho bệnh nhân suy tim gia đoạn cuối. "
        "Đây là ca ghép tim xuyên việc thứ 11 được thực hiện thành công tại bệnh viện Trung vương Huế."
    ),
]

CANDIDATE_CONFIGS = [
    {
        "id": "sailor2-8b",
        "display_name": "Sailor2-8B-Chat (SEA / Vietnamese SOTA)",
        "model_id": "sail/Sailor2-8B-Chat",
        "quantization": None,
    },
    {
        "id": "qwen2.5-7b",
        "display_name": "Qwen2.5-7B-Instruct (Global SOTA 7B)",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": None,
    },
    {
        "id": "seallms-v3-7b",
        "display_name": "SeaLLMs-v3-7B-Chat (SEA Specialized 7B)",
        "model_id": "SeaLLMs/SeaLLMs-v3-7B-Chat",
        "quantization": None,
    },
    {
        "id": "qwen3-14b-4bit",
        "display_name": "Qwen3-14B (4-bit Quantized)",
        "model_id": "Qwen/Qwen3-14B",
        "quantization": "4bit",
    },
]

SYSTEM_PROMPT = """You are an expert Vietnamese transcript editor specializing in Automatic Speech Recognition (ASR) error correction.
Your task is to refine automatically transcribed Vietnamese speech segments.

CRITICAL RULES:
1. PRESERVE SEGMENT TAGS: Every input <SEGMENT_i> must produce an exact corresponding <SEGMENT_i> output in identical order.
2. CONSERVATIVE CORRECTION:
   - Correct spelling, missing Vietnamese diacritics, broken words, and obvious phonetic homophones.
   - Correct proper nouns (people, places, organizations) or numbers/dates/percentages ONLY when strongly supported by the context of the transcript. Otherwise, preserve the original string.
3. NO PARAPHRASING: Do NOT improve style, alter phrasing, or rewrite grammatically valid sentences. Preserve the speaker's original words and speech style.
4. NO HALLUCINATION: Do NOT add facts, commentary, or conversational filler.
5. NO THINKING TAGS: Do NOT output <think> tags or internal reasoning. Return ONLY the tagged segments."""


def build_prompt(segments: List[str]) -> str:
    blocks = []
    for i, text in enumerate(segments):
        blocks.append(f"<SEGMENT_{i}>\n{text.strip()}\n</SEGMENT_{i}>")
    return "\n".join(blocks)


def parse_tags(output_text: str, count: int) -> Dict[int, str]:
    # Strip thinking tags if present
    clean_text = re.sub(r"<think>[\s\S]*?</think>", "", output_text).strip()
    pattern = re.compile(r"<SEGMENT_(\d+)>([\s\S]*?)</SEGMENT_\1>", re.IGNORECASE)
    matches = pattern.findall(clean_text)
    res = {}
    for idx_str, c in matches:
        try:
            res[int(idx_str)] = c.strip()
        except ValueError:
            pass

    if len(res) < count:
        loose = re.compile(r"<SEGMENT_(\d+)>([\s\S]*?)(?=(?:</SEGMENT_\1>|<SEGMENT_\d+>|$))", re.IGNORECASE)
        for idx_str, c in loose.findall(clean_text):
            try:
                idx = int(idx_str)
                if idx not in res:
                    res[idx] = c.strip()
            except ValueError:
                pass
    return res


def evaluate_single_model(cand: Dict[str, Any], segments: List[str], device: str = "cuda") -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = cand["model_id"]
    display_name = cand["display_name"]
    quant = cand.get("quantization")

    print(f"\n{'='*75}")
    print(f"🔄 Loading & Benchmarking: {display_name}")
    print(f"   Model ID: {model_id} (Quantization: {quant or 'FP16/BF16'})")
    print(f"{'='*75}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        load_kwargs["dtype"] = dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)
        except TypeError:
            if "dtype" in load_kwargs:
                load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)
    model.eval()
    load_time = time.time() - t0
    print(f"✓ Model loaded in {load_time:.2f}s")

    # Format Prompt
    user_content = build_prompt(segments)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Measure Generation Time
    t_gen_start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_gen_end = time.time()
    gen_time = t_gen_end - t_gen_start

    gen_tokens = outputs[0][inputs.input_ids.shape[1] :]
    num_tokens = len(gen_tokens)
    tok_per_sec = num_tokens / max(gen_time, 0.001)

    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    parsed = parse_tags(raw_output, len(segments))

    # Clean up GPU memory immediately
    del model, tokenizer, inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "candidate": cand,
        "load_time_sec": round(load_time, 2),
        "gen_time_sec": round(gen_time, 2),
        "total_tokens": num_tokens,
        "tokens_per_sec": round(tok_per_sec, 1),
        "parsed_segments": parsed,
        "tag_count_match": len(parsed) == len(segments),
        "raw_output": raw_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate top candidate LLMs on Vietnamese transcript refinement.")
    parser.add_argument("--device", type=str, default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=None, help="Filter specific model IDs")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🇻🇳 TOP CANDIDATE VIETNAMESE ASR REFINEMENT ARENA")
    print(f" Device: {args.device} | Benchmark Samples: {len(BENCHMARK_SEGMENTS)} Segments")
    print("=" * 80)

    selected_cands = CANDIDATE_CONFIGS
    if args.models:
        filter_set = set(args.models)
        selected_cands = [c for c in CANDIDATE_CONFIGS if c["id"] in filter_set or c["model_id"] in filter_set]

    results = []
    for cand in selected_cands:
        try:
            res = evaluate_single_model(cand, BENCHMARK_SEGMENTS, device=args.device)
            results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[!] Error evaluating {cand['display_name']}: {e}", file=sys.stderr)

    # Print Detailed Side-by-Side Quality Comparison
    print("\n\n" + "#" * 80)
    print("                  SIDE-BY-SIDE REFINEMENT QUALITY REPORT")
    print("#" * 80)

    for seg_idx, orig_text in enumerate(BENCHMARK_SEGMENTS):
        print(f"\n{'='*80}")
        print(f"📝 TEST SEGMENT [{seg_idx + 1}/{len(BENCHMARK_SEGMENTS)}]")
        print(f"RAW ASR INPUT:")
        print(f"  {orig_text}")
        print(f"{'-'*80}")

        for res in results:
            cand_name = res["candidate"]["display_name"]
            parsed_dict = res["parsed_segments"]
            refined_val = parsed_dict.get(seg_idx, "[PARSE FAILED]")

            print(f"▶ {cand_name}:")
            print(f"  {refined_val}")
            print()
        print("=" * 80)

    # Summary Performance Leaderboard
    print("\n" + "=" * 80)
    print("                     PERFORMANCE & TIMING LEADERBOARD")
    print("=" * 80)
    print(f"{'Model Name':<38} | {'Load (s)':<8} | {'Gen (s)':<8} | {'Speed (tok/s)':<13} | {'Tags':<6}")
    print("-" * 80)
    for res in results:
        cand_name = res["candidate"]["display_name"]
        load_t = res["load_time_sec"]
        gen_t = res["gen_time_sec"]
        tps = res["tokens_per_sec"]
        tags_ok = "PASS" if res["tag_count_match"] else "FAIL"
        print(f"{cand_name:<38} | {load_t:<8.2f} | {gen_t:<8.2f} | {tps:<13.1f} | {tags_ok:<6}")
    print("=" * 80)


if __name__ == "__main__":
    main()
