#!/usr/bin/env python3
import os, sys, glob, json, argparse

def json_to_txt(json_dir="asr_transcripts/cache/asr_transcripts", txt_dir="asr_transcripts/cache/asr_text"):
    os.makedirs(txt_dir, exist_ok=True)
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    print(f"[JSON -> TXT] Converting {len(json_files)} files from {json_dir} to {txt_dir}...")
    count = 0
    for j_path in json_files:
        vid_name = os.path.splitext(os.path.basename(j_path))[0]
        out_txt = os.path.join(txt_dir, f"{vid_name}.txt")
        try:
            with open(j_path, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
            t_lines = []
            for seg in data:
                st = seg.get("start_sec", 0.0)
                et = seg.get("end_sec", 0.0)
                sf = seg.get("start_frame", int(round(st * 25.0)))
                ef = seg.get("end_frame", int(round(et * 25.0)))
                txt = (seg.get("text") or "").strip()
                t_lines.append(f"{st} | {et} | {sf} | {ef} | {txt}")
            with open(out_txt, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(t_lines) + "\n")
            count += 1
        except Exception as e:
            print(f"Error converting {j_path}: {e}")
    print(f"Ð Successfully converted {count} files to TXT format!")

def txt_to_json(txt_dir="asr_transcripts/cache/asr_text", json_dir="asr_transcripts/cache/asr_transcripts_restored"):
    os.makedirs(json_dir, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    print(f"[TXT -> JSON] Restoring {len(txt_files)} files from {txt_dir} to {json_dir}...")
    count = 0
    for t_path in txt_files:
        vid_name = os.path.splitext(os.path.basename(t_path))[0]
        out_json = os.path.join(json_dir, f"{vid_name}.json")
        segments = []
        try:
            with open(t_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line: continue
                    parts = [p.strip() for p in line.split("|", 4)]
                    if len(parts) == 5:
                        st, et, sf, ef, text = parts
                        segments.append({
                            "video_id": vid_name,
                            "start_sec": float(st),
                            "end_sec": float(et),
                            "start_frame": int(sf),
                            "end_frame": int(ef),
                            "text": text
                        })
            with open(out_json, "w", encoding="utf-8") as f_out:
                json.dump(segments, f_out, indent=2, ensure_ascii=False)
            count += 1
        except Exception as e:
            print(f"Error restoring {t_path}: {e}")
    print(f" Successfully restored {count} files to JSON format!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-way converter between AIR JSON transcripts and structured TXT files.")
    parser.add_argument("--mode", choices=["json2txt", "txt2json"], default="json2txt")
    parser.add_argument("--src", default=None)
    parser.add_argument("--dst", default=None)
    args = parser.parse_args()
    if args.mode == "json2txt":
        src = args.src or "asr_transcripts/cache/asr_transcripts"
        dst = args.dst or "asr_transcripts/cache/asr_text"
        json_to_txt(src, dst)
    else:
        src = args.src or "asr_transcripts/cache/asr_text"
        dst = args.dst or "asr_transcripts/cache/asr_transcripts_restored"
        txt_to_json(src, dst)
