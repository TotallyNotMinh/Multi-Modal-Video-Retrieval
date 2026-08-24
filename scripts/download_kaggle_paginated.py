#!/usr/bin/env python3
"""
Automated Paginated Kaggle Kernel Output Downloader:
- Automatically extracts the next page token from output logs.
- Continues fetching next pages in a loop until all files are downloaded.
- Retries with exponential backoff on network timeouts.
- Persists the latest page token to 'cache/last_kaggle_page_token.txt' for instant resume.
"""

import os
import sys
import re
import time
import shutil
import argparse
import subprocess

STATE_FILE = "cache/last_kaggle_page_token.txt"

def find_kaggle_executable():
    which_path = shutil.which("kaggle")
    if which_path and os.path.exists(which_path):
        return which_path
    
    candidates = [
        os.path.expanduser("~/miniconda3/bin/kaggle"),
        os.path.expanduser("~/anaconda3/bin/kaggle"),
        os.path.expanduser("~/.local/bin/kaggle"),
        "/usr/local/bin/kaggle",
        "/usr/bin/kaggle"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "kaggle"

def download_paginated(
    kernel: str,
    output_dir: str,
    initial_token: str = None,
    max_retries: int = 5,
    kaggle_bin: str = None
):
    if kaggle_bin is None:
        kaggle_bin = find_kaggle_executable()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    current_token = initial_token
    if not current_token and os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = f.read().strip()
            if saved:
                current_token = saved
                print(f"[*] Resuming from saved page token in {STATE_FILE}")

    page_num = 1
    retry_count = 0

    print(f"[*] Starting automated paginated download for: {kernel}")
    print(f"[*] Destination directory: {os.path.abspath(output_dir)}")
    print(f"[*] Using Kaggle binary: {kaggle_bin}\n")

    token_regex = re.compile(r"Next page token:\s*([A-Za-z0-9+/=_]+)")

    while True:
        cmd = [kaggle_bin, "kernels", "output", kernel, "-p", output_dir]
        if current_token:
            cmd.extend(["--page-token", current_token])

        token_preview = f"{current_token[:25]}..." if current_token else "START (page 1)"
        print(f"\n{'='*70}")
        print(f"📥 [Page {page_num}] Fetching with token: {token_preview}")
        print(f"{'='*70}")

        next_token = None
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        import select
        poller = select.poll()
        poller.register(process.stdout, select.POLLIN)

        timed_out = False
        t_start = time.time()
        last_activity = time.time()
        line_timeout = 60.0   # 60s per file / response start
        total_timeout = 240.0 # 4 minutes max per 20-file batch

        while True:
            if process.poll() is not None:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    match = token_regex.search(line)
                    if match:
                        next_token = match.group(1).strip()
                break

            events = poller.poll(500)
            if events:
                line = process.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    last_activity = time.time()
                    match = token_regex.search(line)
                    if match:
                        next_token = match.group(1).strip()

            if (time.time() - last_activity) > line_timeout or (time.time() - t_start) > total_timeout:
                print(f"\n[!] Kaggle network connection stalled (>20s idle). Terminating stalled socket...")
                timed_out = True
                process.kill()
                process.wait()
                break

        if process.returncode == 0 and not timed_out:
            retry_count = 0
            if next_token and next_token != current_token:
                current_token = next_token
                page_num += 1
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    f.write(current_token)
                time.sleep(1.0)
            else:
                print(f"\n{'='*70}")
                print(f"🎉 [✓] All pages downloaded successfully! Finished at page {page_num}.")
                print(f"{'='*70}")
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)
                break
        else:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\n[!] Error: Exceeded maximum retries ({max_retries}) on page {page_num}.")
                print(f"[!] Last saved token: {current_token}")
                sys.exit(1)

            wait_time = 2 ** min(retry_count, 4)
            print(f"\n[!] Page {page_num} stalled or dropped. Retrying same token in {wait_time}s (Attempt {retry_count}/{max_retries})...")
            time.sleep(wait_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated paginated downloader for Kaggle kernel outputs.")
    parser.add_argument("--kernel", type=str, default="knuckleizmad/notebook1173dac344", help="Kaggle kernel slug")
    parser.add_argument("-p", "--output-dir", type=str, default="output", help="Target output folder")
    parser.add_argument("--page-token", type=str, default=None, help="Specific starting page token")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per page on failure")
    parser.add_argument("--force-start", action="store_true", help="Ignore saved token and start from beginning")
    args = parser.parse_args()

    if args.force_start and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    download_paginated(
        kernel=args.kernel,
        output_dir=args.output_dir,
        initial_token=args.page_token,
        max_retries=args.max_retries
    )
