import os
import json
import subprocess
import time

UPLOADS = [
    {
        "zip_file": "aic2026_part2_L25_L27.zip",
        "title": "AIC 2026 Part 2 L25 L27",
        "slug": "aic2026-part2-l25-l27"
    },
    {
        "zip_file": "aic2026_part3_L26ab_L30.zip",
        "title": "AIC 2026 Part 3 L26ab L30",
        "slug": "aic2026-part3-l26ab-l30"
    },
    {
        "zip_file": "aic2026_part4_L26cd.zip",
        "title": "AIC 2026 Part 4 L26cd",
        "slug": "aic2026-part4-l26cd"
    },
    {
        "zip_file": "aic2026_part5_L26e_L28.zip",
        "title": "AIC 2026 Part 5 L26e L28",
        "slug": "aic2026-part5-l26e-l28"
    },
    {
        "zip_file": "aic2026_part6_L29.zip",
        "title": "AIC 2026 Part 6 L29",
        "slug": "aic2026-part6-l29"
    }
]

def prepare_and_upload(username: str = "knuckleizmad", uploads_root: str = "kaggle_uploads", zips_root: str = "kaggle_zips"):
    os.makedirs(uploads_root, exist_ok=True)

    for item in UPLOADS:
        zip_name = item["zip_file"]
        title = item["title"]
        slug = item["slug"]
        
        src_zip = os.path.join(zips_root, zip_name)
        if not os.path.exists(src_zip):
            print(f"❌ Warning: {src_zip} not found! Skipping...")
            continue

        folder = os.path.join(uploads_root, slug)
        os.makedirs(folder, exist_ok=True)

        meta_path = os.path.join(folder, "dataset-metadata.json")
        meta = {
            "title": title,
            "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        dst_zip = os.path.join(folder, zip_name)
        if not os.path.exists(dst_zip):
            try:
                os.link(src_zip, dst_zip)  # Fast zero-copy hardlink
            except Exception:
                import shutil
                shutil.copy2(src_zip, dst_zip)

        print(f"\n=======================================================")
        print(f"🚀 Uploading: {title} ({slug})")
        print(f"=======================================================")
        
        cmd = ["kaggle", "datasets", "create", "-p", folder]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✅ Successfully initiated upload for {title}")
        else:
            print(f"⚠️ Non-zero exit for {title}. Checking if update is needed...")
            # If dataset already exists, try version update
            cmd_update = ["kaggle", "datasets", "version", "-p", folder, "-m", "Initial upload"]
            subprocess.run(cmd_update)

if __name__ == "__main__":
    prepare_and_upload()
