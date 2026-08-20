import os
import zipfile
import time
from tqdm import tqdm

GROUPS = {
    "aic2026_part1_L21_L24_meta": [
        "data/Videos_L21_a",
        "data/Videos_L22_a",
        "data/Videos_L23_a",
        "data/Videos_L24_a",
        "data/media-info-aic25-b1",
        "data/map-keyframes-aic25-b1",
    ],
    "aic2026_part2_L25_L27": [
        "data/Videos_L25_a",
        "data/Videos_L27_a",
    ],
    "aic2026_part3_L26ab_L30": [
        "data/Videos_L26_a",
        "data/Videos_L26_b",
        "data/Videos_L30_a",
    ],
    "aic2026_part4_L26cd": [
        "data/Videos_L26_c",
        "data/Videos_L26_d",
    ],
    "aic2026_part5_L26e_L28": [
        "data/Videos_L26_e",
        "data/Videos_L28_a",
    ],
    "aic2026_part6_L29": [
        "data/Videos_L29_a",
    ]
}

def create_kaggle_zip(group_name: str, target_paths: list, output_dir: str = "kaggle_zips"):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{group_name}.zip")

    # Collect all files
    files_to_zip = []
    for item in target_paths:
        if os.path.isfile(item):
            files_to_zip.append((item, os.path.relpath(item, "data")))
        elif os.path.isdir(item):
            for root, _, files in os.walk(item):
                for f in files:
                    full_p = os.path.join(root, f)
                    arc_p = os.path.relpath(full_p, ".")
                    files_to_zip.append((full_p, arc_p))

    print(f"\n📦 Creating {zip_path} ({len(files_to_zip)} files)...")
    t0 = time.time()

    # Use ZIP_STORED because MP4 videos are already compressed; saves hours of CPU time
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for full_p, arc_p in tqdm(files_to_zip, desc=f"Zipping {group_name}"):
            zf.write(full_p, arcname=arc_p)

    size_gb = os.path.getsize(zip_path) / (1024 ** 3)
    elapsed = time.time() - t0
    print(f"✅ Finished {group_name}.zip: {size_gb:.2f} GB in {elapsed:.1f}s")
    return zip_path, size_gb

def main():
    output_dir = "kaggle_zips"
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for grp_name, paths in GROUPS.items():
        zip_p, size_gb = create_kaggle_zip(grp_name, paths, output_dir=output_dir)
        summary.append((grp_name, size_gb))

    print("\n" + "=" * 50)
    print("🎉 ALL KAGGLE DATASET ZIP PACKAGES CREATED:")
    print("=" * 50)
    for name, sz in summary:
        print(f"  • {name}.zip: {sz:.2f} GB (Under 20 GB limit ✅)")
    print("=" * 50)

if __name__ == "__main__":
    main()
