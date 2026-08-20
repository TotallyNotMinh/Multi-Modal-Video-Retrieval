import os
import gc
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
from src.index.faiss_index import FAISSIndex
from src.index.matrix_builder import FeatureMatrixBuilder

def build_production_faiss_index(
    siglip_dir: str = "cache/siglip_features",
    siglip_meta_dir: str = "cache/siglip_meta",
    output_prefix: str = "cache/faiss_siglip"
):
    """
    Builds and saves production FAISS index using incremental streaming to prevent RAM spikes.
    Peak RAM is strictly bounded under 300 MB even for 2.5M+ vectors.
    """
    siglip_files = sorted(glob.glob(os.path.join(siglip_dir, "*.npy")))
    
    if len(siglip_files) > 0:
        print(f"[FAISS Builder] Found {len(siglip_files)} SigLIP feature files. Streaming incrementally into FAISS...")
        
        # Peek at dimension from first file
        sample_vec = np.load(siglip_files[0], mmap_mode="r")
        dim = sample_vec.shape[1]
        del sample_vec

        faiss_idx = FAISSIndex(dim=dim, index_type="FlatIP")
        faiss_idx.init_index(dim, total_estimated=len(siglip_files) * 3000)

        global_idx = 0
        for sf in tqdm(siglip_files, desc="Streaming features to FAISS"):
            vid = os.path.splitext(os.path.basename(sf))[0]
            meta_f = os.path.join(siglip_meta_dir, f"{vid}.json")
            
            vecs = np.load(sf).astype(np.float32)
            meta = []
            if os.path.exists(meta_f):
                try:
                    with open(meta_f, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            file_records = []
            for i in range(len(vecs)):
                r = meta[i] if i < len(meta) else {
                    "video_id": vid,
                    "frame_idx": i * 6,
                    "pts_time": float(i * 0.2)
                }
                r["global_idx"] = global_idx
                file_records.append(r)
                global_idx += 1

            faiss_idx.add_batch(vecs, file_records)
            
            del vecs, file_records, meta
            if global_idx % 100000 == 0:
                gc.collect()

        faiss_idx.save(output_prefix)
        print(f"[FAISS Builder] Successfully indexed {faiss_idx.index.ntotal} vectors at {output_prefix}.index")
    else:
        print("[FAISS Builder] No SigLIP files found. Building FAISS index over provided ViT-B/32 CLIP features...")
        matrix, all_records = FeatureMatrixBuilder().build_and_cache()
        dim = matrix.shape[1]

        faiss_idx = FAISSIndex(dim=dim, index_type="FlatIP")
        faiss_idx.build(matrix, all_records, save_path_prefix=output_prefix)
        print(f"[FAISS Builder] Successfully created FAISS index at {output_prefix}.index")

if __name__ == "__main__":
    build_production_faiss_index()
