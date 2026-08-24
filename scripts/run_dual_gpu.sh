#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "🚀 Dual-GPU SigLIP-SO400M Extraction Pipeline"
echo "======================================================================"

mkdir -p cache/logs cache/siglip_features cache/siglip_meta

MANIFEST="manifests/encoded_siglip_videos.txt"
if [ ! -f "$MANIFEST" ]; then
    echo "[*] Manifest $MANIFEST not found. Checking cache..."
    python -c "
import glob, os
vids = sorted(glob.glob('cache/siglip_features/*.npy'))
os.makedirs('manifests', exist_ok=True)
with open('', 'w') as f:
    for v in vids:
        f.write(os.path.splitext(os.path.basename(v))[0] + '
')
"
fi

NUM_DONE=$(wc -l < "$MANIFEST")
echo "[*] Total completed videos in manifest: $NUM_DONE"

# 1. Warm up Hugging Face model cache in single process first
echo "[*] Pre-caching SigLIP-SO400M weights..."
python -c "
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoImageProcessor, SiglipVisionModel
AutoImageProcessor.from_pretrained('google/siglip-so400m-patch14-384')
SiglipVisionModel.from_pretrained('google/siglip-so400m-patch14-384', low_cpu_mem_usage=False)
print('[✓] Model cached successfully.')
"

# 2. Launch isolated GPU processes
echo ""
echo "[*] Spawning GPU 0 process (CUDA:0, Shard 0/2)..."
python -u scripts/extract_siglip_features.py \
    --num-shards 2 \
    --shard-id 0 \
    --device cuda:0 \
    --batch-size 64 \
    --exclude-list "$MANIFEST" > cache/logs/gpu_0.log 2>&1 &
PID0=$!

sleep 3

echo "[*] Spawning GPU 1 process (CUDA:1, Shard 1/2)..."
python -u scripts/extract_siglip_features.py \
    --num-shards 2 \
    --shard-id 1 \
    --device cuda:1 \
    --batch-size 64 \
    --exclude-list "$MANIFEST" > cache/logs/gpu_1.log 2>&1 &
PID1=$!

echo "[✓] GPU 0 (PID: $PID0) and GPU 1 (PID: $PID1) running in background."
echo ""
echo "======================================================================"
echo "📊 Live Progress Monitor"
echo "======================================================================"

T0=$(date +%s)
while kill -0 $PID0 2>/dev/null || kill -0 $PID1 2>/dev/null; do
    sleep 15
    NOW=$(date +%s)
    ELAPSED=$(( (NOW - T0) / 60 ))
    TOTAL_NPY=$(ls cache/siglip_features/*.npy 2>/dev/null | wc -l || echo 0)
    
    L0=$(tail -n 1 cache/logs/gpu_0.log 2>/dev/null | tr '' '
' | tail -n 1 | cut -c 1-85 || echo "Initializing...")
    L1=$(tail -n 1 cache/logs/gpu_1.log 2>/dev/null | tr '' '
' | tail -n 1 | cut -c 1-85 || echo "Initializing...")
    
    echo "[${ELAPSED}m] Total Encoded: ${TOTAL_NPY}/873"
    echo "   • GPU 0: $L0"
    echo "   • GPU 1: $L1"
    echo "----------------------------------------------------------------------"
done

wait $PID0 || true
wait $PID1 || true

TOTAL_FINAL=$(ls cache/siglip_features/*.npy 2>/dev/null | wc -l || echo 0)
echo ""
echo "[🎉] Extraction Complete! Total videos encoded: ${TOTAL_FINAL}/873"

# 3. Create single archive
echo "[*] Creating unified download archive..."
tar -czf /kaggle/working/siglip_features_completed.tar.gz -C cache siglip_features siglip_meta 2>/dev/null || tar -czf siglip_features_completed.tar.gz -C cache siglip_features siglip_meta 2>/dev/null || true
echo "[✓] Archive created: siglip_features_completed.tar.gz"
ls -lh *siglip_features_completed.tar.gz /kaggle/working/siglip_features_completed.tar.gz 2>/dev/null || true
