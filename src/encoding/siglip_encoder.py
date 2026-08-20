import torch
import numpy as np
import gc
from PIL import Image
from typing import List, Union, Optional
from transformers import AutoProcessor, AutoModel


class SigLIPEncoder:
    """
    High-resolution Vision and Text Encoder using Google SigLIP (SO400M-patch14-384).
    Outputs 1152-dimensional L2-normalized embeddings.

    VRAM Maximization & Adaptive OOM Auto-Recovery:
      - Default batch size (256 / 128) saturates 15GB GPU memory for maximum FP16 throughput.
      - Automatic GPU Out-Of-Memory (OOM) auto-recovery: when peak VRAM limits are reached,
        the sub-batch is dynamically halved (256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1) and retried.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_fp16 = use_fp16 and ("cuda" in str(self.device)) and torch.cuda.is_available()
        self.model_name = model_name

        print(f"[SigLIPEncoder] Loading {model_name} on {self.device} (FP16={self.use_fp16})...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        if self.use_fp16:
            self.model = self.model.half()

        self.model.to(self.device)
        self.model.eval()
        print(f"[SigLIPEncoder] Model loaded successfully.")

    def _encode_single_chunk(self, batch_images: List[Image.Image]) -> np.ndarray:
        """
        Internal chunk encoder with automatic GPU OOM handling and recursive sub-batch halving.
        """
        try:
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            if self.use_fp16:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            image_features = self.model.get_image_features(**inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(image_features, "pooler_output", image_features[0])
            norms = image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            image_features = image_features / norms
            out = image_features.cpu().numpy().astype(np.float16 if self.use_fp16 else np.float32)

            del inputs, image_features, norms
            return out
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"[SigLIPEncoder] ⚠️ GPU OutOfMemory caught on sub-batch size {len(batch_images)}! Halving and retrying...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if len(batch_images) <= 1:
                    raise RuntimeError("OOM even with sub-batch_size=1 on GPU.")

                mid = len(batch_images) // 2
                part1 = self._encode_single_chunk(batch_images[:mid])
                part2 = self._encode_single_chunk(batch_images[mid:])
                return np.vstack([part1, part2])
            else:
                raise e

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        batch_size: int = 256
    ) -> np.ndarray:
        """
        Encodes a list of PIL Images or RGB numpy arrays into (N, 1152) float16 / float32 array.
        Maximizes VRAM utilization and automatically halves sub-batches upon OOM.
        """
        if not images:
            return np.empty((0, 1152), dtype=np.float32)

        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        all_embeddings = []
        total = len(pil_images)

        for start_idx in range(0, total, batch_size):
            batch = pil_images[start_idx : start_idx + batch_size]
            emb = self._encode_single_chunk(batch)
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)

    @torch.inference_mode()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        ensemble: bool = True
    ) -> np.ndarray:
        """
        Encodes English text strings into normalized 1152-dim vector.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        text_features = self.model.get_text_features(**inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "pooler_output", text_features[0])
        norms = text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_features = text_features / norms
        feats_np = text_features.cpu().numpy().astype(np.float32)

        del inputs, text_features, norms

        if ensemble and len(texts) > 1:
            mean_vec = np.mean(feats_np, axis=0, keepdims=True)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-12:
                mean_vec = mean_vec / norm
            return mean_vec[0]

        return feats_np[0] if len(texts) == 1 else feats_np
