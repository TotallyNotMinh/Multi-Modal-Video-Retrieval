import torch
import numpy as np
from typing import List, Union
import open_clip

class CLIPTextEncoder:
    """
    Encodes English text strings into 512-dimensional normalized vectors
    using CLIP ViT-B/32 (matching the visual features provided in AIC 2026).
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[CLIPTextEncoder] Loading {model_name} ({pretrained}) on {self.device}...")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("[CLIPTextEncoder] Ready.")

    @torch.inference_mode()
    def encode_text(self, texts: Union[str, List[str]], ensemble: bool = True) -> np.ndarray:
        """
        Returns normalized 512-dim embedding vector.
        If texts is a list and ensemble=True, averages normalized prompt vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(tokens)
        
        # Robust L2 Normalize with zero-clamp
        norms = text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_features = text_features / norms
        feats_np = text_features.cpu().numpy().astype(np.float32)

        if ensemble and len(texts) > 1:
            mean_vec = np.mean(feats_np, axis=0, keepdims=True)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-12:
                mean_vec = mean_vec / norm
            return mean_vec[0]
            
        return feats_np[0] if len(texts) == 1 else feats_np
