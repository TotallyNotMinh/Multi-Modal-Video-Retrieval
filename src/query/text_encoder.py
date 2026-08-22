import torch
import torch.nn.functional as F
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
        
        # GPU L2 Normalize
        text_features = F.normalize(text_features, p=2, dim=-1)

        if ensemble and len(texts) > 1:
            mean_feat = text_features.mean(dim=0, keepdim=True)
            mean_feat = F.normalize(mean_feat, p=2, dim=-1)
            return mean_feat.squeeze(0).cpu().numpy().astype(np.float32)

        feats_np = text_features.cpu().numpy().astype(np.float32)
        return feats_np[0] if len(texts) == 1 else feats_np

