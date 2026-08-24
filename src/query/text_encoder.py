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


class SigLIPTextEncoder:
    """
    Encodes English text strings into 1152-dimensional normalized vectors
    using Google SigLIP (SO400M-patch14-384).
    """
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import AutoTokenizer, SiglipTextModel

        print(f"[SigLIPTextEncoder] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SiglipTextModel.from_pretrained(model_name, low_cpu_mem_usage=False).to(self.device)
        self.model.eval()
        print("[SigLIPTextEncoder] Ready.")

    @torch.inference_mode()
    def encode_text(self, texts: Union[str, List[str]], ensemble: bool = True) -> np.ndarray:
        """
        Returns normalized 1152-dim embedding vector.
        If texts is a list and ensemble=True, averages normalized prompt vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        text_features = self.model(**inputs).pooler_output
        text_features = F.normalize(text_features, p=2, dim=-1)

        if ensemble and len(texts) > 1:
            mean_feat = text_features.mean(dim=0, keepdim=True)
            mean_feat = F.normalize(mean_feat, p=2, dim=-1)
            return mean_feat.squeeze(0).cpu().numpy().astype(np.float32)

        feats_np = text_features.cpu().numpy().astype(np.float32)
        return feats_np[0] if len(texts) == 1 else feats_np

