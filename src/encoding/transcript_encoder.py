"""
Vietnamese Transcript Semantic Encoder using Multilingual E5 / Bi-Encoder models.
Provides dense text embeddings for Vietnamese speech transcripts and user search queries.
"""

import os
import time
from typing import List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TranscriptEncoder:
    """
    Dense Vietnamese/Multilingual Text Encoder for speech transcripts and search queries.
    Uses intfloat/multilingual-e5-large (1024-dim) with GPU acceleration and mean pooling.
    """

    def __init__(
        self,
        model_name_or_path: str = "intfloat/multilingual-e5-large",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: Optional[int] = None,
        use_fp16: bool = True
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self.max_length = max_length
        self.use_fp16 = use_fp16 and (self.device == "cuda" or "cuda" in str(self.device))
        self.batch_size = batch_size if batch_size is not None else (128 if self.device == "cuda" else 32)

        print(f"[TranscriptEncoder] Loading '{model_name_or_path}' on device '{self.device}' (fp16={self.use_fp16}, batch_size={self.batch_size})...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.is_e5 = "e5" in model_name_or_path.lower()
        self.dim = self.model.config.hidden_size
        print(f"[TranscriptEncoder] Loaded encoder (dim={self.dim}) in {time.time() - t0:.2f}s.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.inference_mode()
    def encode_passages(self, passages: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encodes a list of transcript passages / segments.
        Prefixes with 'passage: ' if using an E5 model.
        Returns (N, dim) normalized float32 numpy array.
        """
        if not passages:
            return np.empty((0, self.dim), dtype=np.float32)

        bs = batch_size or self.batch_size
        formatted = [
            f"passage: {p.strip()}" if (self.is_e5 and not p.startswith("passage:")) else p.strip()
            for p in passages
        ]

        embeddings = np.empty((len(passages), self.dim), dtype=np.float32)

        for i in range(0, len(formatted), bs):
            batch_texts = formatted[i : i + bs]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
            normalized = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings[i : i + len(batch_texts)] = normalized.cpu().numpy()

        return embeddings

    @torch.inference_mode()
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single search query string.
        Prefixes with 'query: ' if using an E5 model.
        Returns (dim,) normalized float32 numpy array.
        """
        formatted = f"query: {query.strip()}" if (self.is_e5 and not query.startswith("query:")) else query.strip()
        encoded_input = self.tokenizer(
            [formatted],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        normalized = F.normalize(sentence_embeddings, p=2, dim=1)
        return normalized.cpu().numpy().squeeze(0).astype(np.float32)
