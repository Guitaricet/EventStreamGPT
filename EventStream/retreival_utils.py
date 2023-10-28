import json
from pathlib import Path

import torch
import torch.nn as nn
import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModel


class Retreiver:
    def __init__(
        self,
        faiss_index_path,
        documents_dir,
        *,
        muse_weights_path=None,
        embedder_name_or_path=None,
        embedder_device="cpu",
        embedder_dtype=None,
        load_in_8bit=False,
    ):
        self.faiss_index_path = faiss_index_path
        self.documents_dir = documents_dir

        self.muse = nn.Identity()
        if muse_weights_path is not None:
            weights: torch.Tensor = torch.load(muse_weights_path)
            self.muse = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
            self.muse.weight.data = weights

        self.tokenizer = None
        self.embedder = None
        embedder_device = embedder_device or embedder_device
        embedder_dtype = embedder_dtype or torch.bfloat16
        if embedder_name_or_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(embedder_name_or_path)
            self.embedder = AutoModel.from_pretrained(embedder_name_or_path, load_in_8bit=load_in_8bit)
            self.embedder.to(dtype=embedder_dtype, device=embedder_device)

        self.faiss_index = faiss.read_index(self.faiss_index_path)

        self.documents = []
        for shard in sorted(Path(self.documents_dir).glob("node_*.jsonl")):
            with open(shard, "r") as f:
                for line in f:
                    self.documents.append(json.loads(line))

    @classmethod
    def from_config(cls, config, dtype=torch.bfloat16):
        return cls(
            faiss_index_path=config.faiss_index_path,
            documents_dir=config.documents_dir,
            muse_weights_path=config.muse_weights_path,
            embedder_name_or_path=config.embedder_name_or_path,
            embedder_device=config.embedder_device,
            embedder_dtype=dtype,
            load_in_8bit=config.load_in_8bit,
        )

    @torch.no_grad()
    def get_documents_and_embed(self, queries, n_neighbours=1):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu()

        if len(queries.shape) == 1:
            queries.unsqueeze_(0)

        queries = self.muse(queries)

        queries = queries.numpy().astype(np.float32)
        if queries.shape != (len(queries), self.faiss_index.d):
            raise RuntimeError(f"Expected queries to be of shape (n_queries, {self.faiss_index.d}), got {queries.shape}")

        _, indices = self.faiss_index.search(queries, n_neighbours)
        
        documents = [self.documents[i] for i in indices.flatten()]

        if self.embedder is None:
            raise RuntimeError(f"To embed documents, provide embedder_name_or_path to the constructor of {self.__class__.__name__}")
        
        texts = [doc["text"] for doc in documents]
        return None, None
        # inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.embedder.device)
        # embeddings = self.embedder(**inputs).last_hidden_state
        # seq_len = embeddings.shape[-2]
        # embeddings = embeddings.reshape(len(queries), n_neighbours, seq_len, embeddings.shape[-1])
        # return documents, embeddings
