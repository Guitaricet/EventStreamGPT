import json
from pathlib import Path

import torch
import torch.nn as nn
import faiss
import numpy as np

from datasets import Dataset


class Retreiver:
    def __init__(
        self,
        *,
        faiss_index_path,
        dataset_path,
        muse_weights_path=None,
    ):
        """Assumes that faiss_index_path indexes the dataset at dataset_path.
        The dataset should contain a field `hidden_state` of shape [seq_len, hidden] for each document.
        Seq_len is assumed to be a constant across all documents.

        You can get such a dataset by running scripts/compute_embeddings.py.
        """
        self.faiss_index_path = faiss_index_path
        self.dataset_path = dataset_path

        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.dataset = Dataset.load_from_disk(dataset_path)

        self.muse = nn.Identity()
        if muse_weights_path is not None:
            weights: torch.Tensor = torch.load(muse_weights_path)
            self.muse = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
            self.muse.weight.data = weights

    @classmethod
    def from_config(cls, config):
        return cls(
            faiss_index_path=config.faiss_index_path,
            dataset_path=config.dataset_path,
            muse_weights_path=config.muse_weights_path,
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

        doc_ids = indices.flatten()
        embeddings = np.stack(self.dataset[doc_ids]["hidden_state"])

        seq_len = embeddings.shape[-2]
        embeddings = embeddings.reshape(len(queries), n_neighbours, seq_len, embeddings.shape[-1])
        documents = [self.dataset[i] for i in indices.flatten()]
        return documents, embeddings
