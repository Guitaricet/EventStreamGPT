#!/usr/bin/env python
"""Gets the embeddings of a pre-trained model for a user-specified fine-tuning dataset."""

try:
    import stackprinter
    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass

import hydra
import torch
from EventStream.transformer.lightning_modules.embedding import FinetuneConfig, get_embeddings

torch.set_float32_matmul_precision("high")

def train(cfg: PretrainConfig):
    logger.info("Starting training")
    seed_everything(cfg.seed)

    device = "cuda"
    dtype = torch.float32
    global_rank = 0
    torch.multiprocessing.set_sharing_strategy("file_system")

    model_config: StructuredTransformerConfig = cfg.model_config
    data_config: PytorchDatasetConfig = cfg.data_config
    retriever_config: RetrieverConfig = cfg.retriever_config

    logger.info("Building train dataset")
    train_dataset = PytorchDataset(cfg.data_config, split="train")

    if global_rank != 0:
        logger.remove()

    model: ConditionallyIndependentRetrievalAugTransformer = get_model(model_config)
    model = model.to(dtype=dtype, device=device)
    retriever: EventStream.Retriever = EventStream.Retriever.from_config(retriever_config, dtype=dtype)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_config.per_device_train_batch_size,
        num_workers=model_config.num_dataloader_workers,
        collate_fn=train_dataset.collate,
        shuffle=True,
    )

    chunk_len = model_config.chunked_cross_attention_chunk_len
    batch_size = model_config.per_device_train_batch_size
    n_neighbours = retriever_config.n_neighbours

    embeddings_list = []

    for batch in train_dataloader:
        batch = batch.to(dtype=dtype, device=device)
        retriever_query = model.first_half_forward(batch).last_hidden_state
        n_chunks = retriever_query.shape[1] // chunk_len

        retrieval_queries = model.reshape_to_retrieval_queries(retriever_query, allow_padding=True)

        assert retrieval_queries.shape == (batch_size * n_chunks, model_config.hidden_size)
        _, hidden = retriever.get_documents_and_embed(
            retrieval_queries,
            n_neighbours=n_neighbours,
        )

        embeddings_list.append(hidden)

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    torch.save(embeddings_tensor, cfg.data_config.save_dir / "embeddings.pth")
    
    logger.info("Embeddings saved successfully!")
    logger.info("Script finished successfully!")

@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    
    return train(cfg)

if __name__ == "__main__":
    main()
