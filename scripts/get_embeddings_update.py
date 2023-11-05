#!/usr/bin/env python
"""Gets the embeddings of a pre-trained model for a user-specified fine-tuning dataset."""

import hydra
import torch
from loguru import logger

from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.data.pytorch_dataset import PytorchDataset
from EventStream.config import PretrainConfig
from EventStream.transformer.modeling_retro import ConditionallyIndependentRetreivalAugTransformer
from EventStream.transformer.conditionally_independent_model import CondIndepModelForGenerativeSequenceModeling


def get_embeddings(cfg: PretrainConfig):
    logger.info("Starting training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    model_config: StructuredTransformerConfig = cfg.model_config

    logger.info("Building train dataset")
    train_dataset = PytorchDataset(cfg.data_config, split="train")

    from_pretrained = None
    if from_pretrained is not None:
        model = CondIndepModelForGenerativeSequenceModeling.from_pretrained(from_pretrained)
    else:
        model = CondIndepModelForGenerativeSequenceModeling(model_config)
        if cfg.model_weights_path and Path(cfg.model_weights_path).is_file():
            logger.info(f"Loading model weights from {cfg.model_weights_path}")
            model = CondIndepModelForGenerativeSequenceModeling(cfg.model_weights_path)
        else:
            logger.info("Initializing model from configuration")
            model = CondIndepModelForGenerativeSequenceModeling(model_config)
    

    model: ConditionallyIndependentRetreivalAugTransformer = model.to(dtype=dtype, device=device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_config.per_device_train_batch_size,
        num_workers=model_config.num_dataloader_workers,
        collate_fn=train_dataset.collate,
        shuffle=False,
    )

    chunk_len = model_config.chunked_cross_attention_chunk_len
    batch_size = model_config.per_device_train_batch_size

    save_dir = cfg.data_config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    batch_counter = 0
    embeddings_list = []
    save_every_n_batches = 10

    for batch in train_dataloader:
        batch = batch.to(dtype=dtype, device=device)
        retriever_query = model.first_half_forward(batch).last_hidden_state
        n_chunks = retriever_query.shape[1] // chunk_len

        retriever_query = model.reshape_to_retreival_queries(retriever_query, allow_padding=True)
        assert retriever_query.shape == (batch_size * n_chunks, model_config.hidden_size)

        # VERY BIG ASSUMPTION: we assume that the way we align events is that
        # given an event from the dataset with associated text, we look for
        # an ESGPT sequence that ends on this event
        # In this case we only need the last chunk of the retriever query
        retriever_query = retriever_query.reshape(batch_size, n_chunks, model_config.hidden_size)
        retriever_query = retriever_query[:, -1, :]

        # Move the tensor to CPU before appending
        embeddings_list.append(retriever_query.cpu())

        batch_counter += 1

        # Save every N batches
        if batch_counter % save_every_n_batches == 0:
            embeddings_tensor = torch.cat(embeddings_list, dim=0)
            batch_file = os.path.join(save_dir, f'embeddings_batch_{batch_counter}.pth')
            torch.save(embeddings_tensor, batch_file)
            logger.info(f"Saved batch {batch_counter} embeddings to disk.")
            embeddings_list = []  # Reset list to free memory

    # Save any remaining embeddings after the last batch
    if embeddings_list:
        embeddings_tensor = torch.cat(embeddings_list, dim=0)
        batch_file = os.path.join(save_dir, f'embeddings_final_batch.pth')
        torch.save(embeddings_tensor, batch_file)
        logger.info(f"Saved final batch embeddings to disk.")

    logger.info("Embeddings saved successfully!")
    logger.info("Script finished successfully!")

@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    return get_embeddings(cfg)


if __name__ == "__main__":
    main()
