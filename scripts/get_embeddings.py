#!/usr/bin/env python
"""Gets the emeddings of a pre-trained model for a user-specified fine-tuning dataset."""

try:
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import hydra
import torch

from EventStream.transformer.lightning_modules.embedding import (
    FinetuneConfig,
    get_embeddings,
)

torch.set_float32_matmul_precision("high")

def train(cfg: PretrainConfig):
    logger.info("Starting training")
    seed_everything(cfg.seed)

    device = f"cuda"
    dtype = torch.float32

    global_rank = 0  # assume no distributed for now
    torch.multiprocessing.set_sharing_strategy("file_system")

    model_config: StructuredTransformerConfig = cfg.model_config
    data_config: PytorchDatasetConfig = cfg.data_config
    retreiver_config: RetreiverConfig = cfg.retreiver_config

    # Data
    logger.info("Building train dataset")
    train_dataset = PytorchDataset(cfg.data_config, split="train")

    if global_rank != 0:
        logger.remove()

    # Model
    model: ConditionallyIndependentRetreivalAugTransformer = get_model(model_config)
    model = model.to(dtype=dtype, device=device)
    retreiver: EventStream.Retreiver = EventStream.Retreiver.from_config(retreiver_config, dtype=dtype)

    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_config.per_device_train_batch_size,
        num_workers=model_config.num_dataloader_workers,
        collate_fn=train_dataset.collate,
        shuffle=True,
    )

    chunk_len = model_config.chunked_cross_attention_chunk_len
    batch_size = model_config.per_device_train_batch_size
    n_neighbours = retreiver_config.n_neighbours

    # Create a file to store embeddings
    with open(cfg.save_dir / "embeddings.pth", 'w') as file:

        for batch in train_dataloader:
            batch = batch.to(dtype=dtype, device=device)
            retreiver_query = model.first_half_forward(batch).last_hidden_state
            n_chunks = retreiver_query.shape[1] // chunk_len

            # Get embeddings for the current batch
            retreival_queries = model.reshape_to_retreival_queries(retreiver_query, allow_padding=True)

            assert retreival_queries.shape == (batch_size * n_chunks, model_config.hidden_size)
            _, hidden = retreiver.get_documents_and_embed(
                retreival_queries,
                n_neighbours=n_neighbours,
            )

            # Write the embeddings to the file
            for embed in hidden:
                file.write(' '.join(map(str, embed.tolist())) + '\n')
    
    logger.info("Script finished successfully!")


@hydra.main(version_base=None, config_name="finetune_config")
def main(cfg: FinetuneConfig):
    if type(cfg) is not FinetuneConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.
    return get_embeddings(cfg)


if __name__ == "__main__":
    main()
