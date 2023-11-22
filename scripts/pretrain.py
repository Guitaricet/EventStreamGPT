#!/usr/bin/env python
"""Pre-trains a model from scartch."""
import copy
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.utils.data
import wandb
from transformers import get_polynomial_decay_schedule_with_warmup

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from loguru import logger

import EventStream
from EventStream import Retreiver, RetreiverConfig, PretrainConfig, OptimizationConfig
from EventStream.config import Split
from EventStream.transformer.config import StructuredTransformerConfig
from EventStream.data.pytorch_dataset import PytorchDataset, PytorchDatasetConfig
from EventStream.transformer.config import StructuredEventProcessingMode
from EventStream.transformer.modeling_retro import ConditionallyIndependentRetreivalAugTransformer
from EventStream.transformer.conditionally_independent_model import CondIndepModelForGenerativeSequenceModeling


def seed_everything(seed):
    """Sets the seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main()
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")
    # TODO(mmd): This isn't the right return value for hyperparameter sweeps.

    if cfg == {}:
        raise ValueError("No configuration was provided, specify configuration with --config")

    print(cfg)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_fp = cfg.save_dir / "pretrain_config.yaml"
        cfg_fp.parent.mkdir(exist_ok=True, parents=True)

        cfg_dict = copy.deepcopy(cfg)
        cfg_dict.model_config = cfg_dict.model_config.to_dict()
        OmegaConf.save(cfg_dict, cfg_fp)

    train(cfg)


def get_model(config, from_pretrained: str|Path = None):
    if config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION:
        raise NotImplementedError("Nested attention RETRO is not yet implemented")

    model_cls = CondIndepModelForGenerativeSequenceModeling
    if from_pretrained is not None:
        return model_cls.from_pretrained(from_pretrained)
    return model_cls(config)


@torch.no_grad()
def evaluate_model(model, *, dataloader, metrics_logger, dtype, device, step):
    logger.warning("Evaluation is currently not implemented, need to figure out how exactly metrics should be averaged")
    return

    model.eval()
    for batch in dataloader:
        batch = batch.to(dtype=dtype, device=device)
        out = model(batch)
        metrics_logger.log_metrics(out, step=step, split=Split.VALIDATION)

    model.train()


def train(cfg: PretrainConfig):
    logger.info("Starting training")
    seed_everything(cfg.seed)

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    world_size = 1
    global_rank = 0  # assume no distributed for now
    torch.multiprocessing.set_sharing_strategy("file_system")

    model_config: StructuredTransformerConfig = cfg.model_config
    optimization_config: OptimizationConfig = cfg.optimization_config
    data_config: PytorchDatasetConfig = cfg.data_config
    retreiver_config: RetreiverConfig = cfg.retreiver_config

    # Data
    logger.info("Building train dataset")
    train_dataset = PytorchDataset(cfg.data_config, split="train")
    logger.info("Building eval dataset")
    eval_dataset = PytorchDataset(cfg.data_config, split="tuning")

    logger.info("Setting up config")
    model_config.set_to_dataset(train_dataset)
    optimization_config.set_to_dataset(train_dataset)

    if global_rank != 0:
        logger.remove()

    if global_rank == 0:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving config files")
        config_fp = cfg.save_dir / "config.json"
        if config_fp.exists() and not cfg.do_overwrite:
            raise FileExistsError(f"{config_fp} already exists!")
        else:
            logger.info(f"Writing to {config_fp}")
            model_config.to_json_file(config_fp)

        data_config.to_json_file(cfg.save_dir / "data_config.json", do_overwrite=cfg.do_overwrite)
        optimization_config.to_json_file(cfg.save_dir / "optimization_config.json", do_overwrite=cfg.do_overwrite)
        cfg.pretraining_metrics_config.to_json_file(cfg.save_dir / "pretraining_metrics_config.json", do_overwrite=cfg.do_overwrite)
        cfg.final_validation_metrics_config.to_json_file(cfg.save_dir / "final_validation_metrics_config.json", do_overwrite=cfg.do_overwrite)

    # Model
    model: ConditionallyIndependentRetreivalAugTransformer = get_model(model_config)
    model = model.to(dtype=dtype, device=device)
    retreiver: Retreiver = Retreiver.from_config(retreiver_config, dtype=dtype)
    logger.info(f"Model: \n{model}")

    gradient_accumulation = optimization_config.total_batch_size // (optimization_config.per_device_train_batch_size * world_size)
    if gradient_accumulation != optimization_config.gradient_accumulation and optimization_config.gradient_accumulation is not None:
        logger.warning(f"Overriding gradient accumulation from {optimization_config.gradient_accumulation} to {gradient_accumulation}")

    # Setting up torch dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=optimization_config.per_device_train_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=train_dataset.collate,
        shuffle=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=optimization_config.per_device_eval_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=eval_dataset.collate,
        shuffle=False,
    )

    trainable_params = model.parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=optimization_config.max_lr,
        weight_decay=optimization_config.weight_decay,
    )
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=optimization_config.lr_num_warmup_steps,
        num_training_steps=optimization_config.max_training_steps,
        power=optimization_config.lr_decay_power,
        lr_end=optimization_config.end_lr,
    )

    if global_rank == 0:
        wandb.init(
            project=cfg.wandb_logger_kwargs["project"],
            config={
                "model_config": model_config.to_dict(),
                "optimization_config": optimization_config.to_dict(),
                "data_config": data_config.to_dict(),
                "retreiver_config": retreiver_config.to_dict(),
            },
        )

    metrics_logger = EventStream.MetricsLogger(
        config=model_config,
        metrics_config=cfg.pretraining_metrics_config,
    )

    # these are useful to write more consise code in the training loop
    batch_size = optimization_config.per_device_train_batch_size
    n_neighbours = retreiver_config.n_neighbours
    chunk_len = model_config.chunked_cross_attention_chunk_len

    # Fitting model
    global_step = 0
    update_step = 0
    update_time = time.time()

    def get_pbar(epoch):
        total = len(train_dataloader) // gradient_accumulation
        return tqdm(total=total, desc=f"Epoch {epoch} out of {optimization_config.max_epochs}", ncols=80)

    model.train()
    for epoch in range(optimization_config.max_epochs):
        pbar = get_pbar(epoch)

        for batch in train_dataloader:
            global_step += 1

            batch = batch.to(dtype=dtype, device=device)
            retreiver_query = model.first_half_forward(batch).last_hidden_state
            n_chunks = retreiver_query.shape[1] // chunk_len

            # TODO: implement pad_to_multiple_of for the model input, so that we don't need to do pooling here
            # shape: [batch_size * n_chunks, hidden] (pooled over chunk len)
            retreival_queries = model.reshape_to_retreival_queries(retreiver_query, allow_padding=True)

            assert retreival_queries.shape == (batch_size * n_chunks, model_config.hidden_size), f"Expected shape {(batch_size * chunk_len, model_config.hidden_size)}, got {retreival_queries.shape}"
            # shape: [batch_size * n_chunks, n_neighbors, neighbor_len, retreived_states_hidden_size]
            _, hidden = retreiver.get_documents_and_embed(
                retreival_queries,
                n_neighbours=n_neighbours,
            )
            # batch_size, n_chunks, n_neighbors, neighbor_len, hidden
            hidden = hidden.reshape(batch_size, n_chunks, n_neighbours, -1, hidden.shape[-1])
            hidden = hidden.to(dtype=dtype, device=device)

            retreival_output = model.second_half_forward(
                batch=batch,
                hidden_states=retreiver_query,
                retreived_hidden_states=hidden,
            )

            loss = retreival_output.loss
            scaled_loss = loss / gradient_accumulation
            scaled_loss.backward()

            if global_step % gradient_accumulation != 0:
                continue
            update_step += 1

            # The below code is only executed during the update step
            if global_rank == 0: pbar.update(1)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if optimization_config.clip_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, optimization_config.clip_grad_norm, error_if_nonfinite=True)
                if global_rank == 0:
                    wandb.log({"grad_norm": grad_norm.item()}, step=global_step)

            update_time = time.time() - update_time

            if global_rank == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "update_time": update_time,
                        "update_step": update_step,
                    },
                    step=global_step,
                )

            if update_step % optimization_config.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                evaluate_model(
                    model,
                    dataloader=eval_dataloader,
                    device=device,
                    dtype=dtype,
                    metrics_logger=metrics_logger,
                    step=global_step,
                )

            if update_step % optimization_config.save_every == 0:
                model.save_pretrained(cfg.save_dir / f"model_{update_step}")
                if optimization_config.keep_last_n_checkpoints:
                    current_checkpoints = sorted(cfg.save_dir.glob("model_*"))
                    for checkpoint_path in current_checkpoints[:-optimization_config.keep_last_n_checkpoints]:
                        shutil.rmtree(checkpoint_path)

            update_time = time.time()
            # end of for-loop over batches
            # ##############################

    model.save_pretrained(cfg.save_dir / "model_final")

    if cfg.do_final_validation_on_metrics:
        test_dataset = PytorchDataset(cfg.data_config, split="held_out")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=optimization_config.validation_batch_size,
            num_workers=optimization_config.num_dataloader_workers,
            collate_fn=test_dataset.collate,
            shuffle=False,
        )

        model.metrics_config = cfg.final_validation_metrics_config
        model.build_metrics()

        metrics_logger = EventStream.MetricsLogger(
            model=model,
            config=model_config,
            optimization_config=optimization_config,
            metrics_config=cfg.pretraining_metrics_config,
        )

        logger.info("Performing final evaluation...")
        logger.info("Evaluating on eval set...")
        eval_metrics = evaluate_model(
            model,
            dataloader=eval_dataloader,
            metrics_logger=metrics_logger,
            dtype=dtype,
            device=device,
            step=global_step,
        )
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(
            model,
            dataloader=test_dataloader,
            metrics_logger=metrics_logger,
            dtype=dtype,
            device=device,
            step=global_step,
        )

        if global_rank == 0:
            logger.info("Saving final metrics...")

            with open(cfg.save_dir / "tuning_metrics.json", mode="w") as f:
                json.dump(eval_metrics, f)
            with open(cfg.save_dir / "held_out_metrics.json", mode="w") as f:
                json.dump(test_metrics, f)
    
    logger.info("Scrip finished successfully!")


if __name__ == "__main__":
    main()
