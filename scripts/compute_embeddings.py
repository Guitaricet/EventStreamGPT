"""Compute pubmed embeddings that will be used after retreival as hidden states to attend over

Usage:
    python scripts/compute_embeddings.py \
        --data_path data/faiss/faiss_index_14-32-52/merged.jsonl \
        --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
        --batch_size 512 \
        --max_length 128 \
        --num_workers 8 \
        --device cuda
"""
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer
from datasets import Dataset

from loguru import logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and embed a dataset using a pretrained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Name or path of the pretrained model.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the processed dataset.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for processing.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the input sequence.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation.")

    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = Path(args.data_path).parent / "huggingface_dataset"
        args.output_path = args.output_path.as_posix()

    return args


def calculate_percentiles(dataset, tokenizer):
    dataset_w_lengths = dataset.select(range(10000)).map(
        lambda x: {"length": len(tokenizer(x["text"])["input_ids"])}
    )
    lengths = np.array(dataset_w_lengths["length"])
    percentiles = np.percentile(lengths, [95, 99])
    return lengths, percentiles


def tokenize_batch(batch, tokenizer, max_length=128):
    torch_batch = tokenizer(batch["text"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return {"input_ids": torch_batch["input_ids"].squeeze(0), "attention_mask": torch_batch["attention_mask"].squeeze(0)}


def embed_batch(batch, model, device):
    input_ids = [torch.LongTensor(i) for i in batch["input_ids"]]
    attention_mask = [torch.LongTensor(i) for i in batch["attention_mask"]]

    input_ids = torch.stack(input_ids).to(device=device)
    attention_mask = torch.stack(attention_mask).to(device=device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = out.last_hidden_state.detach().to(device="cpu", dtype=torch.float32).numpy()
    last_hidden_list = np.split(last_hidden, last_hidden.shape[0], axis=0)
    return {"hidden_state": last_hidden_list}


def collate_fn(batch):
    input_ids = torch.stack([torch.LongTensor(example["input_ids"]) for example in batch])
    attention_mask = torch.stack([torch.LongTensor(example["attention_mask"]) for example in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def main():
    args = parse_arguments()
    logger.info("*" * 40)
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"{k:<20}: {v}")
    logger.info("*" * 40)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model = model.to(device=args.device, dtype=torch.bfloat16)

    dataset = Dataset.from_json(args.data_path)
    _, percentiles = calculate_percentiles(dataset, tokenizer)
    logger.info(f"95 length percentile: {percentiles[0]}")
    logger.info(f"99 length percentile: {percentiles[1]}")
    logger.info(f"Using max length: {args.max_length}")

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
    )

    # NOTE: This approach is not very efficient,
    # as collation and embedidng are synchronous, unlike in pytorch dataloader.
    # However, I didn't find a simple way to store the embeddings in the dataset
    # while they are getting computed in a loop. The problem is that the
    # embeddings table doesn't fit into the table.
    # On a single 3090, embedding 1M documents using BERT Base-side model takes ~25 minutes.
    # If done with a pytorch dataloader this could be sped up to just under 10 minutes.
    dataset = dataset.map(
        embed_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=1,  # should not fork model process
        fn_kwargs={"model": model, "device": args.device},
    )
    dataset.save_to_disk(args.output_path)

    # save args
    args_path = Path(args.output_path).parent / "embedding_args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f)

    logger.info(f"Saved dataset to {args.output_path}")


if __name__ == "__main__":
    main()
