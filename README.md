# Retreival-augmented Event Stream GPT

## Installation

> You can install all required packages via conda with the `env.yml` file: `conda env create -n ${ENV_NAME} -f env.yml`, but you still will need to install the package via `pip install -e .`

Installing the package will install all of the requirements, **except for FAISS**
```
python -m pip install -e .
```

**You need to install FAISS either via conda or from source.**

If you are using conda, execute
```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

Or install FAISS from source following the official instructions [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

## Training example

First, you need to get faiss index and compute hidden states for your document database.
To get faiss index, use this script: 
Faiss index is served for similarity and search, while the hidden states serve as values over which the model attends.

```
python scripts/compute_embeddings.py \
    --data_path data/faiss/faiss_index_14-32-52/merged.jsonl \
    --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --batch_size 512 \
    --max_length 128 \
    --num_workers 8 \
    --device cuda
```

Then, ensure that your trainig config (e.g., `configs/retro_test.yaml`) points to the right data, faiss index, and document dataset (retreival). Make sure that retreived_states_hidden_size is set to the same as the hidden size of the model you are using.

You can then train the model using the following command:
```
python scripts/pretrain.py --config-path="../configs" --config-name="retro_test.yaml"
```
