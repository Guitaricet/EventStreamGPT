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

```
python scripts/pretrain.py --config-path="../configs" --config-name="retro_test.yaml"
```
