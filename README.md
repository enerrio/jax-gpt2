# GPT-2 experiments
---
This repo is an extension of my other [jax-transformer](https://github.com/enerrio/jax-transformer) repo.

* entry_point.py: Main entry script for running training and inference
* config.py: Hyperparameters for different GPT-2 model sizes
* run_train.py: Kicks off training of the model
* run_inference.py: Generates text using a pretrained model for a given prompt
* run_plot.py: Plot the results from a training run
* transformer/data.py: Sets up DataLoaders for model training
* transformer/model.py: GPT-2 model built with Jax and Equinox
* transformer/utils.py: Utility functions for model serialization and plotting
* transformer/train.py: Training loop
* transformer/infer.py: Runs inference on a pretrained model
* tests/: Unit tests
* data/download_and_prep.py: Download various datasets for training and evaluation
* data/the-verdict.txt: Small dataset for training

Some of the code related to downloading and preparing datasets are from Andrej Karpathy's [llm.c repo](https://github.com/karpathy/llm.c/tree/master)

This code was tested using:
* python==3.12.4
* jax==0.4.34
* jaxtyping==0.2.34
* optax==0.2.3
* equinox==0.11.7

`environment.yml` is an export of the conda environment used during development of this codebase. If you have conda installed on your machine and want to create create an identical environment with all the libraries ready to go, run this:
```bash
conda env create -f environment.yml
```

Then activate the environment:
```bash
conda activate jax
```

## Usage
---
To train the model:
```bash
python entry_point.py train --model_size small --data the-verdict --nb_steps 19 --batch_size 2 --lr 4e-4 --warmup 5 --exp_name testbench --seq_len 256 --eval_freq 20
```

To run inference on a model:
```bash
python entry_point.py infer --model_size small --model_name gpt2-small-test01.eqx --prompt "hello my dear, I am" --max_new_tokens 50
```

To run evals on a model:
```bash
python entry_point.py eval --model_size small --exp_name testbench
```