# Extracting NER Skills from LLMs

This repository gather the code for our experiments trying to extract the likely existing NER signals from Encoder only LLMs representations of tokens.

We depend on several key packages:
- [`experimaestro-python`](https://github.com/experimaestro/experimaestro-python) for experiment management
- [`transformer-lens`](https://github.com/TransformerLensOrg/TransformerLens) for wrapping LLMs in a generic `HookedTransformer` class with a unified nomencature for placing Hooks. It is build upon the hugginface `transformers` library.

### Installation

### Using [uv](https://docs.astral.sh/uv/)

This project has been developped with uv, you can manually init a new virtual env with :
#### Dev 
```bash
git clone ... 
uv sync # creates a venv with add required dependencies
uv pip install -e .
```

#### Within you env
```bash
uv pip install -e git+https ... 
```

### Using Pip

Optionnally create a special env for
```bash
python -m venv env 
source env/bin/activate
pip install --upgrade pip
```
and then install the requirements

```bash
pip install -r requirements.txt
```


## Running Experiments

### Demo 
The [`Demo.ipynb`](./Notebooks/Demo.ipynb) notebook enables you to quickly test the models.


#### Running experiments

[Experimaestro](https://github.com/experimaestro/experimaestro-python) is used to run launch and run experiments.
You can run an experiment training an Attn Model on a specified Dataset with the following command:

```bash
uv run experimaestro run-experiment experiments/trainTokenMatching
```