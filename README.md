# Extracting NER Skills from LLMs

This repository gather the code for our experiments trying to extract the likely existing NER signals from Encoder only LLMs representations of tokens.

We depend on several key packages:
- [`experimaestro-python`](https://github.com/experimaestro/experimaestro-python) for experiment management
- [`transformer-lens`](https://github.com/TransformerLensOrg/TransformerLens) for wrapping LLMs in a generic `HookedTransformer` class with a unified nomencature for placing Hooks. It is build upon the hugginface `transformers` library.

## Installation

### Using Pip

```bash
uv pip install -e git+https://github.com/VictorMorand/llm2ner.git
```


### Local install for Dev 

#### Using `uv`

We suggest using [uv](https://docs.astral.sh/uv/), a super fast package manager.

```bash
git clone https://github.com/VictorMorand/llm2ner.git
cd llm2ner
uv sync
```

## Usage

###
```python
import llm2ner
from llm2ner import ToMMeR

tommer = ToMMeR.from_pretrained("llm2ner/saved_models/ToMMeR-Llama-3.2-1B_L6_R64")
# load Backbone llm, optionnally cut it to the releval layer if needed.
llm = llm2ner.utils.load_llm( tommer.llm_name, cut_to_layer=tommer.layer,) 
tommer.to(llm.device)

text = "Large language models are awesome. While trained on language modeling, they exhibit emergent Zero Shot abilities that make them suitable for a wide range of tasks, including Named Entity Recognition (NER). "

#fancy interactive output
outputs = llm2ner.plotting.demo_inference( text, tommer, llm,
    decoding_strategy="threshold",  # or "greedy" for flat segmentation
    threshold=0.5, # default 50%
    show_attn=True,
)
```

### Demo 
The [`ToMMeR_Demo.ipynb`](./Notebooks/ToMMeR_Demo.ipynb) notebook enables you to quickly test the models.


### Running experiments

[Experimaestro](https://github.com/experimaestro/experimaestro-python) is used to launch and monitor experiments.
You can run an experiment training a ToMMeR Model on the specified Dataset with the following command:

```bash
uv run experimaestro run-experiment experiments/trainTokenMatching
```