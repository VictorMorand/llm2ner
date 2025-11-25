<div align="center">

<h1>ToMMeR – Efficient Entity Mention Detection from Large Language Models</h1>
<div>
    <a href='https://victormorand.github.io/' target='_blank'>Victor Morand</a><sup>1</sup>&emsp;
    <a target='_blank'>Nadi Tomeh</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=V-Nyr0wAAAAJ' target='_blank'>Josiane Mothe</a><sup>3</sup>&emsp;
    <a href='https://www.piwowarski.fr' target='_blank'>Benjamin Piwowarski</a><sup>1</sup>&emsp;
</div>
<br>
<div>
    <sup>1</sup>Sorbonne Université, CNRS, ISIR, F-75005 Paris, France&emsp;<br>
    <sup>2</sup>LIPN, Université Sorbonne Paris Nord, UMR7030 CNRS&emsp;<br>
    <sup>3</sup>IRIT, Université de Toulouse, UMR5505 CNRS, F-31400 Toulouse, France&emsp;<br>
</div>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-2408.08656-b31b1b.svg)](https://arxiv.org/abs/2510.19410)
[![Repository version](https://img.shields.io/badge/📄-BlogPost-blue)](https://victormorand.github.io/publications/2025-ToMMeR/)
[![Try in Colab !](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VictorMorand/llm2ner/blob/main/Notebooks/ToMMeR_Demo.ipynb)
[![Repository version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FVictorMorand%2Fllm2ner%2Fmain%2Fpyproject.toml&query=project.version&label=version&color=blue)](https://github.com/VictorMorand/llm2ner)

<img src="Assets/AbstractFig.png" alt="ToMMeR Architecture" width="600"/>
</div>

ToMMeR is a lightweight probing model extracting emergent mention detection capabilities from early layers representations of any LLM backbone, achieving high Zero Shot recall across a wide set of 13 NER benchmarks.


### Abstract
> _Identifying which text spans refer to entities -  mention detection - is both foundational for information extraction and a known performance bottleneck. We introduce ToMMeR, a lightweight model (<300K parameters) probing mention detection capabilities from early LLM layers. Across 13 NER benchmarks, ToMMeR achieves 93\% recall zero-shot, with over 90\% precision using an LLM as a judge showing that ToMMeR rarely produces spurious predictions despite high recall. Cross-model analysis reveals that diverse architectures (14M-15B parameters) converge on similar mention boundaries (DICE >75\%), confirming that mention detection emerges naturally from language modeling.  When extended with span classification heads, ToMMeR achieves near SOTA NER performance (80-87\% F1 on standard benchmarks). Our work provides evidence that structured entity representations exist in early transformer layers and can be efficiently recovered with minimal parameters._

## Installation

### Using Pip

```bash
uv pip install -e git+https://github.com/VictorMorand/llm2ner.git
```


### Local install for Dev 

#### Using `uv`

We suggest using [uv](https://docs.astral.sh/uv/), a super fast package manager.
The following commands will clone the repo and install it within a new ready-to-use `.venv` with all dependencies in a few minutes.

```bash
git clone https://github.com/VictorMorand/llm2ner.git
cd llm2ner
uv sync
```

## Usage

- All trained models are available on HugginFace: 🤗 https://huggingface.co/llm2ner/models 
- See the demo notebook [`ToMMeR_Demo.ipynb`](./Notebooks/ToMMeR_Demo.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VictorMorand/llm2ner/blob/main/Notebooks/ToMMeR_Demo.ipynb)

### Raw inference
```py

tommer = ToMMeR.from_pretrained("llm2ner/ToMMeR-Llama-3.2-1B_L3_R64")
# load Backbone llm, optionnally cut the unused layer to save GPU space.
llm = llm2ner.utils.load_llm( tommer.llm_name, cut_to_layer=tommer.layer,) 
tommer.to(llm.device)

#### Raw Inference
text = ["Large language models are awesome"]
print(f"Input text: {text[0]}")

#tokenize in shape (1, seq_len)
tokens = model.tokenizer(text, return_tensors="pt")["input_ids"].to(device)
# Output raw scores
output = tommer.forward(tokens, model) # (batch_size, seq_len, seq_len)
print(f"Raw Output shape: {output.shape}")

#use given decoding strategy to infer entities
entities = tommer.infer_entities(tokens=tokens, model=model, threshold=0.5, decoding_strategy="greedy")
str_entities = [ model.tokenizer.decode(tokens[0,b:e+1]) for b, e in entities[0]]
print(f"Predicted entities: {str_entities}")

>>> Input text: Large language models are awesome
>>> Raw Output shape: torch.Size([1, 6, 6])
>>> Predicted entities: ['Large language models']
```

### HTML output
We also provide plotting options, outputting html for fancy notebook / web app display.
```py
import llm2ner
from llm2ner import ToMMeR

tommer = ToMMeR.from_pretrained("llm2ner/ToMMeR-Llama-3.2-1B_L3_R64")
# load Backbone llm, optionnally cut the unused layer to save GPU space.
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


### Running experiments

[Experimaestro](https://github.com/experimaestro/experimaestro-python) is used to launch and monitor experiments.
You can run an experiment training a ToMMeR Model on the specified Dataset with the following command:

```bash
uv run experimaestro run-experiment experiments/trainTokenMatching
```

## Acknowledgements

We depend on several key packages:
- [`experimaestro-python`](https://github.com/experimaestro/experimaestro-python) for experiment management.
- [`transformer-lens`](https://github.com/TransformerLensOrg/TransformerLens) can be used for wrapping LLMs in a generic `HookedTransformer` class with a unified nomencature for placing Hooks. It is build upon the hugginface `transformers` library.

## Citation

If you find this work useful, please cite the associated paper:
```yaml
@misc{morand2025tommerefficiententity,
      title={ToMMeR -- Efficient Entity Mention Detection from Large Language Models}, 
      author={Victor Morand and Nadi Tomeh and Josiane Mothe and Benjamin Piwowarski},
      year={2025},
      eprint={2510.19410},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.19410}, 
}
```
