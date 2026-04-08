---
language:
- en
license: apache-2.0
library_name: llm2ner
base_model: {{llm_name}}
tags:
- ner
- span-detection
- llm
- pytorch
pipeline_tag: token-classification
model_name: {{model_id}}
source: {{repo_url}}
paper: {{paper_url}}
---

# {{model_id}}


[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)]({{paper_url}})
[![All Models](https://img.shields.io/badge/🤗%20Hugging%20Face%20Models-blue)](https://huggingface.co/llm2ner)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/VictorMorand/llm2ner)


ToMMeR is a lightweight probing model extracting emergent mention detection capabilities from early layers representations of any LLM backbone, achieving high Zero Shot recall across a wide set of 13 NER benchmarks.

## Model Details

This model can be plugged at layer {{layer}} of `{{llm_name}}`, with a computational overhead not greater than an additional attention head.

| Property  | Value |
|-----------|-------|
| Base LLM  | `{{llm_name}}` |
| Layer     | {{layer}}|
| #Params   | {{n_params}} |


# Usage

## Installation
To use ToMMeR, you need to install its codebase first.

```bash
pip install git+{{repo_url}}.git
```


## Raw inference
By default, ToMMeR outputs span probabilities, but we also propose built-in options for decoding entities.

- Inputs:
  - tokens (batch, seq): tokens to process,
  - model: LLM to extract representation from.
- Outputs: (batch, seq, seq) matrix (masked outside valid spans)

```python
from xpm_torch.huggingface import TorchHFHub
from llm2ner import ToMMeR, utils

tommer: ToMMeR = TorchHFHub.from_pretrained("llm2ner/{{model_id}}")
# load Backbone llm, optionnally cut the unused layer to save GPU space.
llm = utils.load_llm( tommer.llm_name, cut_to_layer=tommer.layer,)
tommer.to(llm.device)

#### Raw Inference
text = ["Large language models are awesome"]
print(f"Input text: {text[0]}")

#tokenize in shape (1, seq_len)
tokens = llm.tokenizer(text, return_tensors="pt")["input_ids"].to(llm.device)
# Output raw scores
output = tommer.forward(tokens, llm) # (batch_size, seq_len, seq_len)
print(f"Raw Output shape: {output.shape}")

#use given decoding strategy to infer entities
entities = tommer.infer_entities(tokens=tokens, model=llm, threshold=0.5, decoding_strategy="greedy")
str_entities = [ llm.tokenizer.decode(tokens[0,b:e+1]) for b, e in entities[0]]
print(f"Predicted entities: {str_entities}")

>>>INFO:root:Cut LlamaModel with 16 layers to 7 layers
>>> Input text: Large language models are awesome
>>> Raw Output shape: torch.Size([1, 6, 6])
>>> Predicted entities: ['Large language models']
```


## Fancy Outputs

We also provide inference and plotting utils in `llm2ner.plotting`.

```python
from xpm_torch.huggingface import TorchHFHub
from llm2ner import ToMMeR, utils, plotting

tommer: ToMMeR = TorchHFHub.from_pretrained("llm2ner/{{model_id}}")
# load Backbone llm, optionnally cut the unused layer to save GPU space.
llm = utils.load_llm( tommer.llm_name, cut_to_layer=tommer.layer,)
tommer.to(llm.device)

text = "Large language models are awesome. While trained on language modeling, they exhibit emergent Zero Shot abilities that make them suitable for a wide range of tasks, including Named Entity Recognition (NER). "

#fancy interactive output
outputs = plotting.demo_inference( text, tommer, llm,
    decoding_strategy="threshold",  # or "greedy" for flat segmentation
    threshold=0.5, # default 50%
    show_attn=True,
)
```
<div>
<span class="tex2jax_ignore"><div class="spans" style="line-height: 2.5; direction: ltr">
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    Large
    <span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 40px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
<span style="font-weight: bold; display: inline-block; position: relative; height: 77px;">
    language
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 57px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 57px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
<span style="font-weight: bold; display: inline-block; position: relative; height: 77px;">
    models
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 57px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
</span>
are awesome . While trained on
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    language
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 40px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    modeling
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
</span>
, they exhibit
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    emergent
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 40px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    abilities
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
</span>
that make them suitable for a wide range of
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    tasks
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 40px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
, including
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    Named
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 40px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    Entity

<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
</span>
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    Recognition
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
</span>
(
<span style="font-weight: bold; display: inline-block; position: relative; height: 60px;">
    NER
<span style="background: lightblue; top: 40px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
</span>
<span style="background: lightblue; top: 40px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
    <span style="background: lightblue; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
        PRED
    </span>
</span>
</span>
) . </div></span>
</div>

Please visit the [repository]({{repo_url}}) for more details and a demo notebook.

{{eval_par}}
## Citation
If using this model or the approach, please cite the associated paper:
```
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

## License
Apache-2.0 (see repository for full text).
