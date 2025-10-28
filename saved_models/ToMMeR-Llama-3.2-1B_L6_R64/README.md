---
language:
- en
license: apache-2.0
library_name: llm2ner
base_model: meta-llama/Llama-3.2-1B
tags:
- ner
- span-detection
- llm
- pytorch
pipeline_tag: token-classification
model_name: ToMMeR-Llama-3.2-1B_L6_R64
source: https://github.com/VictorMorand/llm2ner
paper: https://arxiv.org/abs/2510.19410
---

# ToMMeR-Llama-3.2-1B_L6_R64

ToMMeR is a lightweight probing model extracting emergent mention detection capabilities from early layers representations of any LLM backbone, achieving high Zero Shot recall across a wide set of 13 NER benchmarks.

## Checkpoint Details

| Property  | Value |
|-----------|-------|
| Base LLM  | `meta-llama/Llama-3.2-1B` |
| Layer     | 6|
| #Params   | 264.2K |


# Usage

## Installation

Our code can be installed with pip+git, Please visit the [repository](https://github.com/VictorMorand/llm2ner) for more details.

```bash
pip install git+https://github.com/VictorMorand/llm2ner.git
```

## Fancy Outputs

```python
import llm2ner
from llm2ner import ToMMeR

tommer = ToMMeR.from_pretrained("llm2ner/ToMMeR-Llama-3.2-1B_L6_R64")
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


## Raw inference
By default, ToMMeR outputs span probabilities, but we also propose built-in options for decoding entities.

- Inputs:
  - tokens (batch, seq): tokens to process, 
  - model: LLM to extract representation from.
- Outputs: (batch, seq, seq) matrix (masked outside valid spans)

```python

tommer = ToMMeR.from_pretrained("llm2ner/ToMMeR-Llama-3.2-1B_L6_R64")
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

Please visit the [repository](https://github.com/VictorMorand/llm2ner) for more details and a demo notebook.

## Evaluation Results

| dataset             |   precision |   recall |     f1 |   n_samples |
|---------------------|-------------|----------|--------|-------------|
| MultiNERD           |      0.2165 |   0.9858 | 0.355  |      154144 |
| CoNLL 2003          |      0.3355 |   0.9479 | 0.4956 |       16493 |
| CrossNER_politics   |      0.3237 |   0.9703 | 0.4855 |        1389 |
| CrossNER_AI         |      0.3503 |   0.9704 | 0.5148 |         879 |
| CrossNER_literature |      0.4026 |   0.9443 | 0.5645 |         916 |
| CrossNER_science    |      0.3816 |   0.9565 | 0.5456 |        1193 |
| CrossNER_music      |      0.4407 |   0.9553 | 0.6032 |         945 |
| ncbi                |      0.1265 |   0.9193 | 0.2224 |        3952 |
| FabNER              |      0.3014 |   0.736  | 0.4276 |       13681 |
| WikiNeural          |      0.2067 |   0.9783 | 0.3413 |       92672 |
| GENIA_NER           |      0.2477 |   0.9572 | 0.3936 |       16563 |
| ACE 2005            |      0.2872 |   0.4197 | 0.341  |        8230 |
| Ontonotes           |      0.2553 |   0.7303 | 0.3783 |       42193 |
| Aggregated          |      0.2319 |   0.9262 | 0.3709 |      353250 |
| Mean                |      0.2981 |   0.8824 | 0.436  |      353250 |

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