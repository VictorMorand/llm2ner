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
paper: https://arxiv.org/abs/???
---

# ToMMeR-Llama-3.2-1B_L6_R64


A span-based Named Entity Recognition model built on top of the frozen LLM `meta-llama/Llama-3.2-1B` (layer 6).  
It predicts a probability (or logit) for every (start, end) token span.

## TL;DR
- Input: raw text
- Output: span scores (start ≤ end)
- Core idea: low-rank projections produce Q/K → masked similarity matrix + end-mention probe → feature combination → span logits
- Training: binary cross-entropy over gold span incidence matrix (optionally restricted / dilated)

## Usage

Install:
```bash
pip install git+https://github.com/VictorMorand/llm2ner.git
```

Quick inference:
```python
from llm2ner import NERmodel
model = NERmodel.from_pretrained("VictorMorand/ToMMeR/ToMMeR-Llama-3.2-1B_L6_R64")
text = "John lives in New York City."
spans = model.predict_spans([text], threshold=0.5)  # implement threshold logic externally if needed
print(spans)
```

## Model Details
- Base LLM: meta-llama/Llama-3.2-1B
- Layer used: 6
- Span scoring: combines raw attention-like match features and an end-entity probe
- Masking: causal / BOS configurable; optional sliding window
- Outputs: dense (seq x seq) matrix (masked outside valid spans)

## Training Procedure
1. Extract hidden states from layer 6.
2. Linear projections → Q, K (low-rank similarity).
3. End-entity classifier gives per-token end logits.
4. Construct feature stack (attention variants + end logits) and linearly combine.
5. Optimize BCE on gold span matrix (with optional entity dilation / pseudo labels).

## Evaluation
Provide precision / recall / F1 per entity type and overall micro/macro scores here.

## Limitations
- May not handle nested or overlapping entities (unless extended).
- Threshold selection impacts precision/recall.
- Quality bounded by underlying meta-llama/Llama-3.2-1B representations.
- Not audited for PII or sensitive-domain deployment.

## Responsible Use
Validate on your domain data. Check outputs for bias or hallucinated spans.

## Citation
If using this model or the approach, cite the paper:
```
@article{tommer2025,
  title = {Token Matching Mention Recognition},
  author = {Victor Morand, },
  journal = {arXiv},
  year = {2025},
  url = "https://arxiv.org/abs/???"
}
```

## License
Apache-2.0 (see repository for full text).

## Source
Repository: https://github.com/VictorMorand/llm2ner