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
