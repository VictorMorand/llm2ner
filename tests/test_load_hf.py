from xpm_torch.huggingface import TorchHFHub
from llm2ner import ToMMeR, utils

tommer: ToMMeR = TorchHFHub.from_pretrained("llm2ner/ToMMeR-Llama-3.2-1B_L6_R64")
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
