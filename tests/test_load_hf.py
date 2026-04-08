import llm2ner
from xpm_torch.huggingface import TorchHFHub


# Load the model using the loader (which can be used for more complex loading scenarios)
tommer_loader = TorchHFHub.pretrained_loader("updated_model_local")
print("loaded config:")
print(tommer_loader)

# Or load as a ready-to-use instance for direct inference
tommer = TorchHFHub.from_pretrained("updated_model_local")

print("loaded model:")
print(tommer)
# load Backbone llm, optionnally cut the unused layer to save GPU space.
llm = llm2ner.utils.load_llm(
    tommer.llm_name,
    cut_to_layer=tommer.layer,
)
tommer.to(llm.device)

text = "Large language models are awesome. While trained on language modeling, they exhibit emergent Zero Shot abilities that make them suitable for a wide range of tasks, including Named Entity Recognition (NER). "


# tokenize in shape (1, seq_len)
tokens = llm.tokenizer(text, return_tensors="pt")["input_ids"].to(llm.device)

# use given decoding strategy to infer entities
entities = tommer.infer_entities(tokens=tokens, model=llm, threshold=0.5)
str_entities = [llm.tokenizer.decode(tokens[0, b : e + 1]) for b, e in entities[0]]
print(f"Predicted entities: {str_entities}")
