import random, torch
from llm2ner.models.model import NERmodel
from llm2ner.data import InferredDataset
from llm2ner.utils import load_llm
from openai import OpenAI
from tqdm import tqdm
import logging


SYSTEM = \
"""
You are an expert in entity mention annotation.
A mention is defined as : "something that exists as itself. It does not need to be of material existence."
In particular, abstractions and legal fictions are usually regarded as entities. 
In general, there is also no presumption that an entity is animate, or present. It may refer to animals; natural features such as mountains; inanimate objects such as tables; numbers or sets as symbols written on a paper; human contrivances such as laws, corporations and academic disciplines; or supernatural beings such as gods and spirits."

## Instructions
- For each text span provided in [[...]], quickly determine if it is a valid mention as defined above, regardless of its type, length, or style, but ensuring it is not a fragment.
- Briefly explain in one concise sentence whether the span fits the definition. Then answer with a clear "yes" or "no".
"""

PROMPT = """Context:"{context}" """

OAI_MODEL = "gpt-4.1-mini" # "gpt-4.1-mini" "gpt-4.1-nano" "gpt-3.5-turbo"

def format_context(ent:dict, text, n_ctx=30):
    b, e = ent["pos"]
    if b - n_ctx <= 0:
        prev_ctx = text[0:b]
    else:
        prev_ctx = "..." + text[b-n_ctx:b]
    if e + n_ctx >= len(text):
        next_ctx = text[e:len(text)]
    else:
        next_ctx = text[e:e+n_ctx] + "..."    
    return prev_ctx + f"[[{ent['name']}]]" + next_ctx

def parse_answer(text):
    end_text = text[-10:].lower().replace(".", "").replace(",", "").strip()
    if "yes" in end_text:
        return True
    elif "no" in end_text:
        return False
    else:
        return None
    
def generate_random_predictions(item, n_preds=5) -> list:
    
    rand_entities = []

    for _ in range(n_preds):
        b = random.randint(0, len(item.text)-5)
        e = b + random.randint(1, 10)
        rand_entities.append({
            "name": item.text[b:e],
            "pos": [b, e]
        })

    return sorted(rand_entities, key=lambda x: x['pos'][0])

def infer_from_LM_chat(data, inf_limit:int, client, model:str = OAI_MODEL):
    answers = []
    n_req = 0

    judged_dataset = InferredDataset(
        data_name=data.data_name, 
        decoding_strategy=data.decoding_strategy,
        threshold=data.threshold,
        data_folder=data.data_folder)

    pbar = tqdm(total=inf_limit, desc="Inference requests")
    
    try : 
        for sample in data:
            # Stop if we reached the inference limit (but finish sample)
            if n_req >= inf_limit:
                logging.info(f"Reached inference limit of {inf_limit} requests")
                return judged_dataset

            for ent in sample.pred_entities:
                n_req += 1
                pbar.update(1)   # advance progress bar by 1
                context = format_context(ent, sample.text)
                message = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": PROMPT.format(context=context)}
                ]
                
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=message,
                    max_tokens=100,
                    temperature=0,
                )
                resp = resp.choices[0].message.content
                verdict = parse_answer(resp)
                # logging.info(f"Entity: {ent['name']}, Context: {context}, Verdict: {verdict}, Response: {resp}")
                # logging.info(f"{context}, -> {verdict}")
                ent['judged_as_entity'] = verdict
                ent['judge_response'] = resp
            
            judged_dataset.samples.append(sample)
    except Exception as e:
        logging.info(f"Error during inference: {e}, returning partial results")
    finally:
        pbar.close()

    return judged_dataset


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    inf_limit = 10000  # max number of inference requests to the LLM
    assert os.environ.get("OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY environment variable"
    
    # data_path = "/home/morand/experiments/llminterp/jobs/llm2ner.models.model.nermodel.eval/a8537a2d4954476a3f7597de42c5ce7c496ae4c32a0d387f739020a048069363/inferences_ACE 2005_threshold_thr0.5.json"
    
    # 92bf3cebce4887dabb0d77f09f6d2a79c47f6e40f0e7cc36c9af5e50fcc14bc1 = best F1 llama 1B at layer 6
    data_path = "/home/morand/experiments/llminterp/jobs/llm2ner.tasks.learntokenmatching/92bf3cebce4887dabb0d77f09f6d2a79c47f6e40f0e7cc36c9af5e50fcc14bc1/model/inferences_GENIA_NER_threshold_thr0.5.json"
    
    
    # Best recall model on MultiNERD test set
    # ~ 100K entities.
    # Will Cost 15$ to annotate them with gpt4.1-mini
    data_path = "/home/morand/experiments/llminterp/jobs/llm2ner.tasks.learntokenmatching/c130a1db02fe3aeb418c87aef5381e3b960fd5259ea8e5bf7af70de91888b5a6/model/inferences_MultiNERD_dev_threshold_0.5.json"
    
    # Best Recall Model on GENIA dev set
    # 31606 entities with (16.7 % precision)
    # Will Cost 4.74$ to annotate them with gpt4.1-mini
    data_path = "/home/morand/experiments/llminterp/jobs/llm2ner.tasks.learntokenmatching/c130a1db02fe3aeb418c87aef5381e3b960fd5259ea8e5bf7af70de91888b5a6/model/inferences_GENIA_NER_dev_threshold_0.5.json"

    data = InferredDataset.from_json(data_path)
    print(f"Loaded data from {data_path}, with {len(data)} samples")
    
    save_path = data_path.replace(".json", f"_judged_{OAI_MODEL}_lim{inf_limit}.json")

    if os.path.exists(save_path):
        print(f"Judged dataset already exists at {save_path}, skipping inference")
        exit(0)
    
    client = OpenAI()
    answers = infer_from_LM_chat(data, inf_limit, client, model=OAI_MODEL)

    answers.to_json(save_path)
    print(f"Saved judged dataset to {save_path}")

### depr

# def infer_from_LM_logits(model, prompts):
#     """Attempt to infer yes/no answers from the LM logits. In practice, this is not very reliable.
#     LLM are not bad at the task but need some tokens to think
#     """
#     print(f"Loaded {len(data)} examples from {data_path}")

#     ##deprc

#     # icl_examples = [
#     #     "Q:'Paris is the capital of France.' Answer: yes",
#     #     "Q:'The sky is green.' Answer: no"
#     # ]
#     # icl_prompt = "\n".join(icl_examples) + "\n"

#     # questions = [
#     #     "Q:'The Earth revolves around the Moon.' Answer:",
#     #     "Q:'Water boils at 100 degrees Celsius.' Answer:",
#     #     "Q:'Python is a type of snake.' Answer:"
#     # ]
#     # prompts = [icl_prompt + q for q in questions]
#     yes_id = model.tokenizer.encode(" yes", add_special_tokens=False)[0]
#     no_id  = model.tokenizer.encode(" no", add_special_tokens=False)[0]
#     print(f"'Yes' token id: {yes_id}, 'No' token id: {no_id}")

#     print(data[0])
#     def extract_answer(prompts, yes_id, no_id, verbose=False):
#         # Tokenize batch
#         inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)

#         # Get logits for the *next token prediction*
#         logits = outputs.logits  # [batch, seq_len, vocab_size]
#         next_token_logits = logits[:, -1, :]  # last position, [batch, vocab_size]

#         yes_logits = next_token_logits[:, yes_id]
#         no_logits  = next_token_logits[:, no_id]

#         print("Yes logits:", yes_logits)
#         print("No logits:", no_logits)

#         return yes_logits > no_logits

#     def format_prompt(ent, text, n_ctx=10):
#         return f"Q:'{text}' Does the entity '{ent}' appear in the text? Answer:"

#     for item in data:
#         gt_entities = item.gt_entities
        
#         for ent in item.pred_entities:
#             print(f"Entity: {ent}")
#             print(f"Text: {item.text}")
#             break
#         prompt = format_prompt(ent, item.text)
#         print(prompt)
#         break

#     predictions = extract_answer(prompts, yes_id, no_id)

