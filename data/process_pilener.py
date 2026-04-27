# from https://github.com/urchade/GLiNER/blob/main/data/process_pilener.py
#
import json
import re
import ast
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def load_data(filepath):
    """Loads data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
#
def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def extract_entity_spans(entry):
    """Extracts entity spans from an entry."""
    len_start = len("What describes ")
    len_end = len(" in the text?")
    entity_types, entity_texts, negative = [], [], []

    for c in entry['conversations']:
        if c['from'] == 'human' and c['value'].startswith('Text: '):
            text = c['value'][len('Text: '):]
            tokenized_text = tokenize_text(text)
        elif c['from'] == 'human' and c['value'].startswith('What describes '):
            entity_type = c['value'][len_start:-len_end]
            entity_types.append(entity_type)
        elif c['from'] == 'gpt' and c['value'].startswith('['):
            if c['value'] == '[]':
                negative.append(entity_types.pop())
                continue
            texts_ents = ast.literal_eval(c['value'])
            entity_texts.extend(texts_ents)
            num_repeat = len(texts_ents) - 1
            entity_types.extend([entity_types[-1]] * num_repeat)

    # print("entity_types:", len(entity_types), entity_types)
    # print("entity_texts:", len(entity_texts), entity_texts)
    # print("negative:", negative)            # entity types that are not in the text
    #old version from GLiNER
    # entity_spans = []
    # for j, entity_text in enumerate(entity_texts):
    #     entity_tokens = tokenize_text(entity_text)
    #     matches = []
    #     for i in range(len(tokenized_text) - len(entity_tokens) + 1):
    #         if " ".join(tokenized_text[i:i + len(entity_tokens)]).lower() == " ".join(entity_tokens).lower():
    #             matches.append((i, i + len(entity_tokens) - 1, entity_types[j]))
    #     if matches:
    #         entity_spans.extend(matches)
    # item = {"tokenized_text": tokenized_text, "ner": entity_spans, "negative": negative}
    # print(f"found {len(entity_spans)} entities")

    entities = []
    for i, (type, mention) in enumerate(zip(entity_types, entity_texts)):
        #find all occurences of the mention in the text
        matches = re.finditer(r'\b' + re.escape(mention) + r'\b', text)
        for match in matches:
            start, end = match.span()
            entities.append(
                {
                    "name": text[start:end],
                    "type": type,
                    "pos" : [start, end],
                })

    return {
        "sentence": text,
        "entities": entities,
        "negative": negative
    }

def process_data(data):
    """Processes a list of data entries to extract entity spans."""
    all_data = [extract_entity_spans(entry) for entry in tqdm(data)]
    return all_data

def save_data_to_file(data, filepath):
    """Saves the processed data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    # download the pile-ner data: "wget https://huggingface.co/datasets/Universal-NER/Pile-NER-type/blob/main/train.json"
    # path_pile_ner = 'train.json'
    # data = load_data(path_pile_ner)
    from datasets import load_dataset
    from pathlib import Path

    data = load_dataset("Universal-NER/Pile-NER-type")['train'] # only train set available

    processed_data = process_data(data)
    output_folder = Path(__file__).parent / "Pile-NER"

    logging.info(f"Saving processed data to {output_folder} with {len(processed_data)} entries.")
    logging.info(f"you can point move it to your desired location or just use it from here by setting NER_DATA env var with `export NER_DATA={output_folder}`")

    output_folder.mkdir(exist_ok=True)
    save_data_to_file(processed_data, output_folder / 'train.json')
