import torch, os, json
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from itertools import product
from tqdm import tqdm
import numpy as np
from pathlib import Path
from enum import Enum
import logging
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformer_lens.loading_from_pretrained import convert_hf_model_config, get_official_model_name
from llm2ner.utils import to_str_tokens

ATTN_MODES = ["first", "last", "block", "block_only"]

# create Enum as in models.py
class PATTERN_MODES(str, Enum):
    """Attention modes for NER datasets"""

    FIRST = "first"
    """All tokens attend to the first token of the entity"""
    LAST = "last"
    """Only the last token of the entity attends to the first token of the entity"""
    BLOCK = "block"
    """All tokens of the entity attend to each other, attend to bos otherwise"""
    BLOCK_ONLY = "block_only"
    """All tokens of the entity attend to each other, **do not** attend to bos otherwise"""


###################### TAGS UTILS ######################
def align_tags_with_tokens(tokens, tags):
    """Word-level CoNLL tags to token-level with tokens from tokenizer, WITHOUT <bos> token, from
    Args:
        tokens: (seq) str tokens from tokenizer WITHOUT <bos> token
        tags: (#words) word-level CoNLL tags
    Return
        token_tags (seq) token-level ConLL tags
    """
    token_tags = [tags[0]]
    word_index = 0

    for token in tokens[1:]:
        if token.startswith(" "):  # or token.strip() in special_tokens:
            # new word
            word_index += 1
            if word_index >= len(tags):
                word_index = len(tags) - 1
            token_tags.append(tags[word_index])
        else:
            # it is a subword, take previous ner tag
            tag = token_tags[-1]
            tag += tag % 2
            # for CoNLL, following tags are +1, so add one to keep only first token even.
            token_tags.append(tag)
    return token_tags


def normalize_tags(b_tags):
    """Normalize tags to 0, 1, 2 for no entity, start of entity, inside entity
    WARNING: this function modifies the input tensor in place
    Args:
        tags is list of list of tags, need to pad and normalize
    Returns:
        b_tags: tensor of shape (batch, seq) with normalized and padded tags
    """

    if type(b_tags[0]) == list:
        b_tags = [torch.tensor(tags, dtype=torch.int32) for tags in b_tags]
    elif type(b_tags[0]) == int:
        b_tags = [torch.tensor(b_tags)]

    # create tensor, pad if necessary
    b_tags = pad_sequence(
        b_tags, batch_first=True, padding_value=0, padding_side="right"
    )
    b_tags[(b_tags != 0) & (b_tags % 2 == 0)] = 2
    b_tags[(b_tags % 2 == 1)] = 1

    if b_tags.size(0) == 1:
        b_tags.squeeze_(0)

    return b_tags


@torch.jit.script
def char_to_token_pos(
    start_char: int, end_char: int, offsets: torch.Tensor
) -> Tuple[int, int]:
    """
    Given a character span, find the corresponding token span in the tokenized input.
    Args:
        char_span: (start_char, end_char) character span
        offsets: list of tuples (start, end) for each token in the tokenized input
            - can be obtained with hf tokenizer using `return_offsets_mapping=True`
    """
    token_start = -1
    token_end = -1

    for idx in range(offsets.size(0)):
        start = offsets[idx][0].item()
        end = offsets[idx][1].item()

        if token_start == -1 and start <= start_char < end:
            token_start = idx
        if token_start != -1 and start < end_char <= end:
            token_end = idx
            break

    if token_start != -1 and token_end == -1:
        token_end = token_start

    return token_start, token_end


###################### PATTERNS ######################


def get_first_attn_pattern(item):
    """compute target attention pattern from data item
    Args:
        item: dictionary with keys 'tokens' and 'ner_tags'
    Returns:
        pattern: (seq, seq) attention pattern
        ent_end_flag: (seq) 1 if token is the end of an entity
    """
    token_tags = item[
        "token_tags"
    ]  # there is an additional bos token not in str tokens
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq)  # pattern[i][j] is attn i -> j
    ent_end_flag = torch.zeros(seq)  # 1 if token is the end of an entity
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0:  # nothing
            if head != 0:  ##end of entity
                ent_end_flag[i - 1] = 1
                head = 0
            pattern[i][0] = 1
        if tag % 2 == 1:  # if tag is even, this token is the first of an entity
            pattern[i][i] = 1
            head = i  # TODO should the first token of a multi token mention attend to himself anyways ?
        else:  # tag is odd, we are continuing an entity
            pattern[i][head] = 1
    return pattern.unsqueeze(0), ent_end_flag


def get_last_attn_pattern(item):
    """compute target attention pattern from data item with only last token attending to the first token of the entity
    Args:
        item: dictionary with keys 'tokens' and 'ner_tags'
    Returns:
        pattern: (seq, seq) attention pattern
        ent_end_flag: (seq) 1 if token is the end of an entity
    """
    token_tags = item[
        "token_tags"
    ]  # there is an additional bos token not in str tokens
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq)  # pattern[i][j] is attn i -> j
    ent_end_flag = torch.zeros(seq)  # 1 if token is the end of an entity
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0:  # nothing
            if head != 0:  ##end of entity
                ent_end_flag[i - 1] = 1
                pattern[i - 1][head] = 1
                pattern[i - 1][0] = 0
                head = 0
            pattern[i][0] = 1
        if (
            tag % 2 == 1
        ):  # if tag is even, this token is the first of an entity, remember head
            head = i
            pattern[i][0] = 1
        else:
            pattern[i][0] = 1
            # else tag is odd, we are continuing an entity, continue

    return pattern.unsqueeze(0), ent_end_flag


def get_lastonly_attn_pattern(item):
    """compute target attention pattern from data item with only last token attending to the first token of the entity
    Args:
        item: dictionary with keys 'tokens' and 'ner_tags'
    Returns:
        pattern: (seq, seq) attention pattern
        ent_end_flag: (seq) 1 if token is the end of an entity
    """
    token_tags = item[
        "token_tags"
    ]  # there is an additional bos token not in str tokens
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq)  # pattern[i][j] is attn i -> j
    ent_end_flag = torch.zeros(seq)  # 1 if token is the end of an entity
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0:  # nothing
            if head != 0:  ##end of entity
                ent_end_flag[i - 1] = 1
                pattern[i - 1][head] = 1
                head = 0
        if (
            tag % 2 == 1
        ):  # if tag is even, this token is the first of an entity, remember head
            head = i  # TODO should the first token of a multi token mention attend to himself anyways ?
    return pattern.unsqueeze(0), ent_end_flag


def get_block_attn_pattern(item):
    """compute target block attention pattern from data item
    Args:
        item: dictionary with keys 'tokens' and 'ner_tags'
    Returns:
        pattern: (seq, seq) block attention pattern
        ent_end_flag: (seq) 1 if token is the end of an entity
    """
    token_tags = item[
        "token_tags"
    ]  # there is an additional bos token not in str tokens
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq)  # pattern[i][j] is attn i -> j
    ent_end_flag = torch.zeros(seq)  # 1 if token is the end of an entity
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0:  # nothing
            if head != 0:  ##end of entity
                ent_end_flag[i - 1] = 1
                head = 0
            pattern[i][0] = 1
        elif tag % 2 == 1:  # if tag is even, this token is the first of an entity
            pattern[i][i] = 1
            head = i  # TODO should the first token of a multi token mention attend to himself anyways ?
        else:  # tag is odd, we are continuing an entity
            pattern[i][head : i + 1] = 1
    return pattern.unsqueeze(0), ent_end_flag


def get_firstblock_attn_pattern(item):
    """compute target block attention pattern from data item
    Args:
        item: dictionary with keys 'tokens' and 'ner_tags'
    Returns:
        pattern: (seq, seq) block attention pattern
        ent_end_flag: (seq) 1 if token is the end of an entity
    """
    token_tags = item[
        "token_tags"
    ]  # there is an additional bos token not in str tokens
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq)  # pattern[i][j] is attn i -> j
    ent_end_flag = torch.zeros(seq)  # 1 if token is the end of an entity
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0:  # nothing
            if head != 0:  ##end of entity
                ent_end_flag[i - 1] = 1
                head = 0
            pattern[i][0] = 1
        elif tag % 2 == 1:  # if tag is even, this token is the first of an entity
            pattern[i][i] = 1
            head = i  # TODO should the first token of a multi token mention attend to himself anyways ?
        else:  # tag is odd, we are continuing an entity
            pattern[head : i + 1, head : i + 1] = 1
    return pattern.unsqueeze(0), ent_end_flag


def get_blockonly_attn_pattern(item):
    """compute target block attention pattern from data item
    Args:
        item: dictionary with keys 'tokens' and 'ner_tags'
    Returns:
        pattern: (seq, seq) block attention pattern
        ent_end_flag: (seq) 1 if token is the end of an entity
    """
    token_tags = item[
        "token_tags"
    ]  # there is an additional bos token not in str tokens
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq)  # pattern[i][j] is attn i -> j
    ent_end_flag = torch.zeros(seq)  # 1 if token is the end of an entity
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0:  # nothing
            if head != 0:  ##end of entity
                ent_end_flag[i - 1] = 1
                head = 0
        elif tag % 2 == 1:  # if tag is even, this token is the first of an entity
            pattern[i][i] = 1
            head = i
        else:  # tag is odd, we are continuing an entity
            pattern[head : i + 1, head : i + 1] = 1
    return pattern.unsqueeze(0), ent_end_flag


def get_attn_pattern(item, mode: str = PATTERN_MODES.FIRST):
    """get attention pattern for the item, depending on the mode"""
    if mode == PATTERN_MODES.FIRST:
        return get_first_attn_pattern(item)
    elif mode == PATTERN_MODES.LAST:
        return get_lastonly_attn_pattern(item)
    elif mode == PATTERN_MODES.BLOCK:
        return get_firstblock_attn_pattern(item)
    elif mode == PATTERN_MODES.BLOCK_ONLY:
        return get_blockonly_attn_pattern(item)
    else:
        raise ValueError(f"mode must be one of {', '.join(PATTERN_MODES)}, got {mode}")


###################### MAIN DATA CLASS ######################

class NERDataset:
    """Wrapper for all implemented NER datasets
    SCHEMA OF DATA ITEM:
    {
        "text": "John Smith is a person",
        "ner_tags": [1, 2, 0, 0, 0],
        "str_tokens": ["<bos>", "John", "Smith", "is", "a", "person"],
        "token_tags": [0, 1, 2, 0, 0, 0],
        "entities": [{
                    "name": "John Smith",
                    "type": "person",
                    "class": 1,
                    "pos": (0, 10)},
                    "tok_pos": (1,2), # position of the entity in the token_tags
                    },
                    ... ]
        ----- Computed on the fly depending on data mode -----
        "id": 0,
        "pattern": (seq, seq) attention pattern,
        "end_ent": (seq) 1 if token is the end of an entity
        "ent_pos": List of tuples: (beg, end) positions of the entity token-wise
    }
    """

    def __init__(
        self,
        model: Optional[Union[HookedTransformer, AutoModel]] = None,
        tokenizer: Optional[object] = None,
        mode: str = PATTERN_MODES.FIRST,
        max_ent_length=None,
        max_length=None,
    ):
        """NER Dataset class
        Args:
            model: model that will be used, will use its tokenizer to preprocess the data
            mode: mode of attention pattern to use, one of 'first', 'last', 'block'
            max_ent_length: maximum length of an entity
            max_length: maximum TOKEN length of a context
        """
        self.data = []  # list of items, populated by child classes
        self.max_ent_length = max_ent_length
        self.model = model
        
        # TODO remove if we can just pass when model is not set.
        # for now we keep this to avoid breaking changes
        if self.model is not None:
            assert isinstance(model, (HookedTransformer, PreTrainedModel)), f"got model of type {type(model)}"
            self.model = model
            # in our code, we attach the tokenizer to the model (in utils.load_llm)
            self.tokenizer = model.tokenizer
        else:
            assert tokenizer is not None, "tokenizer must be set if no model provided"
            self.tokenizer = tokenizer

        if max_length is not None:  
            
            self.max_length = max_length  # default to model max context length

        elif isinstance(model, HookedTransformer):
            self.max_length = model.cfg.n_ctx
            logging.info(
                f"max_length not specified, defaulting to model context length {self.max_length}"
            )
        else:
            self.max_length = 1024
            logging.info(
                f"max_length not specified, defaulting to {self.max_length}"
            )

        self.type2id = {
            "O": 0,
        }
        self.id2type = {
            0: "O",
        }

        if mode not in list(PATTERN_MODES):
            raise ValueError(
                f"mode {mode} is not one one of {' ,'.join(PATTERN_MODES) }"
            )
        self.mode = mode
        # self.data = []  # list of items, populated by child classes

    #getter for n_classes 
    @property
    def n_classes(self):
        """Number of classes in the dataset"""
        return len(self.type2id)
    
    
    def __post_init__(self):
        """Post init function to be called after the class is initialized"""
        
        assert type(self.data) is list, f"Data must be a list of items, got {type(self.data)}"
        assert type(self.data[0]) is dict, f"Data must be a list of dictionaries, got first item {self.data[0]}"
        
        # sanity checks
        for key in [
            "text",
            "ner_tags",
            "str_tokens",
            "token_tags",
            "entities",
            ]:
            if key not in self.data[0].keys():
                raise ValueError(f"Key {key} not found in data item")
        prev_len = len(self.data)
        # filter out items that are too long
        for item in self.data:
            if len(item["str_tokens"]) > self.max_length or len(item["entities"]) == 0: # type: ignore
                self.data.remove(item)
        if (prev_len - len(self.data)) > 0:
            logging.info(
                f"Filtered {prev_len - len(self.data)} items with more than {self.max_length} tokens out of {prev_len} from dataset."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pattern, end_ent_flag = get_attn_pattern(self.data[idx], mode=self.mode)
        # add pattern to the item
        return {
            "id": idx,
            "pattern": pattern,
            "end_ent": end_ent_flag,
            "ent_pos": [ent["tok_pos"] for ent in self.data[idx]["entities"]],
            **self.data[idx],
        }

    def get_loader(self, batch_size: int = 16):
        """Return a specific DataLoader for this dataset with batches sorted by text length and shuffled"""
        batched_dataset = BatchedDataset(self, batch_size=batch_size)
        return DataLoader(batched_dataset, batch_size=1, collate_fn=lambda x: x[0])

    def get_classes(self, return_counts: bool = False):
        """Get the classes of the dataset"""
        return np.unique(
            np.concatenate(
                [[ent["type"] for ent in item["entities"]] for item in self.data]
            ),
            return_counts=return_counts,
        )

    def clone(self):
        """Clone the dataset into a new instance with the same data"""
        new = NERDataset(
            model=self.model,
            mode=self.mode,
            max_ent_length=self.max_ent_length,
            max_length=self.max_length,
        )
        new.data = self.data.copy()
        new.type2id = self.type2id
        new.id2type = self.id2type
        new.__post_init__()
        return new


# Inference dataset, to store model predictions
class InferredDataset:

    data_name: str
    data_folder: str
    samples: List["Sample"]

    def __init__(self, data_name: str, decoding_strategy: str, threshold: float = 0.5,  data_folder: str = ""):
        self.data_name = data_name
        self.data_folder = data_folder
        self.samples = []
        self.decoding_strategy = decoding_strategy
        self.threshold = threshold

    @dataclass
    class Sample:
        data_id: List[int]
        """Sample id in original dataset"""
        
        text: str
        """Raw text"""
        
        pred_entities: List[dict]
        """dict (name: str, pos: (start: int, end: int))"""

        gt_entities: List[dict]
        """dict (name: str, pos: (start: int, end: int))"""
    
    # 🔹 Save dataset to JSON
    def to_json(self, filepath: str):
        """Save dataset to JSON file"""
        data = {
            "data_name": self.data_name,
            "decoding_strategy": self.decoding_strategy,
            "threshold": self.threshold,
            "data_folder": str(self.data_folder),
            "samples": [asdict(sample) for sample in self.samples],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    @classmethod
    def from_json(cls, filepath: str) -> "InferredDataset":
        """Load dataset from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(
            data_name=data["data_name"], 
            decoding_strategy=data.get("decoding_strategy", "unknown"),
            threshold=data.get("threshold", -1),
            data_folder=data["data_folder"])
        for sample in data["samples"]:
            dataset.samples.append(cls.Sample(**sample))
        return dataset
    
    def __getitem__(self, index):
        return self.samples[index]
    
    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        return f"InferredDataset {self.data_name} with {len(self)} samples"

    @classmethod
    def get_samples_from_outputs(
            self, batch, b_pred_entities, b_span_probs, offsets
            ) -> List["InferredDataset.Sample"]:
        """Creates a Sample from nerModel outputs for a given batch prediction"""
        
        samples = []
        for i in range(len(batch["text"])):
            entities = b_pred_entities[i] # entities token ids
            gt_entities = batch["entities"][i] # entities token ids
            offset = offsets[i]
            text = batch["text"][i]

            ents = []
            for b, e in entities:
                # convert token indices to char indices
                score = b_span_probs[i, e, b].item()
                c_start = offset[b][0].item()
                c_end = offset[e][1].item()
                # remove leading space
                if text[c_start] == " ": c_start += 1  
                ents.append({
                    "name": text[c_start:c_end],
                    "score": score,
                    "pos": (c_start, c_end)
                })

            samples.append(InferredDataset.Sample(
                data_id=batch["id"][i],
                text=text,
                pred_entities=ents,
                gt_entities=gt_entities,
                )
            )
        return samples


    def extract_entities_set(self):
        """Extract entities from the dataset as a set of (data_id, start, end) tuples"""
        entities = set()
        for item in self:
            for entity in item.pred_entities:
                entities.add( (item.data_id, *entity["pos"]) )
        return entities

def compare_inferences(inferences: List[InferredDataset]) -> dict:
    """Compute Intersection over Union of entities in the inferred datasets.
    If more than two datasets are provided, recursively compute the IOU bewteen each pair and return the grid of IOU scores.
    """

    if len(inferences) < 2:
        print("Need at least two inferred datasets to compute IOU")
        return None
    if len(inferences) == 2:
        #extract entities as sets of (data_id, start, end) tuples
        entitiesA, entitiesB = [data.extract_entities_set() for data in inferences]

        # Compute intersection and union
        intersection = entitiesA.intersection(entitiesB)
        union = entitiesA.union(entitiesB)

        iou = len(intersection) / len(union) if len(union) > 0 else 0.0
        dice = 2 * len(intersection) / (len(entitiesA) + len(entitiesB)) if (len(entitiesA) + len(entitiesB)) > 0 else 0.0
        
        return {
            "iou": iou,
            "dice": dice,
            "n_entitiesA": len(entitiesA),
            "n_entitiesB": len(entitiesB),
            "n_intersection": len(intersection),
            "n_union": len(union),
        }
    
    else:
        n = len(inferences)
        iou_matrix = np.zeros((n, n))
        dice_matrix = np.zeros((n, n))

        for i, j in tqdm(product(range(n), repeat=2), total=n**2, desc="Computing IOU/Dice matrices..."):
            if i == j:
                iou_matrix[i, j] = 1.0
                dice_matrix[i, j] = 1.0
            elif i < j:
                result = compare_inferences([inferences[i], inferences[j]])
                iou_matrix[i, j] = result["iou"]
                dice_matrix[i, j] = result["dice"]
            else:
                iou_matrix[i, j] = iou_matrix[j, i]
                dice_matrix[i, j] = dice_matrix[j, i]
        return {
            "iou_matrix": iou_matrix.tolist(),
            "dice_matrix": dice_matrix.tolist(),
        }


### Child classes for specific datasets
class CoNLLDataset(NERDataset):
    def __init__(
        self,
        CoNLLdata,
        model: HookedTransformer,
        mode: str = PATTERN_MODES.FIRST,
        max_ent_length=40,
        max_length=512,
        limit_samples: int = 0,
    ):
        """CoNLL dataset class
        Args:
            CoNLLdata: list of CoNLL dataset items
            model: model that will be used, will use its tokenizer
            max_ent_length: maximum length of an entity
            max_length: maximum length of a context
        """
        super().__init__(
            model=model, mode=mode, max_ent_length=max_ent_length, max_length=max_length
        )

        for i, item in enumerate(CoNLLdata):
            self.data += self.extract_from_item(item)
            if limit_samples and i > limit_samples:
                break

        # reindex
        for i, item in enumerate(self.data):
            item["id"] = i

        self.tokenize_and_augment(model)

    def extract_from_item(self, item):
        ###  CoNLL 2003 tags
        # beg_tags = [1, 3, 5, 7]
        # i_tags   = [2, 4, 6, 8]
        words = item["tokens"]
        item["text"] = " ".join(words)

        # filters
        if all(np.unique(item["ner_tags"]) == [0] ):  
            # if no entities in context
            return []
        else:
            return [item]

    def tokenize_and_augment(self, model: HookedTransformer, verbose: bool = True):
        """Tokenize the texts and compute token-level NER tags from a CoNLL item
        Args:
            model: model that will be used, will use its tokenizer
        """
        for item in tqdm(self.data, disable=not verbose, desc="Processing Data"):
            # Tokenize the sentence
            # raise NotImplementedError("Need to reimplement tokenization for HuggingFace models")
            item["str_tokens"] = model.to_str_tokens( # no more to_str_tokens in HF models
                item["text"], prepend_bos=True
            )  # include bos token

            # Align the tags
            item["token_tags"] = [0] + align_tags_with_tokens(
                item["str_tokens"][1:],  # the function expects tokens without bos token
                item["ner_tags"],
            )


class JSONDataSet(NERDataset):
    def __init__(
        self,
        data_path,
        model: HookedTransformer,
        mode: str = PATTERN_MODES.FIRST,
        max_ent_length=None,
        max_length=None,
        limit_samples: Optional[int] = None,
        **kwargs,
    ):
        """Load a dataset from a json file containing a list of dictionaries with 'sentence' and 'entities' keys.
        Will tokenize the sentence and tag compute tags for each token based on the given entities.

        Args:
            data_path: path to the json file
            model: the HookedTransformer model we will work with, used to tokenize the text.
            max_ent_length: maximum length of an entity
            max_length: maximum length of a sentence
        """
        super().__init__(
            model=model,
            mode=mode, 
            max_ent_length=max_ent_length, 
            max_length=max_length,
            **kwargs
        )

        raw_data = json.load(open(data_path))

        if limit_samples is not None:
            raw_data = raw_data[:limit_samples]

        for item in tqdm(raw_data, desc="Processing Data"):
            if len(item["entities"]) == 0:
                continue
            self.data.append(self.extract_from_item(item))

        self.__post_init__()

    def extract_from_item(self, item):
        text = item["sentence"]
        entities = item["entities"]
        found_entities = []
        data_idx = item.get("data_id", -1)

        # tokenize text
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=True
        )

        offsets = encoding["offset_mapping"][0]  # shape: (num_tokens, 2)
        tokens = encoding["input_ids"][0]  # shape: (num_tokens,)
        str_tokens = to_str_tokens(tokens, self.tokenizer)
        ner_tags = [0] * len(str_tokens)

        # print(text)
        # print("str_tokens", str_tokens)
        # print("entities", entities)
        for entity in entities:
            token_start, token_end = char_to_token_pos(*entity["pos"], offsets)

            if token_start != -1:  # entity found
                entity["tok_pos"] = (token_start, token_end)
                ent_type = entity["type"]
                if ent_type not in self.type2id:
                    self.type2id[ent_type] = len(self.type2id)
                    self.id2type[self.type2id[ent_type]] = ent_type
                entity["class"] = self.type2id[ent_type]

                ner_tags[token_start] = 1
                ner_tags[token_start + 1 : token_end + 1] = [2] * (
                    token_end - token_start
                )
                found_entities.append(entity)
            else:
                logging.warning(f"Sample {data_idx}: Entity {entity} not found in Text, probably due to tokenization truncation.")
                # logging.warning(''.join(str_tokens))
                # logging.warning(f"char_start: {char_start}, char_end: {char_end}")
                # logging.warning(f"str_tokens: {str_tokens}")
                # logging.warning(f"ner_tags: {ner_tags}")
                continue

        return {
            "text": text,
            "token_tags": ner_tags,
            "str_tokens": str_tokens,
            "entities": found_entities,
            "ner_tags": ner_tags,
            "data_id": data_idx,
        }


# Batching is delicate here because of the broad distribution of context lengths
# it is therefore managed manually here and we use a Dataloader with batch size 1 on top of it
class BatchedDataset:
    def __init__(self, dataset: NERDataset, batch_size: int = 16):
        """Wrapper for a dataset to create batches with proper dynamic tokenization"""
        self.dataset = dataset
        self.model = dataset.model
        self.batch_size = batch_size

        # get data indexes sorted by length
        self.data_indx = sorted(
            range(len(self.dataset.data)),
            key=lambda x: len(self.dataset.data[x]["str_tokens"]),
            # reverse=True,
        )

        self.batch_indx = [
            self.data_indx[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(0, len(self.dataset.data) // self.batch_size)
        ]
        # self.data = sorted(self.dataset.data, key=lambda x: len(x["str_tokens"]))
        # self.batches = [self.dataset[i:i+self.batch_size] for i in range(0, len(self.dataset), self.batch_size)]

        # randomize batches
        np.random.shuffle(self.batch_indx)

    # define setter and getter for dataset.mode
    @property
    def mode(self):
        return self.dataset.mode

    @mode.setter
    def mode(self, mode):
        self.dataset.mode = mode
        if mode not in list(PATTERN_MODES):
            raise ValueError(
                f"mode {mode} is not one one of {' ,'.join(PATTERN_MODES) }"
            )

    def __len__(self):
        return len(self.batch_indx)

    def __getitem__(self, idx):
        items_inds = self.batch_indx[idx]

        items = [self.dataset[i] for i in items_inds]

        # Collate the batch
        batch = {key: [item[key] for item in items] for key in items[0].keys()}

        # pad patterns and stack
        max_len = max([pattern.shape[-1] for pattern in batch["pattern"]])
        batch["pattern"] = torch.vstack(
            [
                F.pad(
                    pattern,
                    (0, max_len - pattern.size(1), 0, max_len - pattern.size(2)),
                )
                for pattern in batch["pattern"]
            ]
        )
        batch["token_tags"] = normalize_tags(
            batch["token_tags"]
        )  # normalize tags to 0, 1, 2, pad and stack
        batch["end_ent"] = normalize_tags(
            batch["end_ent"]
        )  # normalize tags to 0, 1 pad and stack
        return batch


def get_span_patterns(batch):
    """Get the span patterns for a batch of this batched dataset
    Args:
        batch: batch of data from a batched dataset
    Returns:
        span_patterns (Tensor[batch, seq, seq]) attention patterns with 1s at
    """
    span_patterns = torch.zeros_like(batch["pattern"])
    for i, ents in enumerate(batch["ent_pos"]):
        for start, end in ents:
            span_patterns[i, end, start] = 1
    return span_patterns


###### DATA LOADING ######
def load_dataset_splits(
    dataset_name: str,
    data_folder: Union[Path, str],
    splits: Optional[List[str]] = None,
    model: HookedTransformer = None,
    tokenizer = None,
    mode: str = PATTERN_MODES.FIRST,
    max_ent_length = None,
    max_length = None,
    val_limit: Optional[int] = None,
) -> dict[str, NERDataset]:
    """Load all data splits for a given dataset name
    Args:
        dataset_name: name of the dataset to load
        model: model that will be used, will use its tokenizer
        splits: list of splits to load, if None, will load all available splits
        limit_samples: limit the number of samples
    Returns:
        data: dictionary of datasets with keys 'train', 'validation', 'test'
    """

    if splits is not None:
        if type(splits) is str:
            splits = [splits]
        assert (
            type(splits) is list
        ), f"splits must be a list of strings, got {type(splits)}"

    val_labels = ["validation", "dev", "val"]
    splits_to_load = ["train", "validation", "test", "dev"]

    if type(data_folder) is not Path:
        data_folder = Path(data_folder)

    if dataset_name.lower() != "conll2003":
        # get all folder names in the data folder
        try:
            folders = [f for f in data_folder.iterdir() if f.is_dir()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data folder {data_folder} not found")

        if dataset_name in [f.name for f in folders]:
            folder = data_folder / dataset_name
            # find json file names in folder, each json file is a data split
            av_splits = [
                f.stem for f in folder.iterdir() if f.is_file() and f.suffix == ".json"
            ]

            if splits is None:
                splits = av_splits
                print(f"Found {len(splits)} files in {folder}: {splits}")
            else:
                # check if splits are in available splits
                for split in splits:
                    if split not in av_splits:
                        raise ValueError(
                            f"Split {split} not found in {folder}, available splits are {av_splits}"
                        )
                # print(f"Found {splits} files in {folder}")

            # filter splits to load
            splits = [split for split in splits if split in splits_to_load]

            if len(splits) == 0:
                raise ValueError(
                    f"No splits found in {folder}, available splits are {splits_to_load}"
                )

            data = {}
            for split in splits:
                print(f"Loading {split} dataset from {folder / f'{split}.json'}")
                data.update(
                    {
                        split: JSONDataSet(
                            folder / f"{split}.json",
                            model=model,
                            tokenizer=tokenizer,
                            mode=mode,
                            max_ent_length=max_ent_length,
                            max_length=max_length,
                            limit_samples=val_limit if split in val_labels else None,
                        )
                    }
                )

            # Harmonize classes between datasets
            harmonize_classes(list(data.values()))

            if len(data.keys()) == 1 and val_limit is not None:
                # if only one split, we add a train and dev split
                print(
                    f"Got only train set of len {len(data['train'])}, splitting into train and dev ({val_limit})"
                )
                # if only one split, we add a train and dev split
                # clone the train set
                data["dev"] = data["train"].clone()
                data["dev"].data = data["train"].data[:val_limit]
                data["train"].data = data["train"].data[val_limit:]

            return data
        else:
            raise ValueError(
                f"Dataset {dataset_name} not found in {data_folder}, available datasets are {[f.name for f in folders]}"
            )

    else:
        # load conll2003 dataset from huggingface
        ds = load_dataset("eriktks/conll2003", trust_remote_code=True)
        # process sepatately each split
        data = {}
        for split in ds.keys():
            data.update(
                {
                    split: CoNLLDataset(
                        ds[split],
                        model,
                        tokenizer=tokenizer,
                        mode=mode,
                        max_ent_length=max_ent_length,
                        max_length=max_length,
                        limit_samples=val_limit if split in val_labels else None,
                    )
                }
            )
        # Harmonize classes between datasets
        harmonize_classes(list(data.values()))
        return data


def fuse_class_dicts(
    type2id_list: List[dict], id2type_list: List[dict]
) -> Tuple[dict, dict]:
    """Fuse multiple type2id and id2type dictionaries into one
    Args:
        type2id_list: list of type2id dictionaries
        id2type_list: list of id2type dictionaries
    Returns:
        type2id: fused type2id dictionary
        id2type: fused id2type dictionary
    """
    type2id = {}
    id2type = {}

    for type2id_dict, id2type_dict in zip(type2id_list, id2type_list):
        for type, id in type2id_dict.items():
            if type not in type2id:
                type2id[type] = id
                id2type[id] = type

    return type2id, id2type


def harmonize_classes(datasets: List[NERDataset]):
    """Harmonize classes between datasets
    Args:
        datasets: list of datasets to harmonize
    """
    assert len(datasets) > 0, "No datasets provided to harmonize classes"

    type2id, id2type = fuse_class_dicts(
        [dataset.type2id for dataset in datasets],
        [dataset.id2type for dataset in datasets],
    )

    
    for dataset in datasets[:]:
        for item in dataset.data:
            for ent in item["entities"]:
                if ent["type"] not in type2id:
                    # if type not in type2id, add it
                    new_id = len(type2id)
                    type2id[ent["type"]] = new_id
                    id2type[new_id] = ent["type"]
                ent["class"] = type2id[ent["type"]]

        dataset.type2id = type2id
        dataset.id2type = id2type
    return


def fuse_datasets(datasets: List[NERDataset], verbose:bool=False) -> NERDataset:
    """
    Fuse multiple NER datasets into a single dataset. will use parameters from the first dataset.
    This function assumes that all datasets have the same model and compatible configurations.
    Args:
        datasets (List[NERDataset]): List of NER datasets to fuse.
    Returns:
        NERDataset: A single fused NER dataset containing all samples from the input datasets.
    """
    model = datasets[0].model  # Assuming all datasets use the same model

    type2id, id2type = fuse_class_dicts(
        [dataset.type2id for dataset in datasets],
        [dataset.id2type for dataset in datasets],
    )
    
    data = []
    for dataset in datasets:
        assert dataset.model == model, "All datasets must use the same model"
        data.extend(dataset.data)

    fused_dataset =  NERDataset(
        model=model,
        max_length=datasets[0].max_length,
        max_ent_length=datasets[0].max_ent_length,
        mode=datasets[0].mode,
    )
    fused_dataset.data = data
    fused_dataset.__post_init__()  # Run post init to filter and reindex
    fused_dataset.type2id = type2id
    fused_dataset.id2type = id2type
    if verbose:
        print(f"Fused dataset of len {len(fused_dataset)}, Harmonized classes: {fused_dataset.get_classes()}")
    
    return fused_dataset


def load_all_splits(dataset_name: str,
    data_folder: Union[Path, str],
    model: HookedTransformer,
    mode: str = PATTERN_MODES.FIRST,
    max_ent_length=None,
    max_length=None,
    val_limit: Optional[int] = None,
    verbose: bool = False,
) -> NERDataset:
    """Load all data splits for a given dataset name and fuse them into a single dataset
    Args:
        dataset_name: name of the dataset to load
        model: model that will be used, will use its tokenizer
        splits: list of splits to load, if None, will load all available splits
        limit_samples: limit the number of samples
    Returns:
        data: NERDataset instance with all data loaded and fused
    """
    datasets = load_dataset_splits(
        dataset_name,
        data_folder=data_folder,
        model=model,
        mode=mode,
        max_ent_length=max_ent_length,
        max_length=max_length,
        val_limit=val_limit,

    )
    return fuse_datasets(list(datasets.values()), verbose=verbose)

##### DATA UTILS #####

def get_ent_pos_classes_flat(item: dict):
    """
    Args:
        item: item from the dataset
    Returns:
        entities (List[#entities, 3 (batch, start, end, class)])
    """
    return [[*ent["tok_pos"], ent["class"]] for ent in item["entities"]]


def b_get_ent_pos_classes_flat(batch: dict):
    """
    Args:
        batch: batch of data
    Returns:
        entities (Tensor[#entities, 3 (batch, start, end, class)])
    """
    return [
        [b, *ent["tok_pos"], ent["class"]]
        for (b, entities) in enumerate(batch["entities"])
        for ent in entities
    ]


# deprecated
# def load_datasets(dataset_name:str = "conll2003",
#                   data_folder:str = "",
#                   model:HookedTransformer = None,
#                   mode:str = PATTERN_MODES.FIRST,
#                   val_limit:int = None,
#                   max_ent_length = 60,
#                   max_length = 400,
#                   ):
#     """Train, Val and Test Dataset loaders for this Experiment
#     Args:
#         dataset_name: name of the dataset to load
#         model: model that will be used, will use its tokenizer
#         val_limit: limit the number of validation samples
#     Returns:
#         train_dataset, val_dataset, test_dataset : NERDataset instances
#     """
#     if not data_folder and dataset_name.lower() != "conll2003":
#         raise ValueError("data_folder must be provided if not using CoNLL2003 !")
#     elif type(data_folder) == str:
#         data_folder = Path(data_folder)


#     else:
#         train_dataset = load_dataset_split(dataset_name, "train", model,
#                                     data_folder=data_folder,
#                                     mode=mode,
#                                     max_ent_length=max_ent_length,
#                                     max_length=max_length)
#         val_dataset = load_dataset_split(dataset_name, "validation", model,
#                                     data_folder=data_folder,
#                                     mode=mode,
#                                     max_ent_length=max_ent_length,
#                                     max_length=max_length,
#                                     limit_samples=val_limit)
#         test_dataset = load_dataset_split(dataset_name, "test", model,
#                                     data_folder=data_folder,
#                                     mode=mode,
#                                     max_ent_length=max_ent_length,
#                                     max_length=max_length)

#     # Harmonize classes between datasets
#     harmonize_classes([train_dataset, val_dataset, test_dataset])

#     return train_dataset, val_dataset, test_dataset
