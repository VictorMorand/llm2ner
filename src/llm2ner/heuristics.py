"""author: Victor Morand
This module contains heuristics to compute for the llm2ner models
"""
import numpy as np
import torch
from jaxtyping import Float, Bool, Int
DEFAULT_THRESHOLD = 0 # threshold for attention LOGIT scores in heuristic



########################################################################
##### Hard-Coded Heuristics to commpute NER Tags from attention patterns
########################################################################
#TODO compile these functions with numba or torch compile


########### first MODE HEURISTICS ###########
def NER_tags_from_first_scores(
        scores: torch.Tensor, # shape ( seq, seq)
        max_ent_length: int, 
        threshold:float) -> torch.Tensor:
    """Heuristic to build NER tags from attention scores
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
        max_ent_length (int): maximum length of an entity
        threshold (float): threshold for attention scores
    Returns:
        ner_tags (seq): NER tags for each token
    """
    # print(scores.shape)
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)
    
    for i in range(seq):
        matches = scores[i,:]
        sorted_idx = torch.argsort(matches, descending=True)
        first_idx = 0
        while 1:
            if sorted_idx[first_idx] == 0: #biggest attention to bos token, no entity
                ner_tags[i] = 0
                break
            # print(i, sorted_idx[first_idx], matches[sorted_idx[first_idx]])
            if threshold is not None and matches[sorted_idx[first_idx]] < threshold: # no match above threshold
                ner_tags[i] = 0
                break
            elif i - sorted_idx[first_idx] > max_ent_length: #entity too long, ignore match and continue
                first_idx += 1
            else: #entity found
                if i == sorted_idx[first_idx]: #first detected token
                    ner_tags[i] = 1
                else: #token points to first detected token
                    # we overwrite previous entities, we could keep them and enable nested entities
                    ner_tags[i] = 2
                    # first = sorted_idx[first_idx]
                    # ner_tags[first] = 1
                    # ner_tags[first+1:i+1] = 2
                break
    return ner_tags

@torch.no_grad()
def NER_tags_from_firstonly_scores_sum(
        scores: torch.Tensor, # shape (1, seq, seq)
        max_ent_length: int, 
        threshold:float) -> torch.Tensor:
    """Heuristic to build NER tags from attention scores in first mode, 
    - CAUSAL ONLY 
    - mask bos token
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
        max_ent_length (int): maximum length of an entity
        threshold (float): threshold for attention scores
    Returns:
        ner_tags (seq): NER tags for each token
    """
    if threshold is None: threshold = DEFAULT_THRESHOLD
  
    # print(scores.shape)
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)

    i = seq - 1
    while i > 0:
        #filter out tokens with no attention
        min_idx = max(1, i - max_ent_length) # consider only spans in eligible range WITOUT BOS TOKEN

        if max(scores[0,i,min_idx:i+1]) > threshold: #no entity
            sum_scores = scores[0,min_idx:i+1,min_idx:i+1].sum(dim=0) #sum on each column 
            # sort first token candidates
            sorted_idx = torch.argsort(sum_scores, descending=True).detach().cpu().tolist() 
            # print(i, sorted_idx, sum_scores)
            first_idx = sorted_idx.pop(0)
            while len(sorted_idx): # we still have candidates
                if sum_scores[first_idx] > threshold: #entity found
                    first_idx += min_idx # we add min_idx to get the real index
                    ner_tags[first_idx:i+1] = 2
                    ner_tags[first_idx] = 1
                    i = first_idx # we go back to the beginning of the entity and continue
                    break
                else: # entity too long, ignore match and continue
                    first_idx = sorted_idx.pop(0)
        i -= 1
    return ner_tags


def NER_tags_from_firstonly_scores(
        scores: torch.Tensor, # shape (1, seq, seq)
        max_ent_length: int, 
        threshold:float = DEFAULT_THRESHOLD) -> torch.Tensor:
    """Heuristic to build NER tags from attention scores in first mode, 
    - CAUSAL ONLY 
    - mask bos token
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
        max_ent_length (int): maximum length of an entity
        threshold (float): threshold for attention scores
    Returns:
        ner_tags (seq): NER tags for each token
    """
    if threshold is None: threshold = DEFAULT_THRESHOLD
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)
    cum_scores = scores.view(seq,seq).cumsum(dim=0) # (seq , seq)
    max_cum_scores_indx = torch.argmax(cum_scores, dim=0) # (seq)
    _, candidates = torch.where( cum_scores[max_cum_scores_indx] > threshold ) # (seq)
    # we start from the end of the sequence
    for idx in candidates.tolist()[::-1]: 
        idx_last = max_cum_scores_indx[idx]
        ner_tags[idx] = 1
        ner_tags[idx+1 : idx_last +1] = 2
    return ner_tags


def NER_tags_from_first_probe(
        scores: Float[torch.Tensor, "seq seq"],
        end_ent: Float[torch.Tensor, "seq"],
        threshold:float,
        max_ent_length: int, 
        ) -> torch.Tensor:
    """Heuristic to build NER tags from attention scores in first mode, 
    - CAUSAL ONLY 
    - mask bos token
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
        max_ent_length (int): maximum length of an entity
        threshold (float): threshold for attention scores
    Returns:
        ner_tags (seq): NER tags for each token
    """
    if threshold is None: threshold = DEFAULT_THRESHOLD

    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)
    cum_scores = scores.view(seq,seq).cumsum(dim=0) # (seq , seq)
    
    # print(cum_scores.shape)
    # print(end_ent)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(cum_scores)
    # plt.colorbar()
    # plt.show()

    candidates = torch.where( end_ent > threshold )[0] # (seq)
    # we start from the end of the sequence
    for idx_last in candidates.tolist()[::-1]: 
        min_idx = max(1, idx_last - max_ent_length) # consider only spans in eligible range WITOUT BOS TOKEN
        cand_scores = cum_scores[idx_last, min_idx:idx_last+1]
        cand_idx_first = torch.where(cand_scores > threshold)[0]
        
        if len(cand_idx_first) == 0: 
            continue #no entity
        else:
            idx_first = cand_idx_first[-1].item()
            idx_first += min_idx # we add min_idx to get the real index
            ner_tags[idx_first:idx_last+1] = 2
            ner_tags[idx_first] = 1
        
    return ner_tags

########### LAST MODE HEURISTICS ###########


def NER_tags_from_last_scores(
        scores: torch.Tensor, # shape (1, seq, seq)
        max_ent_length: int, 
        threshold:float) -> torch.Tensor:
    """Build NER tags from attention scores in last mode
    TODO: implement Nested, for now we keep only the widest span
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
    Returns:
        ner_tags (seq): NER tags for each token
    """
    # print(scores.shape)
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)

    for i in range(seq):
        matches = scores[0,i,:]
        sorted_idx = torch.argsort(matches, descending=True)
        if sorted_idx[0] == 0: #biggest attention to bos token, no entity
            ner_tags[i] = 0
        else :
            first_idx = 0
            while 1:
                # print(i, sorted_idx[first_idx], matches[sorted_idx[first_idx]])
                if threshold is not None and matches[sorted_idx[first_idx]] < threshold: # no match above threshold
                    ner_tags[i] = 0
                    break
                elif i - sorted_idx[first_idx] > max_ent_length: #entity too long, ignore match and continue
                    first_idx += 1
                else: #entity found
                    first_idx = sorted_idx[first_idx]
                    ner_tags[first_idx:i+1] = 2
                    ner_tags[first_idx] = 1
                    break
    return ner_tags


def NER_tags_from_lastonly_scores(
        scores: Float[torch.Tensor, "seq seq"],
        max_ent_length: int, 
        threshold:float) -> torch.Tensor:
    """Build NER tags from attention scores in last mode
    TODO: implement Nested, for now we keep only the widest span
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
    Returns:
        ner_tags (seq): NER tags for each token
    """
    # print(scores.shape)
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)

    for i in range(seq-1,1,-1):
        min_idx = max(1, i - max_ent_length) # consider only spans in eligible range WITOUT BOS TOKEN
        matches = scores[i,min_idx:i+1]
        # print(i, matches)
        first_idx = torch.argmax(matches)
        if matches[first_idx] > threshold: #entity found

            first_idx += min_idx
            ner_tags[first_idx:i+1] = 2
            ner_tags[first_idx] = 1
    return ner_tags

########### BLOCK MODE HEURISTICS ###########

def NER_tags_from_block_scores(
        scores: torch.Tensor, # shape (1, seq, seq)
        max_ent_length: int, 
        threshold:float,
        causal: bool = True,
        patience: int = 3
        ) -> torch.Tensor:
    """Build NER tags from attention scores in 'block' mode
    TODO: implement Nested, for now we keep only the widest span
    Args:
        scores (1, seq, seq): attention scores between all pairs of tokens
        max_ent_length (int): maximum length of an entity
        patience (int): number of tokens to wait for sum of attention scores to be maximal
    Returns:
        ner_tags (seq): NER tags for each token
    """
    # print(scores.shape)
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)

    for i in range(1,seq): # for all tokens in the sequence
        if causal:
            matches = scores[0,i,:i+1] # attention scores for token i
        else:
            matches = scores[0,i,:] # attention scores for token i
        sorted_idx = torch.argsort(matches, descending=True)
        if sorted_idx[0] == 0: #biggest attention to bos token, no entity
        # if matches[0] > matches[i]: #more attention to bos token, no entity
            ner_tags[i] = 0
        else :
            sc = []
            seq = scores.shape[-1]
            # biggest attn is not on bos, we are likely in an entity:
            causal_mask = ~torch.triu(torch.ones(seq, seq), diagonal=1).to(torch.bool)
            for j in range(i,max(0, i - max_ent_length - 1), -1):
                # mask of indexes btw i-j and i
                mask = torch.zeros(seq, seq).to(torch.bool)
                mask[j:i+1,j:i+1] = True
                mask = (mask & causal_mask)
                # print(mask.int())
                sc.append(scores.squeeze(0)[mask].sum().item())
            if patience is not None:
                m = -1e9
                wait = 0
                ent_length = 1
                for j, s in enumerate(sc):
                    if s > m:
                        m = s
                        ent_length = j + 1
                        wait = 0
                    else:
                        wait += 1
                    if wait == patience:
                        break
            else : 
                # without patience, we take the longest entity enven if their is a gap in cumulated attention scores
                ent_length = np.argmax(sc) + 1
            # write found entity in tags, overwrite if another entity was found before
            # enable nested entities if there are multiple 1s 
            ner_tags[i-ent_length+1:i+1] = 2
            ner_tags[i-ent_length+1] = 1
    return ner_tags

def NER_tags_from_blockonly_heuristic(
        scores: torch.Tensor, # shape (1, seq, seq)
        max_ent_length: int, 
        threshold:float, 
        patience:int,
        causal:bool = True, ) -> torch.Tensor:
    """Build NER tags from attention scores in 'block_only' mode
    This is a heuristic trying to find the min BCE solution from logit scores.
    Args:
        scores (1, seq, seq): attention LOGIT scores between all pairs of tokens
        max_ent_length (int): maximum length of an entity
        patience (int): number of tokens to wait for sum of attention scores to be maximal
    Returns:
        ner_tags (seq): NER tags for each token
    """
    # print(scores.shape)
    if threshold is None: threshold = DEFAULT_THRESHOLD
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)
    i = seq - 1
    while i > 0: # for all tokens in the sequence in reverse order (except bos token)
        if scores[0,i,i] < threshold:   
            # print(i, "nothing")
            i -= 1 #do nothing
        else: #we might be in an entity
            sc = []
            b = i #begin of entity
            wait = 0
            for j in range(i, max(0, i - max_ent_length - 1), -1):
                s = scores[0,j:i+1,j:i+1].sum().item() # sum logits on square submatrix j -> i
                sc.append(s)
                if s > sc[i-b]: # we have a new max score
                    b = j
                    wait = 0
                elif patience is not None:
                    wait += 1
                    if wait == patience:
                        break
                else: #maximum reached without patience
                    break
            # print("found", b, i, sc )
            # write found entity (b,i) in tags, overwrite if another entity was found before
            # enable nested entities if there are multiple 1s 
            ner_tags[b] = 1
            ner_tags[b+1:i+1] = 2
            i = b - 1 # we go back to the beginning of the entity and continue
    return ner_tags

def NER_tags_from_blockonly_probe(
                        scores: Float[torch.Tensor, "seq seq"],
                        end_ent: Float[torch.Tensor, "seq"],
                        threshold:float,
                        patience:int,
                        max_ent_length:int,
                        ) -> Int[torch.Tensor, "batch seq"]:
    """Compute NER tags for given scores and end_ent
    Args:
        scores: tensor (seq, seq) attention scores (logits) between all pairs of tokens
        end_ent: tensor (seq) end of entity probe logits
        threshold: threshold for attention scores
        patience: (block mode) number of tokens to wait for sum of attention scores to be maximal
        max_ent_length: maximum length of entities
    Returns:
        ner_tags: tensor (batch, seq) NER tags for each token in the text
    """
    seq = scores.size(-1)
    scores = scores
    end_ent = end_ent
    ner_tags = torch.zeros(seq, dtype=torch.int)

    i = seq - 1
    while i > 0: # for all tokens in the sequence in reverse order (except bos token)
        if end_ent[i] < threshold:   
            i -= 1 #do nothing
        else: #we might be in an entity
            sc = []
            b = i #begin of entity
            wait = 0
            for j in range(i, max(0, i - max_ent_length - 1), -1):
                s = scores[j:i+1,j:i+1].sum().item() # sum logits on square submatrix j -> i
                sc.append(s)
                if s > sc[i-b]: # we have a new max score
                    b = j
                    wait = 0
                elif patience is not None:
                    wait += 1
                    if wait == patience:
                        break
                else: #maximum reached without patience
                    break
            # print("found", b, i, sc )
            # write found entity (b,i) in tags, overwrite if another entity was found before
            # enable nested entities if there are multiple 1s 
            ner_tags[b] = 1
            ner_tags[b+1:i+1] = 2
            i = b - 1 # we go back to the beginning of the entity and continue
    
    return ner_tags
