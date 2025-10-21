import time, torch, logging, gc, json
import numpy as np
from pathlib import Path
from jaxtyping import Float, Bool, Int
from typing import (
    List,
    Tuple,
    Callable,
    Type,
    Union,
    Optional,
    Dict,
    Any,
    overload,
)  # , TypeVar
from functools import lru_cache, partial
from tqdm import tqdm
from enum import Enum

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# HF and Tlens
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
)
from transformers import PreTrainedTokenizerBase

# xpm and misc
from experimaestro import (
    Config,
    Param,
    DataPath,
    Task,
    LightweightTask,
    Meta,
    Param,
    Constant,
)
from rotary_embedding_torch import RotaryEmbedding

# Our code
from llm2ner import utils, masks, heuristics
import llm2ner.data as data
from llm2ner.losses import balanced_BCE
from llm2ner.models.model import *


######################  Attn score manipulation ######################


@lru_cache
def get_norm_mat(seq: int) -> Float[torch.Tensor, "seq seq"]:
    """Get normalization matrix for the attention scores
    Args:
        seq: sequence length
    Returns:
        norm_mat: tensor (seq, seq) normalization matrix
    """
    return torch.nan_to_num(
        1 / torch.cumsum(torch.triu(torch.ones(seq, seq), diagonal=0), dim=1).T,
        posinf=0.0,
        neginf=0.0,
    ).unsqueeze(0)


@torch.jit.script
def shift_left(a):
    return F.pad(a, (-1, 1), mode="constant", value=0.0)


@torch.jit.script
def shift_right(a):
    return F.pad(a, (1, -1), mode="constant", value=0.0)


@torch.jit.script
def shift_up(a):
    return F.pad(a, (0, 0, -1, 1), mode="constant", value=0.0)


@torch.jit.script
def shift_down(a):
    return F.pad(a, (0, 0, 1, -1), mode="constant", value=0.0)


@torch.jit.script
def max_pool_right(a: torch.Tensor):
    """max pool over the last dimension, pooling elements on the right of the current element"""
    seq = a.shape[-1]
    return F.max_pool2d(
        F.pad(a, (0, seq - 1), mode="constant", value=0.0),
        stride=(1, 1),
        kernel_size=(1, seq),
    )  # remove the last seq-1 elements which are padded


@torch.jit.script
def max_pool_right_strict(a: torch.Tensor):
    """max pool over the last dimension, pooling elements on the rigth of the current element"""
    seq = a.shape[-1]
    return F.max_pool2d(
        F.pad(a, (-1, seq), mode="constant", value=0.0),
        stride=(1, 1),
        kernel_size=(1, seq),
    )  # remove the last seq-1 elements which are padded


@torch.jit.script
def min_pool_right_strict(a: torch.Tensor):
    """min pool over the last dimension, pooling elements on the rigth of the current element"""
    seq = a.shape[-1]
    return -F.max_pool2d(
        F.pad(-a, (-1, seq), mode="constant", value=0.0),
        stride=(1, 1),
        kernel_size=(1, seq),
    )  # remove the last seq-1 elements which are padded


######################  Blocks DEFINITION ######################


###### Bilinear Match of Tokens
class SelfAttention(nn.Module):

    def __init__(
        self,
        model_dim: int,
        k: int = 0,
        causal_mask: bool = True,
        sliding_window: int = 0,
        mask_bos: bool = True,
        apply_softmax: bool = False,
        scale: float = None,
        init_identity: bool = False,
    ):
        """One head of self attention mechanism, with custom query and key rank
        Args:
            dim: dimension of the representations
            k: rank of the query and key matrices, if 0, will be equal to dim
            scale: scaling factor for the scores, if None, will be set to 1/sqrt(dim)
            init_identity: whether to initialize query and key matrices as identity matrices
            mask_bos: whether block tokens from attending to bos token (first in seqence), default True
            causal_mask: whether to mask future tokens, default True
        """
        super().__init__()
        # register name as buffer
        self.mode = None  # mode to use for NER inference, will be set by the dataset on which the model is trained
        self.llm_name = None  # name of the LLM that produced the representations this model is trained on
        self.model_dim = model_dim
        self.apply_softmax = apply_softmax
        self.scale = scale if scale else 1 / torch.math.sqrt(self.model_dim)
        self.k = k if k else self.model_dim
        self.mask_bos = mask_bos
        self.mask_val = 0
        self.sliding_window = sliding_window
        self.causal_mask = causal_mask

        if init_identity:
            self.Q = nn.Parameter(
                torch.eye(self.model_dim, self.k),
            )
            self.K = nn.Parameter(torch.eye(self.model_dim, self.k))
        else:
            self.Q = nn.Parameter(torch.randn(self.model_dim, self.k))
            self.K = nn.Parameter(torch.randn(self.model_dim, self.k))

    def __str__(self):
        return f"Attention model:\n \
            - dim: {self.model_dim},\n \
            - k: {self.k},\n \
            - causal_mask: {self.causal_mask},\n \
            - sliding_window: {self.sliding_window},\n \
            - scale: {self.scale},\n \
            - mask_bos: {self.mask_bos}"

    def scores(
        self,
        query: Float[torch.Tensor, "batch seq dim"],
        key: Float[torch.Tensor, "batch seq dim"],
    ) -> Float[torch.Tensor, "batch seq"]:
        """Compute match between two batches of representations q and k,
        M(q, k) = q Q K^T k^T
        Args:
            query: tensor (batch, seq, dim) batch of query representations
            key: tensor (batch, seq, dim) batch of key representations
        """
        return self.scale * torch.einsum(
            "...bd,...bd->...b", (query @ self.Q), (key @ self.K)
        )

    def selfAttnScores(
        self, reps: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq seq"]:
        """Compute match between all pairs of representations reps
        M(q, k) = q Q K^T k^T
        Args:
            query: tensor (batch, seq, dim) batch of query representations
            key: tensor (batch, seq, dim) batch of key representations
        """
        return self.scale * torch.einsum(
            "bih,bjh->bij", (reps @ self.Q), (reps @ self.K)
        )

    def forward(
        self,
        reps: Float[torch.Tensor, "batch seq dim"],
        return_mask: bool = False,
    ) -> Float[torch.Tensor, "batch seq seq"]:
        """Compute self attention for given set of representations or hidden states
        Args:
            reps: tensor (batch, seq, dim) batches of representations
            mask: tensor (batch, seq, seq) mask to apply to the scores
        Returns:
            scores: tensor (batch, seq, seq) attention scores between all pairs of tokens
        """
        seq = reps.size(1)
        mask = masks.create_mask_cached(
            1,
            1,
            seq,
            seq,
            causal=self.causal_mask,
            mask_bos=self.mask_bos,
            sliding_window=self.sliding_window,
        ).squeeze(
            1
        )  # shape (1, seq, seq)

        # compute all scores and mask
        scores = self.scale * torch.einsum(
            "b i h, b j h -> b i j", (reps @ self.Q), (reps @ self.K)
        )

        # apply softmax
        if self.apply_softmax:
            scores = F.softmax(scores, dim=-1)

        scores = scores.masked_fill(~mask, self.mask_val)

        if return_mask:
            return scores, mask
        else:
            return scores

    def get_tags_heuristic(
        self,
        scores: Float[torch.Tensor, "batch seq seq"],
        max_ent_length: int = MAX_ENT_LENGTH,
        threshold: float = 0,
        patience: int = 2,
    ):
        """Get NER tags from attention scores, depending on mode"""
        scores = scores.cpu()
        if self.mode == "last":
            if self.mask_bos:
                return heuristics.NER_tags_from_lastonly_scores(
                    scores, max_ent_length=max_ent_length, threshold=threshold
                )
            else:
                return heuristics.NER_tags_from_last_scores(
                    scores, max_ent_length=max_ent_length, threshold=threshold
                )

        elif self.mode == "first":
            if self.mask_bos:
                return heuristics.NER_tags_from_firstonly_scores(
                    scores, max_ent_length=max_ent_length, threshold=threshold
                )
            else:
                return heuristics.NER_tags_from_first_scores(
                    scores, max_ent_length=max_ent_length, threshold=threshold
                )

        elif self.mode == "block":
            if self.mask_bos:
                return heuristics.NER_tags_from_blockonly_heuristic(
                    scores,
                    max_ent_length=max_ent_length,
                    threshold=threshold,
                    causal=self.causal_mask,
                    patience=patience,
                )
            else:
                return heuristics.NER_tags_from_block_scores(
                    scores, max_ent_length=max_ent_length, patience=patience
                )

        elif self.mode == "block_only":
            if threshold is None:
                threshold = 0
            return heuristics.NER_tags_from_blockonly_heuristic(
                scores,
                max_ent_length,
                threshold=threshold,
                causal=self.causal_mask,
                patience=patience,
            )
        else:
            raise ValueError(f"Mode {self.mode} not recognized")

    @torch.no_grad()
    def validate_BCE(
        self, model: HookedTransformer, layer: int, val_loader, verbose: bool = False
    ):
        """Run validation metric on given loader
        Args:
            model: HookedTransformer form TransformerLens to extract representations from
            layer: layer at which to retreive the representations
            attn: attention model to validate
            val_loader: DataLoader for validation data
            metric ("bce", "f1"): metric to use for validation, default "bce"
            verbose: whether to log the validation metric
        Returns:
            val_metric: validation loss with given metric
        """
        val_metric = 0
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(1),  # no positive bias in validation
            reduction="mean",
        )

        for batch in tqdm(val_loader, disable=not verbose):
            texts = batch["text"]
            inputs = model.tokenizer(
                texts, padding=True, padding_side="right", return_tensors="pt"
            )
            tokens = inputs["input_ids"].cuda()
            attn_mask = inputs["attention_mask"].cuda()

            batch_size, seq = tokens.shape
            patterns = batch["pattern"].cuda()  # tensor shape (batch, seq, seq)
            masked_tokens = tokens == model.tokenizer.pad_token_id  # shape (batch, seq)
            padded_mask = masked_tokens.unsqueeze(1) | masked_tokens.unsqueeze(
                2
            )  # transform to padded mask to 2D mask

            # we batch the forward pass of representations and attention scores
            reps = utils.compute_to_layer(
                model, layer, tokens, attn_mask=attn_mask
            ).cuda()  # shape (batch, seq, dim)
            scores, mask = self.forward(
                reps, return_mask=True
            )  #  (batch, seq, seq) | (1, seq, seq)
            mask = mask.repeat(batch_size, 1, 1) & ~padded_mask

            val_metric += criterion(scores[mask], patterns[mask]).item()

            # import matplotlib.pyplot as plt
            # plt.imshow(mask[0].cpu().int())
            # plt.show()
            # raise ValueError("stop")

        val_metric /= len(val_loader) * batch_size
        if verbose:
            logging.info(f"Validation loss: {val_metric}")
        return val_metric

    def train(
        self,
        layer: int,
        train_loader,
        val_loader,
        hist=[],
        epochs=2,
        lr=1e-3,
        grad_clip=0,
        pos_weight=1,
        accumulation_steps=1,
        patience=3,
        min_lr=1e-5,
        n_val=100,
        val_metric="bce",
        dilate_entities=None,
    ):
        """Train attention model on the given dataset
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            end_classifier: (Module, optionnal) binary classifier to train on top of the attention scores
            sliding_window: whether to use a sliding window attention, default 0 : no sliding window
            hist: history of training
            epochs: number of epochs to train
            lr: learning rate
            grad_clip (float): gradient clipping value, if 0, no clipping
            pos_weight: weight for positive examples in BCE loss, default 1 = no weighting
            accumulation_steps: number of steps to accumulate gradients before updating weights
            patience: number of untolerable validations to wait before reducing learning rate
            min_lr: minimum learning rate, if reached, training stops
            n_val: number of steps between validation and logging
            val_metric: metric to use for validation, default "bce"
            dilate_entities: Compute the loss only on entities (dilated) and not the full sequence
        """
        if self.mode != train_loader.dataset.mode:
            logging.info(
                f"Warning: Current mode of the model {self.mode} differs from dataset ({train_loader.dataset.mode}), changing..."
            )
            self.mode = train_loader.dataset.mode
        self.layer = layer
        model = train_loader.dataset.model
        batch_size = (
            train_loader.dataset.batch_size
        )  # batch size is stored in the dataset, loader batch size is 1
        n_val = n_val // batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_weights = self.state_dict()
        best_val = torch.math.inf if val_metric == "bce" else 0
        pad_id = model.tokenizer.pad_token_id

        if dilate_entities is not None:
            try:
                dilate_entities = torch.tensor(dilate_entities).int().view(-1)
            except TypeError:
                raise ValueError(
                    "if not None, dilate_entities should be an int or a list of two ints"
                )
            # now we have a tensor
            if len(dilate_entities) == 1:
                dilate_entities = dilate_entities.item()
                if not dilate_entities:
                    dilate = None
                    dilate_entities = None
                else:
                    dilate = masks.get_dilation(dilate_entities)
            elif len(dilate_entities) == 2:
                dilate = masks.get_2d_dilation(dilate_entities)
            else:
                raise ValueError(
                    "if not None, dilate_entities should be an int or a list of two ints"
                )

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight), reduction="mean"
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min" if val_metric == "bce" else "max",
            factor=0.5,
            patience=patience,
        )

        logging.info(
            f"Training Self Attention at layer {layer} for {epochs} epochs with batch size {batch_size} \n\
            - dilate_entities: {dilate_entities} \n\
            - pos_weight: {pos_weight} \n\
            - Causal mask {self.causal_mask}\n\
            - bos mask {self.mask_bos}\n\
            - Sliding Window {self.sliding_window}... "
        )

        if len(hist):
            prev_samples = hist[-1]["samples"]
            prev_epoch = hist[-1]["epoch"]
        else:
            prev_samples = 0
            prev_epoch = 0

        def validate():
            # validate and log
            nonlocal best_val, best_weights
            lr = optimizer.param_groups[0]["lr"]

            if val_metric == "bce":
                val = self.validate_BCE(model, layer, val_loader, verbose=False)
                if val < best_val:
                    best_val = val
                    best_weights = self.state_dict()
            elif val_metric in ["precision", "recall", "f1"]:
                val = compute_metrics(val_loader, model, self, layer, verbose=False)[
                    val_metric
                ]
                if val > best_val:
                    best_val = val
                    best_weights = self.state_dict()
            hist[-1]["val_metric"] = val
            logging.info(
                f"\nSample {hist[-1]['samples']} mean loss: {np.mean([h['loss'] for h in hist[-n_val:]]):.4f}, val {val_metric}: {val:.3f}, lr: {lr:.2e}"
            )
            scheduler.step(val)

        def debug_train():
            nonlocal item_mask, scores, patterns, mask, tags, padded_mask
            import matplotlib.pyplot as plt

            print(item_mask.shape, scores.shape, patterns.shape, mask.shape, tags.shape)
            n = 3

            fig, ax = plt.subplots(5, n, figsize=(13, 15))
            for i in range(n):
                ind = torch.randint(batch_size, (1,)).item()
                ax[0, i].imshow(patterns[ind, :, :].cpu().int())
                ax[0, i].set_title(f"mask {ind}")
                ax[1, i].imshow(F.sigmoid(scores[ind, :, :]).cpu().int())
                ax[1, i].set_title(f"scores {ind}")
                ax[2, i].imshow(mask[0, :, :].cpu().int())
                ax[2, i].set_title(f"mask {ind}")
                ax[3, i].imshow(padded_mask[ind, :, :].cpu().int())
                ax[3, i].set_title(f"padded_mask {ind}")
                ax[4, i].imshow(item_mask[ind, :, :].cpu().int())
                ax[4, i].set_title(f"item_mask {ind}")
            plt.show()
            raise ValueError("stop")

        # train loop
        i = 0  # step counter
        for epoch in range(prev_epoch + 1, prev_epoch + epochs + 1):
            optimizer.zero_grad()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

                # stop if lr is too low
                if optimizer.param_groups[0]["lr"] < min_lr:
                    logging.info(f"Minimum learning rate reached, stopping training")
                    break

                i += 1
                texts = batch["text"]
                tags = batch["token_tags"].float().cuda()
                patterns = batch["pattern"].cuda()  # tensor shape (batch, seq, seq)
                inputs = model.tokenizer(
                    texts, padding=True, padding_side="right", return_tensors="pt"
                )
                tokens = inputs["input_ids"].cuda()
                attn_mask = inputs["attention_mask"].cuda()

                padded = tokens == pad_id  # shape (batch, seq)
                padded_mask = padded.unsqueeze(1) | padded.unsqueeze(
                    2
                )  # transform to padded mask to 2D mask

                # we batch the computation of representations and attention scores
                with torch.no_grad():  # we don't need gradients for the representations
                    reps = utils.compute_to_layer(
                        model, layer, tokens, attn_mask=attn_mask
                    ).cuda()  # shape (batch, seq, dim)

                ## TRAINING Self Attention
                # TODO: fow now we compute all scores, we could optimize this by computing only the scores we need
                # TODO: add padded mask to forward
                scores, mask = self.forward(
                    reps, return_mask=True
                )  #  (batch, seq, seq) | (1, seq, seq)
                b_size, _, seq = patterns.shape

                # compute loss
                if (
                    dilate_entities is None
                ):  # compute loss on everything, take orginal mask
                    item_mask = mask.repeat(b_size, 1, 1) & ~padded_mask

                elif (
                    type(dilate_entities) == int
                ):  # we only compute the loss on the entity *rows*
                    # dilate and inflate mask to match the shape of the pattern
                    ner_tags = dilate(tags)  # shape (batch, seq)
                    ner_tags = ner_tags.unsqueeze(1) | ner_tags.unsqueeze(
                        2
                    )  # shape (batch, seq, seq)
                    item_mask = mask & ner_tags & ~padded_mask

                elif (
                    len(dilate_entities) == 2
                ):  # we only compute the loss on the matches around entity blocks
                    dilated_mask = dilate(patterns * mask).squeeze(
                        0
                    )  # shape (batch, seq, seq)
                    item_mask = dilated_mask & mask & ~padded_mask
                else:
                    raise NotImplementedError(
                        f"shouldn't be here, dilate_entities should be None, 1 or 2, not {dilate_entities}"
                    )

                loss = criterion(
                    scores[item_mask],
                    patterns[item_mask],
                )

                #### DEBUG plot the masks side by side
                # debug_train()

                loss /= batch_size * accumulation_steps
                loss.backward()  # backward pass

                # Accumulate gradients and update weights every n_val steps
                if (i + 1) % accumulation_steps == 0:
                    if grad_clip:
                        clip_grad_norm_(
                            self.parameters(), grad_clip
                        )  # gradient clipping
                    optimizer.step()
                    optimizer.zero_grad()

                h = {
                    "epoch": epoch,
                    "samples": i * batch_size + prev_samples,
                    "loss": loss.item() * batch_size,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                hist.append(h)

                if (len(hist) - 1) % n_val == 0:
                    validate()

            # validate at the end of the epoch
            if (i + 1) % n_val != 0:
                validate()

            if optimizer.param_groups[0]["lr"] < min_lr:
                break

        self.load_state_dict(best_weights)
        return hist


class ReprClassifier(nn.Module):
    def __init__(self, model_dim: int, n_classes: int = 1):
        """Linear classifier on top of representations
        Args:
            dim: dimension of the representations
            n_classes: number of classes to classify
        """
        super().__init__()
        self.layer = None
        self.model_dim = model_dim
        self.llm_name = None  # name of the LLM that produced the representations this model is trained on
        self.fc = nn.Linear(model_dim, n_classes)

    def forward(
        self, reps: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq"]:
        """Compute NER tags for given text,
        implemented in the child class
        Args:
            reps: tensor (batch, seq, dim) representations of tokens to classify
        Returns:
            tags: tensor (batch, seq) NER tags for each token in the text
        """
        return self.fc(reps).squeeze(-1)  # shape (batch, seq)

    @torch.no_grad()
    def evaluate(self, layer: int, val_loader, val_metric="bce", verbose: bool = True):
        """Run validation metric on given loader
        Args:
            model: HookedTransformer form TransformerLens to extract representations from
            layer: layer at which to retreive the representations
            attn: attention model to validate
            val_loader: DataLoader for validation data
            metric ("bce", "precision", "recall", "f1"): metric to use for validation, default "bce"
            verbose: whether to log the validation metric
        Returns:
            val_metric: validation loss with given metric
        """
        # sanity check
        model = val_loader.dataset.model
        assert (
            self.llm_name == model.cfg.model_name
        ), f"probe model {self.llm_name} and transformer model {model.cfg.model_name} should be the same"
        assert (
            self.model_dim == model.cfg.d_model
        ), f"probe dim {self.model_dim} and transformer dim {model.cfg.d_model} should be the same"
        assert (
            self.layer == layer
        ), f"probe layer {self.layer} and transformer layer {layer} should be the same"

        pad_id = model.tokenizer.pad_token_id
        bos_id = model.tokenizer.bos_token_id
        val = 0
        tp = 0
        fp = 0
        tot = 0
        batch_size = val_loader.dataset.batch_size
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(1),  # no positive bias in validation
            reduction="sum",
        )
        if verbose:
            logging.info(f"Computing validation loss with {val_metric} metric")

        for batch in tqdm(val_loader, disable=not verbose):
            texts = batch["text"]
            inputs = model.tokenizer(
                texts, padding=True, padding_side="right", return_tensors="pt"
            )
            tokens = inputs["input_ids"].cuda()
            attn_mask = inputs["attention_mask"].cuda()

            end_ent = batch["end_ent"].cuda()

            # masking
            mask = (tokens != pad_id) & (tokens != bos_id)  # shape (batch, seq)

            # Forward pass
            reps = utils.compute_to_layer(
                model, layer, tokens, attn_mask=attn_mask
            ).cuda()  # shape (batch, seq, dim)

            # mask tokens
            reps = reps[mask]  # shape (batch, seq, dim)
            end_ent = end_ent[mask]  # shape (batch, seq)

            logits = self.forward(reps)

            # compute loss
            if val_metric == "bce":
                val += criterion(logits, end_ent).item()
            else:
                preds = (logits > 0).float()
                tp += (preds * end_ent).sum()
                fp += (preds * (1 - end_ent)).sum()
                tot += end_ent.sum()

        if val_metric == "bce":
            if verbose:
                logging.info(f"Validation loss: {val}")
            val /= len(val_loader) * batch_size
            return val
        elif val_metric in ["precision", "recall", "f1", "all"]:
            precision = tp / (tp + fp + EPS)
            recall = tp / (tot + EPS)
            f1 = 2 * precision * recall / (precision + recall + EPS)
            metrics = {"precision": precision, "recall": recall, "f1": f1}
            return metrics[val_metric] if val_metric != "all" else metrics
        else:
            raise ValueError(f"metric {val_metric} not implemented")

    def train(
        self,
        layer: int,
        train_loader,
        val_loader,
        hist=[],
        epochs=2,
        lr=1e-3,
        grad_clip=0,
        pos_weight=1,
        accumulation_steps=1,
        patience=3,
        min_lr=1e-5,
        n_val=100,
        dilate_entities=None,
    ):
        """Train attention model on the given dataset

        Args:
            model: HookedTransformer form TransformerLens to extract representations from
            attn: attention model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            end_classifier: (Module, optionnal) binary classifier to train on top of the attention scores
            hist: history of training
            epochs: number of epochs to train
            lr: learning rate
            grad_clip (float): gradient clipping value, if 0, no clipping
            accumulation_steps: number of steps to accumulate gradients before updating weights
            patience: number of untolerable validations to wait before reducing learning rate
            min_lr: minimum learning rate, if reached, training stops
            n_val: number of steps between validation and logging
        """
        self.layer = layer  # layer at which to extract the representations, remember for validation
        model = train_loader.dataset.model
        model_name = model.cfg.model_name
        if self.llm_name != model_name:
            if self.llm_name is None:
                logging.info(
                    f"Current model {self.llm_name} differs from dataset model {model_name}, changing..."
                )
            self.llm_name = model_name

        batch_size = (
            train_loader.dataset.batch_size
        )  # batch size is stored in the dataset, loader batch size is 1
        n_val = n_val // batch_size
        val_metric = "bce"
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min" if val_metric == "bce" else "max",
            factor=0.5,
            patience=patience,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight),  # no positive bias in validation
            reduction="mean",
        )

        best_weights = self.state_dict()
        best_val = torch.math.inf

        pad_id, bos_id = model.tokenizer.pad_token_id, model.tokenizer.bos_token_id
        dilation = (
            masks.get_dilation(dilate_entities) if dilate_entities is not None else None
        )
        if len(hist):
            prev_epoch, prev_samples = hist[-1]["epoch"], hist[-1]["samples"]
        else:
            prev_epoch, prev_samples = 0, 0

        def debug_train():
            nonlocal logits, end_ent, mask, reps, tokens, batch
            print(
                f"original end_ent {batch['end_ent'].shape}: {batch['end_ent']} \n \
                 - Mask {mask.shape}: {mask.int()} \n \
                 - Masked tokens {mask.sum()} \n \
                 - reps {reps.shape} \n \
                 - tokens {tokens.shape}: {tokens} \n \
                 - end_ent {end_ent.shape}: {end_ent} \n \
                 - logits {logits.shape}"
            )
            raise ValueError("stop")

        i = 0
        logging.info(
            f"Training Linear Classifier at layer {layer} of {model_name} for {epochs} epochs with batch size {batch_size} \n\
                            - dilate_entities: {dilate_entities} \n\
                            - pos_weight: {pos_weight}..."
        )

        for epoch in range(prev_epoch + 1, prev_epoch + epochs + 1):
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                i += 1
                texts = batch["text"]
                inputs = model.tokenizer(
                    texts, padding=True, padding_side="right", return_tensors="pt"
                )
                tokens = inputs["input_ids"].cuda()
                attn_mask = inputs["attention_mask"].cuda()

                end_ent = batch["end_ent"].cuda()

                # masking
                mask = (tokens != pad_id) & (tokens != bos_id)  # shape (batch, seq)
                if dilation is not None:
                    mask = mask & dilation(end_ent)  # shape (batch, seq)

                # Forward pass
                reps = utils.compute_to_layer(
                    model, layer, tokens, attn_mask=attn_mask
                ).cuda()  # shape (batch, seq, dim)
                # mask tokens
                reps = reps[mask]  # shape (batch, seq, dim)
                end_ent = end_ent[mask]  # shape (batch, seq)
                logits = self.forward(reps)
                # compute loss
                loss = criterion(logits, end_ent)
                loss.backward()

                # debug_train()
                if i % accumulation_steps == 0:
                    if grad_clip:
                        clip_grad_norm_(self.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                hist.append(
                    {
                        "epoch": epoch,
                        "samples": i * batch_size + prev_samples,
                        "loss": loss.item() * batch_size,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

                if (len(hist) - 1) % n_val == 0:
                    val = self.evaluate(
                        layer, val_loader, val_metric=val_metric, verbose=False
                    )
                    if val < best_val:
                        best_val = val
                        best_weights = self.state_dict()
                    logging.info(
                        f"\nSample {hist[-1]['samples']} mean loss: {np.mean([h['loss'] for h in hist[-n_val:]]):.4f}, val loss: {val:.3f}, lr: {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    scheduler.step(val)
                    hist[-1]["val_metric"] = val
                if optimizer.param_groups[0]["lr"] < min_lr:
                    logging.info(f"Minimum learning rate reached, stopping training")
                    break
            if optimizer.param_groups[0]["lr"] < min_lr:
                break

        self.load_state_dict(best_weights)

        return hist


# deprecated
@torch.no_grad()
def compute_metrics(
    dataloader,
    model: HookedTransformer,
    attn: SelfAttention,
    layer: int,
    sliding_window=None,
    causal: bool = None,
    max_ent_length=MAX_ENT_LENGTH,
    threshold=heuristics.DEFAULT_THRESHOLD,
    patience=2,
    verbose: bool = True,
) -> dict:
    """Compute metrics for a dataset, will use the pattern mode from the dataset to compute metrics
    can temporarily change the attention model parameters like sliding window and causal mask
    Args:
        dataloader: data to compute metrics on
        model: HookedTransformer form TransformerLens to extract representations from
        attn: attention model to use, should be a nn.Module with forward method
        layer: layer at which to retreive the representations
        sliding_window: whether to use a sliding window attention, default None = use attn current sliding window
        causal: whether to mask future tokens, default None = use attn current causal mask
        max_ent_length: maximum length of entities
        patience: (block mode) number of tokens to wait for sum of attention scores to be maximal
        threshold: threshold for attention scores
    Returns:
        metrics: dict of computed metrics with keys "precision", "recall", "f1"
    """
    true_pos = 0
    false_pos = 0
    total = 0

    mode = dataloader.dataset.mode
    prev = attn.causal_mask
    prev_sw = attn.sliding_window

    if sliding_window is not None:
        attn.sliding_window = sliding_window
    if causal is not None:
        attn.causal_mask = causal

    if verbose:
        logging.info(
            f"Computing metrics with mode {mode}, causal {causal}, threshold {threshold}, patience {patience}"
        )

    for batch in tqdm(dataloader, disable=not verbose):
        texts = batch["text"]
        inputs = model.tokenizer(
            texts, padding=True, padding_side="right", return_tensors="pt"
        )
        tokens = inputs["input_ids"].cuda()
        attn_mask = inputs["attention_mask"].cuda()

        reps = utils.compute_to_layer(
            model, layer, tokens, attn_mask=attn_mask
        ).cuda()  # shape (batch, seq, dim)
        b_scores = attn.forward(reps)  # shape (batch, seq, seq)
        str_tokens = batch["str_tokens"]
        targets = batch["token_tags"].cpu()

        for j in range(tokens.size(0)):
            seq = len(str_tokens[j])
            ner_tags = attn.get_tags_heuristic(
                b_scores[j, :seq, :seq],
                max_ent_length=max_ent_length,
                threshold=threshold,
                patience=patience,
            )
            tp, fp, tot = count_perf_tags(ner_tags, targets[j, :seq])
            true_pos += tp
            false_pos += fp
            total += tot

    attn.causal_mask = prev
    attn.sliding_window = prev_sw

    # compute metrics
    precision = true_pos / (true_pos + false_pos + EPS)
    recall = true_pos / (total + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_ent_NLL(
    scores: torch.Tensor,
    mask: torch.Tensor,
    ent_pos: Tuple[int, int],
    mode: str,
    max_dilation: int,
    dilation_fx: Callable,
    end_ent_logits: torch.Tensor = None,
    return_mask_pattern=False,
) -> torch.Tensor:
    """Compute the log likelihood of the pattern for a given entity
    Args:
        scores: attention scores (seq, seq)
        mask: mask for the scores (seq, seq)
        ent_pos: entity position (start, end)
        mode: mode for the entity pattern, "block", "first", "last"
        max_dilation: maximum dilation for the pattern
        dilation_fx: function to dilate the pattern
        end_ent_logit: (Optionnal) logits for the end of the entity probe
    Returns:
        log_likelihood: log likelihood of the pattern for the entity
    """
    start, end = ent_pos
    seq_len = scores.shape[0]
    # indexes of the window of interest in scores
    min_mask = max(1, start - max_dilation)  # ignore the BOS
    max_mask = min(seq_len, end + max_dilation)  # ignore the EOS

    ent_scores = scores[min_mask : max_mask + 1, min_mask : max_mask + 1]
    mask = mask[min_mask : max_mask + 1, min_mask : max_mask + 1].to(scores.device)
    # compute pattern for candidate entity
    ent_pattern = torch.zeros_like(ent_scores, dtype=torch.float32)

    ## Logic for patterns and masking
    if mode == "block" or mode == "block_only":
        ent_pattern[
            start - min_mask : end + 1 - min_mask, start - min_mask : end + 1 - min_mask
        ] = 1.0
    elif mode == "first":
        ent_pattern[start - min_mask : end + 1 - min_mask, start - min_mask] = 1.0
    elif mode == "last":
        ent_pattern[end - min_mask, start - min_mask] = 1.0

    ent_mask = dilation_fx(ent_pattern.unsqueeze(0))
    mask = ent_mask.squeeze(0) & mask

    ## plot mask and pattern
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(5, 5))
    # plt.subplot(2, 2, 1)
    # plt.imshow(ent_pattern.cpu())
    # plt.subplot(2, 2, 2)
    # plt.imshow(mask.cpu())
    # plt.colorbar()
    # #replace mask in global scores
    # plt.subplot(2, 2, 3)
    # plt.imshow(scores.cpu())
    # plt.show()
    # raise ValueError("stop")

    ent_scores = ent_scores[mask]
    ent_pattern = ent_pattern[mask]
    # print(ent_scores, ent_pattern)
    if end_ent_logits is not None and end + 2 < seq_len - 1:
        # add one more logit for the end of the entity
        ent_scores = torch.cat(
            [ent_scores, end_ent_logits[end : end + 2].to(ent_scores.device)]
        )
        ent_pattern = torch.cat(
            [ent_pattern, torch.tensor([1.0, 0.0], device=ent_scores.device)]
        )
    # print(ent_pattern, scores)
    log_likelihood = F.binary_cross_entropy_with_logits(
        ent_scores, ent_pattern, reduction="sum"
    )
    if not return_mask_pattern:
        return log_likelihood
    else:
        return log_likelihood, mask, ent_pattern


######################  MODEL IMPLEMENTATIONS ######################
###### TOKEN MATCHING NER ######


class METHODS(str, Enum):
    """Enum for the different matching methods for both TokenMatchingNER and CLQKNER"""

    # 'exact' Methods computing span probabilities
    INTER_FIRST = "inter_first"
    """Intersection of Span and probe events _logits_: Combines logits for CNN and probe in a single logit """

    INTER_FIRST_SOFT = "inter_first_soft"
    """Intersection of :
    - Attention span event in `first' mode: tokens of mentions attend to the first token, normalized with softmax
    - End of entity probe computed with sigmoid of probe logit
    """

    # relying on the Binding ID hypothesis
    INTER_FIRST_NEG = "inter_first_neg"
    """ p(span) = ( 1 - p(b-1,e) ) x  p(b,e) x p(end probe): Intersection of:
    - Match between mean rep of span and last token
    - End of entity probe computed with sigmoid of probe logit
    - NOT match between last token and the token preceeding the span
    """

    # relying on the Binding ID hypothesis
    INTER_BLOCK_MEAN = "inter_mean"
    """ p(span) = ( 1 - p(b-1,e) ) x  p(mean match on span) x p(end probe): Intersection of:
    - Match between mean rep of span and last token
    - End of entity probe computed with sigmoid of probe logit
    - NOT match between last token and the token preceeding the span
    """

    CL_FIRST_MEAN = "cl_first_mean"
    """ Linear combination of logits from:
    - Match between mean rep of span and last token
    - match between first and last token of the span
    - End of entity probe computed with sigmoid of probe logit
    """

    CL_PREV_FIRST_MEAN = "cl_prev_first_mean"
    """ Linear combination of logits from:
    - Match between mean rep of span and last token
    - match between first and last token of the span
    - End of entity probe computed with sigmoid of probe logit
    """

    CL_FIRST_NEXT_MEAN = "cl_first_next_mean"
    """ Linear combination of logits from:
    - Match between mean rep of span and last token
    - match between first and last token of the span
    - End of entity probe computed with sigmoid of probe logit
    """

    CL_FIRST_NEXT_POOL = "cl_first_next_pool"
    """ Linear combination of logits from:
    - Match between first and last token of the span
    - Maximum match between tokens after the first token of the span
    - End of entity probe computed with sigmoid of probe logit
    - End probe of next token after the span
    """

    CL_FIRST_NEXT_MINPOOL = "cl_first_next_minpool"
    """ Linear combination of logits from:
    - Match between first and last token of the span
    - Maximum match between tokens after the first token of the span
    - End of entity probe computed with sigmoid of probe logit
    - End probe of next token after the span
    """

    CL_FN_MINMAXPOOL = "cl_fn_minmaxpool"
    """ Linear combination of logits from:
    - Match between first and last token of the span
    - Maximum match between tokens after the first token of the span
    - End of entity probe computed with sigmoid of probe logit
    - End probe of next token after the span
    """

    SUM_BLOCK_MEAN = "sum_mean"
    """Same as above but summing _logits_ instead of multiplying probabilities (not equivalent)"""

    # Methods using heuristics to retrieve spans
    BLOCK_ONLY = "block_only"  # Only use the block of the probe
    FIRST = "first"  # Only use the first token of the probe
    LAST = "last"  # Only use the last token of the probe


class TokenMatchingNER(NERmodel):
    """Model performing NER based on token matching scores and end-of Entity probe
    There are many ways to model the problem, each method is detailed in the methods attribute
    """

    llm_name: Param[str]
    """name of LLM  used to extract the representations"""

    layer: Param[int] = 0
    """List of layers to extract the query and key scores from"""

    rank: Param[int] = 100
    """Rank of the Query / Keys projection"""

    method: Param[str] = "inter_first"
    """Method to compute the span probabilities, choose among METHODS)"""

    ## attn config
    causal_mask: Param[bool] = True
    """Whether to use a causal mask for the attention scores, default True"""

    sliding_window: Param[int] = 0
    """Normalization method for attn scores, default none, can be 'cosine', or 'log_sigmoid' """

    use_cosine: Param[bool] = False
    """Whether to use cosine normalization for the attention scores, default False"""

    normalize_scores: Param[str] = ""
    """Normalization method for attn scores, default none, can be 'cosine', or 'log_sigmoid' """

    def __post_init__(self):

        # sanity checks
        assert self.method in list(
            METHODS
        ), f"Aggregation method '{self.method}' not implemented, choose among: {', '.join(METHODS)}"
        assert (
            self.llm_name is not None
        ), f"llm_name should be set to the name of the model used to extract the representations"
        assert self.rank > 0, f"rank should be > 0, got {self.rank}"
        assert (
            self.sliding_window >= 0
        ), f"sliding_window should be >= 0, got {self.sliding_window}"
        assert (
            self.layer is not None
        ), f"layer should be set to the layers of the model used to extract the representations"

        super().__init__()
        super().__post_init__()

        self.apply_softmax = True if self.method == METHODS.INTER_FIRST_SOFT else False
        self.mask_bos = False if self.method == METHODS.INTER_FIRST_SOFT else True

        self.model_dim = self.dim
        # two sub modules

        self.scale = 1 if self.use_cosine else 1 / (self.rank**0.5)

        # parameters definition
        self.W_Q = nn.Parameter(
            torch.randn(self.model_dim, self.rank)
        )  # shape (model_dim, rank * n_heads)
        self.W_K = nn.Parameter(
            torch.randn(self.model_dim, self.rank)
        )  # shape (model_dim, rank * n_heads)
        self.classifier = ReprClassifier(self.model_dim)

        self.__init_cl_weights__()

    def __init_cl_weights__(self):
        """Initialize the weights for the linear combination of logits"""

        if self.method == METHODS.CL_FIRST_MEAN:
            self.cl_w = nn.Parameter(torch.ones(1, 3, 1, 1))
        elif self.method in [
            METHODS.CL_PREV_FIRST_MEAN,
            METHODS.CL_FIRST_NEXT_POOL,
            METHODS.CL_FIRST_NEXT_MINPOOL,
        ]:
            self.cl_w = nn.Parameter(torch.ones(1, 4, 1, 1))
        elif self.method in [METHODS.CL_FIRST_NEXT_MEAN, METHODS.CL_FN_MINMAXPOOL]:
            self.cl_w = nn.Parameter(torch.ones(1, 5, 1, 1))

    def extra_repr(self):
        return (
            super().extra_repr()
            + f"\n\
            rank={self.rank},\n\
            method={self.method},\n\
            causal_mask={self.causal_mask},\n\
            use_cosine={self.use_cosine},\n\
            normalize_scores={self.normalize_scores},\n\
            sliding_window={self.sliding_window}"
        )

    def load_classifier(self, classifier: ReprClassifier):
        self.classifier = classifier
        # free memory
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def get_tags_heuristic(
        self,
        scores: Float[torch.Tensor, "seq seq"],
        end_ent: Float[torch.Tensor, "seq"],
        threshold: float = 0,
        patience: int = 2,
        max_ent_length: int = MAX_ENT_LENGTH,
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
        if self.mode == "blockonly":
            return heuristics.NER_tags_from_blockonly_probe(
                scores, end_ent, threshold, patience, max_ent_length
            )
        elif self.mode == "first":
            return heuristics.NER_tags_from_first_probe(
                scores, end_ent, threshold, max_ent_length
            )
        else:
            raise ValueError(f"mode {self.mode} not implemented")

    def get_QK_reps(self, 
            tokens: Float[torch.Tensor, "batch seq"], 
            model: Union[HookedTransformer, nn.Module],
            attn_mask: Optional[Float[torch.Tensor, "batch seq seq"]] = None,
            ) -> Tuple[
        Float[torch.Tensor, "batch seq rank"],
        Float[torch.Tensor, "batch seq rank"],
        Float[torch.Tensor, "batch seq dim"],
    ]:
        """Compute the query and key scores from the model
        query and key scores are computed as a linear transformation of all computed query and key from the LLM
        """
        # get the query and key scores from the model with shape (len(layers), batch, seq, n_heads, dim_head)
        reps = self.get_representations(tokens, model, attn_mask=attn_mask)  # shape (batch, seq, dim)

        return (reps @ self.W_Q), (reps @ self.W_K), reps

    def get_mask(self, seq: int) -> Float[torch.Tensor, "1 seq seq"]:
        """Create a mask for the attention scores
        Args:
            seq: sequence length
        Returns:
            mask: tensor (1, seq, seq) mask for the attention scores
        """
        return masks.create_mask_cached(
            1,
            1,
            seq,
            seq,
            causal=self.causal_mask,
            mask_bos=self.mask_bos,
            sliding_window=self.sliding_window,
        ).squeeze(1)

    def attn_forward(
        self, Q, K, return_mask: bool = False, return_logits=True
    ) -> Float[torch.Tensor, "batch seq seq"]:
        """Compute attention scores from Q and K
        Args:
            Q: tensor (batch, seq, dim) query tensor
            K: tensor (batch, seq, dim) key tensor
            return_mask: whether to return the mask
            return_logits: whether to return the logits
        Returns:
            scores: tensor (batch, seq, seq) attention scores"""

        b, seq, _ = Q.size()

        if self.use_cosine:
            # compute cosine similarity
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)

        scores = self.scale * torch.einsum("b i h, b j h -> b i j", Q, K)

        scores_mask = self.get_mask(seq)  # shape (1, seq, seq)

        scores.masked_fill_(
            ~scores_mask, FILL_NEG_LOGITS
        )  # fill masked values with FILL_NEG_LOGITS

        if self.normalize_scores == "log_sigmoid":
            # apply log sigmoid normalization
            scores = F.logsigmoid(scores)

        if self.normalize_scores == "atan":
            # apply atan normalization
            scores = torch.atan(scores)

        if self.apply_softmax and not return_logits:
            scores = F.softmax(scores, dim=-1)

        if return_mask:
            return scores, scores_mask
        else:
            return scores

    def combine_scores(
        self,
        attn_scores: Float[torch.Tensor, "batch seq seq"],
        end_ent_scores: Float[torch.Tensor, "batch seq"],
        method: str,
        mask: Float[torch.Tensor, "batch seq seq"],
    ) -> Float[torch.Tensor, "batch seq seq"]:
        """Combine the attention and probe logits to compute span scores (logits or probs), depending on the method (as defined in the class)
        Args:
            attn_scores: tensor (batch, seq, seq) of attention scores ( Logits OR probs)
            end_ent_scores: tensor (batch, seq) of end entity scores (Always logits)
        Returns:
            combined_scores: tensor (batch, seq, seq) of combined scores as logits
        """
        # expand end_ent_scores logprobs to 2D
        end_ent_2d = end_ent_scores.unsqueeze(-1).expand(-1, -1, attn_scores.size(2))

        attn_scores.masked_fill_(~mask, torch.tensor(0))  # shape (batch, seq, seq)

        if method == METHODS.INTER_FIRST:
            # final logit is intersection of attn score and end probe
            return LogitIntersection(attn_scores, end_ent_2d)

        elif method == METHODS.INTER_FIRST_SOFT:
            # attn score is softmaxed, compute probabilities directly
            span_probs = F.softmax(attn_scores, dim=-1) * F.sigmoid(end_ent_2d)

            # remove bos spans
            span_probs[:, :, 0] = 0

            return span_probs

        elif method == METHODS.INTER_BLOCK_MEAN:
            # compute mean match inside spans
            mean_score = utils.rev_cumsum(attn_scores) * get_norm_mat(
                attn_scores.size(-1)
            ).to(attn_scores.device)

            # intersection with border
            shifted = F.pad(
                attn_scores, (1, -1), mode="constant", value=FILL_NEG_LOGITS
            )

            attn_scores = LogitIntersection(mean_score, -shifted)
            # intersection with end probe
            return LogitIntersection(attn_scores, end_ent_2d)

        elif method == METHODS.INTER_FIRST_NEG:
            # compute mean match inside spans
            # intersection with border
            shifted = F.pad(
                attn_scores, (1, -1), mode="constant", value=FILL_NEG_LOGITS
            )

            attn_scores = LogitIntersection(attn_scores, -shifted)

            # intersection with end probe
            return LogitIntersection(attn_scores, end_ent_2d)

        elif method == METHODS.CL_FIRST_MEAN:
            # compute mean match inside spans
            mean_score = utils.rev_cumsum(attn_scores) * get_norm_mat(
                attn_scores.size(-1)
            ).to(attn_scores.device)

            # intersection with border ?
            # shifted = F.pad(attn_scores, (1, -1), mode="constant", value=FILL_NEG_LOGITS)

            stacked_logits = torch.stack(
                [mean_score, attn_scores, end_ent_2d], dim=1
            )  # shape (batch, 3, seq, seq)
            # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
            # Linear combination of the features
            return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

        elif method == METHODS.CL_PREV_FIRST_MEAN:
            # compute mean match inside spans
            mean_score = utils.rev_cumsum(attn_scores) * get_norm_mat(
                attn_scores.size(-1)
            ).to(attn_scores.device)

            # intersection with border ?
            prev_attn = shift_right(attn_scores)

            stacked_logits = torch.stack(
                [prev_attn, attn_scores, mean_score, end_ent_2d], dim=1
            )  # shape (batch, 3, seq, seq)
            # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
            # Linear combination of the features
            return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

        elif method == METHODS.CL_FIRST_NEXT_MEAN:
            # compute mean match inside spans
            mean_score = utils.rev_cumsum(attn_scores) * get_norm_mat(
                attn_scores.size(-1)
            ).to(attn_scores.device)

            stacked_logits = torch.stack(
                [
                    attn_scores,
                    shift_right(attn_scores),
                    shift_left(mean_score),
                    end_ent_2d,  # last token probe score
                    shift_up(
                        end_ent_2d
                    ),  # shift up to get the next token's probe score
                ],
                dim=1,
            )  # shape (batch, 3, seq, seq)

            # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
            # Linear combination of the features
            return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

        elif method == METHODS.CL_FIRST_NEXT_POOL:
            # compute mean match inside spans
            pooled_score = max_pool_right_strict(
                attn_scores,
            )

            stacked_logits = torch.stack(
                [
                    attn_scores,
                    pooled_score,  # max pool right
                    end_ent_2d,  # last token probe score
                    shift_up(
                        end_ent_2d
                    ),  # shift up to get the next token's probe score
                ],
                dim=1,
            )  # shape (batch, 3, seq, seq)

            # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
            # Linear combination of the features
            return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

        elif method == METHODS.CL_FIRST_NEXT_MINPOOL:
            # compute mean match inside spans

            stacked_logits = torch.stack(
                [
                    attn_scores,
                    min_pool_right_strict(attn_scores),  # min pool right
                    end_ent_2d,  # last token probe score
                    shift_up(
                        end_ent_2d
                    ),  # shift up to get the next token's probe score
                ],
                dim=1,
            )  # shape (batch, 3, seq, seq)

            # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
            # Linear combination of the features
            return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

        elif method == METHODS.CL_FN_MINMAXPOOL:
            # compute mean match inside spans

            stacked_logits = torch.stack(
                [
                    attn_scores,
                    min_pool_right_strict(attn_scores),  # min pool right
                    max_pool_right_strict(attn_scores),  # max pool right
                    end_ent_2d,  # last token probe score
                    shift_up(
                        end_ent_2d
                    ),  # shift up to get the next token's probe score
                ],
                dim=1,
            )  # shape (batch, 3, seq, seq)

            # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
            # Linear combination of the features
            return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

        elif method == METHODS.SUM_BLOCK_MEAN:
            # compute mean match inside spans
            mean_score = utils.rev_cumsum(attn_scores) * get_norm_mat(
                attn_scores.size(-1)
            ).to(attn_scores.device)

            # intersection with border
            shifted = F.pad(
                attn_scores, (1, -1), mode="constant", value=FILL_NEG_LOGITS
            )

            attn_scores = mean_score - shifted

            # still intersection with end probe
            return LogitIntersection(attn_scores, end_ent_2d)
        else:
            raise NotImplementedError(f"method {method} not implemented yet")

    # for training
    def token_matching_loss(
        self,
        batch: dict,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        scores: Float[torch.Tensor, "batch seq seq"],
        end_ent_scores: Float[torch.Tensor, "batch seq"],
        mask: Bool[torch.Tensor, "batch seq seq"],
        method: str,
        pos_weight: float,
        dilate_entities: Optional[List[int]] = None,
        PL_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute the loss for the token matching method, common for both TokenMatchingNER and CLQKNER
        Args:
            batch: batch of data to train on
            tokens: tensor (batch, seq) batch of tokens process
            model: hooked transformer model to extract representations from
            scores: attention scores (seq, seq)
            end_ent_scores: end entity scores (seq)
            mask: mask for the attention scores (seq, seq)
            dilate_entities: dilate entities for the loss computation
            method: method to use for the loss in METHODS
        Returns:
            loss: loss for the token matching method
        """

        def debug_train():
            nonlocal batch_mask, batch_mask_1D, scores, attn_patterns, span_patterns, mask, tags, padded_mask, b_size, span_scores, dilated_mask  # type: ignore
            import matplotlib.pyplot as plt

            print(
                "\n - batch_mask",
                batch_mask.shape,
                "\n - dilated_mask",
                dilated_mask.shape,
                "\n - scores",
                scores.shape,
                "\n - attn_patterns",
                attn_patterns.shape,
                "\n - mask",
                mask.shape,
                "\n - tags",
                tags.shape,
            )
            n = 3
            fig, ax = plt.subplots(7, n, figsize=(13, 20))
            for i in range(n):
                ind = torch.randint(b_size, (1,)).item()
                ax[0, i].imshow(attn_patterns[ind, :, :].cpu().int())
                ax[0, i].set_title(f"attention pattern {ind}")
                ax[1, i].imshow(scores[ind, :, :].detach().cpu().numpy())
                ax[1, i].set_title(f"attn scores {ind}")
                ax[2, i].imshow(padded_mask[ind, :, :].cpu().int())
                ax[2, i].set_title(f"padded_mask {ind}")
                ax[3, i].imshow(batch_mask[ind, :, :].cpu().int())
                ax[3, i].set_title(f"item_mask {ind}")
                ax[4, i].imshow(span_patterns[ind, :, :].cpu().int())
                ax[4, i].set_title(f"span_patterns {ind}")
                ax[5, i].imshow(span_scores[ind, :, :].detach().cpu())
                ax[5, i].set_title(f"span logits {ind}")
                ax[6, i].imshow(batch_mask_1D[ind, :].unsqueeze(0).cpu().int())
                ax[6, i].set_title(f"1D_mask {ind}")
            plt.show()
            raise ValueError("stop")

        tags = batch["token_tags"].float().cuda()
        attn_patterns = batch["pattern"].cuda()  # tensor shape (batch, seq, seq)
        end_ent = batch["end_ent"].cuda()
        tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore
        b_size = attn_patterns.size(0)
        # type:ignore shape (batch, seq)
        padded: torch.Tensor = (tokens == tokenizer.pad_token_id)

        # transform to padded mask to 2D mask
        padded_mask = padded.unsqueeze(1) | padded.unsqueeze(2)

        #### mask computation for loss
        if dilate_entities is None:
            # compute loss on everything, take orginal mask
            end_ent_mask_1D = torch.ones_like(end_ent, dtype=torch.bool)
            dilated_mask = torch.ones_like(attn_patterns, dtype=torch.bool)

        else:  # dilate entities -> can be 2D or 1D
            if (
                len(dilate_entities) == 1
            ):  # we only compute the loss on the entity *rows*
                if dilate_entities[0] > 0:
                    # dilate and inflate mask to match the shape of the pattern
                    dilated_mask = masks.get_dilation(dilate_entities[0])(
                        tags
                    )  # shape (batch, seq)
                    dilated_mask = dilated_mask.unsqueeze(1) | dilated_mask.unsqueeze(
                        2
                    )  # shape (batch, seq, seq)
                else:
                    dilated_mask = masks.fill_ones_left(attn_patterns)

            elif (
                len(dilate_entities) == 2
            ):  # we only compute the loss on the matches around entity blocks
                dilated_mask = masks.get_2d_dilation(tuple(dilate_entities))(
                    attn_patterns * mask
                ).squeeze(0)
                # shape (batch, seq, seq)
            else:
                raise NotImplementedError(
                    f"shouldn't be here, dilate_entities should be None, 1 or 2, not {dilate_entities}"
                )

            # (batch, seq)
            # end_ent_mask_1D = masks.get_dilation(max(dilate_entities[0], 1))(
            #     end_ent
            # ).squeeze(0)
            # (batch, seq, seq)

        batch_mask = dilated_mask & mask & ~padded_mask
        batch_mask_1D = batch_mask.diagonal(0, 1, 2)  # shape (batch, seq)

        span_scores = self.combine_scores(
            scores, end_ent_scores, method, batch_mask
        )  # shape (batch, seq, seq)

        # Finally Compute the loss depending on the method.

        if method == METHODS.INTER_FIRST_SOFT:

            span_patterns = data.get_span_patterns(
                batch
            ).cuda()  # tensor shape (batch, seq, seq)

            # attn_scores andend_ent_scores are logits ,  span_scores is probs,
            # attn_loss = F.cross_entropy(
            #     scores[batch_mask_1D, :],
            #     attn_patterns[batch_mask_1D, :],
            #     reduction="mean",
            # )

            # probe_loss = F.binary_cross_entropy_with_logits(
            #     end_ent_scores[end_ent_mask_1D],
            #     end_ent[end_ent_mask_1D],
            #     pos_weight=torch.tensor(pos_weight),
            #     reduction="mean",
            # )
            if self.pos_weight is None or self.pos_weight == 0:

                # balanced BCE loss
                span_loss = balanced_BCE(
                    span_scores,
                    span_labels,
                )
            else:
                span_loss = F.binary_cross_entropy(
                    span_scores[batch_mask],
                    span_patterns[batch_mask],
                )

            loss = span_loss  # + 1 * attn_loss + 1 * probe_loss  # + 0.50 * span_loss

        else:  # default behavior, only train on final prediction (end to end)
            span_labels = attn_patterns[batch_mask]
            span_scores = span_scores[batch_mask]

            if PL_threshold is not None:
                # apply pseudo labeling and compute loss
                thr_mask = span_scores > PL_threshold
                span_labels[thr_mask] = F.sigmoid(span_scores[thr_mask])

            if self.pos_weight is None or self.pos_weight == 0:

                # balanced BCE loss
                span_loss = balanced_BCE(
                    span_scores,
                    span_labels,
                )
            else:
                pos_weigths = torch.ones_like(span_scores)
                pos_weigths[span_labels == 1] = self.pos_weight

                span_loss = F.binary_cross_entropy_with_logits(
                    span_scores,
                    span_labels,
                    pos_weight=pos_weigths,
                )

            loss = span_loss
        # debug_train()
        return loss

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        attn_mask: Optional[Float[torch.Tensor, "batch seq seq"]] = None,
        return_logits: bool = False,
        return_mask: bool = False,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq"]:
        """Compute probabilities for each span in the text,
        Args:
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
            return_logits: Force return of logits instead of probabilities
        Returns:
            span_scores: tensor (batch, seq seq) logits or probabilities for each span in the text
        """
        b, seq = tokens.size()
        # get the query and key scores from the model with shape (len(layers), batch, seq, n_heads, dim_head)
        Q, K, reps = self.get_QK_reps(
            tokens, model, attn_mask=attn_mask
        )  # shape (batch, seq, rank) , " , (batch, seq, dim)

        scores, attn_mask = self.attn_forward(
            Q, K, return_mask=True
        )  # shape (batch, seq, seq)
        end_ent = self.classifier(reps)  # shape (batch, seq)

        b_mask = attn_mask.repeat(b, 1, 1)
        if not self.mask_bos:
            b_mask[:, 0, :] = True

        span_logits = self.combine_scores(
            scores, end_ent, self.method, b_mask
        )  # shape (batch, seq, seq)

        if not return_logits and not self.apply_softmax:
            # return probabilities
            span_logits = F.sigmoid(span_logits)  
            span_logits.masked_fill_(~b_mask, 0)
        else:
            span_logits.masked_fill_(~b_mask, FILL_NEG_LOGITS)

        if return_mask:
            return span_logits, b_mask
        else:
            return span_logits

    def training_step(self, batch: dict, model: HookedTransformer) -> torch.Tensor:
        """Compute loss for a batch of data
        Args:
            batch: dict of data
        Returns:
            loss: loss for the batch
        """

        inputs = model.tokenizer(
            batch["text"],
            padding=True,
            padding_side="right",
            return_tensors="pt",
            truncation=True,
        )
        tokens = inputs["input_ids"].cuda()
        attn_mask = inputs["attention_mask"].cuda()

        dilate_entities = self.dilate_entities if hasattr(self, "dilate_entities") else None

        # get the query and key scores from the model with shape (len(layers), batch, seq, n_heads, dim_head)
        Q, K, reps = self.get_QK_reps(
            tokens, model, attn_mask=attn_mask
        )  # shape (batch, seq, rank) , " , (batch, seq, dim)

        scores, mask = self.attn_forward(
            Q, K, return_mask=True, return_logits=True
        )  # shape (batch, seq, seq)
        end_ent_scores = self.classifier(reps)  # shape (batch, seq)

        # reshape to (batch, seq, seq) putting the end_ent_scores on the diagonal and add to scores
        loss = self.token_matching_loss(
            batch=batch,
            tokens=tokens,
            model=model,
            scores=scores,
            end_ent_scores=end_ent_scores,
            mask=mask,
            method=self.method,
            pos_weight=self.pos_weight,
            dilate_entities=dilate_entities,
        )

        return loss

    def train(
        self,
        train_loader,
        val_loader,
        optimizer=partial(torch.optim.AdamW, weight_decay=1e-4),
        hist: list = [],
        epochs: int = 2,
        lr: float = 1e-3,
        grad_clip: float = 0,
        pos_weight: float = 0.0,
        PL_threshold: float = 1,
        accumulation_steps: int = 1,
        patience: int = 10,
        min_lr: float = 1e-5,
        n_val: int = 1000,
        val_metric: str = "bce",
        dilate_entities: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        """Train attention model on the given dataset
        Args:
            model: HookedTransformer form TransformerLens to extract representations from
            attn: attention model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            end_classifier: (Module, optionnal) binary classifier to train on top of the attention scores
            sliding_window: whether to use a sliding window attention, default 0 : no sliding window
            hist: history of training
            epochs: number of epochs to train
            lr: learning rate
            grad_clip (float): gradient clipping value, if 0, no clipping
            pos_weight: weight for positive examples in BCE loss, default 0 => use balanced_BCE
            PL_threshold: threshold probability for pseudo label, if the probability is above this threshold, the label is set to 1
            accumulation_steps: number of steps to accumulate gradients before updating weights
            patience: number of untolerable validations to wait before reducing learning rate
            min_lr: minimum learning rate, if reached, training stops
            n_val: number of steps between validation and logging
            val_metric: metric to use for validation, default "bce"
            dilate_entities: Compute the loss only on entities (dilated) and not the full sequence
        """
        if self.method in [METHODS.INTER_BLOCK_MEAN, METHODS.SUM_BLOCK_MEAN]:
            data_mode = data.PATTERN_MODES.BLOCK
        else:
            data_mode = data.PATTERN_MODES.LAST

        if train_loader.dataset.mode != data_mode:
            # if the data_mode is not the same, we need to change it
            logging.info(
                f"Current dataset data_mode is {train_loader.dataset.mode}, changing to {data_mode}"
            )
            train_loader.dataset.mode = data_mode

        ## Setup masking and loss
        if dilate_entities is not None:
            assert (
                type(dilate_entities) == list
            ), f"dilate_entities should be a list of ints, got {type(dilate_entities)}"
            if dilate_entities == []:
                dilate_entities = None

        self.dilate_entities = dilate_entities
        self.pos_weight = pos_weight

        if PL_threshold != 1:
            assert (
                PL_threshold > 0 and PL_threshold < 1
            ), f"Pseudo-label threshold should be a probability in [0, 1], got {PL_threshold}"
            self.PL_threshold = torch.tensor(PL_threshold).logit()
        else:
            self.PL_threshold = None

        logging.info(
            f"Training Token Matching model with representations from layer {self.layer} for {epochs} epochs with batch size {train_loader.dataset.batch_size} \n\
                         - optimizer: {optimizer} \n\
                         - dilate_entities: {dilate_entities} \n\
                         - method: {self.method} \n\
                         - pos_weight: {self.pos_weight} \n\
                         - Causal mask {self.causal_mask}\n\
                         - bos mask {self.mask_bos}\n\
                         - Pseudo label logit threshold {self.PL_threshold}\n\
                         - Sliding Window {self.sliding_window}... "
        )
        # set the model to training mode
        gc.collect()
        torch.cuda.empty_cache()

        hist = train(
            self,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            val_metric=val_metric,
            hist=hist,
            epochs=epochs,
            lr=lr,
            grad_clip=grad_clip,
            accumulation_steps=accumulation_steps,
            patience=patience,
            min_lr=min_lr,
            n_val=n_val,
            train_step_args={
                "model": train_loader.dataset.model,

            },
            **kwargs,
        )

        gc.collect()
        torch.cuda.empty_cache()

        return hist


class CLQK_NER(TokenMatchingNER):
    """Token Matching NER model,
    overriding representation extraction to use directly query and key scores from the attention layer of an LLM
    """

    layers: Param[List[int]] = [0]
    """List of layers to extract the query and key scores from, layer will be set as max(layers)"""

    #overrides from parent class
    need_hookedtransformer: bool = True
    """This model needs a HookedTransformer to extract query and key scores"""

    def __post_init__(self):
        super().__init__()

        self.layer = max(
            self.layers
        )  # needed to use get_representations from parent class

        super().__post_init__()

        try :
            from transformer_lens import HookedTransformer
            config = convert_hf_model_config(self.llm_name)
            self.qk_dim = config["d_head"]
            self.n_heads = config["n_heads"]
            self.n_kv_heads = config.get("n_key_value_heads", self.n_heads)
            logging.info(f"Found config for model {self.llm_name}: qk_dim {self.qk_dim}, n_heads {self.n_heads}, n_kv_heads {self.n_kv_heads}")
        except Exception as e:
            raise ValueError(f"Could not load model config for {self.llm_name}, make sure the model name is correct and the model is supported. Original error: {e}")


        # overrides parameters definition
        self.W_Q = nn.Parameter(
            torch.randn(self.qk_dim * self.n_heads * len(self.layers), self.rank)
        )  # shape (model_dim, rank * n_heads)
        self.W_K = nn.Parameter(
            torch.randn(self.qk_dim * self.n_kv_heads * len(self.layers), self.rank)
        )  # shape (model_dim, rank * n_heads)
        self.classifier = ReprClassifier(self.model_dim)

    def get_QK_reps(self, 
                    tokens: torch.Tensor, 
                    model: HookedTransformer, 
                    attn_mask: Optional[torch.Tensor] = None) -> Tuple[
        Float[torch.Tensor, "batch seq rank"],
        Float[torch.Tensor, "batch seq rank"],
        Float[torch.Tensor, "batch seq dim"],
    ]:
        """Compute the query and key scores from the model
        query and key scores are computed as a linear transformation of all computed query and key from the LLM
        """
        # get the query and key scores from the model with shape (len(layers), batch, seq, n_heads, dim_head)
        llm_q, llm_k, reps = utils.get_QK_from_layers(
            model, tokens, layers=self.layers, attn_mask=attn_mask, dtype=self.dtype
        )

        llm_q = rearrange(
            llm_q,
            "l b s h d -> b s (l h d)",
            l=len(self.layers),
            h=self.n_heads,
            d=self.qk_dim,
        )

        llm_k = rearrange(
            llm_k,
            "l b s h d -> b s (l h d)",
            l=len(self.layers),
            h=self.n_kv_heads,
            d=self.qk_dim,
        )
        
        return (llm_q @ self.W_Q), (llm_k @ self.W_K), reps

    def training_step(
        self, batch: dict, model: HookedTransformer, lasso_reg: Optional[float] = 0
    ) -> torch.Tensor:
        """Compute loss for a batch of data
        Args:
            batch: dict of data
        Returns:
            loss: loss for the batch
        """

        inputs = model.tokenizer(
            batch["text"],
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )
        tokens = inputs["input_ids"].cuda()
        attn_mask = inputs["attention_mask"].cuda()
        
        # get the query and key scores from the model with shape (len(layers), batch, seq, n_heads, dim_head)
        Q, K, reps = self.get_QK_reps(
            tokens, model, attn_mask=attn_mask
        )  # shape (batch, seq, rank) , " , (batch, seq, dim)

        scores, mask = self.attn_forward(
            Q, K, return_mask=True, return_logits=True
        )  # shape (batch, seq, seq)
        end_ent_scores = self.classifier(reps)  # shape (batch, seq)

        # reshape to (batch, seq, seq) putting the end_ent_scores on the diagonal and add to scores
        loss = self.token_matching_loss(
            batch=batch,
            tokens=tokens,
            model=model,
            scores=scores,
            end_ent_scores=end_ent_scores,
            mask=mask,
            method=self.method,
            pos_weight=self.pos_weight,
        )
        if lasso_reg:
            # Lasso (L1 norm) regularization on the scores kernel
            loss += lasso_reg * (torch.norm(self.W_Q, 1) + torch.norm(self.W_K, 1))

        return loss


class AttentionLCNER(TokenMatchingNER):
    """Token Matching model using a linear combination of attention scores from a given layer of a LLM as scores for token matching.
    The attention scores are retrieved using the TransformerLens library.
    """

    layers: Param[List[int]] = [0]
    """List of layers to extract the attention scores from, layer will be set as max(layers)"""

    normalize_scores: Param[str] = "logits"
    """Normalization method for attn scores,
    - logits : keep raw logits from scalar product of query and key
    - softmax: will rather extract scores _after_ the softmax in the LLM
    - log_sigmoid: apply log sigmoid normalization to scores
    - sigmoid: apply sigmoid normalization to scores
    """

    need_hookedtransformer: bool = True
    """This model needs a HookedTransformer to extract attention scores"""

    mask_bos: bool = True

    def __post_init__(self):
        
        super().__init__()

        if self.dim is None:
            self.dim = self.get_llm_dim()
            logging.info(f"Found hidden dimension {self.dim} for {self.llm_name}")
        
        self.model_dim = self.dim
        self.layer = max(self.layers)

        #retrieve model configuration to get number of heads
        try :
            from transformer_lens import HookedTransformer
            config = convert_hf_model_config(self.llm_name)
            self.n_heads = config["n_heads"]
            logging.info(f"Found config for model {self.llm_name}: n_heads {self.n_heads}")
        except Exception as e:
            raise ValueError(f"Could not load model config for {self.llm_name}, make sure the model name is correct and the model is supported. Original error: {e}")

        # overrides parameters definition
        self.cl_attn = nn.Linear(len(self.layers) * self.n_heads, 1, bias=True)
        
        self.classifier = ReprClassifier(self.model_dim)

        self.PL_threshold = None
        self.pos_weight = 0.0
        self.dilate_entities = None

        self.__init_cl_weights__()

    def extra_repr(self) -> str:
        TM_repr = super().extra_repr()
        #remove rank=... from parent class
        keys_to_remove = ["rank", "use_cosine"]
        TM_repr = ", ".join([x for x in TM_repr.split(", ") if not x.startswith(tuple(keys_to_remove))])
        return f"{TM_repr}, layers={self.layers}, normalize_scores={self.normalize_scores}, method={self.method}"

    def attn_forward(
        self,
        attn_scores: Float[torch.Tensor, "batch layer n_heads seq seq"],
        return_mask: bool = False,
    ) -> Tuple[Float[torch.Tensor, "batch seq seq"], Float[torch.Tensor, "batch seq seq"]]:
        """Overrides the attn_forward method to use the attention scores from the LLM.
        Args:
            attn_scores: tensor of shape (batch, layer, n_heads, seq, seq) with the attention scores from the LLM
        Returns:
            output: tensor of shape (batch, seq, seq) with the combined attention scores
        """
        #rearrange to (batch, seq, seq, n_heads * layers)
        attn_scores = rearrange(attn_scores, "b l h i j -> b i j (h l)")
        # filter nan values from attn scores
        attn_scores = torch.where(torch.isfinite(attn_scores), attn_scores, torch.zeros_like(attn_scores))
        
        if self.normalize_scores in ["logits", "softmax"]:
            pass  # keep raw logits or already softmaxed values
        elif self.normalize_scores == "log_sigmoid":
            attn_scores = F.logsigmoid(attn_scores)
        elif self.normalize_scores == "sigmoid":
            attn_scores = torch.sigmoid(attn_scores)
        else:
            raise ValueError(f"Unknown normalization method {self.normalize_scores}")
        # create mask and fit to batched scores
        b, seq, _ , _ = attn_scores.size()
        b_mask = self.get_mask(seq).expand(b, -1, -1).to(attn_scores.device)
        output = self.cl_attn(attn_scores).squeeze(-1)

        if return_mask:
            return output, b_mask 
        else:
            return output

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        return_logits: bool = False,
        return_mask: bool = False,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq"]:
        """Compute probabilities for each span in the text,
        Args:
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
            return_logits: Force return of logits instead of probabilities
        Returns:
            span_scores: tensor (batch, seq seq) logits or probabilities for each span in the text
        """
        b, seq = tokens.size()
       
        # get the attention scores from llms
        llm_attn_scores, reps = utils.get_attnScores_from_layers(
            layers=self.layers,
            model=model,
            tokens=tokens,
        )  # shape (batch, len(layers), n_heads, seq, seq)

        scores, b_mask = self.attn_forward(llm_attn_scores, return_mask=True)  # shape (batch, seq, seq)
        end_ent = self.classifier(reps)  # shape (batch, seq)

        span_logits = self.combine_scores(
            scores, end_ent, self.method, b_mask
        )  # shape (batch, seq, seq)

        if return_logits:
            span_logits.masked_fill_(~b_mask, FILL_NEG_LOGITS)
        else:
            # return probabilities
            span_logits = F.sigmoid(span_logits)  
            span_logits.masked_fill_(~b_mask, 0)

        if return_mask:
            return span_logits, b_mask
        else:
            return span_logits


    def training_step(
        self, batch: dict, model: HookedTransformer, lasso_reg: Optional[float] = 0
    ) -> torch.Tensor:
        """Compute loss for a batch of data
        Args:
            batch: dict of data
        Returns:
            loss: loss for the batch
        """

        inputs = model.tokenizer(
            batch["text"],
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )
        tokens = inputs["input_ids"].cuda()
        attn_mask = inputs["attention_mask"].cuda()
        
        b, seq = tokens.size()
        
        # get the attention scores from llms
        llm_attn_scores, reps = utils.get_attnScores_from_layers(
            layers=self.layers,
            model=model,
            tokens=tokens,
            attn_mask=attn_mask,
        )  # shape (batch, len(layers), n_heads, seq, seq)

        attn_scores, b_mask = self.attn_forward(llm_attn_scores, return_mask=True)  # both shape (batch, seq, seq)

        end_ent_scores = self.classifier(reps)  # shape (batch, seq)


        span_logits = self.combine_scores(
            attn_scores, end_ent_scores, self.method, b_mask
        )  # shape (batch, seq, seq)


        # reshape to (batch, seq, seq) putting the end_ent_scores on the diagonal and add to scores
        loss = self.token_matching_loss(
            batch=batch,
            tokens=tokens,
            model=model,
            scores=span_logits,
            end_ent_scores=end_ent_scores,
            mask=b_mask,
            method=self.method,
            pos_weight= 0 if not hasattr(self, "pos_weight") else self.pos_weight,
        )

        if lasso_reg:
            # Lasso (L1 norm) regularization on the scores kernel
            loss += lasso_reg * (torch.norm(self.W_Q, 1) + torch.norm(self.W_K, 1))

        return loss





###### CNN NER MODEL ######


class CNN_METHODS(str, Enum):
    INTER = "inter"
    """ Intesection of span and probe events: Combines logits for CNN and probe in a single logit"""

    LOGITS = "logits"
    """ Output logits dont normalize logit scores \ell_ij"""

    LOGSIGMOID = "logsigmoid"
    """ Output logits, Normalize logit scores wiht logsigmoid"""

    SOFTMAX = "softmax"
    """ Output logits, Normalize logit scores with softmax"""

    MEAN = "mean"
    """ Output logits, Normalize logit scores with mean over summed attention scores"""

    NLL_ALL = "NLL_all"
    """ Output NLL: log(p) = \sum a_ij log(p_ij)"""


class AttentionCNN_NER(NERmodel):
    """Model performing NER based on token matching scores, end-of Entity probe and a CNN on top of the attention scores"""

    # -----  Model config -----
    # Projections
    model_dim: Param[int] = 512
    """Model dimension"""
    rank: Param[int] = 100
    """Rank of the Query / Keys projection"""
    causal_mask: Param[bool] = True
    """Whether to use a causal mask for the attention scores, default True"""
    mask_bos: Param[bool] = True
    """Whether to mask the first token (BOS) in the attention scores, default True"""
    sliding_window: Param[int] = 0
    """Whether to use a sliding window for the attention scores, default 0 (no sliding window)"""

    # CNN config
    kernel_padding: Param[List[int]] = DEFAULT_KERNEL_PADDING  # (left right top bottom)
    """ Kernel padding for the CNN : (left, right, top, bottom)"""
    method: Param[str] = CNN_METHODS.LOGITS
    """Logits aggregation method"""
    init_mean: Param[bool] = False
    """Whether to initialize the kernel with mean pooling"""

    # Meta data -- on which LLM the model is trained
    llm_name: Param[Optional[str]] = None
    """name of LLM  used to extract the representations"""

    layer: Param[int]
    """layer of LLM used to extract the representations"""

    def __post_init__(self):
        """Initialize the model, paramters and parent classes
        This function is called when the model is initialized, with model.instance() (done automatically if given as a Task parameter)
        """
        # Sanity checks
        assert self.method in list(
            CNN_METHODS
        ), f"Aggregation method '{self.method}' not implemented, choose among: {', '.join(CNN_METHODS)}"
        super().__init__()

        self.attn = SelfAttention(
            model_dim=self.model_dim,
            k=self.rank,
            causal_mask=self.causal_mask,
            mask_bos=self.mask_bos,
            sliding_window=self.sliding_window,
            apply_softmax=True if self.method == CNN_METHODS.SOFTMAX else False,
        )
        self.classifier = ReprClassifier(self.model_dim)

        self.probe_padding = (
            self.kernel_padding[1],
            self.kernel_padding[0],
        )  # right, left

        self.k_size = (
            1 + self.kernel_padding[2] + self.kernel_padding[3],  # height (top, bottom)
            1 + self.kernel_padding[0] + self.kernel_padding[1],  # width  (left, right)
        )

        self.scoresKernel = nn.Parameter((torch.randn((1, 1, *self.k_size)) / 5))
        self.probeKernel = nn.Parameter((torch.randn((1, 1, self.k_size[1])) / 5))

        if self.init_mean:
            self.init_kernel_mean()

    def init_kernel_mean(self):
        logging.info(
            f"Initializing scores Aggregator with equal weights (mean pooling)"
        )
        self.scoresKernel.data.fill_((-1 / (self.k_size[0] * self.k_size[1])))
        self.probeKernel.data.fill_(-1 / self.k_size[1])

    def set_kernels_block(self, freeze: bool = False):
        """Set kernels to block attention scores and end_ent probe"""
        left, right, top, bottom = self.kernel_padding
        k_size = (top + bottom + 1, left + right + 1)

        kernel_mask = torch.zeros(k_size)
        causal_mask = torch.tril(
            torch.ones(k_size[0], k_size[1]), diagonal=right + 1 - top
        ).bool()
        kernel_mask[: top + 1, left - 1 : left] = -1
        kernel_mask[: top + 1, left:] = 1
        if bottom:
            kernel_mask[top + 1, left:] = -1
        kernel_mask[~causal_mask] = 0

        self.scoresKernel = nn.Parameter(
            kernel_mask.unsqueeze(0).unsqueeze(0),  # shape (1, 1, k1, k2)
            requires_grad=not freeze,
        )

        probe_kernel = torch.zeros((1, 1, self.k_size[1]))
        probe_kernel[:, :, left - 1 : left + 2] = -1
        probe_kernel[:, :, left] = 1
        self.probeKernel = nn.Parameter(probe_kernel, requires_grad=not freeze)

    def __str__(self):
        return (
            super().__str__()
            + f"\n \
    - trained on representations from {self.llm_name} at layer {self.layer}\n \
    - scores aggregation with method '{self.method}'\n \
    - kernel padding, scores {self.kernel_padding} probe {self.probe_padding} \n \
            - {self.attn} \n "
        )

    def load_attn(self, attn: SelfAttention):
        self.attn = attn
        # free memory
        gc.collect()
        torch.cuda.empty_cache()

    def load_classifier(self, classifier: ReprClassifier):
        self.classifier = classifier
        # free memory
        gc.collect()
        torch.cuda.empty_cache()

    def conv_scores(
        self,
        scores: Float[torch.Tensor, "batch seq seq"],
        kernel: Float[torch.Tensor, "1 1 k1 k2"],
    ) -> Float[torch.Tensor, "batch seq seq"]:
        """Convolve the attention scores with the kernel"""
        return F.conv2d(  # convolve the scores with the kernel
            F.pad(scores.unsqueeze(1), self.kernel_padding, value=0),
            kernel,
            padding=0,
            bias=None,
        ).squeeze(
            1
        )  # shape (batch, seq, seq)

    def conv_probe(
        self,
        end_ent: Float[torch.Tensor, "batch seq"],
        kernel: Float[torch.Tensor, "1 1 k"],
    ) -> Float[torch.Tensor, "batch seq"]:
        return F.conv1d(
            F.pad(end_ent.unsqueeze(1), self.kernel_padding[:2], value=0),
            kernel,
            padding=0,
            bias=None,
        ).squeeze(
            1
        )  # shape (batch, seq)

    def get_representations(self, tokens, model):
        """Compute the representation of the tokens
        Args:
            tokens: tensor (batch, seq) of token ids
            model: model to use to compute the representations
        Returns:
            reps: tensor (batch, seq, dim) of representations
        """
        with torch.no_grad():
            reps = utils.compute_to_layer(
                model, self.layer, tokens, dtype=self.dtype
            ).cuda()
        return reps

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        return_logits: bool = False,
        scores_mask: torch.Tensor = None,
        probe_mask: torch.Tensor = None,
        return_mask: bool = False,
    ) -> Float[torch.Tensor, "batch seq"]:
        """Compute probabilities for each span in the text,
        Args:
            reps: tensor (batch, seq, dim) representations of the tokens
            scores_mask: tensor OPTIONAL (batch, seq, seq) mask for the tokens before the CNN aggregation
            probe_mask: tensor OPTIONAL (batch, seq) mask for the end_ent before the CNN aggregation
            return_probs: Force return of probabilities instead of logits
            return_mask: return the mask used for the scores
        Returns:
            tags: tensor (batch, seq seq) logits or probabilities for each span in the text
        """

        batch, seq = tokens.size()
        reps = self.get_representations(tokens, model)  # shape (batch, seq, dim)

        scores, attn_mask = self.attn.forward(
            reps,
            return_mask=True,
        )  # scores are already masked
        end_ent = self.classifier(reps)  # shape (batch, seq)

        def debug_forward():
            nonlocal scores, output, conv_spans, conv_probe_2d, mask, scores_mask, probe_mask
            logging.info("DEBUG FORWARD")
            import matplotlib.pyplot as plt

            print(
                f"Scores {scores.shape} \n \
                 - End_ent {end_ent.shape} \n \
                 - Mask {attn_mask.shape} \n \
                 - Scores Mask {scores_mask.shape} \n \
                 - Probe Mask {probe_mask.shape} \n \
                 - conv_scores {conv_spans.shape}: \n "
            )

            n = 3
            fig, ax = plt.subplots(6, n, figsize=(10, 15))
            for i in range(n):
                ind = torch.randint(batch, (1,)).item()
                im = ax[0, i].imshow(scores[ind, :, :].detach().cpu())
                ax[0, i].set_title(f"scores {ind}")
                fig.colorbar(im, ax=ax[0, i])

                ax[1, i].imshow(mask[ind, :, :].detach().cpu().int())
                ax[1, i].set_title(f"mask {ind}")

                im = ax[2, i].imshow(conv_spans[ind, :, :].detach().cpu())
                ax[2, i].set_title(f"conv_spans {ind}")
                fig.colorbar(im, ax=ax[2, i])

                im = ax[3, i].imshow(conv_probe_2d[ind, :, :].detach().cpu())
                ax[3, i].set_title(f"conv_probe_2d {ind}")
                fig.colorbar(im, ax=ax[3, i])

                im = ax[4, i].imshow(F.sigmoid(output[ind, :, :].detach()).cpu())
                ax[4, i].set_title(f"output {ind}")
                fig.colorbar(im, ax=ax[4, i])

                # plot scoresAggregator gradients
                # im = ax[5,i].imshow(self.scoresKernel.grad[0,0,:,:].detach().cpu())
                # ax[5,i].set_title(f"grad scoresKernel {ind}")

            plt.show()
            raise ValueError("stop")

        # masking
        if scores_mask is not None:
            mask = scores_mask & attn_mask.repeat(
                batch, 1, 1
            )  # shape (batch, seq, seq)
        else:
            mask = attn_mask.repeat(batch, 1, 1)  # shape (batch, seq, seq)

        if probe_mask is None:
            probe_mask = torch.ones((batch, seq), dtype=torch.bool, device=reps.device)

        if self.method in [CNN_METHODS.LOGITS, CNN_METHODS.SOFTMAX, CNN_METHODS.MEAN]:
            end_ent.masked_fill_(~probe_mask, 0)
            scores.masked_fill_(~mask, 0)

            conv_spans = self.conv_scores(scores, self.scoresKernel)
            conv_probe = self.conv_probe(
                end_ent, self.probeKernel
            )  # shape (batch, seq)

            # expand end_ent logprobs to 2D
            conv_probe_2d = (
                conv_probe.unsqueeze(-1).expand(-1, -1, seq) * mask
            )  # shape (batch, seq, seq)

            if self.method == CNN_METHODS.MEAN:
                conv_spans = conv_spans * get_norm_mat(seq).expand(batch, -1, -1).to(
                    reps.device
                )

            output = (conv_spans + conv_probe_2d).masked_fill_(~mask, FILL_NEG_LOGITS)

            if not return_logits:
                output = F.sigmoid(output)

        elif self.method == CNN_METHODS.LOGSIGMOID:
            end_ent.masked_fill_(~probe_mask, 0)
            conv_spans = self.conv_scores(
                F.logsigmoid(scores) * mask, self.scoresKernel
            )
            conv_probe = self.conv_probe(
                F.logsigmoid(end_ent), self.probeKernel
            )  # shape (batch, seq)
            # expand end_ent logprobs to 2D
            conv_probe_2d = (
                conv_probe.unsqueeze(-1).expand(-1, -1, seq) * mask
            )  # shape (batch, seq, seq)

            output = (conv_spans + conv_probe_2d).masked_fill_(~mask, FILL_NEG_LOGITS)
            if not return_logits:
                output = F.sigmoid(output)

        elif self.method == CNN_METHODS.INTER:
            # fill masked values with 0 before convolution
            end_ent.masked_fill_(~probe_mask, 0)
            scores.masked_fill_(~mask, 0)
            # aggregate scores with Convolution
            conv_spans = self.conv_scores(
                scores, self.scoresKernel
            )  # shape (batch, seq, seq)
            conv_probe = self.conv_probe(
                end_ent, self.probeKernel
            )  # shape (batch, seq)
            # normalize conv_spans with the norm matrix
            # print(f"conv_spans {conv_spans.shape} conv_probe {conv_probe.shape}")
            conv_spans = conv_spans * get_norm_mat(seq).expand(batch, -1, -1).to(
                reps.device
            )
            # expand end_ent logprobs to 2D
            conv_probe_2d = (
                conv_probe.unsqueeze(-1).expand(-1, -1, seq) * mask
            )  # shape (batch, seq, seq)
            # Combine the two logits into one modeling the intersection of events
            span_logits = (LogitIntersection(conv_spans, conv_probe_2d)).masked_fill_(
                ~mask, FILL_NEG_LOGITS
            )

            if return_logits:
                output = span_logits
            else:
                output = F.sigmoid(span_logits)

        elif self.method == CNN_METHODS.NLL_ALL:
            conv_spans = self.conv_scores(
                F.logsigmoid(scores) * mask, F.sigmoid(self.scoresKernel)
            ) + self.conv_scores(
                (F.logsigmoid(scores) - scores) * mask, 1 - F.sigmoid(self.scoresKernel)
            )
            conv_probe = self.conv_probe(
                F.logsigmoid(end_ent) * probe_mask, F.sigmoid(self.probeKernel)
            ) + self.conv_probe(
                (F.logsigmoid(end_ent) - end_ent) * probe_mask,
                1 - F.sigmoid(self.probeKernel),
            )

            # expand end_ent logprobs to 2D
            conv_probe_2d = (
                conv_probe.unsqueeze(-1).expand(-1, -1, seq) * mask
            )  # shape (batch, seq, seq)

            # Output Probabilities
            output = torch.exp(
                (conv_spans + conv_probe_2d).masked_fill_(~mask, FILL_NEG_LOGITS)
            )

        else:
            raise ValueError(
                f"Method {self.method} not implemented, choose among {CNN_METHODS}"
            )

        # debug_forward()

        if return_mask:
            return output, mask
        else:
            return output

    def training_step(
        self, batch: dict, model: HookedTransformer, lasso_reg: float = 0
    ) -> torch.Tensor:
        """Compute loss for a batch of data
        Args:
            batch: dict of data
            model: HookedTransformer form TransformerLens to extract representations from
            lasso_reg: Lasso (L1 norm) regularization on the scores kernel
        Returns:
            loss: loss for the batch
        """

        def debug_train():
            nonlocal batch_mask, mask, span_probs, patterns, tags, padded_mask, b_size
            import matplotlib.pyplot as plt

            m_span_probs = span_probs.detach().clone()
            m_span_probs[~mask] = torch.nan
            print(batch_mask.shape, span_probs.shape, patterns.shape, tags.shape)
            n = 3
            fig, ax = plt.subplots(4, n, figsize=(13, 15))
            for i in range(n):
                ind = torch.randint(b_size, (1,)).item()
                ax[0, i].imshow(patterns[ind, :, :].cpu().int())
                ax[0, i].set_title(f"pattern {ind}")
                im = ax[1, i].imshow(m_span_probs[ind, :, :].detach().cpu())
                ax[1, i].set_title(f"span_probs {ind}")
                # colorbar
                fig.colorbar(im, ax=ax[1, i])
                ax[2, i].imshow(padded_mask[ind, :, :].detach().cpu().int())
                ax[2, i].set_title(f"padded_mask {ind}")
                ax[3, i].imshow(batch_mask[ind, :, :].cpu().int())
                ax[3, i].set_title(f"item_mask {ind}")
            plt.show()
            raise ValueError("stop")

        texts = batch["text"]
        tags = batch["token_tags"].float().cuda()
        patterns = batch["pattern"].cuda()  # tensor shape (batch, seq, seq)
        b_size = patterns.size(0)
        inputs = model.tokenizer(
            texts, padding=True, padding_side="right", return_tensors="pt"
        )
        tokens = inputs["input_ids"].cuda()
        attn_mask = inputs['attention_mask'].cuda()
        padded = tokens == model.tokenizer.pad_token_id  # shape (batch, seq)
        padded_mask = padded.unsqueeze(1) | padded.unsqueeze(
            2
        )  # transform to padded mask to 2D mask (batch, seq, seq)

        span_probs, mask = self.forward(
            tokens,
            model,
            scores_mask=~padded_mask,
            probe_mask=~padded,
            return_logits=True,
            return_mask=True,
        )  #  (batch, seq, seq)

        # Compute loss
        if (
            self.dilate_entities is None
        ):  # compute loss on everything, take orginal mask
            batch_mask = ~padded_mask & mask  # consider all predictions for BCE loss

        elif (
            type(self.dilate_entities) == int
        ):  # we only compute the loss on the entity *rows*
            # dilate and inflate mask to match the shape of the pattern
            ner_tags = self.dilate_fx(tags)  # shape (batch, seq)
            ner_tags = ner_tags.unsqueeze(1) | ner_tags.unsqueeze(
                2
            )  # shape (batch, seq, seq)
            batch_mask = ner_tags & ~padded_mask & mask

        elif (
            len(self.dilate_entities) == 2
        ):  # we only compute the loss on the matches around entity blocks
            dilated_mask = self.dilate_fx(patterns).squeeze(
                0
            )  # shape (batch, seq, seq)
            batch_mask = dilated_mask & ~padded_mask & mask
        else:
            raise NotImplementedError(
                f"shouldn't be here, self.dilate_entities should be None, 1 or 2, not {self.dilate_entities}"
            )

        loss = self.criterion(span_probs[batch_mask], patterns[batch_mask])

        if lasso_reg:
            # Lasso (L1 norm) regularization on the scores kernel
            loss += lasso_reg * (
                torch.norm(self.scoresKernel, 1) + torch.norm(self.probeKernel, 1)
            )

        # debug_train()
        return loss

    def train(
        self,
        train_loader,
        val_loader,
        layer: int,
        hist=[],
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        epochs=2,
        lr=1e-3,
        grad_clip=0,
        lasso_reg=0,
        pos_weight=1,
        accumulation_steps=1,
        patience=3,
        min_lr=1e-5,
        n_val=1000,
        val_metric="bce",
        dilate_entities=None,
    ):
        """Train attention model on the given dataset
        Args:
            model: HookedTransformer form TransformerLens to extract representations from
            attn: attention model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            end_classifier: (Module, optionnal) binary classifier to train on top of the attention scores
            sliding_window: whether to use a sliding window attention, default 0 : no sliding window
            hist: history of training
            epochs: number of epochs to train
            lr: learning rate
            grad_clip (float): gradient clipping value, if 0, no clipping
            pos_weight: weight for positive examples in BCE loss, default 1 = no weighting
            accumulation_steps: number of steps to accumulate gradients before updating weights
            patience: number of untolerable validations to wait before reducing learning rate
            min_lr: minimum learning rate, if reached, training stops
            n_val: number of steps between validation and logging
            val_metric: metric to use for validation, default "bce"
            dilate_entities: Compute the loss only on entities (dilated) and not the full sequence
        """
        train_loader.dataset.mode = "last"
        val_loader.dataset.mode = "last"

        if self.layer != layer:
            logging.info(
                f"Warning: Current layer of the model ({self.layer}) differs from the one given ({layer}), changing..."
            )
        self.layer = layer
        model = train_loader.dataset.model
        if self.llm_name != model.cfg.model_name:
            logging.info(
                f"Warning: Current LLM of the model ({self.llm_name}) differs from dataset ({model.cfg.model_name:}), changing..."
            )
            self.llm_name = model.cfg.model_name

        ## Setup masking and loss
        if dilate_entities is not None:
            try:
                dilate_entities = torch.tensor(dilate_entities).int().view(-1)
            except TypeError:
                raise ValueError(
                    "if not None, dilate_entities should be an int or a list of two ints"
                )
            # now we have a tensor
            if len(dilate_entities) == 1:
                dilate_entities = dilate_entities.item()
                if not dilate_entities:
                    dilate = None
                    dilate_entities = None
                else:
                    dilate = masks.get_dilation(dilate_entities)
            elif len(dilate_entities) == 2:
                dilate = masks.get_2d_dilation(dilate_entities)
            else:
                raise ValueError(
                    "if not None, dilate_entities should be an int or a list of two ints"
                )

        self.dilate_fx = dilate
        self.dilate_entities = dilate_entities

        if self.method == CNN_METHODS.NLL_ALL:
            self.criterion = nn.BCELoss(reduction="mean")
        else:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight), reduction="mean"
            )

        logging.info(
            f"Training Self Attention, Probe and Pattern at layer {self.layer} for {epochs} epochs with batch size {train_loader.dataset.batch_size} \n\
                         - dilate_entities: {dilate_entities} \n\
                         - pattern kernel padding: {self.kernel_padding} \n\
                         - CNN aggrgation method : {self.method} \n\
                         - pos_weight: {pos_weight} \n\
                         - Causal mask {self.attn.causal_mask}\n\
                         - bos mask {self.attn.mask_bos}\n\
                         - Sliding Window {self.attn.sliding_window}... "
        )

        hist = train(
            self,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            val_metric=val_metric,
            hist=hist,
            epochs=epochs,
            lr=lr,
            grad_clip=grad_clip,
            accumulation_steps=accumulation_steps,
            patience=patience,
            min_lr=min_lr,
            n_val=n_val,
            train_step_args={
                "model": train_loader.dataset.model,
                "lasso_reg": lasso_reg,
            },
        )

        del self.criterion
        del self.dilate_fx
        del self.dilate_entities
        gc.collect()
        torch.cuda.empty_cache()

        return hist
