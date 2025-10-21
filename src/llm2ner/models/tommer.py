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

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# HF and Tlens
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

# xpm and misc
from experimaestro import (
    Param,
    Param,
    Constant,
)

# Our code
from llm2ner import utils, masks, heuristics
import llm2ner.data as data
from llm2ner.losses import balanced_BCE
from llm2ner.models.model import *
from llm2ner.models.TokenMatching import (
    ReprClassifier,
    shift_up,
    min_pool_right_strict,
    max_pool_right_strict,
)

CARD_TEMPLATE_FILE = Path(__file__).parent / "ToMMeR.md"
CARD_TEMPLATE = CARD_TEMPLATE_FILE.read_text()

class ToMMeR(NERmodel,
    library_name="VictorMorand/ToMMeR",
    model_card_template=CARD_TEMPLATE,
    tags=["torch", "transformers", "llm", "ner"],
    repo_url="https://github.com/VictorMorand/llm2ner",
    paper_url="https://arxiv.org/abs/???",
):
    """ToMMer Model : Token Matching Mention Recognition
    ToMMeR is an attention based model for Entity Mention Recognition.

    For more information, see the related paper : https://arxiv.org/abs/2510.09421
    """

    llm_name: Param[str]
    """Name or Id of LLM  used to extract the representations"""

    layer: Param[int] = 0
    """List of layers to extract the query and key scores from"""

    rank: Param[int] = 100
    """Rank of the Query / Keys projection"""

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

        self.mask_bos = True
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
        self.cl_w = nn.Parameter(torch.ones(1, 5, 1, 1))

    def extra_repr(self):
        return (
            super().extra_repr()
            + f"\n\
            rank={self.rank},\n\
            causal_mask={self.causal_mask},\n\
            use_cosine={self.use_cosine},\n\
            normalize_scores={self.normalize_scores},\n\
            sliding_window={self.sliding_window}"
        )

    def get_model_id(self):
        llm_name = self.llm_name.split("/")[-1]
        return f"ToMMeR-{llm_name}_L{self.layer}_R{self.rank}"

    def generate_model_card(self, **kwargs):
        # Merge model-specific variables with any additional kwargs
        # load all named parameters from the config
        cfg_dict = self.__config__.__xpm__.values
        cfg_dict['model_id'] = self.get_model_id()
        return super().generate_model_card(**cfg_dict | kwargs)

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

    def get_QK_reps(
        self,
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
        reps = self.get_representations(
            tokens, model, attn_mask=attn_mask
        )  # shape (batch, seq, dim)

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
            device=self.device
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
        elif self.normalize_scores == "atan":
            # apply atan normalization
            scores = torch.atan(scores)

        if return_mask:
            return scores, scores_mask
        else:
            return scores

    def combine_scores(
        self,
        attn_scores: Float[torch.Tensor, "batch seq seq"],
        end_ent_scores: Float[torch.Tensor, "batch seq"],
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

        # compute mean match inside spans

        stacked_logits = torch.stack(
            [
                attn_scores,
                min_pool_right_strict(attn_scores),  # min pool right
                max_pool_right_strict(attn_scores),  # max pool right
                end_ent_2d,  # last token probe score
                shift_up(end_ent_2d),  # shift up to get the next token's probe score
            ],
            dim=1,
        )  # shape (batch, 3, seq, seq)

        # print(f"stacked_logits {stacked_logits.shape} - mean_score {mean_score.shape} - attn_scores {attn_scores.shape} - end_ent_2d {end_ent_2d.shape}")
        # Linear combination of the features
        return (self.cl_w * stacked_logits).sum(dim=1)  # shape (batch, seq, seq)

    # for training
    def token_matching_loss(
        self,
        batch: dict,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        scores: Float[torch.Tensor, "batch seq seq"],
        end_ent_scores: Float[torch.Tensor, "batch seq"],
        mask: Bool[torch.Tensor, "batch seq seq"],
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
        tags = batch["token_tags"].float().to(self.device)
        attn_patterns = batch["pattern"].to(self.device)  # tensor shape (batch, seq, seq)
        end_ent = batch["end_ent"].to(self.device)
        tokenizer: PreTrainedTokenizerBase = model.tokenizer  # type: ignore
        b_size = attn_patterns.size(0)
        # type:ignore shape (batch, seq)
        padded: torch.Tensor = tokens == tokenizer.pad_token_id

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

        batch_mask = dilated_mask & mask & ~padded_mask
        batch_mask_1D = batch_mask.diagonal(0, 1, 2)  # shape (batch, seq)
        span_scores = self.combine_scores(
            scores, end_ent_scores, batch_mask
        )  # shape (batch, seq, seq)
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
            scores, end_ent, b_mask
        )  # shape (batch, seq, seq)

        if not return_logits :
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
        tokens = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)

        dilate_entities = (
            self.dilate_entities if hasattr(self, "dilate_entities") else None
        )

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
