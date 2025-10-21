import torch, logging, gc
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
)  # , TypeVar
from functools import partial

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

# xpm and misc
from experimaestro import (
    Param,
    Constant,
    Config,
    DataPath,
    Task,
    LightweightTask,
    Meta,
)
from rotary_embedding_torch import RotaryEmbedding

# Our code
from llm2ner import utils, masks, heuristics
import llm2ner.data as data
from llm2ner.losses import balanced_BCE
from llm2ner.models import NERmodel, train, FILL_NEG_LOGITS


###### MHSA NER MODEL ######


# Match representations computed with a MHSA layer
class MHSA_NER(NERmodel):

    n_heads: Param[int]

    rank: Param[int]

    sliding_window: Param[int]

    causal_mask: Param[bool] = False

    use_rotary: Param[bool] = False

    use_cosine: Param[bool] = False

    use_pre_LN: Param[bool] = False

    equal_q_k: Param[bool] = False

    mask_bos: Param[bool] = False

    version: Constant[str] = "1.0"

    # Public attributes (initialized in __post_init__)
    rotary: RotaryEmbedding
    Q: nn.Linear
    K: nn.Linear
    V: nn.Linear
    O: nn.Linear

    def __post_init__(self):
        """Method called after instanciation of config"""
        super().__post_init__()

        self.scale = 1 if self.use_cosine else 1 / (self.rank**0.5)

        if self.use_rotary:
            llm_config = convert_hf_model_config(get_official_model_name(self.llm_name))
            self.rotary = RotaryEmbedding(
                dim=self.rank,
                # theta=model.cfg.rotary_base,
                xpos_scale_base=llm_config["rotary_base"],
                use_xpos=True,
            )
        else:
            self.rotary = None

        if self.use_pre_LN:
            self.pre_LN = nn.RMSNorm(self.dim, eps=1e-5)
        else:
            self.pre_LN = nn.Identity()
        # Parameters definition
        #   input shape is batch, seq
        # we double the number of heads to compute queries and keys for final score
        self.QK_dim = 2 * self.n_heads * self.rank
        self.Q = nn.Linear(self.dim, self.QK_dim, bias=False)
        self.V = nn.Linear(self.dim, self.QK_dim, bias=False)
        self.O = nn.Linear(self.QK_dim, self.QK_dim, bias=True)

        if self.equal_q_k:
            self.K = self.Q
        else:
            self.K = nn.Linear(self.dim, self.QK_dim, bias=False)

    def extra_repr(self):
        return (
            f"{super().extra_repr()}\n n_heads={self.n_heads}, rank={self.rank}, equal_q_k={self.equal_q_k}, use_rotary={self.use_rotary} \n"
            f"causal_mask={self.causal_mask}, sliding_window={self.sliding_window}, mask_bos={self.mask_bos},"
        )

    def get_mask(
        self, seq: int, causal: bool, mask_bos: bool
    ) -> Float[torch.Tensor, "seq seq"]:
        """Get mask for the attention scores, used by the MHSA layer as well"""
        if self.sliding_window or self.causal_mask:
            return (
                masks.create_mask_cached(
                    1,
                    1,
                    seq,
                    seq,
                    causal=causal,
                    mask_bos=mask_bos,
                    sliding_window=self.sliding_window,
                    device=self.device,
                )
                .squeeze(0)
                .squeeze(0)
            )  # shape (seq, seq)
        else:
            return None

    def forward_MHSA(
        self, reps: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq rank"]:
        """Apply multi-head self attention to the input representations
        Args:
            reps: tensor (batch, seq, dim) representations of the tokens
        Returns:
            values: tensor (batch, seq, rank) values for each token in the text after mhsa
        """
        b, seq, _ = reps.size()

        # Reps shape is (batch, seq, dim)
        query, key, value = self.Q(reps), self.K(reps), self.V(reps)

        # shape (batch, seq, n_heads * rank)
        n_heads = 2 * self.n_heads  # double the number of heads for query and key

        # Reshape to (batch, n_heads, seq, rank)
        query = rearrange(query, "b s (h r) -> b h s r", h=n_heads)
        key = rearrange(key, "b s (h r) -> b h s r", h=n_heads)
        value = rearrange(value, "b s (h r) -> b h s r", h=n_heads)
        # print(query.shape, key.shape, value.shape)

        if self.rotary:
            query, key = self.rotary.rotate_queries_and_keys(query, key)

        # create mask (can be None)
        mask = self.get_mask(
            seq, causal=self.causal_mask, mask_bos=self.mask_bos
        ).repeat(b, n_heads, 1, 1)

        # run efficient pytorch self attention with mask
        outputs = F.scaled_dot_product_attention(
            query,
            key,
            value,  # Batch, n_head, seq, dim_head
            attn_mask=mask,
            scale=self.scale,
        )  # shape (batch, 2*n_heads, seq, rank)

        # print("sdpa outputs shape:", outputs.shape)

        # concatenate heads and pass to output layer
        return self.O(
            rearrange(
                outputs,
                "b h s r -> b s (h r)",
            )  # shape (batch, seq, 2*n_heads*rank)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        model: HookedTransformer = None,
        attn_mask: torch.Tensor = None,
        return_logits: bool = False,
        return_mask: bool = False,
    ) -> Float[torch.Tensor, "batch seq"]:
        """Compute NER tags for given text,
        Args:
            reps: tensor (batch, seq, dim) representations of tokens to classify
        Returns:
            tags: tensor (batch, seq) NER tags for each token in the text
        """
        if model is not None:
            # assuming inputs is tokens, compute reps
            reps = self.get_representations(tokens=inputs, model=model, attn_mask=attn_mask)
        else:
            # assuming inputs is already the representations
            reps = inputs

        assert len(reps.shape) == 3, "Input tensor must be of shape (batch, seq, dim)"
        b, seq, _ = reps.size()
        
        reps = self.pre_LN(reps)
        
        # Compute attention scores
        outputs = self.forward_MHSA(reps)  # shape (batch, seq, 2*n_heads*rank)
        # print("mhsa outputs shape:", outputs.shape)

        # separate queries and keys
        s_query = outputs[:, :, : self.n_heads * self.rank]
        s_key = outputs[:, :, self.n_heads * self.rank :]

        if self.use_cosine:
            # compute cosine similarity
            Q = F.normalize(s_query, dim=-1)
            K = F.normalize(s_key, dim=-1)

        # score each span by computing the dot product of begin and end tokens
        scores = torch.einsum(
            "b i h, b j h -> b i j", s_query, s_key
        )  # shape (batch, seq, seq)

        # get causal mask for scores
        scores_mask = self.get_mask(seq, causal=True, mask_bos=True).repeat(
            b, 1, 1
        )  # shape (seq, seq)

        if scores_mask is not None:
            scores.masked_fill_(
                ~scores_mask, FILL_NEG_LOGITS
            )  # shape (batch, seq, seq)

        if not return_logits:
            scores = F.sigmoid(scores)

        if return_mask:
            return scores, scores_mask
        else:
            return scores

    def training_step(
        self,
        batch: dict,
        model: HookedTransformer,
    ) -> torch.Tensor:
        """Compute loss for a batch of data
        Args:
            batch: dict of data
            model: HookedTransformer form TransformerLens to extract representations from
            lasso_reg: Lasso (L1 norm) regularization on the scores kernel
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
        attn_patterns = batch["pattern"].cuda()  # tensor shape (batch, seq, seq)

        span_logits, batch_mask = self.forward(
            tokens, model, attn_mask=attn_mask, return_mask=True, return_logits=True
        )  # shape (batch, seq, seq)
        # reshape to (batch, seq, seq) putting the end_ent_scores on the diagonal and add to scores

        loss = balanced_BCE(
            span_logits[batch_mask],
            attn_patterns[batch_mask],
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

        assert PL_threshold == 1, "Pseudo Labeling is not implemented"

        logging.info(
            f"Training MHSA NER model with representations from layer {self.layer} for {epochs} epochs with batch size {train_loader.dataset.batch_size} \n\
                         - optimizer: {optimizer} \n\
                         - dilate_entities: {dilate_entities} \n\
                         - Causal mask {self.causal_mask}\n\
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


### DEPRECATED : manual MHSA implementation, kept for reference ###


class MultiHeadAttention(nn.Module):
    """naive implementation of Multihead attention using einstein summation"""

    def __init__(
        self,
        model_dim: int,
        rank: int,
        n_heads: int = 1,
        use_rotary: bool = False,
        llm_name: str = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.rank = rank
        self.n_heads = n_heads
        self.use_rotary = use_rotary

        self.scale = 1 / (rank**0.5)

        if use_rotary:
            llm_config = convert_hf_model_config(get_official_model_name(llm_name))
            self.rotary = RotaryEmbedding(
                dim=rank,
                xpos_scale_base=llm_config["rotary_base"],
                use_xpos=True,
            )
            logging.info(
                f"Using rotary embeddings with base {llm_config['rotary_base']} for {llm_name}"
            )
        else:
            self.rotary = None
        # Parameters definition
        #   input shape is batch, seq
        self.W_Q = nn.Parameter(
            torch.randn(model_dim, rank * n_heads)
        )  # shape (model_dim, rank * n_heads)
        self.W_K = nn.Parameter(
            torch.randn(model_dim, rank * n_heads)
        )  # shape (model_dim, rank * n_heads)
        self.W_V = nn.Parameter(
            torch.randn(model_dim, rank * n_heads)
        )  # shape (model_dim, rank * n_heads)
        self.W_O = nn.Parameter(
            torch.randn(rank * n_heads, rank)
        )  # shape (rank * n_heads, rank)

    def forward(
        self,
        X: Float[torch.Tensor, "batch seq dim"],
        mask: Bool[torch.Tensor, "batch seq seq"] = None,
        return_scores: bool = False,
    ) -> Float[torch.Tensor, "batch seq rank"]:
        """Compute the attention scores for the input x
        Args:
            x: tensor (batch, seq, dim) representations of tokens to classify
            mask: (Optionnal) tensor (seq, seq) mask for the attention scores
            return_scores: whether to return the attention scores along with the attention output
        Returns:
            attn: tensor (batch, seq, rank) attention scores for each token in the text
        """
        b, seq, _ = X.size()

        # print(f"X {X.shape} mask {mask.shape if mask is not None else None}")

        # Compute QKV, shape (batch, seq, n_heads * rank)
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = rearrange(Q, "b s (h r) -> b h s r", h=self.n_heads)
        K = rearrange(K, "b s (h r) -> b h s r", h=self.n_heads)
        V = rearrange(V, "b s (h r) -> b h s r", h=self.n_heads)
        # print(f"Q {Q.shape} K {K.shape} V {V.shape}")
        # Reshape to (batch, n_heads, seq, rank)
        # Q = Q.view(b, seq, self.n_heads, self.rank).transpose(1, 2)
        # K = K.view(b, seq, self.n_heads, self.rank).transpose(1, 2)
        # V = V.view(b, seq, self.n_heads, self.rank).transpose(1, 2)

        # Apply rotary embedding
        if self.use_rotary:
            Q, K = self.rotary.rotate_queries_and_keys(Q, K)

        # print(f"Q {Q.shape} K {K.shape} V {V.shape}")

        # Compute attention scores
        attn_scores = (
            torch.einsum("b h i r, b h j r -> b h i j", Q, K) * self.scale
        )  # (batch, n_heads, seq, seq)

        if mask is not None:
            # repeat mask to n_heads
            mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            # Apply mask to attention scores
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)  # Apply softmax

        # shape (batch, n_heads, seq, seq)
        # Compute attention output
        attn_output = torch.einsum(
            "b h i j, b h j r -> b h i r", attn_scores, V
        )  # shape (batch, n_heads, seq, rank)

        # Concatenation : Reshape to (batch, seq, rank)
        attn_output = rearrange(
            attn_output, "b h i r -> b i (h r)"
        )  # shape (batch, seq, rank)
        # print(f"attn_output {attn_output.shape} attn_scores {attn_scores.shape}")

        # Apply output projection
        attn_output = attn_output @ self.W_O  # shape (batch, seq, dim)

        if return_scores:
            return attn_output, attn_scores
        else:
            return attn_output


# Match representations computed with a MHSA layer
class MHSA_NER_man(NERmodel):
    def __init__(
        self,
        llm_name: str,
        # MHSA
        model_dim: int,
        rank: int,
        n_heads: int = 1,
        use_rotary: bool = False,
        # masking
        sliding_window: int = 0,
        causal_mask: bool = True,
        # Misc
        layer: int = None,
    ):

        super().__init__()  # sets layer and dim

        llm_config = convert_hf_model_config(get_official_model_name(llm_name))

        # ok, its cleaner with XPM
        self.sliding_window = sliding_window
        self.n_heads = n_heads
        self.causal_mask = causal_mask
        self.dim = model_dim
        self.rank = rank
        self.layer = layer
        self.llm_name = llm_name

        self.scale = 1 / (rank**0.5)

        # Parameters definition
        #   input shape is batch, seq
        self.Q_MHSA = MultiHeadAttention(
            model_dim,
            rank,
            n_heads=n_heads,
            use_rotary=use_rotary,
            llm_name=llm_name,
        )
        self.K_MHSA = MultiHeadAttention(
            model_dim,
            rank,
            n_heads=n_heads,
            use_rotary=use_rotary,
            llm_name=llm_name,
        )

        # self.Q_MHSA = nn.MultiheadAttention(
        #     model_dim,
        #     n_heads,
        #     dropout=0.0,
        #     bias=True,
        #     add_bias_kv=False,
        #     add_zero_attn=False,
        #     batch_first=True)
        # self.K_MHSA = nn.MultiheadAttention(
        #     model_dim,
        #     n_heads,
        #     dropout=0.0,
        #     bias=True,
        #     add_bias_kv=False,
        #     add_zero_attn=False,
        #     batch_first=True)
        # self.W_O = nn.Parameter(torch.randn(self.rank * self.n_heads, self.dim)) ??

    def get_attn_mask(
        self, seq: int, mask_bos: bool = False
    ) -> Float[torch.Tensor, "seq seq"]:
        """Get mask for the attention scores, used by the MHSA"""
        if self.sliding_window or self.causal_mask:
            return (
                masks.create_mask_cached(
                    1,
                    1,
                    seq,
                    seq,
                    causal=self.causal_mask,
                    mask_bos=mask_bos,
                    sliding_window=self.sliding_window,
                )
                .squeeze(0)
                .squeeze(0)
            )  # shape ( seq, seq)
        else:
            return None

    def get_scores_mask(self, seq: int) -> Float[torch.Tensor, "seq seq"]:
        """Get mask for the span scores, always causal"""
        return (
            masks.create_mask_cached(
                1,
                1,
                seq,
                seq,
                causal=True,
                mask_bos=True,
                sliding_window=self.sliding_window,
            )
            .squeeze(0)
            .squeeze(0)
        )  # shape ( seq, seq)

    def debug_forward(self, attn_mask, scores_mask, scores):
        """Debug function to visualize the attention scores"""
        import matplotlib.pyplot as plt

        batch, seq, _ = scores.size()
        print(f"attn mask {attn_mask.shape} scores {scores.shape}")
        n = 3
        fig, ax = plt.subplots(3, n, figsize=(15, 15))
        for i in range(n):
            ind = torch.randint(batch, (1,)).item()
            im = ax[0, i].imshow(attn_mask[ind, :, :].detach().cpu())
            ax[0, i].set_title(f"attn_mask {ind}")
            fig.colorbar(im, ax=ax[0, i])

            im = ax[1, i].imshow(scores[ind, :, :].detach().cpu())
            ax[1, i].set_title(f"scores {ind}")
            fig.colorbar(im, ax=ax[1, i])

            im = ax[2, i].imshow(scores_mask.detach().cpu())
            ax[2, i].set_title(f"scores_mask {ind}")
            fig.colorbar(im, ax=ax[2, i])
        plt.tight_layout()
        plt.show()
        raise ValueError("stop")

    def forward(
        self, reps, return_mask: bool = False, return_logits: bool = False
    ) -> Float[torch.Tensor, "batch seq rank"]:
        """Compute NER tags for given text,
        Args:
            reps: tensor (batch, seq, dim) representations of tokens to classify
        Returns:
            tags: tensor (batch, seq) NER tags for each token in the text
        """
        b, seq, _ = reps.size()
        attn_mask = self.get_attn_mask(seq)
        scores_mask = self.get_scores_mask(seq)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(b, 1, 1).to(
                reps.device
            )  # shape (batch, seq, seq)

        # Compute attention scores
        attn_Q = self.Q_MHSA(reps, mask=attn_mask)  # shape (batch, seq, rank)
        attn_K = self.K_MHSA(reps, mask=attn_mask)  # shape (batch, seq, rank)

        # score each span by computing the dot product of begin and end tokens
        scores = torch.einsum(
            "b i h, b j h -> b i j", attn_K, attn_Q
        )  # shape (batch, seq, seq)
        scores.masked_fill_(~scores_mask, FILL_NEG_LOGITS)  # shape (batch, seq, seq)

        if not return_logits:
            scores = F.sigmoid(scores)

        # self.debug_forward(attn_mask, scores_mask, scores)
        if return_mask:
            return scores, scores_mask
        else:
            return scores

    def training_step(
        self,
        batch: dict,
        model: HookedTransformer,
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
            m_span_probs[:, ~mask] = torch.nan
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

        with torch.no_grad():  # we don't need gradients for the representations
            reps = utils.compute_to_layer(
                model, self.layer, tokens, attn_mask=attn_mask, dtype=self.dtype
            ).cuda()  # shape (batch, seq, dim)

        span_probs, mask = self.forward(
            reps, return_logits=True, return_mask=True
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
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight), reduction="mean"
        )

        logging.info(
            f"Training MHSA Ner model at layer {self.layer} for {epochs} epochs with batch size {train_loader.dataset.batch_size} \n\
                         - dilate_entities: {dilate_entities} \n\
                         - pos_weight: {pos_weight} \n\
                         - Causal mask {self.causal_mask}\n\
                         - RoPE {self.Q_MHSA.use_rotary}\n\
                         - Sliding Window {self.sliding_window}... "
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
            },
        )

        del self.criterion
        del self.dilate_fx
        del self.dilate_entities
        gc.collect()
        torch.cuda.empty_cache()

        return hist
