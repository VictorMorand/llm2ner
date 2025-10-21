import json, logging
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path
from functools import partial
import torch, torch.nn as nn, torch.nn.functional as F
from enum import Enum
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from jaxtyping import Float, Int
from sklearn.metrics import confusion_matrix
from transformers.modeling_utils import PreTrainedModel
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
)

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
    Path,
    field,
    PathGenerator,
)

# Our code
from llm2ner import utils
from llm2ner.losses import balanced_BCE
import llm2ner.data as data
from llm2ner.models import NERmodel, train, EPS
from llm2ner.xpmModel import xpmTorchHubModule


class NERCmodel(xpmTorchHubModule):
    """NER + clasification model, adds a classifier on top of the NER model"""

    ner_model: Param[NERmodel]
    """NER model to use for NER"""
    
    llm_name: str
    """Name of the LLM used for NER, same as provided ner_model"""
    
    layer: Param[Optional[int]]
    """Layer at which to extract representations, extracted from ner_model if None"""
    
    method: Param[str] = "concat"
    """Method to use for span representation, either 'concat' or 'avg'"""

    ent_type_prompt: Param[str] = "Entity type: '"
    """Prompt to use for entity type classification, used to compute class representations in zero shot"""

    embed_dim: Param[int] = 0
    """Embedding dimension for the span representations, if 0, will use the model dimension"""

    n_layers: Param[int] = 2
    """Number of layers in the span embedding MLP"""

    bias: Param[bool] = True
    """Whether to use bias in the linear layers of the model"""

    FN_CLASS_VAL = -1

    model_dim: int
    """Dimension of the LLM model, extracted from ner_model"""


    class SPAN_METHODS(str, Enum):
        """Methods for NER classification"""

        CONCAT = "concat"
        """Concatenate the representations of the first and last tokens of the entity, as done in GliNER"""

        AVG = "avg"
        """Average the representations of the tokens of the entity"""

    def extra_repr(self):
        return f"method: '{self.method}', layer: {self.layer}, embed_dim: {self.embed_dim}" + super().extra_repr()
    
    def __post_init__(
        self,
    ):
        """Initialize NERCmodel
        Args:
            ner_model: NERmodel to use for NER
            n_classes: number of classes to classify
        """
        super().__post_init__()

        self.llm_name = self.ner_model.llm_name
        self.model_dim = self.ner_model.dim

        if self.embed_dim <= 0:
            self.embed_dim = self.model_dim

        if self.layer is None: self.layer = self.ner_model.layer 

        assert self.method in list(
            self.SPAN_METHODS
        ), f"method {self.method} not in { ' '.join(self.SPAN_METHODS)}"

        if self.method == self.SPAN_METHODS.AVG:
            self.span_dim = self.model_dim
        elif self.method == self.SPAN_METHODS.CONCAT:
            self.span_dim = 2 * self.model_dim
        else:
            raise NotImplementedError(f"method {self.method} not implemented")

        # only one projection
        # self.classifier = nn.Linear(self.dim, n_classes)

        # intermediate "entity representation" before classification
        layers = [nn.Linear(self.span_dim, self.embed_dim, bias=self.bias)]
        if self.n_layers > 1:
            layers += [
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias),
            ] * (self.n_layers - 1)
        self.span_embed = nn.Sequential(*layers)

        self.class_embed = nn.Sequential(
            nn.Linear(self.model_dim, self.embed_dim, bias=self.bias),
            # RMSNorm(self.embed_dim, eps=1e-6, dtype=torch.float32),
        )

        self.O_class_embed = nn.Parameter(
            torch.zeros(self.embed_dim, dtype=torch.float32), requires_grad=True
        )  # representation for O class
        # will contain class representations during training
        self.class_reps = None

    def get_dtype(self) -> torch.dtype:
        for param in self.parameters():
            return param.dtype
        for buffer in self.buffers():
            return buffer.dtype
        raise ValueError("Module has no parameters or buffers.")

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model"""
        return self.get_dtype()

    @dtype.setter
    def dtype(self, dtype):
        """Set the dtype of the model"""
        self.to(dtype)

    def get_span_reps(
        self,
        reps: Float[torch.Tensor, "batch seq dim"],
        entities_inds: Int[torch.Tensor, "n_entities 3"],
    ) -> Float[torch.Tensor, "batch n_entities dim"]:
        """Get span representations for given entities, extract LLM reps and embed them
        Args:
            reps: tensor (batch, seq, dim) representations of tokens to classify
            entities: tensor (n_entities, 3) indices (batch, begin, end) of the entities in the batch
        Returns:
            span_reps: tensor (batch, n_entities, dim) representations of entity spans
        """
        if self.method == self.SPAN_METHODS.CONCAT:
            span_reps = torch.cat(
                [
                    reps[entities_inds[:, 0], entities_inds[:, 1], :],
                    reps[entities_inds[:, 0], entities_inds[:, 2], :],
                ],
                dim=-1,
            )
        elif self.method == self.SPAN_METHODS.AVG:
            span_reps = torch.vstack(
                [
                    torch.mean(reps[btc, b : e + 1, :], dim=0)
                    for btc, b, e in entities_inds
                ]
            )
        else:
            raise NotImplementedError(f"method {self.method} not implemented")

        return self.span_embed(span_reps)

    def get_class_reps(
        self,
        classes: List[str],
        model: HookedTransformer,
        layer: int,
    ):
        """
        Computes class representations for given classes, adding self O class representation
        Args:
            classes: list of n classes to compute representations for
            model: HookedTransformer form TransformerLens to extract representations from
            layer: layer at which to extract the representations
        Returns:
            class_reps: tensor (n + 1, dim) representations for each class, including the O class at index 0
        """

        prompts = [self.ent_type_prompt + cls for cls in classes]
        inputs = model.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        )
        tokens = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        reps = utils.compute_to_layer(
            model,
            layer,
            tokens=tokens,
            attn_mask=attention_mask,
            dim=self.model_dim,
            dtype=self.dtype,
        )

        # embed last token representations
        labeled_class_reps = self.class_embed(reps[:, -1, :]).requires_grad_(True)

        # add O class representation at index 0
        return torch.vstack(
            [self.O_class_embed, labeled_class_reps],
        )

    def load_nermodel(self, ner_model: NERmodel):
        """Load NER model to use for NER"""

        assert self.llm_name == ner_model.llm_name
        if self.layer != ner_model.layer:
            logging.warning(
                f"Loading NER model at layer {ner_model.layer} instead of {self.layer}"
            )
            self.layer = ner_model.layer

        self.ner_model = ner_model

    def infer_and_embed(
        self,
        tokens: torch.Tensor,
        model: Union[HookedTransformer, None],
        attn_mask: Optional[torch.Tensor] = None,
        decoding_strategy: str = "threshold",
        threshold: float = 0.5,
    ) -> Tuple[
        Int[torch.Tensor, "n_entities 3"],
        Float[torch.Tensor, "n_entities dim"],
    ]:
        """Infer entities with ner_model, extract residual representations and
        compute inferred spans representations them using the span_embed layer
        Args:
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
        Returns:
            inds: tensor (#entities, 3) indices of the entities in the batch (batch, begin, end)
            span_reps: tensor (#entities, dim) representations of entity spans
        """
        # predict unclassified entitites
        with torch.no_grad():
            cand_entities = self.ner_model.infer_entities(
                tokens,
                model,
                attn_mask=attn_mask,
                decoding_strategy=decoding_strategy,
                threshold=threshold,
            )  # batch of list of entities

        inds = torch.tensor(
            [(b, *ent) for b in range(len(cand_entities)) for ent in cand_entities[b]]
        ).cuda()  # shape (#entities , 3)

        span_reps = self.embed_spans(inds, tokens, model, attn_mask=attn_mask)

        return inds, span_reps  # shape (#entities, dim)

    def embed_spans(
        self,
        entities: torch.Tensor,
        tokens: torch.Tensor,
        model: Union[HookedTransformer, None],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "n_entities dim"]:
        """Embed entity spans using the model's span embedding layer
        Args:
            entities: tensor (n_entities, 3) of entity indices (batch, begin, end)
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
        Returns:
            span_reps: tensor (n_entities, dim) of span representations
        """
        if model is None:
            raise ValueError("Model must be provided for span embedding")

        with torch.no_grad():
            reps = utils.compute_to_layer(
                model, self.layer, tokens=tokens, attn_mask=attn_mask, dim=self.model_dim
            )

        # extract and embed
        span_reps = self.get_span_reps(reps, entities)  # shape (batch, #entities, dim)

        return span_reps

    def classify_zero_shot(
        self,
        span_reps: Float[torch.Tensor, "batch n_entities dim"],
        str_classes: Optional[List[str]] = None,
        model: Optional[HookedTransformer] = None,
        return_logits: bool = False,
    ):
        """Zero - Shot Classification of spans representations using the class_embed layer,
        This differs from classify_sup in that a negative match logit will be returned as class 0.
        will compute new class representations if str_classes is provided, or use pre-computed self.class_reps otherwise

        Args:
            span_reps: tensor (batch, n_entities, dim) representations of entity spans
            str_classes: list of classes to classify the entities, if None, uses precomputed class representations
            model: HookedTransformer form TransformerLens to extract representations from, required if str_classes is provided
            return_logits: if True, returns logits instead of classes

        Returns:
            class_logits: tensor (batch, n_entities, n_str_) logits for each class
        """
        if str_classes is not None:  # compute new class representations
            assert (
                model is not None
            ), "Model should be provided for zero-shot classification"
            class_reps = self.get_class_reps(
                str_classes, model, self.layer
            )  # shape (n_classes + 1, dim)
        else:  # use precomputed class representations (still zero-shot inference)
            assert (
                self.class_reps is not None
            ), "NERC model should have precomputed class representation for zero-shot classification when no str_classes provided."
            class_reps = self.class_reps

        class_logits = span_reps @ class_reps.T  # shape (batch, #entities, n_classes)

        if return_logits:
            return class_logits
        else:
            # Get argmax
            classes = torch.argmax(class_logits, dim=-1)  # tensor shape (#entities)

            # Replace by 0 for no match if logit is negative
            classes[class_logits.max(dim=-1).values < 0] = 0

            return (
                classes.detach().cpu()
            )  # shape (batch, #entities, n_classes), array (#entities)

    def classify_sup(
        self,
        span_reps: Float[torch.Tensor, "batch n_entities dim"],
        return_logits: bool = False,
    ):
        """Classify spans representations using the class representations self.class_reps.
        In this setup the class is inferred as the closest class representation to the span representation, including the O class.
        This is used for supervised training where class representations are precomputed.
        Args:
            span_reps: tensor (batch, n_entities, dim) representations of entity spans
            return_logits: if True, returns logits instead of classes
        Returns:
            class_logits: tensor (batch, n_entities, n_str_) logits for each class
        """

        assert (
            self.class_reps is not None
        ), "Model should have precomputed class representation for supervised classification"
        class_reps = self.class_reps

        class_logits = span_reps @ class_reps.T  # shape (batch, #entities, n_classes)

        if return_logits:
            return class_logits
        else:
            return torch.argmax(class_logits, dim=-1)  # array shape (#entities)

    def forward(
        self,
        tokens: torch.Tensor,
        model: Optional[Union[HookedTransformer, PreTrainedModel]],
        attn_mask: Optional[torch.Tensor] = None,
        decoding_strategy: str = "threshold",
        threshold: float = 0.5,
        str_classes: Optional[List[str]] = [],
        zero_shot: Optional[bool] = None,
        return_logits: bool = False,
    ):
        """Predict and classify entities
        Args:
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
            str_classes: list of classes to classify the entities, if None, uses precomputed class representations
            zero_shot: if True, uses zero-shot classification, if None, uses str_classes to determine the classification method
            return_logits: if True, returns logits instead of classes
        Returns:
            entities (Array (#entities, (batch_ind, beging, end))) list of predicted entities
            classes (Array (#entities)) predicted class for each entity
        """

        inds, span_reps = self.infer_and_embed(
            tokens,
            model,
            attn_mask=attn_mask,
            decoding_strategy=decoding_strategy,
            threshold=threshold,
        )  # shape (batch, #entities, dim)

        if str_classes or zero_shot:  # zero-shot classification
            classes = self.classify_zero_shot(
                span_reps, str_classes, model=model, return_logits=return_logits
            )
        else:  # supervised classification
            assert (
                self.class_reps is not None
            ), "Model should have precomputed class representation for supervised classification"
            classes = self.classify_sup(span_reps, return_logits=return_logits)

        return inds.cpu().numpy(), classes

    def tags_from_ents_classes(self, shape, entities, classes):
        """Convert entities and classes to tags
        Args:
            shape: shape of the output tensor (batch, seq)
            entities: tensor of entities (#entities, (batch, start, end))
            classes: list of classes (batch, n_classes)
        Returns:
            tags: tensor (batch, seq) NER tags for each token in the text
        """

        tags = torch.zeros(shape, dtype=torch.int64)
        batch, first, last = entities[:, 0], entities[:, 1], entities[:, 2]

        for b, f, l, cls in zip(batch, first, last, classes):
            tags[b, f] = 2 * cls + 1  # start of entity
            tags[b, f + 1 : l + 1] = 2 * cls + 2  # inside of entity

        return tags

    @torch.no_grad()
    def infer_tags(
        self,
        tokens: torch.Tensor,
        model: Optional[Union[HookedTransformer, PreTrainedModel]],
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """Infer NER tags from representations, WARNING, overwrites nested entities...
        Args:
            reps: tensor (batch, seq, dim) representations of the tokens
        Returns:
            tags: tensor (batch, seq) NER tags for each token in the text
        """
        entities, classes = self.forward(
            tokens, model, attn_mask=attn_mask, return_logits=False
        )
        b, seq = tokens.shape
        # convert entities and classes to tags
        tags = self.tags_from_ents_classes((b, seq), entities, classes)
        # convert tags to numpy
        return tags.cpu()

    def training_step(
        self,
        batch,
        model,
        id2type: Dict[int, str],
        loss_fn: str = "balanced_bce",
        temperature: float = 1.0,
        PL_threshold: float = 0.99,
    ):
        """Train step for the model in the PileNER setting: we train the span embedding to match class embedding in cosine similarity.
        Args:
            batch: batch of data to train on
        """

        PL_threshold = torch.tensor(PL_threshold).logit()
        # convert to logit and get float

        b_size = len(batch["text"])
        entities = batch[
            "entities"
        ]  # list of list of entities for each text in the batch
        inputs = model.tokenizer(
            batch["text"], return_tensors="pt", padding=True, padding_side="right",  truncation=True,
        )
        tokens = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        with torch.no_grad():  # we don't need gradients for the representations
            reps = utils.compute_to_layer(
                model,
                self.layer,
                tokens,
                attn_mask=attention_mask,
                dim=self.model_dim,
                dtype=self.dtype,
            ).cuda()  # shape (batch, seq, dim)

        # predict unclassified entitites, We are doing two LLMs forward pass here ... TODO
        pred_ents = self.ner_model.infer_entities(
            tokens, model
        )  # batch of list of entities

        inds, cls = [], []

        for i in range(b_size):
            for ent in entities[i]:
                inds.append((i, *ent["tok_pos"]))
                cls.append(ent["class"])
            for ent in pred_ents[i]:
                index = (i, *ent)
                if not index in inds:
                    inds.append(index)
                    cls.append(0)  # 0 for no entity

        inds = torch.tensor(inds).cuda()  # shape (#entities, 3)
        cls = np.array(cls)  # shape (#entities)

        span_reps = self.get_span_reps(reps, inds)  # shape (batch, #entities, dim)

        # get unique class ids and map them to indexes
        un_cls = np.unique(cls)
        clsid2ind = {
            c: i for i, c in enumerate(un_cls)
        }  # map class id to index in un_cls

        # get class types str, remove O_CLASS_VAL, all class indexes are shifted by 1
        classes_str = [
            id2type[c] for c in un_cls if c != 0
        ]  # convert class ids to class names

        # TODO why not choose another layer to extract class representations from ?
        class_reps = self.get_class_reps(
            classes_str, model, self.layer
        )  # shape (n_classes +1, dim)
        # print(class_reps.shape, class_reps.dtype, class_reps.device, class_reps)
        # compute cross match
        scores = (
            span_reps @ class_reps.T / temperature
        )  # shape (batch, #entities, n_classes)

        targets = torch.zeros_like(scores)
        pred_classes = scores.argmax(dim=-1)  # shape (batch, #entities)
        for i, c in enumerate(cls):
            # set target to 1 for the class of the entity, including 0 class
            if c != 0:  # labeled data
                targets[i, clsid2ind[c]] = 1
            else:  # unlabeled data
                pseudo_cls = pred_classes[i]  # get the class with highest score
                # if the logit score is higher than threshold, we consider it as a pseudo label
                # leave as 0 otherwise
                if scores[i, pseudo_cls] > PL_threshold:
                    targets[i, pseudo_cls] = 1
                # else:
                #     targets[i, 0] = 1

        if loss_fn.lower() == "balanced_bce":
            # Balanced BCE loss, we use the logits as scores, and the targets are the class indices
            loss = balanced_BCE(
                scores[1:].flatten(),
                targets[1:].flatten(),
            )
            # print(f"loss {loss} - pos_weight {self.pos_weight}")
        elif loss_fn.lower() == "bce":

            loss = F.binary_cross_entropy_with_logits(
                scores[1:].flatten(),
                targets[1:].flatten(),
                pos_weight=self.pos_weight,
            )

        elif loss_fn.lower() == "infonce":
            # InfoNCE loss,

            # We use the logits as scores, and the targets are the class indices
            log_probs = F.log_softmax(scores, dim=-1)
            loss = -log_probs[targets == 1].mean()  # sum log probs fo positive examples
            # loss += F.binary_cross_entropy_with_logits(
            #     scores.flatten(),
            #     targets.flatten(),
            #     reduction="mean",
            #     )
            # loss = sum(CE[pos_inds]) - sum(CE[neg_inds])  # sum over positive and negative samples
        else:
            raise NotImplementedError(
                f"Loss function {loss_fn} not implemented, should be 'bce' or 'infoNCE'"
            )

        # DEBUG
        # print(f"classes_str {classes_str} - un_cls {un_cls} - clsid2ind {clsid2ind}")
        # print(f"- scores shape {scores.shape}\n targets ({targets.shape}) {targets}\n  - cls {cls}\n - inds {inds.shape}\n")
        # raise ValueError("stop")
        return loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: str = "infoNCE",
        val_metric: str = "BCE",
        pos_weight: float = 1.0,
        temperature: float = 1.0,
        PL_threshold: float = 0.99,
        hist: list = [],
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        **kwargs,
    ):
        """Train the model on the given dataset
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            val_metric: metric to use for validation, default "micro" for Micro F1
            hist: history of training
            epochs: number of epochs to train
            lr: learning rate
            grad_clip (float): gradient clipping value, if 0, no clipping
            accumulation_steps: number of steps to accumulate gradients before updating weights
            patience: number of untolerable validations to wait before reducing learning rate
            optimizer: optimizer to use for training
            hist: history of training
            kwargs: additional arguments for the training loop
        """
        # freeze ner model
        for param in self.ner_model.parameters():
            param.requires_grad = False
        
        self.train_zero_shot = True  # use zero-shot classification during training

        logging.info(
            f"Launching Training for NERC model with \
            \n- temperature {temperature} \
            \n- pos_weight {pos_weight} \
            \n- loss_fn {loss_fn} \
            \n- PL_threshold {PL_threshold} \
            "
        )
        self.pos_weight = torch.tensor(pos_weight)

        hist = train(
            self,
            train_loader,
            val_loader,
            optimizer=optimizer,
            val_metric=val_metric,
            train_step_args={
                "model": train_loader.dataset.model,
                "id2type": train_loader.dataset.dataset.id2type,
                "loss_fn": loss_fn,
                "temperature": temperature,
                "PL_threshold": PL_threshold,  # pseudo label threshold
            },
            **kwargs,
        )
        
        del self.train_zero_shot
        del self.pos_weight

        # unfreeze ner model
        for param in self.ner_model.parameters():
            param.requires_grad = True

        return hist

    def sup_training_step(
        self, batch: dict, model: HookedTransformer, loss_fn: str = "CrossEntropy"
    ) -> torch.Tensor:
        """Train step for the model in the supervised FineTunning setting.
        Args:
            batch: batch of data to train on
            model: HookedTransformer form TransformerLens to extract representations from
            loss_fn: loss function to use, either "CrossEntropy" or "InfoNCE"
        """

        def debug_train():
            nonlocal reps, scores, cls, inds, span_reps, loss
            print(
                f"reps {reps.shape} \n \
                 - class_logits {scores.shape} \n \
                 - cls {cls.shape}: {cls} \n \
                 - inds {inds.shape}: {inds} \n \
                 - loss {loss} \n \
                 - span_reps {span_reps.shape} \n \
                 - entities {entities}"
            )

            raise ValueError("stop")

        b_size = len(batch["text"])
        # list of list of entities for each text in the batch
        entities = batch["entities"]

        inputs = model.tokenizer(
            batch["text"], return_tensors="pt", padding=True, padding_side="right"
        )
        tokens = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        
        with torch.no_grad():  # we don't need gradients for the representations
            reps = utils.compute_to_layer(
                model,
                self.layer,
                tokens,
                attn_mask=attention_mask,
                dim=self.model_dim,
                dtype=self.dtype,
            ).cuda()  # shape (batch, seq, dim)

        # predict unclassified entitites, We are doing two LLMs forward pass here ... TODO ?
        # Keep it like that, we extract representations at diff layers anyway
        pred_ents = self.ner_model.infer_entities(
            tokens, model
        )  # batch of list of entities

        inds, cls = [], []

        for i in range(b_size):
            for ent in entities[i]:
                inds.append((i, *ent["tok_pos"]))
                cls.append(ent["class"])
            for ent in pred_ents[i]:
                index = (i, *ent)
                if not index in inds:
                    inds.append(index)
                    cls.append(0)  # 0 for no entity

        inds = torch.tensor(inds).cuda()  # shape (#entities, 3)
        cls = np.array(cls)  # shape (#entities)

        span_reps = self.get_span_reps(reps, inds)  # shape (batch, #entities, dim)

        scores: torch.Tensor = self.classify_sup(
            span_reps, return_logits=True
        )  # shape (#entities, n_classes)

        targets = torch.zeros_like(scores)

        for i, c in enumerate(cls):
            # set target to 1 for the class of the entity, including 0 class
            targets[i, c] = 1

        if loss_fn.lower() == "crossentropy":
            # Cross entropy loss, we use the logits as scores, and the targets are the class indices
            loss = F.cross_entropy(
                scores,
                targets,
                reduction="mean",
            )
            # debug_train()

        elif loss_fn.lower() == "infonce":
            # InfoNCE loss,
            # We use the logits as scores, and the targets are the class indices
            log_probs = F.log_softmax(scores, dim=-1)
            loss = -log_probs[targets == 1].mean()  # sum log probs fo positive examples

        elif loss_fn.lower() == "bce":
            loss = F.binary_cross_entropy_with_logits(
                scores.flatten(),
                targets.flatten(),
                reduction="mean",
            )

        else:
            raise NotImplementedError(
                f"Loss function {loss_fn} not implemented, should be in 'CrossEntropy', 'bce' or 'infoNCE'"
            )

        # DEBUG
        # print(f"classes_str {classes_str} - un_cls {un_cls} - clsid2ind {clsid2ind}")
        # print(f"- scores shape {scores.shape}\n targets ({targets.shape}) {targets}\n  - cls {cls}\n - inds {inds.shape}\n")
        # raise ValueError("stop")
        return loss

    def train_supervised(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: str = "CrossEntropy",
        val_metric: str = "micro",
        hist: list = [],
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        force_reset: bool = False,
        train_class_only: bool = False,
        **kwargs,
    ):
        """Train the model on the given dataset
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            val_metric: metric to use for validation, default "micro" for Micro F1
            hist: history of training
            epochs: number of epochs to train
            lr: learning rate
            grad_clip (float): gradient clipping value, if 0, no clipping
            accumulation_steps: number of steps to accumulate gradients before updating weights
            patience: number of untolerable validations to wait before reducing learning rate
            optimizer: optimizer to use for training
            hist: history of training
            kwargs: additional arguments for the training loop
        """
        if train_class_only:
            # freeze model (and ner model) parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # freeze only ner model parameters
            for param in self.parameters():
                param.requires_grad = True
            for param in self.ner_model.parameters():
                param.requires_grad = False

        # compute initial class representations, this keeps grads to class_embed !
        self.init_on_dataset(
            train_loader.dataset.dataset, force=force_reset, verbose=True
        )

        # unfreeze class representations
        self.class_reps: torch.Tensor = torch.nn.Parameter(self.class_reps.detach())

        logging.info(
            f"Launching Training for NERC model with \
            \n - loss_fn {loss_fn} \
            \n - val_metric {val_metric} \
            \n - optimizer {optimizer} \
            "
        )

        hist = train(
            self,
            train_loader,
            val_loader,
            optimizer=optimizer,
            hist=hist,
            val_metric=val_metric,
            train_step_fn=self.sup_training_step,
            train_step_args={
                "model": train_loader.dataset.model,
                "loss_fn": loss_fn,
            },
            **kwargs,
        )

        # unfreeze
        for param in self.parameters():
            param.requires_grad = True

        return hist

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> float:
        """Validate the model on the given dataset
        Args:
            val_loader: DataLoader for validation data
            verbose: whether to print the results or not
        """
        model = val_loader.dataset.model
        id2type = val_loader.dataset.dataset.id2type
        loss = 0

        for batch in tqdm(val_loader, disable=not verbose):
            loss += self.training_step(batch, model, id2type).item()
        loss /= len(val_loader)

        if verbose:
            logging.info(f"Validation loss: {loss:.4f}")
        return loss

    def init_on_dataset(
        self, dataset: data.NERDataset, force: bool = False, verbose: bool = True
    ) -> None:
        """Initialize the model on a given dataset, setting the id2type and type2id attributes"""

        if verbose:
            logging.info(f"Initializing NERC model on dataset ...")

        if self.class_reps is not None:
            if not force:
                assert hasattr(self, "class_reps"), "Class representations not set ! "
                assert hasattr(self, "id2type"), "id2type not set ! "
                assert hasattr(self, "type2id"), "type2id not set ! "
                if verbose:
                    logging.info("NERC model already initialized on dataset")
                return
            else:
                logging.warning(
                    "Overwriting existring Class embeddings with zero-shot representations"
                )
                del self.class_reps  # remove existing class representations

        self.id2type = dataset.id2type | {self.FN_CLASS_VAL: "FN"}
        self.type2id = {v: k for k, v in self.id2type.items()}

        # str classes, ignoring 0 class !
        str_classes = [
            self.id2type[key]
            for key in self.id2type.keys()
            if key not in [0, self.FN_CLASS_VAL]
        ]

        if len(str_classes) > 1e3:
            if verbose:
                logging.info(
                    "Too many classes in dataset, would have to compute class representation on the fly"
                )
            raise NotImplementedError("Computation on the fly not managed yet")
        else:
            if verbose:
                logging.info(f"Computing representations for classes {str_classes}")
            self.class_reps = self.get_class_reps(
                str_classes, dataset.model, self.layer
            )  # shape (n_classes + 1, dim), including O class at index 0

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader,
        val_metric: str = "micro",
        zero_shot: bool = False,
        decoding_strategy: str = "threshold",
        threshold: float = 0.5,
        confusion: bool = False,
        verbose: bool = True,
    ):
        """Evaluate the model on the test set
        Args:
            eval_loader: test loader
            val_metric: metric to use for evaluation, default "micro" for Micro F1
            zero_shot: if True, will reset the model to zero-shot mode, recomputing class representations from the dataset
            decoding_strategy: strategy to use for entity decoding, see llm2ner.decoders for details
            threshold: threshold to use for entity decoding
            confusion: if True, will compute and return the confusion matrix
            verbose: whether to print the results or not
        returns:
            perfs_by_class: performance by class,
        """
        if val_metric not in ["micro", "macro", "BCE", "all"]:
            logging.warning(f"Unknown metric {val_metric}, default 'BCE'")
            val_metric = "BCE"

        if hasattr(self, "train_zero_shot") and self.train_zero_shot:
            #we are training in zero-shot mode, Simply validate with BCE
            return {"BCE": self.validate(eval_loader, verbose=verbose)}

        model = eval_loader.dataset.model

        self.init_on_dataset(
            eval_loader.dataset.dataset, force=zero_shot, verbose=verbose
        )

        all_pred_labels = {}
        all_true_labels = {}
        perfs_by_class = {
            cls: torch.zeros(3, dtype=torch.int) for cls in self.id2type.keys()
        }
        logging.info(
            f"Evaluating NERC model on dataset using strategy {decoding_strategy} (thr={threshold}) ..."
        )
        for batch in tqdm(eval_loader, disable=not verbose):
            ids = batch["id"]
            inputs = model.tokenizer(
                batch["text"], padding=True, padding_side="right", return_tensors="pt", truncation=True
            )
            tokens = inputs["input_ids"].cuda()
            attention_mask = inputs["attention_mask"].cuda()

            pred_entities, pred_classes = self.forward(
                tokens,
                model,
                attn_mask=attention_mask,
                zero_shot=False,  # we precomputed 0-shot class representations, still zero-shot inference
                decoding_strategy=decoding_strategy,
                threshold=threshold,
                return_logits=False,
            )
            gt_entities = batch["entities"]
            # print(ids, pred_entities, pred_classes, gt_entities)

            for b in range(len(gt_entities)):
                for ent in gt_entities[b]:
                    entity = (ids[b], *ent["tok_pos"])
                    cls = ent["class"]
                    all_true_labels[entity] = cls  # class of the entity
                    perfs_by_class[cls][2] += 1  # total count for class
                    # False Negative, will be updated if predicted
                    all_pred_labels[entity] = self.FN_CLASS_VAL
                    perfs_by_class[self.FN_CLASS_VAL][2] += 1  # total count for class

            for i in range(len(pred_entities)):
                cls = int(pred_classes[i])
                b, beg, end = pred_entities[i]
                entity = (ids[b], int(beg), int(end))
                all_pred_labels[entity] = cls

                if entity not in all_true_labels:
                    # entity not in dataset, should be classified as 0
                    all_true_labels[entity] = 0  # 0 for no entity
                    perfs_by_class[0][2] += 1  # total count for class 0
                else:
                    # entity was found, remove one from FN class count.
                    perfs_by_class[self.FN_CLASS_VAL][2] -= 1

                if all_true_labels[entity] == cls:
                    # True Positive
                    perfs_by_class[cls][0] += 1
                else:
                    # False Positive
                    perfs_by_class[cls][1] += 1

        # convert to lists
        entities = list(all_true_labels.keys())
        pred_classes = [all_pred_labels[ent] for ent in entities]
        gt_classes = [all_true_labels[ent] for ent in entities]
        # entities = np.array(entities)  # convert to numpy array for easier indexing

        # print perfs for each class
        for cls in self.id2type.keys():
            tp, fp, tot = perfs_by_class[cls]
            precision = tp / (tp + fp + EPS)
            recall = tp / (tot + EPS)
            f1 = 2 * precision * recall / (precision + recall + EPS)

            if verbose:
                print(
                    f"{str(cls).ljust(2)} {self.id2type.get(cls, 'none').ljust(25)} - tp: {tp} - fp: {fp} - tot: {tot} - precision: {precision:.2f} - recall: {recall:.2f} - f1: {f1:.2f}"
                )

        # --- Confusion Matrix ---
        if confusion:
            cm = confusion_matrix(
                gt_classes, pred_classes, labels=list(self.id2type.keys())
            )

        if val_metric == "micro" or val_metric == "all":
            # Compute ignore O class
            summed = torch.zeros(3)
            for cls in perfs_by_class.keys():
                if cls in [0, self.FN_CLASS_VAL]:
                    continue  # ignore # O class and -1 class
                else:
                    summed += perfs_by_class[cls]

            tp, fp, tot = summed
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tot + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            if verbose:
                print(f"Micro - tp: {tp} - fp: {fp} - tot: {tot}")
                print(
                    f"Micro - precision: {precision:.3f} - recall: {recall:.3f} - f1: {f1:.3f}"
                )

            if val_metric == "all":
                m = {
                    "micro": f1.item(),
                    "recall": recall.item(),
                    "precision": precision.item(),
                }
            else :
                m = f1.item()

            if confusion:
                return m, cm
            else:
                return m
        elif val_metric == "macro":
            raise NotImplementedError("Macro F1 not implemented")
        else:
            raise ValueError(f"Unknown metric {val_metric}")

class LearnSupervisedNER(Task):
    """From a NER model, learn a representation for the entities for zero shot inference."""

    nerc_model: Param[NERCmodel]
    """NERC model to train"""

    finetune_ner: Param[bool] = False
    """whether to finetune the NER model along with the class embeddings"""

    # Evaluation
    decoding_strategies: Param[List[str]] = ["threshold", "greedy"]
    """decoding strategy to use for entity extraction when evaluating model, see llm2ner.decoders for details"""

    eval_thresholds: Param[List[float]] = [0.5]
    """thresholds to use for entity decoding when evaluating model"""

    # training
    epochs: Param[int] = 4
    batch_size: Param[int] = 16
    optimizer: Param[str] = "AdamW"
    """optimizer to use for training, can be "Adam", "SGD", etc."""
    lr: Param[float] = 1e-3
    grad_clip: Param[float] = 0.0
    """gradient clipping value, if 0, no clipping"""

    accumulation_steps: Param[int] = 1
    """number of steps to accumulate gradients before updating weights"""
    
    patience: Param[int] = 5
    """number of untolerable validations to wait before reducing learning rate"""
    
    early_stopping: Param[bool] = False
    """whether to stop training if no improvement on validation set"""

    min_lr: Param[float] = 1e-5
    """minimum learning rate, if 0, no minimum learning rate"""
    
    n_val: Param[int] = 1000
    """number of validation steps"""

    val_limit: Param[int] = 1000  
    """limit the number of validation samples, if None, no limit"""
    
    val_metric: Param[str] = "BCE"  # metric to use for validation,
    """metric to use for validation, default "micro" for Micro F1"""
    
    w_decay: Param[float] = 1e-5 
    """weight decay for the optimizer, if 0, no weight decay"""

    # data
    dataset_name: Param[str] = "CoNLL 2003"

    max_length: Param[int] = 1200
    
    max_ent_length: Param[int] = 20
    """maximum length of the input sequence, if None, no limit"""

    # Meta params, not used to compute signature
    data_folder: Meta[str] = ""  # Folder where the data is stored, not a parameter
    """Path to the data folder"""

    runpath: Meta[Path] = field(default_factory=PathGenerator("runs"))
    """Path to store tensorboard logs"""
    
    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""
    
    result_path: Meta[Path] = DataPath("Eval.json")
    """Path to store the evaluation results"""

    # Misc
    run: Param[int] = 0
    """Run number, used if we want to run the same task multiple times"""

    version: Constant[str] = "1.0"
    """Version of this task: Can change if code has been updated and need to recompute"""

    def task_outputs(self, dep):
        return dep(
            NERCmodel.Loader.C(model=self.nerc_model, parameters=self.parameters_path)
        )

    def get_partial_optimizer(self):
        # optimizer
        optim_name = self.optimizer.lower()

        if optim_name == "adam":
            return partial(
                torch.optim.Adam,
                lr=self.lr,
                weight_decay=self.w_decay,
            )
        elif optim_name == "sgd":
            return partial(
                torch.optim.SGD,
                momentum=0.9,
                nesterov=True,
                lr=self.lr,
                weight_decay=self.w_decay,
            )
        elif optim_name == "adamw":
            return partial(
                torch.optim.AdamW,
                lr=self.lr,
                weight_decay=self.w_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

    def finetune_ner_model(self, model, train_loader: DataLoader, val_loader: DataLoader):
        """Finetune the NER model on the given dataset"""
        logging.info("Finetuning NER model...")
        # freeze LLM
        for param in model.parameters():
            param.requires_grad = False
        
        nerModel:NERmodel = self.nerc_model.ner_model.to(model.device)

        for param in nerModel.parameters():
            param.requires_grad = True
        
        hist = nerModel.train(
            train_loader,
            val_loader,
            optimizer=torch.optim.AdamW,
            epochs=5,
            lr=1e-3,
            grad_clip=0.5,
            patience=5000,
            min_lr=1e-6,
            n_val=5000,
            val_metric="recall",
            early_stopping=True,  # stop training if no improvement on validation set
        )

        return hist

    def execute(self):
        """Called when this task is run"""

        ### Load model and dataset
        nerc_model = self.nerc_model  # .instance() ?
        logging.info(
            f"Loaded NERC model with {sum(p.numel() for p in nerc_model.parameters())} parameters"
        )

        # Load model
        logging.info(
            f"Loading llm {nerc_model.llm_name} and dataset {self.dataset_name}..."
        )
        model = utils.load_llm(
            nerc_model.llm_name,
            to_hookedtransformer=self.nerc_model.ner_model.need_hookedtransformer,
            cut_to_layer=nerc_model.layer,
        ).eval()

        logging.debug(
            f"Done ! {nerc_model.llm_name} dimension is {nerc_model.model_dim}"
        )

        # Load dataset + Tokenize and compute token-level NER tags
        dataset = data.load_dataset_splits(
            dataset_name=self.dataset_name,
            data_folder=self.data_folder,
            model=model,
            val_limit=self.val_limit,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
        )

        train_dataset, val_dataset, test_dataset = (
            dataset.get("train", None),
            dataset.get("dev", None),
            dataset.get("test", None),
        )

        # Loaders
        logging.info("Building loaders...")
        train_loader = train_dataset.get_loader(batch_size=self.batch_size)
        val_loader = val_dataset.get_loader(batch_size=self.batch_size)

        if self.finetune_ner:
            logging.info("Finetuning NER before training class embeddings...")
            for param in nerc_model.ner_model.parameters():
                param.requires_grad = True
            self.finetune_ner_model(model, train_loader, val_loader)
            logging.info("Done !")

        # freeze LLM and NER model if not finetuning
        for module in [model, nerc_model.ner_model]:
            for param in module.parameters():
                param.requires_grad = False       

        nerc_model.to(model.device)

        hist = nerc_model.train_supervised(
            train_loader,
            val_loader,
            train_class_only=False,
            # training parameters
            optimizer=self.get_partial_optimizer(),
            epochs=self.epochs,
            lr=self.lr,
            grad_clip=self.grad_clip,
            # accumulation_steps=1,
            patience=self.patience,
            min_lr=self.min_lr,
            n_val=self.n_val,
            early_stopping=self.early_stopping,  # stop training if no improvement on validation set
        )

        # save model
        torch.save(nerc_model.state_dict(), self.parameters_path)
        # nerc_model.cpu().save_pretrained("model")
        del train_loader, val_loader, train_dataset, val_dataset

        logging.info(f"Training finished, model saved to {self.parameters_path}")

        metrics = {}
        for strat in self.decoding_strategies:
            for thr in self.eval_thresholds:
                eval_id = f"{strat}_thr{thr}"
                logging.info(f"Evaluating model with strategy {strat} (thr={thr})...")
                m, confusion_m = nerc_model.evaluate(
                    test_dataset.get_loader(batch_size=self.batch_size),
                    decoding_strategy=strat, 
                    threshold=thr,
                    verbose=True, 
                    val_metric="all",
                    confusion=True, 
                    zero_shot=False,
                )
                metrics[eval_id] = m | {"decoding_strategy": strat, "threshold": float(thr)}
                logging.info(f"Test set performance: {m}")
                logging.info(f"Test set confusion matrix:\n {confusion_m}")

                # save confusion matrix
                np.savetxt(
                    f"confusion_matrix_{eval_id}.csv", confusion_m, fmt="%d", delimiter=","
                )

        with open(self.result_path, "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "version": self.version,
                },
                f,
            )
        logging.info(f"Results saved !")

class LearnZeroShotNER(Task):
    """From a NER model, learn a representation for the entities for zero shot inference."""

    nerc_model: Param[NERCmodel]

    # loss
    loss_fn: Param[str] = (
        "balanced_bce"  # loss function to use, can be "balanced_bce", "bce", "infonce"
    )
    """loss function to use, can be "balanced_bce", "bce", "infonce" """
    pos_weight: Param[float] = (
        1.0  # weight for the positive class in the loss function, if 1.0, no weight
    )
    """weight for the positive class in the loss function, if 1.0, no weight"""
    PL_threshold: Param[float] = 0.99  # threshold for pseudo labeling
    """Probability threshold for pseudo labeling, if the logit score is higher than this value, we consider it as a pseudo label"""

    # training
    epochs: Param[int] = 4
    batch_size: Param[int] = 16
    optimizer: Param[str] = "AdamW"
    """optimizer to use for training, can be "Adam", "SGD", etc."""
    lr: Param[float] = 1e-3
    grad_clip: Param[float] = 0.0
    """gradient clipping value, if 0, no clipping"""
    accumulation_steps: Param[int] = 1
    """number of steps to accumulate gradients before updating weights"""
    patience: Param[int] = 5
    """number of untolerable validations to wait before reducing learning rate"""
    min_lr: Param[float] = 1e-5
    """minimum learning rate, if 0, no minimum learning rate"""
    n_val: Param[int] = 1000
    """number of validation steps"""

    val_limit: Param[int] = (
        1000  # limit the number of validation samples, if None, no limit
    )
    val_metric: Param[str] = "BCE"  # metric to use for validation,
    """metric to use for validation, default "micro" for Micro F1"""
    w_decay: Param[float] = (
        1e-5  # weight decay for the optimizer, if 0, no weight decay
    )

    # data
    dataset_name: Param[str] = "Pile-NER"
    max_length: Param[int] = 1200
    max_ent_length: Param[int] = 20
    """maximum length of the input sequence, if None, no limit"""

    # Meta params, not used to compute signature
    data_folder: Meta[str] = ""  # Folder where the data is stored, not a parameter
    """Path to the data folder"""
    runpath: Meta[Path] = field(default_factory=PathGenerator("runs"))
    """Path to store tensorboard logs"""
    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""

    # Misc
    run: Param[int] = 0
    """Run number, used if we want to run the same task multiple times"""
    version: Constant[str] = "1.0"
    """Version of this task: Can change if code has been updated and need to recompute"""

    def task_outputs(self, dep):
        return dep(
            NERCmodel.Loader.C(model=self.nerc_model, parameters=self.parameters_path)
        )

    def execute(self):
        """Called when this task is run"""

        ### Load model and dataset
        nerc_model = self.nerc_model  # .instance() ?
        logging.info(
            f"Loaded NERC model with {sum(p.numel() for p in nerc_model.parameters())} parameters"
        )

        # Load model
        logging.info(
            f"Loading llm {nerc_model.llm_name} and dataset {self.dataset_name}..."
        )
        model = utils.load_llm(
            nerc_model.llm_name,
            to_hookedtransformer=False,
            cut_to_layer=nerc_model.layer,
        ).eval()
        logging.debug(
            f"Done ! {nerc_model.llm_name} dimension is {nerc_model.model_dim}"
        )

        # Load dataset + Tokenize and compute token-level NER tags
        dataset = data.load_dataset_splits(
            dataset_name=self.dataset_name,
            data_folder=self.data_folder,
            model=model,
            val_limit=self.val_limit,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
        )

        train_dataset, val_dataset, test_dataset = (
            dataset.get("train", None),
            dataset.get("dev", None),
            dataset.get("test", None),
        )

        # Loaders
        logging.info("Building loaders...")
        train_loader = train_dataset.get_loader(batch_size=self.batch_size)
        val_loader = val_dataset.get_loader(batch_size=self.batch_size)

        # freeze LLM
        for param in model.parameters():
            param.requires_grad = False

        # optimizer
        if self.optimizer.lower() == "adam":
            optimizer = partial(
                torch.optim.Adam,
                lr=self.lr,
                weight_decay=self.w_decay,
            )
        elif self.optimizer.lower() == "sgd":
            optimizer = partial(
                torch.optim.SGD,
                momentum=0.9,
                nesterov=True,
                lr=self.lr,
                weight_decay=self.w_decay,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = partial(
                torch.optim.AdamW,
                lr=self.lr,
                weight_decay=self.w_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

        hist = nerc_model.train(
            train_loader,
            val_loader,
            optimizer=optimizer,
            lr=self.lr,
            epochs=self.epochs,
            grad_clip=self.grad_clip,
            accumulation_steps=self.accumulation_steps,
            patience=self.patience,
            min_lr=self.min_lr,
            # loss
            pos_weight=self.pos_weight,
            loss_fn=self.loss_fn,
            PL_threshold=self.PL_threshold,
            # validation
            n_val=self.n_val,
            early_stopping=False,
            val_metric=self.val_metric,
        )

        # save model
        torch.save(nerc_model.state_dict(), self.parameters_path)
        # nerc_model.cpu().save_pretrained("model")
        del train_loader, val_loader, train_dataset, val_dataset

        logging.info(f"Training finished, model saved to {self.parameters_path}")

        # save history
        with open(self.parameters_path.with_suffix(".hist"), "w") as f:
            json.dump(hist, f, indent=4)
