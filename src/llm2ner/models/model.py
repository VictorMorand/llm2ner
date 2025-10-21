import os, time, torch, logging, gc, json
from pathlib import Path
from tqdm import tqdm
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
)

# PyTorch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
)

# HF and Tlens
from transformer_lens import HookedTransformer
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel

# xpm and misc
from experimaestro import Param, Task, Meta, Param, Constant, DataPath

# Our code
from llm2ner import utils
from llm2ner.utils import PathOutput
from llm2ner.xpmModel import xpmTorchHubModule
import llm2ner.data as data
import llm2ner.decoders

MAX_ENT_LENGTH = 10
DEFAULT_PATIENCE = (
    3  # number of tokens to wait for sum of attention scores to be maximal in heuristic
)
EPS = 1e-8
# value to fill negative logits in BCE loss, should be low enough to not affect the loss, but high enough to not cause numerical issues
FILL_NEG_LOGITS = -1e6
DEFAULT_KERNEL_PADDING = [2, MAX_ENT_LENGTH + 1, MAX_ENT_LENGTH + 1, 2]
# T = TypeVar("T")


# Combine two logits with a logit intersection
# This is a numerically stable way to combine two logits into a new one modeling the intersection of the two events
# we have Logit(p(x)*p(y)) = logit(p(x)) + logit(p(y)) - log(1 + exp(logit(p(x)) + exp(logit(p(y)))
LogitIntersection = (
    lambda x, y: x + y - torch.logsumexp(torch.stack([torch.ones_like(x), x, y]), dim=0)
)

###### NER Utils


def get_entities_from_tags(tags):
    """Extract entities from NER tags, first token of entity is tagged 1, following tokens are tagged 2, lone 2 are ignored
    Args:
        tags: NER tags
    Returns:
        entities (List (tuples)): list of extracted entities
    """
    entities = []
    entity = None

    for i, tag in enumerate(tags):
        if tag % 2 == 1:  # we start a new entity
            if entity is not None:
                entities.append(entity)
            entity = (i, i)
        elif tag >= 2 and entity is not None:
            entity = (entity[0], i)
        else:  # tag is 0
            if entity is not None:
                entities.append(entity)
                entity = None

    if entity is not None:
        entities.append(entity)
    return entities


def count_perf_tags(tags, target):
    """Compares tags with target and returns number of correct entities and total number of entities
    Args:
        tags: tags from NER_inference
        target: target tags
    Returns:
        true_pos: number of correct entities
        false_pos: number of incorrect entities
        total: total number of entities in target
    """
    inferred = get_entities_from_tags(tags)
    targets = get_entities_from_tags(target)

    return count_perf(inferred, targets)


def count_perf(
    b_entities: Union[List[List[Tuple]], List[Tuple]], b_targets: List[List[Tuple]]
) -> Tuple[int, int, int]:
    """Count number of true pos, false pos and total entities in a batch of inferred entities vs given targets
    Args:
        entities: List OR list of list of extracted entities as tuples (first, last) token
        targets: List OR list of list of target entities as tuples (first, last) token
    Returns:
        true_pos: number of correct entities
        false_pos: number of incorrect entities
        total: total number of entities in target
    """

    if not len(b_targets):
        return 0, len(b_entities), 0
    if len(b_entities) == 0:
        return 0, 0, len(b_targets)

    if type(b_entities[0]) == tuple:
        b_entities = [b_entities]
    if type(b_targets[0]) == tuple:
        b_targets = [b_targets]

    assert (
        type(b_entities) == list and type(b_targets) == list
    ), f"entities and targets should be lists, not {type(b_entities)} and {type(b_targets)}"

    total, true_pos, false_pos = 0, 0, 0

    for entities, targets in zip(b_entities, b_targets):
        total += len(targets)
        for entity in entities:
            if entity in targets:
                true_pos += 1
            else:
                false_pos += 1

    return true_pos, false_pos, total


def get_str_entities(tags, str_tokens):
    """Extract entities from NER tags and return them as strings
    Args:
        tags: NER tags
        text: text to extract entities from
    Returns:
        entities (List (str)): list of extracted entities
    """
    entities = get_entities_from_tags(tags)
    str_entities = []
    for start, end in entities:
        str_entities.append("".join(str_tokens[start : end + 1]))
    return str_entities


def flat_tags_from_probs(span_probs, threshold=0.5, strategy="line_col_greedy"):
    """Get NER tags from span pro
    Args:
        span_probs: span probabilities
        threshold: threshold to use for NER tags, default 0.5
    Returns:
        tags: NER tags for each token in the text
    """
    batch, seq, _ = span_probs.shape
    tags = torch.zeros((batch, seq), dtype=torch.int)

    if strategy == "line_longest":
        max_lines = torch.argmax(span_probs, dim=1)
        # erase all other values except the max
        span_probs[
            torch.zeros_like(span_probs).scatter_(1, max_lines.unsqueeze(1), 1) == 0
        ] = 0
    else:
        logging.warning(f"Strategy {strategy} unknown, using default: line_col_greedy")
        max_lines = torch.argmax(span_probs, dim=1)
        # erase all other values except the max
        span_probs[
            torch.zeros_like(span_probs).scatter_(1, max_lines.unsqueeze(1), 1) == 0
        ] = 0
        max_cols = torch.argmax(span_probs, dim=0)
        span_probs[
            torch.zeros_like(span_probs).scatter_(0, max_cols.unsqueeze(0), 1) == 0
        ] = 0

    batch, last, first = torch.where(span_probs > threshold)
    idx_end = torch.argsort(first, descending=True)
    batch, last, first = batch[idx_end], last[idx_end], first[idx_end]

    for b, f, l in zip(batch, first, last):
        tags[b, f] = 1
        tags[b, f + 1 : l + 1] = 2

    return tags


######################  TRAINING ######################


# only training logic
def train(
    module: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: type[torch.optim.Optimizer],
    val_metric: str = "bce",
    hist: list = [],
    epochs: int = 2,
    lr: float = 1e-3,
    grad_clip: float = 0.0,
    accumulation_steps: int = 1,
    w_decay: float = 0.0,
    patience: int = 3,
    min_lr: float = 1e-5,
    early_stopping: bool = True,
    n_val: int = 100,
    train_step_fn: Optional[Callable] = None,
    train_step_args: dict = {},
    evaluate_args: dict = {},
):
    """Training loop for given module on the given loaders, implements all specific logics for training
    Module must have a training_step and init_training method.

    Args:
        module: module to train
        model: HookedTransformer form TransformerLens to extract representations from
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        val_metric: metric to use for validation, default "bce" that will be passed to module.evaluate
        end_classifier: (Module, optionnal) binary classifier to train on top of the attention scores
        hist: history of training
        epochs: number of epochs to train
        lr: learning rate
        grad_clip (float): gradient clipping value, if 0, no clipping
        accumulation_steps: number of steps to accumulate gradients before updating weights
        patience: number of untolerable validations to wait before reducing learning rate
        min_lr: minimum learning rate, if reached, training stops
        n_val: number of steps between validation and logging
        dilate_entities: Compute the loss only on entities (dilated) and not the full sequence
    Returns:
        hist: updated history of training
    """
    writer = SummaryWriter()  # will write to ./runs/ folder by default

    #log working dir to tensorboard
    writer.add_text("workdir", os.getcwd(), 0)
    
    batch_size = (
        train_loader.dataset.batch_size
    )  # batch size is stored in the dataset, loader batch size is 1
    n_val = n_val // batch_size
    optimizer = optimizer(module.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        # mode='min' if val_metric == "bce" else 'max',
        factor=0.5,
        patience=patience,
        min_lr=min_lr,
    )
    best_weights = module.state_dict()
    best_val = torch.inf

    if train_step_fn is None:
        assert hasattr(
            module, "training_step"
        ), "Module must have a training_step method or train_step_fn must be provided"
        train_step_fn: Callable = module.training_step  # type: ignore

    # move model to device if available
    if torch.cuda.is_available():
        use_cuda = True
        module.cuda()
    else:
        use_cuda = False
        module.cpu()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    beg_time = time.time()
    # get the dtype of the module

    if len(hist):
        prev_epoch, prev_samples = hist[-1]["epoch"], hist[-1]["samples"]
    else:
        prev_epoch, prev_samples = 0, 0


    i = 0
    loss = torch.tensor(0, dtype=torch.float32).to("cuda" if use_cuda else "cpu")
    for epoch in range(prev_epoch + 1, prev_epoch + epochs + 1):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

            i += 1
            step = i * batch_size + prev_samples

            try:
                loss += train_step_fn(batch, **train_step_args)

            except Exception as e:
                logging.error(
                    f"Error in training step {i}, batch is {batch['id']} {batch['text']}: {e}"
                )
                raise e

            if i % accumulation_steps == 0:
                
                # final loss and backward pass
                loss /= accumulation_steps
                loss.backward()

                if grad_clip > 0:
                    total_norm = clip_grad_norm_(module.parameters(), grad_clip)
                    writer.add_scalar("train/GradNorm", total_norm, step)
                
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step(loss.item())
                hist.append(
                    {
                        "epoch": epoch,
                        "samples": step,
                        "loss": loss.item() * batch_size,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                # write to tensorboard as well
                writer.add_scalar("train/Loss", loss.item() * batch_size, step)
                writer.add_scalar("train/LR", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("train/Epoch", epoch, step)

                # reset loss
                loss = 0

            # Validation and logging
            if (i * batch_size) % n_val == 0:
                metrics = module.evaluate(val_loader, val_metric="all", verbose=False)
                val = metrics.get(val_metric, None)
                if val is None:
                    logging.error(
                        f"Validation metric {val_metric} not found in metrics {metrics}, available metrics are {list(metrics.keys())}"
                    )
                    raise ValueError(f"Validation metric {val_metric} not found")

                if val < best_val:
                    best_val = val
                    best_weights = module.state_dict()

                logging.info(
                    f"\nSample {hist[-1]['samples']} mean loss: {hist[-1]['loss']:.4f}, val {val_metric}: {val:.3f}, lr: {optimizer.param_groups[0]['lr']:.2e}"
                )
                for k, v in metrics.items():
                    logging.info(f"  {k}: {v:.3f}")
                    writer.add_scalar(f"Val/{k}", v, step)
                hist[-1]["val_metric"] = val

            if optimizer.param_groups[0]["lr"] <= min_lr and early_stopping:
                break
        if optimizer.param_groups[0]["lr"] <= min_lr and early_stopping:
            logging.info(f"Minimum learning rate reached, stopping training")
            break

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = 0
    gc.collect()
    end_time = time.time()
    hr_time = time.strftime(
        "%H:%M:%S", time.gmtime(end_time - beg_time)
    )  # format time in hours:minutes:seconds
    logging.info(
        f"Training Done ! Peak memory usage: {peak_mem / 1024**3:.2f} GB, duration: {hr_time}"
    )
    # module.load_state_dict(best_weights)
    return hist


######################  MODEL INTERFACE ######################
CARD_TEMPLATE = """
# {model_card_template}
A NER model trained on top of LLM representations.
This model was trained on top of {llm_name} representations at layer {layer}.
It outputs span probabilities for each token in the input text.

For more details, see the [paper]({paper_url}) and the [repo]({repo_url}).

usage
First install the package: 
```bash
pip install git+{repo_url}.git
```

```python 
from llm2ner import NERmodel

NERmodel.from_pretrained("{model_name}")
```
"""


class NERmodel(
    xpmTorchHubModule,
    # library_name="my-org/my-model",
    model_card_template=CARD_TEMPLATE,
    tags=["torch", "experimaestro"],
    repo_url="https://github.com/VictorMorand/test-model",
    paper_url="https://arxiv.org/abs/???",
):
    """Main interface for NER models, should be subclassed
    A NER model is a custom pytorch model that processes given LLM representations to predict entities span probabilities.
    """

    llm_name: Param[str]
    """name of the LLM that produced the representations this model is trained on"""

    layer: Param[int]
    """max layer at which to extract the representations, avoid to compute all LLM layers"""

    dim: int = None
    """dimension of the LLM representations, initialized in __post_init__"""

    need_hookedtransformer: bool = False
    """whether the model needs the llm loaded as a HookedTransformer to compute representations"""

    def __post_init__(self):
        super().__post_init__()
        if self.dim is None:
            self.dim = self.get_llm_dim()
            logging.info(f"Found hidden dimension {self.dim} for {self.llm_name}")

    def extra_repr(self):
        return f"llm_name={self.llm_name}, layer={self.layer}"

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

    def get_llm_dim(self) -> int:
        """Get the dimension of the LLM representations if model is known
        Returns:
            dim: dimension of the LLM representations
        """
        try:
            return utils.get_model_dim(self.llm_name)
        except ValueError as e:
            logging.error(
                f"Error getting LLM config for {self.llm_name}, try to set dim manually"
            )
            raise e

    def get_representations(
        self,
        tokens: torch.Tensor,
        model: HookedTransformer,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch seq dim"]:
        """Get representations for given tokens at the specified layer
        Args:
            tokens: tokens to get representations for
            model: hooked transformer model to extract representations from
        Returns:
            reps: tensor (batch, seq, dim) representations for each token in the text
        """
        assert isinstance(tokens, torch.Tensor), "tokens should be a tensor"
        assert self.layer is not None, "model layer should be set"
        return utils.compute_to_layer(
            model, self.layer, tokens=tokens, attn_mask=attn_mask, dim=self.dim
        )

    # forward version taking tokens and model
    @overload
    def forward(
        self,
        inputs: Float[torch.Tensor, "batch seq"],
        model: Union[HookedTransformer, PreTrainedModel],
        attn_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq seq"]: ...

    # forward version taking representations directly
    @overload
    def forward(
        self,
        inputs: Float[torch.Tensor, "batch seq dim"],
        return_logits: bool = False,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq seq"]: ...

    def forward(
        self,
        input: torch.Tensor,
        model: HookedTransformer = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs: Any,
    ) -> Float[torch.Tensor, "batch seq seq"]:
        """Compute span probabilities or logits for given tokized text,
        Implemented in the child class
        Args:
            input: tensor (batch, seq) batch of either:
                - token IDs, in this case we need the model to compute embeddings
                - token embeddings
            model: hooked transformer model to extract representations from

        Returns:
            scores: tensor (batch, seq, seq) scores for each span of the text
        """
        raise NotImplementedError("forward should be implemented in the child class")

    @torch.no_grad()
    def infer_tags(
        self,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        attn_mask: Optional[torch.Tensor] = None,
        decoding_strategy: str = None,
        threshold: float = 0.5,
    ) -> Int[torch.Tensor, "batch seq"]:
        """Infer NER tags from representations,
        /!\ overwrites nested entities...
        /!\  Assuming model.forward() outputs span probabilities

        Args:
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
        Returns:
            tags: tensor (batch, seq) NER tags for each token in the text
        """
        return torch.hstack(
            [
                self.get_tags(ents, tokens.size(1))
                for ents in self.infer_entities(
                    tokens,
                    model,
                    attn_mask=attn_mask,
                    decoding_strategy=decoding_strategy,
                    threshold=threshold,
                )
            ]
        )

    @torch.no_grad()
    def get_tags(
        self,
        entities: List[Tuple[int, int]],
        seq_len: int,
    ) -> Int[torch.Tensor, "seq"]:
        """Get NER tags from entities, overwrites nested entities...
        Please ensure entities does not overlap.
        Args:
            entities: list of entities as (start, end) tuples
            seq_len: length of the sequence
        Returns:
            tags: tensor (seq) NER tags for each token in the text
        """
        tags = torch.zeros((seq_len), dtype=torch.int)
        entities = sorted(entities, key=lambda x: x[0])
        # sort
        for f, l in entities:
            tags[f] = 1
            tags[f + 1 : l + 1] = 2

        return tags

    @torch.no_grad()
    def infer_entities(
        self,
        tokens: Float[torch.Tensor, "batch seq"],
        model: HookedTransformer,
        attn_mask: Optional[torch.Tensor] = None,
        decoding_strategy: str = "threshold",
        max_ent_length=MAX_ENT_LENGTH,
        threshold=0.5,
        return_probs: bool = False,
    ) -> List[List[Tuple[int, int]]]:
        """Default implementations to Infer entities from representations.
        /!\ Assuming model.forward() outputs span probabilities
        This method is called by the evaluate method to get the entities from the representations
        Args:
            tokens: tensor (batch, seq) of tokens
            model: HookedTransformer form TransformerLens to extract representations from
            max_ent_length: maximum length of entities
            threshold: threshold for span probabilities
        Returns:
            entities: list of entities as (start, end) tuples for each text in the batch
        """
        b_size, _ = tokens.size()
        spans_probs = self.forward(
            tokens, model, attn_mask=attn_mask
        ).detach()  # shape (batch, seq, seq)

        if decoding_strategy == "threshold" or decoding_strategy is None:
            logging.debug(f"Decoding Using threshold {threshold}")
            batch, last, first = torch.where(spans_probs > threshold)
            entities = [[] for _ in range(b_size)]
            for b, f, l in zip(
                batch.cpu().numpy(), first.cpu().numpy(), last.cpu().numpy()
            ):
                entities[b].append((f, l))
        else:
            logging.debug(f"Using strategy {decoding_strategy}")
            entities = []
            for b in range(b_size):
                spans = llm2ner.decoders.decode(
                    spans_probs[b:b+1],  # keep batch dim
                    decoding_strategy=decoding_strategy,
                    threshold=threshold,
                )
                entities.append([(s.start, s.end) for s in spans])
            
        if return_probs:
            return entities, spans_probs
        else:
            return entities

    def get_ckpt_name(self):
        return f"{type(self).__name__}_{self.llm_name}_{self.layer}_{self.mode}"

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        decoding_strategy: str = None,
        threshold: float = 0.5,
        val_metric: str = "all",
        inference_path: Optional[Path] = None,
        verbose: bool = True,
    ):
        """Evaluate attention model on the given dataset, computes F1, precision and recall
        Args:
            model: HookedTransformer form TransformerLens to extract representations from
            eval_loader: DataLoader for evaluation data
            decoding_strategy: the stategy used to contraint NER prediction, for instance when considering only flat ner tags.
            threshold: threshold to use for strategies requiring one
            val_metric (deprecated) wether to export only one metric, default send prec, recall and f1 as dict.
            inference_path: path to save inferences as data.InferredDataset, if None, do not save
            verbose: whether to print progress and results
        """
        model: HookedTransformer = eval_loader.dataset.model

        # special loader for batch size 1, batching is done in the batched dataset wrapper
        batch_size = (
            eval_loader.dataset.batch_size
        )  # batch size is stored in the dataset, loader batch size is 1
        if verbose:
            logging.info(
                f"Evaluating NER model with decoding strategy {decoding_strategy} (thr {threshold}), batch size {batch_size}... "
            )
        true_pos = 0
        false_pos = 0
        total = 0
        total_spans = 0
        if inference_path is not None:
            if not isinstance(inference_path, Path):
                inference_path = Path(inference_path)
            inference_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Will save inferences to {inference_path}...")
            
            inferred_data = data.InferredDataset(
                data_name=inference_path.stem, 
                decoding_strategy=decoding_strategy,
                threshold=threshold,
                data_folder=inference_path.parent,
            )
        for batch in tqdm(eval_loader, disable=not verbose, desc="Evaluation"):
            texts = batch["text"]
            b_target_entities = [
                [ent["tok_pos"] for ent in entities] for entities in batch["entities"]
            ]  # list of list of Tuples (start, end) for each entity

            inputs = model.tokenizer(
                texts, padding=True, padding_side="right", return_tensors="pt", return_offsets_mapping=True, truncation=True,
            )
            tokens = inputs["input_ids"].cuda()
            offsets = inputs.pop("offset_mapping")
            attn_mask = inputs["attention_mask"].cuda()

            # get NER tags
            b_pred_entities, b_span_probs = self.infer_entities(
                tokens,
                model,
                attn_mask=attn_mask,
                decoding_strategy=decoding_strategy,
                threshold=threshold,
                return_probs=True,
            )  # batch of list of entities

            if inference_path is not None:
                inferred_data.samples += (
                    data.InferredDataset.get_samples_from_outputs(
                        batch, b_pred_entities, b_span_probs, offsets
                    )
                )
            # use count_perf to compute metrics
            tp, fp, tot = count_perf(b_pred_entities, b_target_entities)
            true_pos += tp
            false_pos += fp     
            total += tot    # total number of labeled entities
            n_tok = tokens.size(1)
            total_spans += (b_span_probs > 0).sum().item()  # total number of predicted spans

        # compute metrics
        precision = true_pos / (true_pos + false_pos + EPS)
        recall = true_pos / (total + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)

        metrics = {
            "precision": precision, 
            "recall": recall, 
            "f1": f1,
            "true_pos": true_pos,
            "false_pos": false_pos,
            "total": total,
            "total_spans": total_spans,
        }

        if inference_path is not None:
            inferred_data.to_json(inference_path)
            logging.info(f"Saved inferences to {inference_path}")
            
        if verbose:
            logging.info(
                f"Precision: {precision:.2f} % Recall: {recall:.2f} % F1: {f1:.2f} %"
            )

        if val_metric == "all":
            return metrics
        else:
            return metrics[val_metric]
    
    @torch.no_grad()
    def save_inference(self,
        model,
        save_path,
        dataset,
        data_name,
        data_path,
        decoding_strategy="threshold",
        threshold=0.50,
        batch_size=32,
    ):
        data = dataset["test"]
        loader = data.get_loader(batch_size=batch_size)
        inferred_data = data.InferredDataset(data_name, data_folder=data_path)
        logging.info(f"Running inference on {data_name} and saving to {save_path}...")

        for batch in tqdm(loader):
            inputs = model.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
            )
            offsets = inputs.pop("offset_mapping")
            tokens = inputs["input_ids"].cuda()
            attn_mask = inputs["attention_mask"].cuda()

            b_entities, b_span_probs = self.infer_entities(
                tokens,
                model,
                attn_mask=attn_mask,
                decoding_strategy=decoding_strategy,
                threshold=threshold,
                return_probs=True,
            )

            inferred_data.samples.append(
                data.InferredDataset.get_sample_from_outputs(
                    batch, b_entities, b_span_probs, offsets
                )
            )

        inferred_data.to_json(save_path)
        logging.info(f"Saved inference on {data_name} to {save_path}")

    class Eval(Task):
        """Generic Evaluation task for any NER model"""

        ner_model: Param["NERmodel"]
        """Model to evaluate"""

        eval_datasets: Param[List]
        """List of datasets to evaluate on"""

        decoding_strategy: Param[str] = "threshold"
        """Decoding strategy to use, default 'threshold'"""

        threshold: Param[float] = 0.5
        """Threshold to use for decoding strategy, default 0.5"""

        write_inferences: Param[bool] = False
        """Whether to write inferences to file, default False"""

        data_folder: Meta[PathOutput]
        """Path to the data folder"""

        version: Constant[str] = "1.4"
        """Version of the task, 1.3: adds max_length, max_ent_length 100->400"""

        result_path: Meta[Path] = DataPath("Eval.json")

        @torch.no_grad()
        def execute(self):
            """Called when this task is run"""

            ner_model = self.ner_model.cuda()
            logging.info(f"Loaded model {ner_model}")

            llm_name = ner_model.llm_name
            
            # Load model
            logging.info(f"Loading llm {llm_name}...")
            model = utils.load_llm(
                llm_name,
                to_hookedtransformer=ner_model.need_hookedtransformer,
                cut_to_layer=ner_model.layer,
                ).eval()

            c_length = utils.get_model_max_length(llm_name)

            logging.info(f"Evaluating model on datasets {self.eval_datasets}")
            metrics = {}

            data_path = self.data_folder.path
            assert data_path.exists(), f"Data folder {data_path} does not exist"

                
            # Load datasets
            for dataset_name in self.eval_datasets:
                logging.info(f"Loading {dataset_name}")
                eval_dataset = data.load_all_splits(
                    dataset_name=dataset_name,
                    model=model,
                    data_folder=data_path,
                    mode="last",
                    max_ent_length=400,
                    max_length=c_length,
                    verbose=True,
                )

                if self.write_inferences:
                    inference_path = f"inferences_{dataset_name}_{self.decoding_strategy}_thr{self.threshold}.json"
                else:
                    inference_path = None

                metrics[dataset_name] = ner_model.evaluate(
                    eval_dataset.get_loader(batch_size=30), 
                    decoding_strategy=self.decoding_strategy,
                    threshold=self.threshold,
                    inference_path=inference_path,
                    verbose=True
                )
                metrics[dataset_name]["n_samples"] = len(eval_dataset)
                logging.info(f" --> Metrics on {dataset_name}: {metrics[dataset_name]}")

            # Save results in json file
            logging.info("Saving results...")

            with open(self.result_path, "w") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "version": self.version,
                    },
                    f,
                )
            logging.info(f"Results saved !")
