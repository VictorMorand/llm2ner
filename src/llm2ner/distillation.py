from typing import Optional, Dict, Any
from functools import partial
import logging, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer

from llm2ner.models import NERmodel, train
from llm2ner.losses import balanced_BCE


class DistillLoss:

    def __init__(
        self,
        teacher_thr_prob: float = 0.9,
        temperature: float = 1.0,
        sparse: bool = True,
    ) -> torch.Tensor:
        """
        Distillation loss.

        Args:
            student_span_logits (torch.Tensor): The span probabilities from the student model.
            teacher_span_logits (torch.Tensor): The span probabilities from the teacher model.
            gt_spans (torch.Tensor): The ground truth spans. (likely with false negatives)
            batch_mask (torch.Tensor): The mask for the batch.
            teacher_thr_prob (float): Threshold for teacher probabilities to consider a span as positive.
            temperature (float): Temperature for distillation.
            sparse (bool): Whether to use sparse loss computation.
        """
        self.teacher_thr_prob = teacher_thr_prob
        self.temperature = temperature
        self.sparse = sparse

    def __call__(
        self,
        student_span_logits: torch.Tensor,
        teacher_span_logits: torch.Tensor,
        gt_spans: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the distillation loss.
        Args:
            student_span_logits (torch.Tensor): The span probabilities from the student model.
            teacher_span_logits (torch.Tensor): The span probabilities from the teacher model.
            gt_spans (torch.Tensor): The ground truth spans. (likely with false negatives)
            batch_mask (torch.Tensor): The mask for the batch.
        Returns:
            torch.Tensor: The computed loss.
        """

        def debug_print():
            nonlocal student_span_logits, teacher_span_logits, gt_spans, gt_mask, batch_mask
            import matplotlib.pyplot as plt

            # fill with nans
            student_span_logits = student_span_logits.masked_fill(
                ~batch_mask, torch.nan
            ).detach()
            teacher_span_logits = teacher_span_logits.masked_fill(
                ~batch_mask, torch.nan
            ).detach()
            gt_spans = gt_spans.masked_fill(~batch_mask, torch.nan).detach()
            print(
                "\n - batch_mask",
                batch_mask.shape,
                "\n - student_span_logits",
                student_span_logits.shape,
                "\n - teacher_span_logits",
                teacher_span_logits.shape,
                "\n - gt_spans",
                gt_spans.shape,
            )
            n = 3
            b_size = gt_spans.shape[0]
            fig, ax = plt.subplots(4, n, figsize=(13, 20))
            for i in range(n):
                ind = torch.randint(b_size, (1,)).item()
                im0 = ax[0, i].imshow(gt_spans[ind, :, :].cpu())
                ax[0, i].set_title(f"gt_spans {ind}")
                plt.colorbar(im0, ax=ax[0, i], fraction=0.046, pad=0.04)
                im1 = ax[1, i].imshow(student_span_logits[ind, :, :].cpu())
                ax[1, i].set_title(f"student_span_logits {ind}")
                plt.colorbar(im1, ax=ax[1, i], fraction=0.046, pad=0.04)
                im2 = ax[2, i].imshow(teacher_span_logits[ind, :, :].cpu())
                ax[2, i].set_title(f"target probs {ind}")
                plt.colorbar(im2, ax=ax[2, i], fraction=0.046, pad=0.04)
                im3 = ax[3, i].imshow(batch_mask[ind, :, :].cpu())
                ax[3, i].set_title(f"batch_mask {ind}")
                plt.colorbar(im3, ax=ax[3, i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
            raise ValueError("stop")

        gt_mask = gt_spans.bool()  # mask for ground truth spans
        teacher_span_logits[gt_mask] = (
            1e4  # force teacher logits to be high for ground truth spans
        )
        # target[gt_mask] = 1 # force teacher probs are 1 for the ground truth spans

        # We use sigmoid on the teacher logits to get probabilities
        b_target = F.sigmoid(teacher_span_logits[batch_mask] / self.temperature)
        # print(f"b_target shape: {b_target.shape}, batch_mask shape: {batch_mask.shape}")

        if self.sparse:
            # and then compute BCE with logits on the student logits
            b_target = (b_target > self.teacher_thr_prob).float()
            # print(f"b_target shape: {b_target}, batch_mask shape: {batch_mask.shape}")
            loss = balanced_BCE(
                student_span_logits[batch_mask] / self.temperature,
                b_target,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                input=student_span_logits[batch_mask] / self.temperature,
                target=F.sigmoid(teacher_span_logits[batch_mask] / self.temperature),
                reduction="mean",
            )

        # Compute negative log likelihood for the ground truth spans (= BCE loss on positive labels only )
        # loss +=  - F.logsigmoid(student_span_logits[gt_mask] / temperature).mean() * pos_weight

        # print(F.logsigmoid(student_span_logits[gt_mask] / temperature))
        # debug_print()

        # return neg_softBCE
        return loss


def distill_step(
    batch: dict,
    student: NERmodel,
    teacher: NERmodel,
    model: HookedTransformer,
    loss_fn: DistillLoss = DistillLoss(),
):
    """
    Perform a single training step with distillation.
    Args:
        model (NERmodel): The student model.
        teacher (NERmodel): The teacher model.
        batch (dict): The batch of data.
        optimizer (torch.optim.Optimizer): The optimizer.
        temperature (float): Temperature for distillation.
        pos_weight (float): Positive weight for the loss.
    """
    inputs = model.tokenizer(
        batch["text"],
        padding=True,
        padding_side="right",
        return_tensors="pt",
        truncation=True,
    )
    tokens = inputs["input_ids"].cuda()  # shape (batch, seq)
    attn_mask = inputs["attention_mask"].cuda()

    gt_spans = batch[
        "pattern"
    ].cuda()  # data must be in 'last' mode <-> patterns are span labels

    # Forward pass through the teacher model
    with torch.no_grad():
        # make it a leaf tensor to avoid tracking gradients
        teacher_span_logits = teacher(tokens, model, attn_mask=attn_mask, return_logits=True).detach()

    # Forward pass through the student model
    student_span_logits, b_mask = student.forward(
        tokens, model, attn_mask=attn_mask, return_logits=True, return_mask=True
    )

    # Compute the loss
    return loss_fn(
        student_span_logits,
        teacher_span_logits,
        gt_spans,
        b_mask,
    )


def train_distill(
    student: NERmodel,
    teacher: NERmodel,
    model: HookedTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher_thr_prob: float = 0.9,
    temperature: float = 1.0,
    sparse: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    val_metric: str = "recall",
    hist: Optional[list] = None,
    **kwargs,
):
    """
    Train the Token Matching NER model with distillation.
    Args:
        student (NERmodel): The student model to be trained.
        teacher (NERmodel): The teacher model to distill from.
        model (HookedTransformer): The base transformer model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to use, defaults to Adam.
        **kwargs: Additional arguments for training, such as epochs, hist, etc.
    """
    if optimizer is None:
        optimizer = partial(
            torch.optim.AdamW,
            weight_decay=1e-4,  # default weight decay
        )
    if hist is None:
        hist = []
    # change mode
    data_mode = "last"
    for loader in [train_loader, val_loader]:
        if loader.dataset.mode != data_mode:
            # if the data_mode is not the same, we need to change it
            logging.info(
                f"Current dataset data_mode is {loader.dataset.mode}, changing to {data_mode}"
            )
            loader.dataset.mode = data_mode

    for param in student.parameters():
        param.requires_grad = True

    for param in teacher.parameters():
        param.requires_grad = False

    logging.info(
        f"Training Student model {student}\n with distillation from Teacher model {teacher} with \n\
            - data_mode: {data_mode}\n\
            - val_metric: {val_metric}\n\
            - optimizer: {optimizer}\n\
            - teacher_thr_prob: {teacher_thr_prob}\n\
            - temperature: {temperature}\n\
            - sparse BCE: {sparse}\n\
            "
    )

    return train(
        student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        hist=hist,
        train_step_fn=partial(
            distill_step,
            student=student,
            teacher=teacher,
            model=model,
            loss_fn=DistillLoss(
                teacher_thr_prob=teacher_thr_prob,
                temperature=temperature,
                sparse=sparse,
            ),
        ),
        val_metric=val_metric,
        **kwargs,
    )
