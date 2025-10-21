import torch
import torch.nn.functional as F


@torch.jit.script
def balanced_BCE(input, target):
    """
    Balanced Binary Cross Entropy (BBCE) loss.

    Each _batch_ is re-weighted so that positive and negative samples 
    contribute equally, regardless of imbalance.

    Args:
        input (torch.Tensor): The input logits.
        target (torch.Tensor): The target *binary* labels (same shape as input).
    Returns:
        torch.Tensor: The computed loss.
    """

    # Avoid division by zero
    pos = target.sum()
    neg = target.numel() - pos
    if pos * neg == 0:
        pos_weight = torch.tensor(1.0, device=input.device, dtype=input.dtype)
    else:
        pos_weight = neg / pos

    loss = F.binary_cross_entropy_with_logits(
        input=input, target=target, pos_weight=pos_weight, reduction="mean"
    )
    return loss

