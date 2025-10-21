from functools import lru_cache, partial
import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float
from typing import Callable, List, Optional, Tuple, Union

#### Filters
def sliding_window_causal_mask(sliding_window, mask_bos, b , h,  q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    windowed_mask = (
        q_idx - kv_idx <= sliding_window
    )  # We dont need to check the right side of the sliding window since we are applying the causal mask
    if mask_bos:
        return (causal_mask & windowed_mask) & (kv_idx != 0)
    else: 
        return causal_mask & windowed_mask | (kv_idx == 0)

def sliding_window_mask(sliding_window, mask_bos, b, h, q_idx, kv_idx):
    mask = torch.abs(q_idx - kv_idx) <= sliding_window
    if mask_bos:
        return mask & (kv_idx != 0)
    else:
        return mask | (kv_idx == 0)

def causal_mask(mask_bos, b, h, q_idx, kv_idx):
    if not mask_bos:
        return q_idx >= kv_idx
    else:
        return (q_idx >= kv_idx) & (kv_idx != 0)


### Helpers, code extracted from torch.nn.attention.flex_attention
# Need to define it here so that Dynamo doesn't skip it
def _vmap_for_bhqkv(
    fn: Callable,
    prefix: Tuple[Optional[int], ...],
    suffix: Tuple[Optional[int], ...] = (),
    out_dims: Union[int, List[Optional[int]]] = 0,
    group_dim: bool = False,
):
    """Used to vmap both score_mods and mask_mods over 4-dimensional/5-dimension inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (callable): The function to vmap.
        prefix (tuple): The prefix of the vmap. For score mod functions,
                        this should be set to (0,). For mask_mods = ()
        suffix (tuple): We need to add (0,) if gradOut is being mapped over,
                        and (None,) * len(other_buffers).
        out_dims (tuple): For forward cases, keep this as the default 0 since
                          we are only returning 1 output. For backwards, the joint
                          graph returns grads for B, H, Q_idx, KV_idx and other_buffers,
                          so we set this to (0, None, None, None, None) + (None,) * len(other_buffers).

    Returns:
        callable: The vmapped function.
    """
    # We vamp a function 4 times, broadcasting the [b, h, q_idx, kv_idx] dimensions
    dimensions: List[Tuple[None | int, None | int, None | int, None | int]] = []
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
    ]

    if group_dim:
        dimensions += [
            (None, 0, None, None),
        ]

    dimensions += [
        (0, None, None, None),
    ]

    for dims in dimensions:
        fn = torch.vmap(fn, in_dims=prefix + dims + suffix, out_dims=out_dims)
    return fn


def create_mask(
    mod_fn: Callable,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
) -> Tensor:
    r"""This function creates a mask tensor from a mod_fn function.

    Args:
        mod_fn (Union[_score_mod_signature, _mask_mod_signature]): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    if B is None:
        B = 1
    if H is None:
        H = 1
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)
    
    mask_mod = _vmap_for_bhqkv(mod_fn, prefix=())
    mask = mask_mod(b, h, m, n)
    return mask
        

#### Main Mask generator
@lru_cache #use cache to store the mask, so that it is not recomputed for each batch with the same size
def create_mask_cached(B, H, M, N, 
        causal=True, 
        sliding_window=0,
        mask_bos=False,
        device="cpu") -> Float[Tensor, "B H M N"]:
    """Create mask for the attention scores"""
    if sliding_window:
        if causal:
            score_mod = partial(sliding_window_causal_mask, sliding_window, mask_bos)
        else:
            score_mod = partial(sliding_window_mask, sliding_window, mask_bos)
    else: 
        if causal:
            score_mod = partial(causal_mask, mask_bos)
        else: # no mask 
            score_mod = lambda b, h, q_idx, kv_idx: torch.ones_like(q_idx, device=device, dtype=torch.bool)

    return create_mask(score_mod, B, H, M, N, device=device)


@torch.jit.script
def fill_ones_left(a:torch.Tensor):
    """Fill ones on the left of the ones in a 2D tensor"""
    return (a + a.sum(dim=-1, keepdim=True) - a.cumsum(dim=-1)) > 0

### Dilation functions

# create masks around labeled entities for the loss
@lru_cache
def get_dilation(dilation):
    """Get dilation function for NER tags that creates a mask for the tokens"""
    kernel = torch.ones(
        (1, 1, 2 * dilation + 1)
    ).cuda()  # (out_channels, in_channels, kernel_size)
    dilation_fx = lambda tags: torch.where(
        F.conv1d(tags.unsqueeze(1), kernel, padding=dilation).squeeze(1) != 0,
        True,
        False,
    )
    return dilation_fx


@lru_cache
def get_2d_dilation(dilation:Tuple):
    """Get dilation function for NER tags that creates a mask for the tokens
    Args;
        dilation: list (dilate_x, dilate_y), dimension of the dilation"""

    try:
        b, a = dilation
    except ValueError:
        a = b = dilation
    k_size = 2 * max(dilation) + 1
    a = (k_size - (2 * a + 1)) // 2
    b = (k_size - (2 * b + 1)) // 2

    kernel = torch.zeros((1, 1, k_size, k_size))
    kernel[:, :, a : -a if a else k_size, b : -b if b else k_size] = 1
    kernel = kernel.cuda()  # (out_channels, in_channels, kernel_size, kernel_size)
    # pattern input shape (batch, seq, seq)
    dilation_fx = lambda pattern: torch.where(
        F.conv2d(pattern.unsqueeze(1), kernel, padding="same", stride=1).squeeze(1)
        != 0,
        torch.tensor(True, device="cuda"),
        torch.tensor(False, device="cuda"),
    )
    return dilation_fx

