import os, gc, re, torch, logging
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
from tqdm import tqdm
from typing import Optional, Union, List
from jaxtyping import Float
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformer_lens import HookedTransformer
import transformer_lens as tl

from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
)


CLASS_TO_BLOCK_NAME = {
    "LlamaModel": "layers",
    "GPTNeoXModel": "layers",
    "MistralModel": "layers",
    "ModernBertModel": "layers",
    "PhiModel": "layers",
    "gpt2": "h",
    "bert-base-uncased": "layer",
    "RobertaModel": "encoder.layer",
    "BertModel": "encoder.layer",
}


########################################################################
### Python utils

### has / get attr nested
def hasattr_nest(obj, attr_path: str) -> bool:
    """
    Check if an object has a nested attribute like 'encoder.layer.0.attention'.
    """
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return True
    except AttributeError:
        return False

def getattr_nest(obj, attr_path: str, default=None):
    """
    Safely get a nested attribute, or return default if not found.
    """
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default



########################################################################
############################## XPM utils  ##############################
from experimaestro import Config, Param, Meta
from pathlib import Path


class PathOutput(Config):
    """Path to store the output of the task"""

    path: Param[Path]


########################################################################
############################## Parameters ##############################

checkpoints_folder = ""
data_folder = ""

########################################################################
############################## FUNCTIONS ##############################


@torch.jit.script
def rev_cumsum(a: torch.Tensor):
    """cumulative sum in reverse order on the last dimension"""
    return a + a.sum(dim=-1, keepdim=True) - a.cumsum(dim=-1)


############################## HUGGINGFACE FUNCTIONS ##############################


def check_hf_cache(model_id):
    """Check if the model is already downloaded in the cache
    Args:
        model_id: the id of the model to check
    Returns:
        True if the model is already downloaded, False otherwise
    """
    # check if the model is already downloaded

    try:
        # Try downloading the config (or another known file)
        hf_hub_download(repo_id=model_id, filename="config.json", local_files_only=True)
        return True

    except EntryNotFoundError:
        # If the file is not found, the model is not downloaded
        return False


def check_internet():
    try:
        import socket

        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False


def load_llm(
    model_name: str,
    to_hookedtransformer: bool = False,
    to_causal: bool = False,
    return_config: bool = False,
    cut_to_layer: int = None,
    dtype=torch.float32,
    force_download: bool = False,
    **kwargs,
) -> Union[HookedTransformer, AutoModel]:
    """Use fixed parameters to load models from Tlens
    Args:
        model_name: the name of the model to load
        to_hookedtransformer: if True, load the model as a HookedTransformer, else as an [AutoModel](https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes)
        return_config: if True, return the model config as a second argument
        cut_to_layer: if not None, cut the model to the given layer (inclusive)
            - Note, cutting is performed _after_ loading the model, so the model is first fully loaded in memory (although we don't load LM heads with `transformers.AutoModel`)
        force_download: if True, force re-download of the model even if it is in cache
        dtype: the dtype to use for the model
        **kwargs: additional keyword arguments to pass to the AutoModel.from_pretrained
    Returns:
        model: the loaded model in HookedTransformer format, or in AutoModel format if to_hookedtransformer is False
    """

    try:
        model_id = get_official_model_name(model_name)
        config = convert_hf_model_config(model_id)
    except Exception as e:
        logging.warning(
            f"Model alias {model_name} not found, assuming its a valid HuggingFace model name"
        )
        model_id = model_name

    # check if the model is already downloaded
    if check_hf_cache(model_id) or force_download:
        # hugginface in offline mode by default
        logging.info(f"Model {model_name} is in cache")
        # get config directly from HF
        config = AutoConfig.from_pretrained(model_id)  # check if model exists

    else:
        logging.info(
            f"Trying to download {model_name}... (will need internet) \n you can also try to download it manually with `python scripts/download_llm.py {model_id}`"
        )
        # pass hugginface in online mode
        # Unset offline environment variables to allow online download
        import os

        for var in [
            "HF_DATASETS_OFFLINE",
            "HF_EVALUATE_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_OFFLINE",
        ]:
            os.environ[var] = "0"

        if not check_internet():
            raise EnvironmentError(
                "No internet connection available, cannot download the model"
            )
        print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"))
        config = AutoConfig.from_pretrained(
            model_id, local_files_only=False
        )  # check if model exists
        snapshot_download(model_id, local_files_only=False)

    if to_hookedtransformer:
        model = HookedTransformer.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            fold_ln=False,
            fold_value_biases=False,
            device_map="auto",
            dtype=dtype,
            local_files_only=True,
        )
    else:
        Base_class = AutoModelForCausalLM if to_causal else AutoModel
        model = Base_class.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            # dtype=dtype,
            **kwargs,
        )
        model.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            # dtype=dtype,
            **kwargs,
        )

    # check if model has pad_token, if not add it
    if not hasattr(model.tokenizer, "pad_token") or model.tokenizer.pad_token is None:
        logging.warning(
            f"Model tokenizer {model.tokenizer.__class__.__name__} has no pad_token, setting it to eos_token"
        )
        if (
            hasattr(model.tokenizer, "eos_token")
            and model.tokenizer.eos_token is not None
        ):
            model.tokenizer.pad_token = model.tokenizer.eos_token
        else:
            # set to a new token
            model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(model.tokenizer))
            logging.warning(
                f"Model tokenizer {model.tokenizer.__class__.__name__} had no eos_token, added new pad_token [PAD]"
            )
            assert (
                model.tokenizer.pad_token is not None
            ), "pad_token should be defined now"

    logging.info(
        f"{model_id} loaded as {model.__class__.__name__} \n \
          with {count_parameters(model)/1e9:.3f} B parameters \n \
          GPU allocated memory : {torch.cuda.memory_allocated()/1024**3:.3f} GB"
    )

    if cut_to_layer is not None:
        model = cut_llm(model, cut_to_layer)

    gc.collect()
    torch.cuda.empty_cache()

    if return_config:
        return model, config
    return model


def cut_llm(model: HookedTransformer, layer: int) -> HookedTransformer:
    """Cut a HookedTransformer model at a given layer
    Args:
        model: the HookedTransformer model to cut
        layer: the layer to cut the model at (inclusive)
    Returns:
        cut_model: the cut HookedTransformer model
    """
    logging.info(f"Trying to cut model after layer {layer}..")
    if isinstance(model, HookedTransformer):
        n_layers = model.cfg.n_layers
        assert (
            0 <= layer < n_layers
        ), "layer must be between 0 and cfg.n_layers-1"

        # keep only the layers up to the given layer (inclusive)
        model.blocks = model.blocks[: layer + 1]
        if hasattr(model, "ln_final"):
            del model.ln_final  # delete final layer norm
        if hasattr(model, "unembed"):
            del model.unembed  # delete unembed layer

        model.cfg.n_layers = layer + 1
        logging.info(f"Cut HookedTransformer with {n_layers} layers to {len(model.blocks)} layers")
    else:
        className = model.__class__.__name__
        blocks_attr = CLASS_TO_BLOCK_NAME.get(className, None)
        if blocks_attr is None:
            logging.warning(
                f"Model class {className} not supported for cutting yet"
            )
            return model

        blocks = getattr_nest(model, blocks_attr, None)
        assert isinstance(
            blocks, (list, nn.ModuleList)
        ), f"Could not find {blocks_attr} in model of class {className}, got {type(blocks)}"
        n_layers = len(blocks)
        assert (
            0 <= layer < n_layers
        ), f"layer must be between 0 and {n_layers-1} for model of class {className}"

        # set the blocks attribute
        setattr(model, blocks_attr, blocks[: layer + 1])
        # clear the cache if any
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"Cut {className} with {n_layers} layers to {len(getattr_nest(model, blocks_attr, []))} layers")

    return model


def get_model_max_length(model_name: str) -> int:
    from transformers import AutoConfig

    max_length = None
    possible_attr = ["max_position_embeddings", "n_ctx", "max_length", "context_length"]

    try:
        llm_config = convert_hf_model_config(get_official_model_name(model_name))
        max_length = llm_config["n_ctx"]

    except Exception as e:
        logging.warning(
            f"Could not get max length from transformer_lens config for model {model_name}, trying from AutoModel config"
        )
        llm_config = AutoConfig.from_pretrained(model_name)
        for attr in possible_attr:
            if hasattr(llm_config, attr):
                max_length = getattr(llm_config, attr)
                logging.info(
                    f"Found max length attribute {attr} with value {max_length}"
                )
                break
        if max_length is None:
            logging.error(
                f"Could not find max length attribute in config: {llm_config}"
            )
            max_length = 512
    return max_length


def get_model_dim(model_name: str):
    from transformers import AutoConfig

    dim = None
    possible_attr = ["hidden_size", "d_model"]

    try:
        llm_config = convert_hf_model_config(get_official_model_name(model_name))
        dim = llm_config["d_model"]

    except Exception as e_tlens:
        hf_config = AutoConfig.from_pretrained(model_name)
        dim = hf_config.hidden_size

        for attr in possible_attr:
            if hasattr(hf_config, attr):
                dim = getattr(hf_config, attr)
                logging.info(f"Found model dimension attribute {attr} with value {dim}")
                break
        if dim is None:
            raise ValueError(
                f"Could not find model dimension attribute in config: {hf_config}"
            )
    return dim


############################## TORCH FUNCTIONS ##############################


def reinit_weights(module: nn.Module):
    """Reinitialize the weights of a module with default initialization for Linear and Embedding layers
    Consider implement your own reinitialization function for more specific cases
    """
    for name, param in module.named_parameters():
        if "weight" in name:
            init.normal_(param)
        elif "bias" in name:
            init.zeros_(param)
        else:
            init.normal_(param)
            logging.warning(
                f"Parameter {name} not recognized, applying normal_ initialization"
            )

    # you can add other layers (e.g., Conv2d) if needed


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_str_tokens(
    tokens: Union[torch.Tensor, List[int]],
    tokenizer,
    skip_special_tokens=False,
    **kwargs,
):
    """Convert token ids to string tokens using the tokenizer
    Args:
        tokenizer: the tokenizer to use
        tokens: the token ids to convert
        skip_special_tokens: if True, skip special tokens
        **kwargs: additional keyword arguments to pass to the tokenizer
    Returns:
        str_tokens: the string tokens
    """
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    tokens = tokens.view(-1)
    return [
        token.replace("▁", " ").replace("Ġ", " ")
        for token in tokenizer.convert_ids_to_tokens(
            tokens, skip_special_tokens=skip_special_tokens, **kwargs
        )
    ]


def print_all_submodules(model: torch.nn.Module, max_depth: int = float("inf")):
    """Print all submodules of a model, with optional depth limit."""
    for name, module in model.named_modules():
        depth = name.count(".")
        if depth <= max_depth:
            logging.info(f"{'  ' * depth}├── {name}: {module.__class__.__name__}")


def get_residual_rep_hook_name(layer: int):
    """Get the residual representation hook for a given layer
    Args:
        layer: layer to get the hook for
    Returns:
        hook: function to call to get the residual representation
    """
    # get the hook name for the layer output
    if layer < 0:
        # baseline, extract embedding only
        hook_name = tl.utils.get_act_name("embed")
    else:
        hook_name = tl.utils.get_act_name(
            "resid_post", layer=layer
        )  # get hook name for the layer output
    return hook_name


@torch.no_grad()
def compute_to_layer_HF(
    model: nn.Module,
    tokens: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    hook_name: str,
    dim: int,
    verbose: bool = False,
    move_to_device: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Run a transformer-like model up to a given layer and return hidden states.

    Args:
        model (nn.Module): PyTorch llm (must support forward(tokens))
        tokens (Tensor): input tokens of shape (batch, seq)
        hook_name (str): name of the submodule where to stop (e.g., "layers.5")
        dim (int): hidden dimension size of the model
        verbose (bool): if True, print hook name and debug info
        move_to_device (bool): if True, put buffer on the same device as model
        dtype (torch.dtype): output dtype

    Returns:
        hidden_states (Tensor): _detached_ tensor of shape (batch, seq, dim)
    """
    b_size, seq = tokens.shape
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # buffer to store activations
    buffer = torch.zeros(b_size, seq, dim, dtype=model_dtype)
    if move_to_device:
        buffer = buffer.to(device)

    if verbose:
        logging.debug(f"Registering hook on {hook_name}")

    def save_activation(module, input, output):
        """
        Save the forward activations into buffer
        """
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"Unexpected output type from {hook_name}: {type(output)}")
        if output.shape != (b_size, seq, dim):
            raise ValueError(
                f"Unexpected output shape from {hook_name}: {output.shape}, expected {(b_size, seq, dim)}"
            )
        if verbose:
            logging.info(
                f"Saving activations from {hook_name} with shape {output.shape}"
            )
        buffer.copy_(output.detach())
        raise StopForwardException  # custom stop

    class StopForwardException(Exception):
        """Raised to stop the forward pass early"""

        pass

    # Find the submodule
    submodule = dict(model.named_modules()).get(hook_name, None)
    if submodule is None:
        raise ValueError(f"No submodule found with name {hook_name} in model {model}")

    handle = submodule.register_forward_hook(save_activation)

    try:
        with torch.no_grad():
            # run model until hook triggers
            _ = model(tokens, attention_mask=attn_mask)
    except StopForwardException:
        if verbose:
            logging.info(f"Stopped forward after {hook_name}")
    finally:
        handle.remove()  # clean up hook

    return buffer.to(dtype)


@torch.no_grad()
def compute_to_layer(
    model: Union[HookedTransformer, nn.Module],
    layer: int,
    tokens: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dim: int = None,
    verbose: bool = False,
    move_to_device: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """Compute the transformer up to a given layer, return the hidden states
    Args:
        model: HookedTransformer from TransformerLens, or nn.Module
        layer: layer to compute the transformer up to
        tokens (tensor 'batch, seq'): tokens to compute the transformer on
        verbose: if True, print the hook names and the buffer shapes
        move_to_device: if True, move the buffer to the device of the model
        dtype: the output dtype to use, can be different from the model dtype
    Returns:
        hidden_states: tensor of shape (batch, len, dim) with the hidden states at the given layer
    """
    if not isinstance(model, HookedTransformer):
        assert dim is not None, "dim must be specified for non-HookedTransformer models"

        #try to get the block name for the model, default to "layers"
        block_name = CLASS_TO_BLOCK_NAME.get(model.__class__.__name__, "layers")
        #check if the block name exists in the model
        if not hasattr_nest(model, block_name):
            raise NotImplementedError(
                f"cannot find block {block_name} in model of class {model.__class__.__name__}"
            )
        hook_name = f"{block_name}.{layer}"
        if hook_name is None:
            raise NotImplementedError(
                f"cannot find block hook name for model of class {model.__class__.__name__}"
            )

        return compute_to_layer_HF(
            model=model,
            tokens=tokens,
            attn_mask=attn_mask,
            hook_name=hook_name,
            dim=dim,
            verbose=verbose,
        )
    # else normal behavior
    dim = model.cfg.d_model
    b_size, seq = tokens.shape
    model_dtype = next(model.parameters()).dtype
    buffer = torch.zeros(
        b_size, seq, dim, dtype=model_dtype
    )  # create buffer where to store representations
    if move_to_device:
        buffer = buffer.to(next(model.parameters()).device)

    hook_name = get_residual_rep_hook_name(layer)  # get hook name for the layer output
    if verbose:
        logging.info(f"compute hidden states at hook {hook_name}")

    def save_activation(tensor, buffer):
        """Save wanted activation in buffer
        Args:
            tensor: the act cache to modify
            buffer (Tensor): the buffer where to store the wanted activations
            inds (List): the token index that we want to overwrite
        """
        # just store the wanted activations in the buffer
        buffer[:] = tensor
        # stop the forward pass
        raise ValueError("Stopping the forward pass")

    # run the model with the hook
    try:
        model.run_with_hooks(
            tokens,
            return_type=None,
            fwd_hooks=[
                (hook_name, lambda tensor, hook: save_activation(tensor, buffer))
            ],
        )
    except ValueError as e:
        if verbose:
            logging.info(f"Caught exception {e}")

    return buffer.to(dtype)


def get_QK_from_layers(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layers: list,
    attn_mask: Optional[torch.Tensor] = None,
    verbose: bool = False,
    move_to_device: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor]:
    """Compute the transformer up to a given layer, return the computed Queries, Keys and representations

    -> WARNING: For some models (e.g Llama), the number of heads for the keys is different from the number of heads for the queries: This function will return all of them.

    Args:
        model: HookedTransformer from TransformerLens
        tokens (tensor 'batch, seq'): tokens to compute the transformer on
        layers (list): the layer to compute the transformer
        verbose: if True, print the hook names and the buffer shapes
        move_to_device: if True, move the buffer to the device of the model
        dtype: the output dtype to use, can be different from the model dtype

    Returns:
        queries: tensor of shape (len(layers), batch, seq, n_heads, head_dim) with the queries at the given layers
        keys: tensor of shape (len(layers), batch, seq, n_heads, head_dim) with the keys at the given layers
        reps: tensor of shape (batch, seq, dim) with the residual representations at last given layer
    """
    # process model configuration to get right hooks
    head_dim = model.cfg.d_head
    n_heads = model.cfg.n_heads
    n_kv_heads = model.cfg.n_key_value_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads

    b_size, seq = tokens.shape
    model_dtype = next(model.parameters()).dtype
    
    layers = sorted(layers)  # sort the layers to get the right hook names
    use_rotary = True if model.cfg.positional_embedding_type == "rotary" else False

    # get the hook names for the queries and keys
    hook_q_key = "rot_q" if use_rotary else "q"
    hook_k_key = "rot_k" if use_rotary else "k"
    hooks_q_name = [
        tl.utils.get_act_name(hook_q_key, layer=layer) for layer in layers
    ]  # get hook name for the layer output
    hooks_k_name = [
        tl.utils.get_act_name(hook_k_key, layer=layer) for layer in layers
    ]  # get hook name for the layer output
    hook_rep_name = tl.utils.get_act_name(
        "resid_post", layer=layers[-1]
    )  # get hook name for the layer output

    # create buffer where to store representations with the right configuration
    queries = torch.zeros(
        len(layers), b_size, seq, n_heads, head_dim, dtype=model_dtype
    )
    keys = torch.zeros(
        len(layers), b_size, seq, n_kv_heads, head_dim, dtype=model_dtype
    )
    # create buffer where to store representations
    reps = torch.zeros(b_size, seq, model.cfg.d_model, dtype=model_dtype)

    if verbose:
        logging.info(
            f"Compute query and keys for {model.cfg.model_name} at layers {layers}, hooks are:"
        )
        logging.info(hooks_q_name)
        logging.info(hooks_k_name)
        logging.info(f"Buffer shape: {queries.shape} {keys.shape}")

    # hook functions to save the activations

    def save_activation_layer(tensor, buffer, layer_ind):
        # just store the wanted activations in the buffer and continue the forward pass
        buffer[layer_ind, :] = tensor
        return tensor

    def save_activation(tensor, buffer):
        buffer[:] = tensor
        raise ValueError("Stopping the forward pass")

    hooks_q = [  # hook name, function to call  for queries
        (
            hook_name,
            lambda tensor, hook: save_activation_layer(
                tensor, queries, layer_ind=(layer - layers[0])
            ),
        )
        for hook_name, layer in zip(hooks_q_name, layers)
    ]

    hooks_k = [  # hook name, function to call for keys
        (
            hook_name,
            lambda tensor, hook: save_activation_layer(
                tensor, keys, layer_ind=(layer - layers[0])
            ),
        )
        for hook_name, layer in zip(hooks_k_name, layers)
    ]

    hook_rep = [(hook_rep_name, lambda tensor, hook: save_activation(tensor, reps))]
    # stop the forward pass at last layer as it is the last call that we need

    with torch.no_grad():
        # run the model with the hook
        try:
            model.run_with_hooks(
                tokens,
                attention_mask=attn_mask,
                return_type=None,
                fwd_hooks=hooks_q + hooks_k + hook_rep,
            )
        except ValueError as e:
            if verbose:
                logging.info(f"Caught exception {e}")

    if move_to_device:
        device = next(model.parameters()).device
        queries = queries.to(device)
        keys = keys.to(device)
        reps = reps.to(device)

    return queries.to(dtype), keys.to(dtype), reps.to(dtype)


def get_attnScores_from_layers(
    layers: list,
    model: HookedTransformer,
    tokens: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    before_softmax: bool = True,
    verbose: bool = False,
    move_to_device: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[Float[torch.Tensor, "batch layer n_heads seq seq"], Float[torch.Tensor, "batch seq dim"]]:
    """Compute the transformer up to a given layer.
    - Catch and return the attention scores at the given layers.
    - Also returns the residual stream at the last given layer.

    Args:
        model: HookedTransformer from TransformerLens
        tokens (tensor 'batch, seq'): tokens to compute the transformer on
        layers (list): the layers from which to extract the attention scores
        verbose: if True, print the hook names and the buffer shapes
        move_to_device: if True, move the buffer to the device of the model
        dtype: the output dtype to use, can be different from the model dtype

    Returns:
        attn_scores_q: tensor of shape (len(layers), batch, seq, n_heads, head_dim) with the attention scores for the queries at the given layers
        reps: tensor of shape (batch, seq, dim) with the residual representations at last given layer
    """
    # process model configuration to get right hooks
    head_dim = model.cfg.d_head
    n_heads = model.cfg.n_heads
    
    b_size, seq = tokens.shape
    model_dtype = next(model.parameters()).dtype
    layers = sorted(layers)  # sort the layers to get the right hook names

    # get the hook names for the attn scores
    attn_hook_names = [
        tl.utils.get_act_name(
            "attn_scores" if before_softmax else "pattern", 
            layer=layer) for layer in layers
    ]
    # get hook name for the layer output
    hook_rep_name = tl.utils.get_act_name("resid_post", layer=layers[-1])  

    # create buffer where to store representations with the right configuration
    attn_buffer = torch.zeros(
        b_size, len(layers), n_heads, seq, seq, dtype=model_dtype
    )
    # create buffer to store representations
    reps = torch.zeros(b_size, seq, model.cfg.d_model, dtype=model_dtype)

    if verbose:
        logging.info(
            f"Compute query and keys for {model.cfg.model_name} at layers {layers}, hooks are:"
        )
        logging.info(f" - attn hooks: {attn_hook_names}")
        logging.info(f"Buffer shape: {attn_buffer.shape} {reps.shape}")

    # hook functions to save the activations

    def save_activation_layer(tensor, buffer, layer_ind):
        # just store the wanted activations in the buffer and continue the forward pass
        buffer[:,layer_ind, :] = tensor
        return tensor

    def save_activation(tensor, buffer):
        buffer[:] = tensor
        raise ValueError("Stopping the forward pass")

    # stop the forward pass at last layer as it is the last call that we need
    hook_rep = [(hook_rep_name, lambda tensor, hook: save_activation(tensor, reps))]
    # hooks to store the attention scores at the given layers
    attn_hooks = [
        (
            hook_name,
            lambda tensor, hook: save_activation_layer(
                tensor, attn_buffer, layer_ind=(layer - layers[0])
            ),
        )
        for hook_name, layer in zip(attn_hook_names, layers)
    ]
    
    # run the model with the hook
    with torch.no_grad():
        try:
            model.run_with_hooks(
                tokens,
                attention_mask=attn_mask,
                return_type=None,
                fwd_hooks=attn_hooks + hook_rep,
            )
        except ValueError as e:
            if verbose:
                logging.info(f"Caught exception {e}")

    if move_to_device:
        device = next(model.parameters()).device
        attn_buffer = attn_buffer.to(device)
        reps = reps.to(device)

    return attn_buffer.to(dtype), reps.to(dtype)


# place a hook to replace representation of "_"
def get_replace_with_rep_hook(reps, inds):

    def replace_with_rep(tensor, reps, inds):
        """replace first token with representation
        Args:
            tensor: (batch, sent_len, H) the act cache to modify
            reps tensor(batch, H) the representations to overwrite at hook
            inds tensor(batch), the indexes that we want to overwrite in each sentence of batch
        """
        # replace the token a ind by the given representations
        # tensor[torch.arange(tensor.shape[0]),inds,:] = reps # slower
        inds_expanded = (
            inds.unsqueeze(1)
            .expand(-1, tensor.shape[-1])
            .unsqueeze(1)
            .to(tensor.device)
        )
        tensor.scatter_(1, inds_expanded, reps.unsqueeze(1))
        return tensor

    return lambda tensor, hook: replace_with_rep(tensor, reps, inds)


def get_representation(
    model: HookedTransformer,
    tokens,
    token_inds,
    layer: int,
    hooks: list = [],
    verbose: bool = False,
):
    """Extract model representation of token `token_inds` at layer `layer` from batch
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        tokens: tensor(batch, N) tokenized texts to process
        token_inds: (batch) index of tokens where to extract representation
        hooks: (Optionnal) List [(hook_name, hook_fx)] hooks that will also be placed on the model during computation
        layer: layer at which to retreive the representations
    """

    dim = model.QK.shape[-1]
    b_size = tokens.shape[0]
    dtype = model.W_U.dtype
    assert len(token_inds) == b_size
    buffer = torch.zeros(
        b_size, dim, dtype=dtype
    )  # create buffer where to store representations

    if layer < 0:
        # baseline, extract embedding only
        hook_name = tl.utils.get_act_name("embed")
    else:
        hook_name = tl.utils.get_act_name(
            "resid_post", layer=layer
        )  # get hook name for the layer output

    if verbose:
        logging.info(
            f"extract representation of tokens {token_inds} at hook {hook_name}"
        )

    def save_activation(tensor, buffer, inds):
        """Save wanted activation in buffer
        Args:
            tensor: the act cache to modify
            buffer (Tensor): the buffer where to store the wanted activations
            inds (List): the token index that we want to overwrite
        """
        # just store the wanted activations in the buffer
        buffer[:] = torch.vstack([tensor[i, inds[i], :] for i in range(len(inds))])
        return tensor

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            return_type=None,
            fwd_hooks=hooks
            + [
                (
                    hook_name,
                    lambda tensor, hook: save_activation(tensor, buffer, token_inds),
                )
            ],
        )
    return buffer


def get_avg_representation(
    model: HookedTransformer, prompts, entities, layer: int, verbose: bool = False
):
    """extract average model representation of entities at given layer
    representations will be extracted at entity tokens after reading f"{prompt}{entity}"
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        prompts: list of prompts to use for extraction
        entities: list of entities to extract representations from
        layer: layer at which to retreive the representations
    """

    dim = model.QK.shape[-1]
    b_size = len(prompts)
    dtype = model.W_U.dtype
    assert len(entities) == b_size

    if layer < 0:
        # baseline, extract embedding only
        hook_name = tl.utils.get_act_name("embed")
    else:
        hook_name = tl.utils.get_act_name(
            "resid_post", layer=layer
        )  # get hook name for the layer output

    if verbose:
        logging.info(
            f"extract average representation of {entities} at hook {hook_name}"
        )

    # tokenize prompts and entities
    prompts_inputs = model.tokenizer(
        prompts, padding=True, padding_side="left", return_tensors="pt"
    )
    prompt_tokens = prompts_inputs["input_ids"].cuda()
    prompt_mask = prompts_inputs["attention_mask"].cuda()

    entity_inputs = model.tokenizer(
        entities,
        padding=True,
        padding_side="right",
        return_tensors="pt",
        add_special_tokens=False,
    )
    entity_tokens = entity_inputs["input_ids"].cuda()
    entity_mask = entity_inputs["attention_mask"].cuda()

    # concatenate prompts and entities
    tokens = torch.hstack([prompt_tokens, entity_tokens])
    attn_mask = torch.hstack([prompt_mask, entity_mask])
    # TODO may need to adjust attention mask for left padding -> TBD

    # check that decoder string is correct
    # logging.info(model.tokenizer.decode(tokens.view(-1).cpu().numpy())) # OK !

    ind_min = prompt_tokens.shape[-1]
    ind_max = prompt_tokens.shape[-1] + entity_tokens.shape[-1] - 1

    # mask with zeros where there is an eos
    mask = torch.ones_like(entity_tokens)
    mask[entity_tokens == model.tokenizer.eos_token_id] = 0

    buffer = torch.zeros(
        b_size, entity_tokens.shape[-1], dim, dtype=dtype
    )  # create buffer where to store representations
    buffer.requires_grad = False
    buffer = buffer.cuda()

    def save_activations(tensor, buffer, min_ind, max_ind):
        """Save wanted activations in buffer
        Args:
            tensor: the act cache to modify
            buffer (Tensor): the buffer where to store the wanted activations
            min_ind: the first token index that we want to save
            max_ind: the last token index that we want to save
        """
        # just store the wanted activations in the buffer
        buffer[:] = tensor[:, min_ind : max_ind + 1, :]
        return tensor

    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            return_type=None,
            fwd_hooks=[
                (
                    hook_name,
                    lambda tensor, hook: save_activations(
                        tensor, buffer, ind_min, ind_max
                    ),
                )
            ],
        )

    buffer = buffer * mask.unsqueeze(-1)  # apply mask to zero out eos tokens
    buffer = buffer.sum(dim=1) / mask.sum(dim=1).unsqueeze(
        -1
    )  # sum over tokens and divide by number of non-zero tokens
    return buffer  # return obtained average representation


def project_on_vocab(model, rep, k=10):
    """project a representation on the vocabulary of the model
    Args:
        model: the model to use
        rep: the representation to project
        k: number of top tokens to return
    """
    logits = torch.tensor(rep).float().cuda() @ model.W_U
    topk_inds = torch.topk(logits, k).indices
    topk_tokens = model.tokenizer.convert_ids_to_tokens(topk_inds.cpu().numpy())
    return topk_tokens
