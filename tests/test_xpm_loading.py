import torch
import logging
from pathlib import Path
from experimaestro import deserialize
from llm2ner.models.tommer import ToMMeR
from xpm_torch import Random
from xpm_torch.huggingface import TorchHFHub
from experimaestro.huggingface import ExperimaestroHFHub
import numpy as np

logging.basicConfig(level=logging.INFO)

def test_save_load():
    save_path = Path("test_model_save").absolute()
    hf_save_path = Path("test_model_hf").absolute()

    for p in [save_path, hf_save_path]:
        if p.exists():
            import shutil
            shutil.rmtree(p)

    # 1. Create model from config
    llm_name = "meta-llama/Llama-3.2-1B"
    model_config = ToMMeR.C(
        llm_name=llm_name,
        layer=4,
        rank=64,
    )
    model = model_config.instance()

    # Initialize structural parts
    model.initialize()

    # Set some weights to non-random values to verify loading
    with torch.no_grad():
        model.W_Q.fill_(1.0)
        model.W_K.fill_(2.0)

    logging.info(f"Model initialized: {model}")

    # 2. Save model
    logging.info(f"Saving model to {save_path}")
    model.save_model(save_path)

    # 3. Reload model using loader_config
    loader_config = model_config.loader_config(save_path)
    loader = loader_config.instance()
    loader.execute()
    reloaded_model = loader.model

    logging.info(f"Model reloaded: {reloaded_model}")

    # 4. Verify weights
    assert torch.allclose(model.W_Q, reloaded_model.W_Q), "W_Q weights mismatch!"
    assert torch.allclose(model.W_K, reloaded_model.W_K), "W_K weights mismatch!"

    logging.info("SUCCESS: Model saved and reloaded correctly with identical weights.")

    # 5. HF Export to disk
    logging.info(f"Exporting model to HF format at {hf_save_path}")
    # TorchHFHub takes a Loader configuration
    hub = TorchHFHub(loader_config)
    hub.save_pretrained(hf_save_path)

    # 6. Load as if from HF
    logging.info(f"Loading model from HF-formatted directory {hf_save_path}")

    hf_model = TorchHFHub.from_pretrained(hf_save_path, as_instance=True)


    logging.info(f"HF Model reloaded: {hf_model}")

    # 7. Verify HF weights
    assert torch.allclose(model.W_Q, hf_model.W_Q), "HF W_Q weights mismatch!"
    assert torch.allclose(model.W_K, hf_model.W_K), "HF W_K weights mismatch!"

    logging.info("SUCCESS: Model exported to HF format and reloaded correctly.")

    hf_loader = TorchHFHub.pretrained_loader(hf_save_path, as_instance=True)
    hf_loader.execute()
    hf_model = hf_loader.model

    # 7. Verify HF weights
    assert torch.allclose(model.W_Q, hf_model.W_Q), "HF W_Q weights mismatch!"
    assert torch.allclose(model.W_K, hf_model.W_K), "HF W_K weights mismatch!"

    logging.info("SUCCESS: Model reloaded correctly using HF loader.")

if __name__ == "__main__":

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    test_save_load()
