import torch
import logging
from pathlib import Path
from llm2ner.models.tommer import ToMMeR
from xpm_torch import Random
import numpy as np

logging.basicConfig(level=logging.INFO)

def test_save_load():
    save_path = Path("test_model_save")
    if save_path.exists():
        import shutil
        shutil.rmtree(save_path)
    
    # 1. Create model from config
    # We use a real LLM name that we know exists in utils.py (e.g., 'gpt2') 
    # or one that has its dimension hardcoded in utils.py
    llm_name = "gpt2" 
    model_config = ToMMeR.C(
        llm_name=llm_name,
        layer=6,
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
    loader = model_config.loader_config(save_path).instance()
    loader.execute()
    reloaded_model = loader.model
    
    logging.info(f"Model reloaded: {reloaded_model}")
    
    # 4. Verify weights
    assert torch.allclose(model.W_Q, reloaded_model.W_Q), "W_Q weights mismatch!"
    assert torch.allclose(model.W_K, reloaded_model.W_K), "W_K weights mismatch!"
    
    logging.info("SUCCESS: Model saved and reloaded correctly with identical weights.")

if __name__ == "__main__":
    test_save_load()
