import click
import os
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformer_lens.loading_from_pretrained import get_official_model_name



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


@click.command()
@click.argument("model_name")
def main(model_name):
    """
    Downloads the weights of the specified model to the local cache.
    """
    try :
        official_name = get_official_model_name(model_name)
    except Exception as e:
        official_name = model_name
        
    if official_name is None:
        click.echo(f"Model '{model_name}' not found.")
        return

    if check_hf_cache(official_name):
        click.echo(f"Model '{model_name}' is already downloaded in the cache.")
        return
    
    click.echo(f"Downloading model '{model_name}' from Hugging Face Hub...")
    snapshot_download(repo_id=official_name)
    
    click.echo(f"Successfully dowloaded Model '{model_name}' !")


if __name__ == "__main__":

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["HF_EVALUATE_OFFLINE"] = "0"

    main()
