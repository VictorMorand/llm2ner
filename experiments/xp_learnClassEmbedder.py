"""
In this experiment, we load a NER model, and learn an embedding model on top of it for classification.
The learned model is then evaluated on several datasets.
- in a Zero-Shot setting
- in a supervised setting (TODO - implement Gradient boosting classifier)
"""

### IMPORTS 
import logging, os, sys
from shutil import rmtree
from pathlib import Path
from typing import List
from attr import dataclass
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, serialize, Task, deserialize
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experimaestro import settings

logging.basicConfig(level=logging.DEBUG)

from llm2ner import NERmodel
from llm2ner.classification import LearnZeroShotNER, NERCmodel

N_TASKS_LIMIT = 300
WORKSPACE = "LLMinterp"


def get_alias(task: Task):
    alias = task.__xpmtype__.name().split(".")[-1] + '_' + tagspath(task)
    alias = alias[:200] if len(alias) > 200 else alias
    return alias

def find_model(hash: str):
    ws = settings.get_workspace(WORKSPACE)
    try :
        path = next((ws.path / "jobs").glob(f"*/{hash}/model"))
    except StopIteration:
        print(f"Could not find model for hash {hash} in {list((ws.path / 'jobs').glob(f'*/{hash}/*'))}")
        path = None
    return path


@configuration
class Configuration(ConfigurationBase):
    """Configuration of the whole experiment"""

    learner_hashs: List[str] = [""]
    
    # NERC model    
    n_layers: List[int] = [1]
    embed_dim: List[int] = [0] 
    span_method: str = NERCmodel.SPAN_METHODS.CONCAT
    

    # Zero Shot Training
    epochs: int = 5
    launcher: str = """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""
    n_val: int = 1000  # number of steps between validation and logging
    lr: float = 1e-2
    batch_size: int = 32
    accumulation_steps: int = 2
    grad_clip: float = 10.0
    patience: int = 3  # lr scheduler patience
    min_lr: float = 1e-5


    # Data
    max_length: int = 1000  # tokens
    max_ent_length: int = 400  # Chars
    val_limit: int = 1000
    val_metric: str = "f1"

    sup_eval_datasets: list = ["ncbi", "CoNLL 2003"]

    # Misc
    data_folder: str = ""
    n_runs: int = 1



@dataclass
class processed_model:
    nerc_model: NERCmodel
    id: str
    loader: Task
    sup_evals: List[Task]



def run(helper: ExperimentHelper, cfg: Configuration):


    # Build result directories
    llm_eval_folder = helper.xp.resultspath / "llm_annotations"
    runpath = helper.xp.resultspath / "runs"
    model_dir = helper.xp.resultspath / "models"

    for folder in [llm_eval_folder, runpath, model_dir]:
        folder.mkdir(exist_ok=True, parents=True)
    
    tasks = []
    evals = []  
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])

    for hash in cfg.learner_hashs:
        # Load NER model
        model_dir = find_model(hash)
        if model_dir is None:
            print(f"Could not find model for hash {hash}, skipping.")
            continue
        print(f"Loading model from {model_dir}")
        ner_model, loader = deserialize(model_dir)
        

        for n_layers in cfg.n_layers:
            for hidden_size in cfg.hidden_size:
                        
                        
                # launch Classification overhead
                nerc_model = NERCmodel.C(ner_model=ner_model)

                embed_learner = LearnZeroShotNER.C(
                    nerc_model=nerc_model,
                    loss_fn="balanced_bce",
                    PL_threshold=0.9,
                    # Training
                    epochs=2,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    optimizer="adamw",
                    accumulation_steps=cfg.accumulation_steps,
                    grad_clip=cfg.grad_clip,
                    patience=cfg.patience,
                    n_val=cfg.n_val,
                    val_metric="BCE", # Binary Cross Entropy on val set
                    # Data
                    dataset_name=tag(cfg.dataset_name),
                    max_length=cfg.max_length,
                    max_ent_length=cfg.max_ent_length,
                    run=run,
                    data_folder=cfg.data_folder,
                )
                embed_learner.submit(launcher=gpulauncher, init_tasks=[loader])
                (runpath / get_alias(embed_learner)).symlink_to(
                    embed_learner.runpath, target_is_directory=True
                )
                tasks.append(task)
                evals.append(evaluation)
                            
    # Wait that everything finishes
    helper.xp.wait()
    import pandas as pd
    for i in range(len(evals)):
        eval = evals[i]
        eval_path = eval.jobpath / eval.result_path
        if eval_path.is_file():
            #load json
            with open(eval_path, "r") as f:
                results = f.read()
            results_df = pd.DataFrame.from_dict(results["metrics"])
            
            print(results_df.to_csv(sep="\t"))