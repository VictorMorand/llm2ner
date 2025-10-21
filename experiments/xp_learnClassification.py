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
from llm2ner.classification import NERCmodel, LearnZeroShotNER, LearnSupervisedNER
from experiments.xp_learnTokenMatching import get_alias

N_TASKS_LIMIT = 300
WORKSPACE = "LLMinterp"


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
    extract_layer: int = 10
    """Layer from which to extract the embeddings, can be different from the layer used in the NER model"""

    # Zero Shot Training
    epochs: int = 5
    launcher: str = """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""
    n_val: int = 1000  # number of steps between validation and logging
    lr: float = 1e-2
    batch_size: int = 32
    accumulation_steps: int = 2
    grad_clip: float = 10.0
    patience: int = 3  # lr scheduler patience
    early_stopping: bool = False
    min_lr: float = 1e-5


    # Data
    max_length: int = 1000  # tokens
    max_ent_length: int = 400  # Chars
    val_limit: int = 1000
    val_metric: str = "f1"

    zero_shot_data: str = ""  
    """Name of the dataset for zero shot training"""
    sup_datasets: list = ["ncbi", "CoNLL 2003"]
    """List of datasets for supervised evaluation"""
    flat_decoder_strategies: List[str] = ["threshold"]
    """List of decoding strategies for flat evaluation"""
    eval_thresholds: List[float] = [0.5]
    """Thresholds to use for entity decoding when evaluating model"""

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
    
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])
    processed_models: List[dict] = []

    for hash in cfg.learner_hashs:
        # Load NER model
        model_dir = find_model(hash)
        if model_dir is None:
            print(f"Could not find model for hash {hash}, skipping.")
            continue
        print(f"Loading model from {model_dir}")
        ner_model, init_tasks = deserialize(model_dir)
        print(f"Loaded model {ner_model} with loader {init_tasks[0]}")

        for n_layers in cfg.n_layers:
            for embed_dim in cfg.embed_dim:
                        
                # launch Classification overhead
                nerc_model = NERCmodel.C(
                    ner_model=ner_model,
                    n_layers=n_layers,
                    embed_dim=embed_dim,
                    layer=tag(cfg.extract_layer),
                )
                if cfg.zero_shot_data:
                    embed_learner = LearnZeroShotNER.C(
                        nerc_model=nerc_model,
                        loss_fn="bce",
                        PL_threshold=0.99,
                        # Training
                        epochs=3,
                        batch_size=8,
                        lr=cfg.lr,
                        optimizer="adamw",
                        accumulation_steps=cfg.accumulation_steps,
                        grad_clip=cfg.grad_clip,
                        patience=cfg.patience,
                        n_val=cfg.n_val,
                        # Data
                        dataset_name=tag(cfg.zero_shot_data),
                        max_length=cfg.max_length,
                        max_ent_length=cfg.max_ent_length,
                        data_folder=cfg.data_folder,
                    ).tag(
                        "llm", nerc_model.ner_model.llm_name
                    ).tag(
                        "n_layers", n_layers
                    ).tag(
                        "embed", embed_dim
                    )

                    zero_shot_loader = embed_learner.submit(launcher=gpulauncher, init_tasks=init_tasks)
                    run_dir = runpath / get_alias(embed_learner)
                    if not run_dir.exists():
                        run_dir.unlink(missing_ok=True)
                        run_dir.symlink_to(embed_learner.runpath, target_is_directory=True)
                    else:
                        logging.warning(f"Run dir {run_dir} already exists, skipping symlink")

                for sup_dataset in cfg.sup_datasets:
                    for ft_ner in [False, True]: # True, False]:
                        #if we learned a pretrained representation, we also train from this checkpoint
                        for zero_shot_pretrain in [False] + ([True] if cfg.zero_shot_data else []): 

                            # Supervised evaluation
                            sup_learner = LearnSupervisedNER.C(
                                nerc_model=nerc_model,
                                finetune_ner=tag(ft_ner),
                                # Training
                                epochs=cfg.epochs,
                                batch_size=cfg.batch_size,
                                lr=cfg.lr,
                                optimizer="adamw",
                                accumulation_steps=cfg.accumulation_steps,
                                grad_clip=cfg.grad_clip,
                                patience=cfg.patience,
                                early_stopping=cfg.early_stopping,
                                min_lr=cfg.min_lr,
                                n_val=cfg.n_val,
                                val_metric=cfg.val_metric, # F1 on val set
                                # Evaluation
                                eval_thresholds = cfg.eval_thresholds,
                                decoding_strategies=cfg.flat_decoder_strategies,
                                # Data
                                dataset_name=tag(sup_dataset),
                                max_length=cfg.max_length,
                                max_ent_length=cfg.max_ent_length,
                                data_folder=cfg.data_folder,
                            ).tag(
                                "llm", nerc_model.ner_model.llm_name
                            ).tag(
                                "n_layers", n_layers
                            ).tag(
                                "embed", embed_dim
                            ).tag(
                                "zs_pretrain", zero_shot_pretrain
                            )
                            
                            if zero_shot_pretrain:
                                init_tasks_sup = [zero_shot_loader]
                            else:
                                init_tasks_sup = init_tasks

                            loader = sup_learner.submit(launcher=gpulauncher, init_tasks=init_tasks_sup)
                            processed_models.append({
                                "sup_learner": sup_learner,
                                "loader": loader,
                                "model": nerc_model,
                                "ner_model_hash": hash,
                            })
                            run_dir = runpath / get_alias(sup_learner)
                            if not run_dir.exists():
                                run_dir.unlink(missing_ok=True)
                                run_dir.symlink_to(sup_learner.runpath, target_is_directory=True)
                            else:
                                logging.warning(f"Run dir {run_dir} already exists, skipping symlink")

                            
    # Wait that everything finishes
    helper.xp.wait()
    logging.info("All tasks finished")
    import pandas as pd
    import json
    def save_df(df, path):
        """Save dataframe to csv, if exists, append or merge existing"""
        if path.is_file():
            try:
                old_df = pd.read_csv(path, sep=",")
            except pd.errors.EmptyDataError:
                print(f"File {path} is empty, overwriting")
                old_df = pd.DataFrame()
            if not old_df.empty:
                print(f"Merging with existing file {path}, now {len(df)} rows")
                df = pd.concat([old_df, df], ignore_index=True)
                df = df.drop_duplicates()
                print(f"After merging, {len(df)} rows")
            else:
                print(f"Existing file {path} is empty, overwriting")
        print(f"Saving results to {path}, {len(df)} rows")
        df.to_csv(path, sep=",", index=False)

    by_dataset = {
        sup_dataset: []
        for sup_dataset in cfg.sup_datasets
    }
    by_model = {
        p_model["ner_model_hash"]: []
        for p_model in processed_models
    }
    for p_model in processed_models:
        task = p_model["sup_learner"]
        ner_hash = p_model["ner_model_hash"]
        nerc_hash = task.jobpath.name
        eval_path = task.jobpath / task.result_path
        if eval_path.is_file():
            #load json
            with open(eval_path, "r") as f:
                results = json.load(f)
            print(f"Got Results for {tagspath(task)}\nat {eval_path}")
            #"add hash and tags path to evals
            for res in results["metrics"].values():
                res["ner_model_hash"] = ner_hash
                res["nerc_model_hash"] = nerc_hash
                res["tags"] = tagspath(task)
                res["sup_dataset"] = task.dataset_name
            by_dataset[task.dataset_name] += results["metrics"].values()
            by_model[ner_hash] += results["metrics"].values()
        else:
            print(f"Could not find eval file {eval_path}")

    by_dataset =  {sup_dataset: pd.DataFrame(metrics) for sup_dataset, metrics in by_dataset.items()}
    for sup_dataset, df in by_dataset.items():
        save_df(df, helper.xp.resultspath / f"results_{sup_dataset.replace(' ', '_')}.csv")

    # Save by model
    by_model =  {ner_hash: pd.DataFrame(metrics) for ner_hash, metrics in by_model.items()}
    for ner_hash, df in by_model.items():
        print(f"\n#### Results for model {ner_hash}")
        print(df)
        save_df(df, helper.xp.resultspath / f"results_model_{ner_hash}.csv")

    print(f"Done, written results by dataset and by model to {helper.xp.resultspath}")

    