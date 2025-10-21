"""Author : Victor MORAND
"""
import logging, os, sys
from pathlib import Path
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug

# import task
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root)) # add grandparent directory to path = repo root
logging.info(f"Added {root} to sys.path")

from llm2ner.tasks import LearnNERselfAttn
from llm2ner.data import ATTN_MODES

N_TASKS_LIMIT = 500

# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    model_name: str = "gpt2-small"
    rank: int = 100
    dataset_name: str = "CoNLL2003"
    data_folder: str = ""
    
    # Training
    epochs: int = 5
    launcher: str =  """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""
    n_val: int = 1000  # number of steps between validation and logging
    lr: float = 1e-2
    batch_size: int = 32
    accumulation_steps: int = 2
    grad_clip: float = 10.0
    patience: int = 3 # lr scheduler patience
    min_lr: float = 1e-5
    
    #grid search params
    layers: dict = {'from':0, 'to':0}
    dilate_entities: list = [None]
    mode: list = ["full"] # full or last, will 
    causal_mask: list = [True]
    pos_weight: list = [1.0]

    # Data
    max_length: int = 200
    max_ent_length: int = 20
    val_limit: int = 1000
    val_metric: str = "f1"
    # Misc
    n_runs: int = 1

def run( helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])

    logging.info(f"Launching Tasks using launcher: {gpulauncher}")
    
    tasks = {}
    layers = range(cfg.layers["from"],cfg.layers["to"] + 1)
    # revert the order of layers to start with the last one that takes longer to train
    layers = list(layers)[::-1]
    # compute the number of jobs to launch
    n_jobs = len(layers) \
            * len(cfg.mode) \
            * len(cfg.dilate_entities) \
            * len(cfg.pos_weight) \
            * cfg.n_runs \
            * len(cfg.causal_mask)
    
    if n_jobs > N_TASKS_LIMIT:
        logging.warning(f"Too many tasks would be launched: {n_jobs} > {N_TASKS_LIMIT}")
        logging.warning(f"Please reduce the number of runs or the number of layers to avoid overloading the system")
        return

    logging.info(f"will launch jobs for {cfg.model_name} with: \n - layers {layers}\n - modes {cfg.mode}\n - dilate_entities {cfg.dilate_entities}\n - pos_weight {cfg.pos_weight} \n - causal {cfg.causal_mask}")
    for layer in layers:
        for causal in cfg.causal_mask:
            for pos_w in cfg.pos_weight:
                for dilation in cfg.dilate_entities:
                    for mode in cfg.mode:
                        if mode not in ATTN_MODES:
                            raise ValueError(f"mode should be in {ATTN_MODES}, not {cfg.mode}")
                        for run in range(cfg.n_runs):
                            task = LearnNERselfAttn(
                                        model_name= tag(cfg.model_name), 
                                        layer=tag(layer), 
                                        rank=tag(cfg.rank),
                                        mode=tag(mode),
                                        causal_mask=tag(causal),
                                        # Training
                                        epochs=cfg.epochs,
                                        n_val=cfg.n_val,
                                        lr=cfg.lr,
                                        batch_size = cfg.batch_size,
                                        pos_weight=pos_w,
                                        accumulation_steps=cfg.accumulation_steps,
                                        grad_clip=cfg.grad_clip,
                                        patience=cfg.patience,
                                        dilate_entities=dilation,
                                        # Data
                                        dataset_name= tag(cfg.dataset_name), 
                                        max_length=cfg.max_length,
                                        max_ent_length=cfg.max_ent_length,
                                        val_limit=cfg.val_limit,
                                        val_metric=cfg.val_metric,
                                        # Misc
                                        run = run,
                                        #Meta 
                                        data_folder=cfg.data_folder
                                        )
                            tasks[tagspath(task)] = task.submit(launcher=gpulauncher).jobpath

    # Build a central "runs" directory to easily plot the metrics
    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    
    for key, jobath in tasks.items():
        path = (runpath / key)
        if path.exists():
            # remove the old symlink
            os.remove(path)
        path.symlink_to(jobath)
