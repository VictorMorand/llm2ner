"""Author : Victor MORAND
"""
import logging, os, sys
from pathlib import Path
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experimaestro.launchers.slurm import SlurmLauncher


# import task
sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm2ner import LearnNERselfAttn

# logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug


# Configuration of the whole experiment
@configuration
class Configuration(ConfigurationBase):
    model_name: str = "gpt2-small"
    rank: int = 100
    dataset_name: str = "CoNLL2003"
    layers: dict = {'from':0, 'to':0}
    # Training
    epochs: int = 5
    launcher: str =  """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""
    n_val: int = 100  # number of steps between validation and logging
    lr: float = 1e-2
    batch_size: int = 32
    pos_weight: float = 1.0
    accumulation_steps: int = 2
    grad_clip: float = 10.0
    patience: int = 3 # lr scheduler patience
    # Data
    max_length: int = 200
    max_ent_length: int = 20
    # Misc
    n_runs: int = 1

def run( helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])

    logging.info(f"Launching Tasks using launcher: {gpulauncher}")

    tasks = {}
    layers = range(cfg.layers["from"],cfg.layers["to"] + 1)
    logging.info(f"will launch jobs for layers {layers} of {cfg.model_name} ")
    for layer in layers:
        for run in range(cfg.n_runs):
            task = LearnNERselfAttn(
                        model_name= tag(cfg.model_name), 
                        layer=tag(layer), 
                        rank=tag(cfg.rank),
                        # Training
                        epochs=cfg.epochs,
                        n_val=cfg.n_val,
                        lr=cfg.lr,
                        batch_size = cfg.batch_size,
                        pos_weight=cfg.pos_weight,
                        accumulation_steps=cfg.accumulation_steps,
                        grad_clip=cfg.grad_clip,
                        patience=cfg.patience,
                        # Data
                        dataset_name= tag(cfg.dataset_name), 
                        max_length=cfg.max_length,
                        max_ent_length=cfg.max_ent_length,
                        # Misc
                        run = run,
                        )
            tasks[tagspath(task)] = task.submit(launcher=gpulauncher).jobpath

    # Build a central "runs" directory to easily plot the metrics
    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    
    for key, jobath in tasks.items():
        path = (runpath / key)
        if path.exists():
            path.unlink()
        path.symlink_to(jobath)
