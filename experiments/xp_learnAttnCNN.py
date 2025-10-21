import logging, os, sys
from shutil import rmtree
from pathlib import Path
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, serialize
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    get_official_model_name,
)

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug

# import task, should not be needed
# root = Path(__file__).resolve().parents[1]
# sys.path.append(str(root)) # add grandparent directory to path = repo root
# logging.info(f"Added {root} to sys.path")

from llm2ner.models import AttentionCNN_NER, CNN_METHODS, DEFAULT_KERNEL_PADDING
from llm2ner.tasks import Learn_AttentionCNN, EvalModel

N_TASKS_LIMIT = 500


@configuration
class Configuration(ConfigurationBase):
    """Configuration of the whole experiment"""

    model_name: str = "gpt2-small"
    rank: int = 64
    mask_bos: bool = True
    sliding_window: int = 0
    kernel_padding: list = DEFAULT_KERNEL_PADDING
    dataset_name: str = "MultiNERD"
    data_folder: str = ""

    # Training
    epochs: int = 5
    launcher: str = """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""
    n_val: int = 1000  # number of steps between validation and logging
    lasso_reg: float = 0.0
    lr: float = 1e-2
    batch_size: int = 32
    accumulation_steps: int = 2
    grad_clip: float = 10.0
    patience: int = 3  # lr scheduler patience
    min_lr: float = 1e-5

    # grid search params
    layers: dict = {"from": 0, "to": 0}
    dilate_entities: list = [None]
    pos_weight: list = [1.0]
    methods: list = ["inter"]

    # Data
    max_length: int = 200
    max_ent_length: int = 20
    val_limit: int = 1000
    val_metric: str = "f1"
    eval_datasets: list = ["ncbi", "CoNLL 2003"]

    # Misc
    n_runs: int = 1


def run(helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])
    logging.info(f"Launching Tasks using launcher: {gpulauncher}")

    tasks_loaders = {}
    layers = range(cfg.layers["from"], cfg.layers["to"] + 1)
    # revert the order of layers to start with the last one that takes longer to train
    layers = list(layers)[::-1]
    # compute the number of jobs to launch
    n_jobs = (
        len(layers)
        * len(cfg.methods)
        * len(cfg.dilate_entities)
        * len(cfg.pos_weight)
        * cfg.n_runs
    )
    if n_jobs > N_TASKS_LIMIT:
        logging.warning(f"Too many tasks would be launched: {n_jobs} > {N_TASKS_LIMIT}")
        logging.warning(
            f"Please reduce the number of runs or the number of layers to avoid overloading the system"
        )
        return

    # This path will contain all the tensorboard data
    runpath = (
        helper.xp.resultspath / "runs"
    )  # using pathlib.Path for cross-platform compatibility

    if runpath.is_dir():
        rmtree(runpath)
    runpath.mkdir(exist_ok=True, parents=True)

    # get llm config thanks to transformer_lens
    llm_config = convert_hf_model_config(get_official_model_name(cfg.model_name))

    logging.info(
        f"will launch jobs for {cfg.model_name} with: \n - layers {layers}\n - methods {cfg.methods}\n - dilate_entities {cfg.dilate_entities}\n - pos_weight {cfg.pos_weight} \n "
    )
    for layer in layers:
        for pos_w in cfg.pos_weight:
            for dilation in cfg.dilate_entities:
                for method in cfg.methods:
                    for run in range(cfg.n_runs):

                        # Build model
                        ner_model = AttentionCNN_NER(
                            model_dim=llm_config["d_model"],
                            rank=tag(cfg.rank),
                            causal_mask=True,  # always true for now
                            mask_bos=cfg.mask_bos,
                            sliding_window=tag(cfg.sliding_window),
                            kernel_padding=tag(cfg.kernel_padding),
                            method=tag(method),
                            layer=tag(layer),
                            llm_name=tag(cfg.model_name),
                        )

                        print(
                            f"Will train {ner_model.__json__()} \n for {cfg.dataset_name} with layer {layer}, method {method}, dilation {dilation}, pos_weight {pos_w}"
                        )
                        
                        task = Learn_AttentionCNN(
                            # Model
                            ner_model=ner_model,
                            # Loss
                            pos_weight=tag(pos_w),
                            lasso_reg=tag(cfg.lasso_reg),
                            dilate_entities=tag(dilation),
                            # Training
                            epochs=cfg.epochs,
                            lr=cfg.lr,
                            batch_size=cfg.batch_size,
                            accumulation_steps=cfg.accumulation_steps,
                            grad_clip=cfg.grad_clip,
                            patience=cfg.patience,
                            n_val=cfg.n_val,
                            # Data
                            dataset_name=tag(cfg.dataset_name),
                            max_length=cfg.max_length,
                            max_ent_length=cfg.max_ent_length,
                            val_limit=cfg.val_limit,
                            val_metric=cfg.val_metric,
                            # Misc
                            run=run,
                            # Meta
                            data_folder=cfg.data_folder,
                        )

                        # submit the task and get the model loader
                        loader = task.submit(launcher=gpulauncher)

                        #create the directory for the model
                        model_dir = task.jobpath / "model"
                        model_dir.mkdir(exist_ok=True, parents=True)

                        # save the model separately for easier loading
                        task.on_completed(
                            lambda : serialize(
                                ner_model,
                                model_dir,
                                init_tasks=[loader]
                            )
                        )

                        evaluation = EvalModel(
                            ner_model=ner_model,
                            eval_datasets=tag(cfg.eval_datasets),
                            data_folder=cfg.data_folder,
                        )

                        evaluation.submit(launcher=gpulauncher, init_tasks=[loader])

    # Wait that everything finishes
    helper.xp.wait()
