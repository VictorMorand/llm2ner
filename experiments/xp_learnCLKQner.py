import logging, os, sys
from shutil import rmtree
from typing import List
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, serialize
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experiments.xp_learnTokenMatching import (
    ProcessedModel,
    process_results,
    process_model,
)


logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug

from llm2ner.models import CLQK_NER
from llm2ner.tasks import LearnTokenMatching

N_TASKS_LIMIT = 300


@configuration
class Configuration(ConfigurationBase):
    """Configuration of the whole experiment"""

    model_names: List[str] = ["gpt2-small"]
    ranks: List[int] = [64]
    sliding_window: int = 0
    use_cosine: bool = True
    normalize_scores: str = ""
    # grid search params
    layers: dict = {"from": 0, "to": 0, "n": 1}
    methods: list = ["cl_fn_minmaxpool"]
    pos_weight: list = [0.0]
    dilate_entities: list = [[]]

    # Training
    epochs: int = 5
    launcher: str = """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""
    n_val: int = 1000
    """number of steps between validation and logging"""
    lr: float = 1e-2
    batch_size: int = 32
    accumulation_steps: int = 2
    grad_clip: float = 10.0
    patience: int = 3  # lr scheduler patience
    min_lr: float = 1e-5
    # Distillation
    self_distillation_phases: int = 0
    reset_student_weights: bool = True
    """if True, reinitialize the student weights at each distillation phase"""
    sparse_distill_loss: bool = True
    teacher_thr_prob: float = 0.9
    """threshold for teacher probability, 0.0 means no threshold"""

    # Data
    dataset_name: str = "MultiNERD"
    max_length: int = 1000  # tokens
    max_ent_length: int = 400  # Chars
    val_limit: int = 1000
    val_metric: str = "f1"
    
    # Evaluation
    eval_launcher: str = """duration=10h & cuda(mem=12G)*1 & cpu(cores=4)"""
    flat_eval_datasets: list = ["ncbi", "CoNLL 2003"]
    flat_decoder_strategies: list = ["threshold"]
    eval_threshold: float = 0.9
    nested_eval_datasets: list = []
    llm_annotated_datasets: list = []  # datasets to evaluate with LLM annotations
    llm_annotated_threshold: float = 0.7


    # LLM Annotation
    annot_names: List[str] = ["llama3.2:1b"]  # gpt-oss:20b
    annot_datasets: List[str] = []
    limit_samples: int = 10000  # 0 means no limit
    annot_launcher: str = """duration=19h & cuda(mem=25G)*1 & cpu(cores=4)"""

    # Misc
    data_folder: str = ""
    n_runs: int = 1


def run(helper: ExperimentHelper, cfg: Configuration):

    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    # revert the order of layers to start with the last one that takes longer to train
    layers = range(cfg.layers["from"], cfg.layers["to"] + 1)
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
        logging.warning(
            f"Too many tasks would be launched: {n_jobs} > {N_TASKS_LIMIT}. "
            f"Please reduce the number of runs or the number of layers to avoid overloading the system"
        )
        return

    logging.info(
        f"will launch {n_jobs} jobs for {cfg.model_names} with: \n - layers {layers}\n - methods {cfg.methods}\n - dilate_entities {cfg.dilate_entities}\n - pos_weight {cfg.pos_weight} \n "
    )
    processedModels: List[ProcessedModel] = []
    for model_name in cfg.model_names:
        for layer in layers:
            qk_layers = list(range(layer - cfg.layers["n"] + 1, layer + 1))
            for pos_w in cfg.pos_weight:
                for rank in cfg.ranks:
                    for method in cfg.methods:
                        for run in range(cfg.n_runs):

                            # Build model
                            ner_model = CLQK_NER.C(
                                llm_name=tag(model_name),
                                layers=tag(qk_layers),
                                rank=tag(rank),
                                method=method,
                                causal_mask=True,  # always true for now
                                sliding_window=cfg.sliding_window,
                                use_cosine=cfg.use_cosine,
                                normalize_scores=cfg.normalize_scores,
                            )

                            print(
                                f"Will train {ner_model.__json__()} \n for {cfg.dataset_name} with layer {layer}, method {method}, pos_weight {pos_w}"
                            )
                    processedModels.append(
                        process_model(
                            ner_model,
                            LearnTokenMatching,
                            cfg,
                            runpath=runpath,
                            learner_kwargs={
                                "dilate_entities": [],
                                "use_hookedtransformer": True
                            },
                        )
                    )

    # Wait that everything finishes
    helper.xp.wait()
    # Process results
    process_results(helper, processedModels)
