import logging
from typing import List
from functools import partial
from shutil import rmtree
from pathlib import Path
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, serialize
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from llm2ner.models import MHSA_NER
from llm2ner.tasks import LearnMHSAmodel
from experiments.xp_learnTokenMatching import ProcessedModel, process_results, process_model
logging.basicConfig(level=logging.DEBUG)

N_TASKS_LIMIT = 300


@configuration
class Configuration(ConfigurationBase):
    """Configuration of the whole experiment"""


    ### Model Params
    model_name: str = "gpt2-small"
    sliding_window: int = 0
    mask_bos: bool = True
    causal_mask: bool = False
    use_cosine: bool = False
    use_pre_LN: bool = True
    use_rotary: bool = False
    normalize_scores: str = ""
    # grid search params
    layers: dict = {"from": 0, "to": 0, 'n':1}
    ranks: list = [32]
    n_heads: list = [1]


    ### Training
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
    teacher_thr_prob: float =  0.9
    """threshold for teacher probability, 0.0 means no threshold"""

    # Data
    dataset_name: str = "MultiNERD"
    max_length: int = 1000  # tokens
    max_ent_length: int = 400  # Chars
    val_limit: int = 1000
    val_metric: str = "f1"
    flat_eval_datasets: list = ["ncbi", "CoNLL 2003"]
    flat_decoder_strategies: list = ["threshold", "greedy"]
    eval_threshold: float = 0.9
    nested_eval_datasets: list = []
    llm_annotated_datasets: list = []  # datasets to evaluate with LLM annotations
    llm_annotated_threshold: float = 0.7

    # Misc
    data_folder: str = ""
    n_runs: int = 1


def run(helper: ExperimentHelper, cfg: Configuration):

    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)

    
    ###### Training + Evaluation of NER models ######
    layers = range(cfg.layers["from"], cfg.layers["to"] + 1)
    # revert the order of layers to start with the last one that takes longer to train
    layers = list(layers)[::-1]
    n_jobs = len(layers) * len(cfg.ranks) * len(cfg.n_heads) * cfg.n_runs
    
    processedModels: List[ProcessedModel] = []
    logging.info(
        f"will launch {n_jobs} jobs for {cfg.model_name} with: \n - layers {layers}\n - ranks {cfg.ranks}\n - n_heads {cfg.n_heads}\n - runs {cfg.n_runs} each"
    )
    for layer in layers:
        for rank in cfg.ranks:
            for n_head in cfg.n_heads:

                # Build model
                ner_model = MHSA_NER.C(
                    llm_name=tag(cfg.model_name),
                    layer=tag(layer),
                    rank=tag(rank),
                    n_heads=tag(n_head),
                    causal_mask=cfg.causal_mask,
                    use_pre_LN=tag(cfg.use_pre_LN),
                    use_cosine=tag(cfg.use_cosine),
                    sliding_window=tag(cfg.sliding_window),
                    mask_bos=tag(cfg.mask_bos),
                )

                print(
                    f"Will train {ner_model.__json__()} \n for {cfg.dataset_name}"
                )
                processedModels.append(
                    process_model(ner_model, LearnMHSAmodel, cfg, runpath=runpath)
                )
                
    # Wait that everything finishes
    helper.xp.wait()
    # Process results
    process_results(helper, processedModels)
