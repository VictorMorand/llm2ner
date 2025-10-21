import logging, os, sys
from shutil import rmtree
from typing import List
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, serialize
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experiments.xp_learnTokenMatching import ProcessedModel, process_results, process_model
from experiments.xp_learnCLKQner import Configuration, N_TASKS_LIMIT

logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG) # in order to set experimaestro to debug

from llm2ner.tasks import LearnTokenMatching
from llm2ner.models import AttentionLCNER


def run(helper: ExperimentHelper, cfg: Configuration):

    runpath = helper.xp.resultspath / "runs"
    runpath.mkdir(exist_ok=True, parents=True)
    # revert the order of layers to start with the last one that takes longer to train
    layers = range(cfg.layers["from"], cfg.layers["to"] + 1)
    layers = list(layers)[::-1]
    n_layers = cfg.layers["n"]

    # compute the number of jobs to launch
    n_jobs = (
        len(layers)
        * len(cfg.methods)
        * cfg.n_runs
    )
    if n_jobs > N_TASKS_LIMIT:
        logging.warning(
            f"Too many tasks would be launched: {n_jobs} > {N_TASKS_LIMIT}. "
            f"Please reduce the number of runs or the number of layers to avoid overloading the system"
        )
        return

    processedModels: List[ProcessedModel] = []

    for model_name in cfg.model_names:
        for layer in layers:
            for norm_method in cfg.methods:
                for run in range(cfg.n_runs):
                    attn_layers = list(range(layer+1))
                    if n_layers != -1:
                        attn_layers = attn_layers[-n_layers:]
                    
                    # Build model
                    ner_model = AttentionLCNER.C(
                        llm_name=tag(model_name),
                        layers=tag(attn_layers),
                        normalize_scores=tag(norm_method),
                        method="cl_fn_minmaxpool",
                        causal_mask=True,
                        sliding_window=cfg.sliding_window,
                    )
                    print(
                        f"Will train {ner_model.__json__()} \n for {cfg.dataset_name} with layer {layer} and norm_method {norm_method}"
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
