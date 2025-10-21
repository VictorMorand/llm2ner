import logging, os, sys, json
from shutil import rmtree
from pathlib import Path
from typing import List
from attr import dataclass
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath, serialize, Task
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher

logging.basicConfig(level=logging.DEBUG)

from llm2ner.models import NERmodel, TokenMatchingNER
from llm2ner.tasks import LearnTokenMatching
from llm2ner.classification import LearnZeroShotNER, NERCmodel
from llm2ner.baselines.annotate_api import LLMannotation
from llm2ner.utils import PathOutput
from llm2ner.results import process_eval, add_agg_metrics
from llm2ner.data import compare_inferences, InferredDataset

N_TASKS_LIMIT = 300


# def dataclass containing tasks
@dataclass
class ProcessedModel:
    ner_model: TokenMatchingNER
    id: str
    learner: Task
    loader: Task
    evals: List[Task] = []
    inferences: List[Path] = []
    llm_evals: List[Task] = []


def get_alias(task: Task):
    alias = task.__xpmtype__.name().split(".")[-1] + f"_{tagspath(task)}_{task.jobpath.name}"
    alias = alias if len(alias) < 150 else alias[:150]
    return alias


@configuration
class Configuration(ConfigurationBase):
    """Configuration of the whole experiment"""

    ## Model
    model_names: List[str] = ["gpt2-small"]
    ranks: List[int] = [64]
    sliding_window: int = 0
    normalize_scores: str = ""
    use_cosine: bool = True
    # grid search params
    layers: dict = {"from": 0, "to": 0, "n": 1}
    methods: list = []
    pos_weight: list = [0.0]
    dilate_entities: list = [[]]

    # Training
    launcher: str = """duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"""

    epochs: int = 5
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
    eval_launcher: str = """duration=10h & cuda(mem=4G)*1 & cpu(cores=4)"""
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


def process_model(
    ner_model: NERmodel,
    LearnerClass: Task,
    cfg: Configuration,
    runpath: Path,
    learner_kwargs: dict = {},
) -> ProcessedModel:
    """Process a single model: train and evaluate it
    Returns a ProcessedModel dataclass containing the model and its tasks
    """
    assert Path(cfg.data_folder).exists(), f"Data folder {cfg.data_folder} does not exist"

    task = LearnerClass.C(
        # Model
        ner_model=ner_model,
        # Training
        epochs=tag(cfg.epochs),
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        patience=cfg.patience,
        accumulation_steps=cfg.accumulation_steps,
        grad_clip=cfg.grad_clip,
        n_val=cfg.n_val,
        val_metric=cfg.val_metric,
        # Distillation
        self_distillation_phases=tag(cfg.self_distillation_phases),
        reset_student_weights=cfg.reset_student_weights,
        sparse_distill_loss=cfg.sparse_distill_loss,
        teacher_thr_prob=tag(cfg.teacher_thr_prob),
        # Data
        dataset_name=cfg.dataset_name,
        val_limit=cfg.val_limit,
        max_length=cfg.max_length,
        max_ent_length=cfg.max_ent_length,
        # Meta
        data_folder=cfg.data_folder,
        **learner_kwargs,
    )

    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])
    eval_launcher = find_launcher(cfg.eval_launcher, tags=["slurm"])

    logging.info(f"Launching Task using launcher: {gpulauncher}")
    # Submit the task and get the model loader
    loader = task.submit(launcher=gpulauncher)

    processed_model = ProcessedModel(
        ner_model=ner_model,
        id=get_alias(task),
        learner=task,
        loader=loader,
        evals=[],
        inferences=[],
        llm_evals=[],
    )

    # Symlink so we can watch all this on tensorboard
    run_dir = runpath / processed_model.id

    if not run_dir.exists():
        run_dir.unlink(missing_ok=True)
        run_dir.symlink_to(
            task.runpath,
            target_is_directory=True,
        )
    else:
        logging.warning(f"Run dir {run_dir} already exists, skipping symlink")

    # Launch Flat benchmark evaluations
    for strat in cfg.flat_decoder_strategies:
        evaluation: Task = TokenMatchingNER.Eval.C(
            ner_model=ner_model,
            eval_datasets=cfg.flat_eval_datasets,
            data_folder=PathOutput.C(path=cfg.data_folder),
            decoding_strategy=tag(strat),
            threshold=tag(cfg.eval_threshold),
        ).tag("name", "Flat")

        evaluation.submit(launcher=eval_launcher, init_tasks=[loader])

        processed_model.evals.append(evaluation)

    # Launch Nested benchmark evaluations
    if len(cfg.nested_eval_datasets) > 0:
        nested_evaluation: Task = TokenMatchingNER.Eval.C(
            ner_model=ner_model,
            eval_datasets=cfg.nested_eval_datasets,
            data_folder=PathOutput.C(path=cfg.data_folder),
            decoding_strategy=tag("threshold"),  
            threshold=tag(cfg.eval_threshold),
            write_inferences=True,
        ).tag("name", "Nested")

        nested_evaluation.submit(launcher=eval_launcher, init_tasks=[loader])

        processed_model.evals.append(nested_evaluation)

    if len(cfg.llm_annotated_datasets) > 0:
        llm_annot_evaluation: Task = TokenMatchingNER.Eval.C(
            ner_model=ner_model,
            eval_datasets=cfg.llm_annotated_datasets,
            data_folder=PathOutput.C(path=cfg.data_folder),
            decoding_strategy=tag("threshold"),
            threshold=tag(cfg.eval_threshold),
            write_inferences=True,
        ).tag("name", "LLM_Annotated")

        llm_annot_evaluation.submit(launcher=eval_launcher, init_tasks=[loader])

        processed_model.evals.append(llm_annot_evaluation)

    return processed_model


def process_results(helper: ExperimentHelper, processedModels: List[ProcessedModel]):
    """Process the results of the experiment: save models and evaluation results"""

    import pandas as pd

    # Create model directory
    model_dir = helper.xp.resultspath / "models"
    model_dir.mkdir(exist_ok=True, parents=True)

    for processed_model in processedModels:

        if not processed_model.learner.jobpath.exists() or (len(list(processed_model.learner.jobpath.glob("*.done"))) == 0):
            logging.warning(
                f"Model {tagspath(processed_model.learner)} job path {processed_model.learner.jobpath} Not trained yet: skipping"
            )
            continue
        
        ## save model
        print(f"\n\n# Processing model {tagspath(processed_model.learner)}\n")
        link_dir = model_dir / processed_model.id
        # save model in its job path
        save_dir = processed_model.learner.jobpath / "model"
        if save_dir.exists():
            rmtree(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        if link_dir.is_symlink() or link_dir.is_file():
            link_dir.unlink()
        if not link_dir.exists():
            link_dir.symlink_to(save_dir, target_is_directory=True)

        serialize(
            processed_model.ner_model, save_dir, init_tasks=[processed_model.loader]
        )
        print(f"Model saved to :\n{save_dir}")

        ## process Gold evaluation results
        
        print("Gold Evaluation results:")
        for i, eval in enumerate(processed_model.evals):
            # Process each raw evaluation and save results to csv in the model folder
            eval_name = f"Eval_{i}_{eval.tags()['name']}_{eval.tags()['decoding_strategy']}(thr={eval.threshold})"
            eval_path = eval.jobpath / eval.result_path
            if not eval_path.exists():
                logging.warning(f"skipping Eval path {eval_path} does not exist, for model {processed_model.id}")
                continue
            results_df = process_eval(eval_path)
            
            if len(processedModels) < 10: print(f"\n## Results for eval {eval_name}\n", results_df)
            
            res_file = save_dir / f"{eval_name}.csv"
            with open(res_file, "w") as f:
                f.write(results_df.to_csv(sep="\t"))
            logging.info(f"Results written to {res_file.name}")

            # if we find inferences, move them to the model folder as well
            inf_paths = eval.jobpath.glob("inferences_*.json")
            for inf_path in inf_paths:
                processed_model.inferences.append(inf_path)
                inf_save_path = save_dir / inf_path.name
                if inf_save_path.exists():
                    #remove the file or symlink
                    inf_save_path.unlink()
                # move the file
                inf_save_path.hardlink_to(inf_path)
                logging.info(f"Inference hardlinked to {inf_save_path.name}")

        # depr : we store al levals in p_model.evals
        ## process LLM evaluation results, 
        # for llmEval in processed_model.llm_evals:
        #     print(llmEval.__xpm__.job.dependencies)
        # print(f"Processing {len(processed_model.llm_evals)} LLM evals for model")
        # all_llm_results = [
        #     process_eval(llm_eval.jobpath / llm_eval.result_path)
        #     for llm_eval in processed_model.llm_evals
        # ]
        # if len(all_llm_results) > 0:
        #     llm_results_df = pd.concat(all_llm_results, ignore_index=True)
        #     llm_results_df = add_agg_metrics(llm_results_df)
        #     print("LLM Evaluation results:")
            
        #     if len(processedModels) < 10: print(llm_results_df)

        #     # write results to csv
        #     llm_res_file = save_dir / "llmEval_results.csv"

        #     with open(llm_res_file, "w") as f:
        #         f.write(llm_results_df.to_csv(sep="\t"))
        #     logging.info(f"LLM Results written to {llm_res_file}")

        #     # concatenate gold and llm results
        #     combined_results = pd.concat(
        #         [results_df, llm_results_df], ignore_index=True
        #     )
        # else:
        #     combined_results = results_df
        # concatenate all results


def run(helper: ExperimentHelper, cfg: Configuration):

    # Build result directories
    llm_eval_folder = helper.xp.resultspath / "llm_annotations"
    runpath = helper.xp.resultspath / "runs"

    for folder in [llm_eval_folder, runpath]:
        folder.mkdir(exist_ok=True, parents=True)

    ###### LLM Annotation ######
    # Launch LLM annotation task for each annotator
    llm_annot_paths = []
    llm_annot_datasets = []

    annotLauncher = find_launcher(cfg.launcher, tags=["slurm"])
    for annot in cfg.annot_names:
        for dataset in cfg.annot_datasets:

            llm_annot_task = LLMannotation.C(
                llm_name=tag(annot),
                data_name=tag(dataset),
                limit_samples=tag(cfg.limit_samples),
                max_length=cfg.max_length,
                max_ent_length=cfg.max_ent_length,
                data_folder=cfg.data_folder,
                api_url="http://localhost:11434",  # Ollama default API URL
            )
            annotated_data_path = llm_annot_task.submit(launcher=annotLauncher)

            dataFolder = llm_annot_task.res_folder_name()
            (llm_eval_folder / dataFolder).symlink_to(
                llm_annot_task.jobpath / dataFolder, target_is_directory=True
            )
            # store the output folder of annotation task
            llm_annot_paths.append(annotated_data_path)
            llm_annot_datasets.append(llm_annot_task.res_folder_name())

    ###### Training + Evaluation of NER models ######
    layers = range(cfg.layers["from"], cfg.layers["to"] + 1, cfg.layers.get("n", 1))
    # revert the order of layers to start with the last one that takes longer to train
    layers = list(layers)[::-1]
    # compute the number of jobs to launch
    n_jobs = (
        len(layers)
        * len(cfg.methods)
        * len(cfg.dilate_entities)
        * len(cfg.pos_weight)
        * len(cfg.ranks)
    )
    if n_jobs > N_TASKS_LIMIT:
        logging.warning(f"Too many tasks would be launched: {n_jobs} > {N_TASKS_LIMIT}")
        logging.warning(
            f"Please reduce the number of runs or the number of layers to avoid overloading the system"
        )
        return
    logging.info(f"Will launch {n_jobs} tasks")

    logging.info(
        f"will launch jobs for {cfg.model_names} with: \n - layers {layers}\n - methods {cfg.methods}\n - dilate_entities {cfg.dilate_entities}\n - pos_weight {cfg.pos_weight} \n "
    )

    processedModels: List[ProcessedModel] = []

    # main loop over models and hyperparameters
    for model_name in cfg.model_names:
        for layer in layers:
            for pos_w in cfg.pos_weight:
                for method in cfg.methods:
                    for rank in cfg.ranks:

                        # Build model
                        ner_model = TokenMatchingNER.C(
                            llm_name=tag(model_name),
                            layer=tag(layer),
                            rank=tag(rank),
                            method=tag(method),
                            causal_mask=True,  # always true for now
                            sliding_window=cfg.sliding_window,
                            use_cosine=cfg.use_cosine,
                            normalize_scores=cfg.normalize_scores,
                        )
                        logging.info(
                            f"Training {ner_model.__json__()} \n for {cfg.dataset_name} with layer {layer}, method {method}, pos_weight {pos_w}"
                        )

                        p_model = process_model(
                            ner_model, LearnTokenMatching, cfg, runpath=runpath, learner_kwargs={
                                "dilate_entities": [],
                            }
                        )

                        # LLM Annotation evaluation
                        for annot_task, data_name in zip(
                            llm_annot_paths, llm_annot_datasets
                        ):
                            llm_evaluation: Task = TokenMatchingNER.Eval.C(
                                ner_model=ner_model,
                                eval_datasets=tag([data_name]),
                                data_folder=annot_task,  # parent folder of annot task is the data folder
                                decoding_strategy="threshold",  # for now only threshold makes sense
                                threshold=tag(cfg.eval_threshold),
                            )

                            llm_evaluation.submit(
                                launcher=find_launcher(
                                    cfg.launcher, tags=["slurm"]
                                ),
                                init_tasks=[p_model.loader],
                            )
                            p_model.llm_evals.append(llm_evaluation)

                        # Store tasks for later processing
                        processedModels.append(p_model)

    # Wait that everything finishes
    helper.xp.wait()
    import pandas as pd
    # Process results
    process_results(helper, processedModels)

    print("Finished training and evaluating models: ")
    for processed_model in processedModels:
        print(f" - {processed_model.id} : {processed_model.learner.__identifier__()}")
    
    ### Compare inferences if they exist
    inferences = {}
    for processed_model in processedModels:
        for inf_path in processed_model.inferences:
            if inf_path.name not in inferences.keys(): 
                inferences[inf_path.name] = []
            inferences[inf_path.name].append(
                {"model_id": processed_model.id, "path": str(inf_path)}
            )
    print(f"Found {len(inferences)} different inference files to compare")

    #save the df of inferences, ensuring Paths are converted to strings
    inf_df = pd.DataFrame.from_dict(inferences, orient="index")
    inf_df_file = helper.xp.resultspath / "inferences_summary.json"
    with open(inf_df_file, "w") as f:
        json.dump(inferences, f, indent=4)
    
    for inf_name in inferences.keys():
        if len(inferences[inf_name]) < 2:
            print(f"Only one inference for {inf_name}, skipping comparison")
        else:
            models = [inf["model_id"] for inf in inferences[inf_name]]
            paths = [inf["path"] for inf in inferences[inf_name]]

            print(f"Comparing {len(paths)} inferences for {inf_name}...")
            
            if len(paths) > 20:
                print(f"Too many inferences to compare ({len(paths)}), only 20 allowed written df to filter and process in {inf_df_file}")
                continue

            comp = compare_inferences(
                [InferredDataset.from_json(path) for path in paths]
            )
            comp["models"] = models
            comp["layers"] = [m.ner_model.layer for m in processedModels if m.id in models]
            comp["paths"] = [str(p) for p in paths]

            comp_file = helper.xp.resultspath / f"Comparison_{inf_name}"
            with open(comp_file, "w") as f:
                json.dump(comp, f, indent=4)
            print(f"Comparison written to {comp_file}")
