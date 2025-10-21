import torch, json, logging
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
from experimaestro import Task, Param, Constant, Meta, field, PathGenerator, Config
# Import all NER models
from llm2ner.models.TokenMatching import (
    compute_metrics,
    SelfAttention,
    AttentionCNN_NER,
    TokenMatchingNER,
    CLQK_NER,
)
from llm2ner.models.mhsa import MHSA_NER
from llm2ner.data import load_dataset_splits
from llm2ner.distillation import train_distill
import llm2ner.utils as utils

#########################  CONSTANTS #########################

METRICS_VERSION = "1.0"

#####################################################################
######################  task For Experimaestro ######################
#####################################################################


class LearnTokenMatching(Task):
    """Task that learns a NER model from LLM representations"""

    # model
    ner_model: Param[TokenMatchingNER]
    """Model to train"""
    use_hookedtransformer: Param[bool] = False
    """If True, will use HookedTransformer to load the model, otherwise standard HF transformers"""

    # Loss params
    pos_weight: Param[Optional[float]] = 0.0
    """Weight for the positive class in the BCE loss, 0 means Balanced BCE"""
    dilate_entities: Param[Optional[List[int]]] = None
    """List of dilation values to use for the entity labels, None means no dilation"""

    # Training
    epochs: Param[int] = 5
    batch_size: Param[int] = 32
    lr: Param[float] = 1e-2
    patience: Param[int] = 3  # lr scheduler patience
    min_lr: Param[float] = 5e-5
    """Minimum learning rate for the scheduler"""
    accumulation_steps: Param[int] = 2
    grad_clip: Param[float] = 1.0
    n_val: Param[int] = 3000  # number of steps between validation and logging
    val_metric: Param[str] = "recall"  # metric to use for early stopping
    # Distillation
    reset_student_weights: Param[bool] = False
    """if True, will reset the student weights to their initial values before each distillation phase"""
    self_distillation_phases: Param[int] = 0
    """if > 0, will run distillation phases with the trained model as a teacher for itself"""
    sparse_distill_loss: Param[bool] = True
    """if True, will use the sparse distillation loss, otherwise the dense one"""
    teacher_thr_prob: Param[float] = 0.9
    """Threshold probability for the teacher model in distillation phases."""

    # Data
    dataset_name: Param[str]
    val_limit: Meta[int] = 1000  # limit the validation set size, not a parameter
    max_length: Param[int] = 2000
    max_ent_length: Param[int] = 20

    # Meta params, not used to compute signature
    data_folder: Meta[str] = ""  # Folder where the data is stored, not a parameter
    """Path to the data folder"""
    runpath: Meta[Path] = field(default_factory=PathGenerator("runs"))
    """Path to store tensorboard logs"""
    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""
    # Misc
    run: Param[int] = 0  
    """Run number, used if we want to run the same task multiple times"""
    version: Constant[str] = "1.0" 
    """Version of this task: Can change if code has been updated and need to recompute"""

    def task_outputs(self, dep):
        return dep(
            TokenMatchingNER.Loader.C(model=self.ner_model, parameters=self.parameters_path)
        )

    def execute(self):
        """Called when this task is run"""

        ### Load model and dataset
        ner_model = self.ner_model  # .instance() ?

        # Load model
        logging.info(
            f"Loading llm {ner_model.llm_name} and dataset {self.dataset_name}..."
        )
        
        c_length = utils.get_model_max_length(ner_model.llm_name)
        if self.max_length > c_length:
            logging.warning(f"max_length {self.max_length} is greater than the model's max length {c_length}, setting max_length to {c_length}")
            self.max_length = c_length
        
        model = utils.load_llm(ner_model.llm_name, to_hookedtransformer=self.use_hookedtransformer, cut_to_layer=ner_model.layer).eval()
        logging.debug(f"Done ! {ner_model.llm_name} dimension is {ner_model.dim}")

        # Load dataset + Tokenize and compute token-level NER tags
        dataset = load_dataset_splits(
            dataset_name=self.dataset_name,
            data_folder=self.data_folder,
            model=model,
            val_limit=self.val_limit,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
        )

        train_dataset, val_dataset, test_dataset = (
            dataset.get("train", None),
            dataset.get("dev", None),
            dataset.get("test", None),
        )

        # freeze model
        for param in model.parameters():
            param.requires_grad = False

        # Loaders
        logging.info("Building loaders...")
        train_loader = train_dataset.get_loader(batch_size=self.batch_size)
        val_loader = val_dataset.get_loader(batch_size=self.batch_size)

        hist = ner_model.train(
            train_loader,
            val_loader,
            epochs=self.epochs,
            lr=self.lr,
            grad_clip=self.grad_clip,
            accumulation_steps=self.accumulation_steps,
            patience=self.patience,
            min_lr=self.min_lr,
            # loss
            pos_weight=self.pos_weight,
            dilate_entities=self.dilate_entities,
            # validation
            n_val=self.n_val,
            val_metric=self.val_metric,
        )
        

        # If self distillation is enabled, run it
        phase = 0
        for phase in range(self.self_distillation_phases):
            save_name = f"model_phase_{phase}.pth"
            torch.save(ner_model.state_dict(), save_name)
            ner_model.cuda()
            logging.info(f"Saved model parameters for phase {phase} to {save_name}")

            logging.info(f"Distillation Phase {phase + 1}/{self.self_distillation_phases}")
            teacher_model = deepcopy(ner_model).cuda()

            if self.reset_student_weights:
                logging.info(f"Reset student weights random initialization")
                utils.reinit_weights(ner_model)

            hist = train_distill(
                student=ner_model,  # the student model
                teacher=teacher_model,  # the teacher model
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                hist=hist,
                epochs=self.epochs,
                lr=self.lr,
                n_val=self.n_val,
                grad_clip=self.grad_clip,
                accumulation_steps=self.accumulation_steps,
                patience=self.patience,
                min_lr=self.min_lr,
                early_stopping=True,
                
                teacher_thr_prob= self.teacher_thr_prob,
                sparse=self.sparse_distill_loss,
            )
            

        # save model
        torch.save(ner_model.state_dict(), self.parameters_path)
        logging.info(f"Saved model parameters after {self.self_distillation_phases} phases")
        llm_label = ner_model.llm_name.split("/")[-1]
        del train_loader, val_loader, train_dataset, val_dataset

        if test_dataset is not None:
            ner_model.cuda()
            test_loader = test_dataset.get_loader(batch_size=30)
            # Compute metrics
            metrics = ner_model.evaluate(test_loader)
            logging.info(f"Metrics: {metrics}")

            # Save results in json file
            logging.info("Saving results...")
            jsonname = (
                f"results_{llm_label}_rank{ner_model.rank}_{self.dataset_name}.json"
            )

            with open(jsonname, "w") as f:
                json.dump(
                    {
                        "hist": hist,
                        "metrics": metrics,
                        "version": self.version,
                    },
                    f,
                )
        else:
            logging.warning("No test loader found, skipping metrics computation")
            return



class LearnMHSAmodel(Task):
    """Task that learns a NER model from LLM representations"""

    # model
    ner_model: Param[MHSA_NER]
    """Model to train"""

    # Training
    epochs: Param[int] = 5
    batch_size: Param[int] = 32
    lr: Param[float] = 1e-2
    patience: Param[int] = 3  # lr scheduler patience
    min_lr: Param[float] = 5e-5
    """Minimum learning rate for the scheduler"""
    accumulation_steps: Param[int] = 2
    grad_clip: Param[float] = 1.0
    n_val: Param[int] = 3000  # number of steps between validation and logging
    val_metric: Param[str] = "recall"  # metric to use for early stopping
    # Distillation
    self_distillation_phases: Param[int] = 0
    """if > 0, will run distillation phases with the trained model as a teacher for itself"""
    reset_student_weights: Param[bool] = False
    """if True, will reset the student weights to their initial values before each distillation phase"""
    sparse_distill_loss: Param[bool] = True
    """if True, will use the sparse distillation loss, otherwise the dense one"""
    teacher_thr_prob: Param[float] = 0.9
    """Threshold probability for the teacher model in distillation phases."""

    # Data
    dataset_name: Param[str]
    val_limit: Meta[int] = 1000  # limit the validation set size, not a parameter
    max_length: Param[int] = 2000
    max_ent_length: Param[int] = 20

    # Misc
    run: Param[int] = (
        0  # Run number, used if we want to run the same task multiple times
    )
    version: Constant[str] = (
        "1.0"  # Can change if code has been updated and need to recompute
    )

    # Meta params, not used to compute signature
    data_folder: Meta[str] = ""  # Folder where the data is stored, not a parameter
    """Path to the data folder"""
    runpath: Meta[Path] = field(default_factory=PathGenerator("runs"))
    """Path to store tensorboard logs"""
    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""

    def task_outputs(self, dep):
        return dep(
            MHSA_NER.Loader.C(model=self.ner_model, parameters=self.parameters_path)
        )

    def execute(self):
        """Called when this task is run"""

        ### Load model and dataset
        ner_model = self.ner_model

        # Load model
        logging.info(
            f"Loading llm {ner_model.llm_name} and dataset {self.dataset_name}..."
        )
        model = utils.load_llm(ner_model.llm_name, to_hookedtransformer=False, cut_to_layer=ner_model.layer).eval()
        logging.debug(f"Done ! {ner_model.llm_name} dimension is {ner_model.dim}")

        # Load dataset + Tokenize and compute token-level NER tags
        dataset = load_dataset_splits(
            dataset_name=self.dataset_name,
            data_folder=self.data_folder,
            model=model,
            val_limit=self.val_limit,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
        )

        train_dataset, val_dataset, test_dataset = (
            dataset.get("train", None),
            dataset.get("dev", None),
            dataset.get("test", None),
        )

        # freeze model
        for param in model.parameters():
            param.requires_grad = False

        # Loaders
        logging.info("Building loaders...")
        train_loader = train_dataset.get_loader(batch_size=self.batch_size)
        val_loader = val_dataset.get_loader(batch_size=self.batch_size)

        hist = ner_model.train(
            train_loader,
            val_loader,
            epochs=self.epochs,
            lr=self.lr,
            grad_clip=self.grad_clip,
            accumulation_steps=self.accumulation_steps,
            patience=self.patience,
            min_lr=self.min_lr,
            # validation
            n_val=self.n_val,
            val_metric=self.val_metric,
        )

        # If self distillation is enabled, run it
        for phase in range(self.self_distillation_phases):
            save_name = f"model_phase_{phase}.pth"
            torch.save(ner_model.cpu().state_dict(), save_name)
            logging.info(f"Saved model parameters for phase {phase} to {save_name}")
           
            logging.info(f"Distillation Phase {phase + 1}/{self.self_distillation_phases}")
            teacher_model = deepcopy(ner_model)

            if self.reset_student_weights:
                logging.info(f"Reset student weights random initialization")
                utils.reinit_weights(ner_model)

            hist += train_distill(
                student=ner_model.cuda(),  # the student model
                teacher=teacher_model.cuda(),  # the teacher model
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.epochs,
                lr=self.lr,
                n_val=self.n_val,
                grad_clip=self.grad_clip,
                accumulation_steps=self.accumulation_steps,
                patience=self.patience,
                min_lr=self.min_lr,
                early_stopping=True,
                
                teacher_thr_prob= self.teacher_thr_prob,
                sparse=self.sparse_distill_loss,
            )
           

        # save last model and send to task output
        llm_label = ner_model.llm_name.split("/")[-1]
        torch.save(ner_model.state_dict(), self.parameters_path)

        del train_loader, val_loader, train_dataset, val_dataset

        with open("hist.json", "w") as f:
            json.dump(hist, f)

        if test_dataset is not None:
            ner_model.cuda()
            test_loader = test_dataset.get_loader(batch_size=30)
            # Compute metrics
            metrics = ner_model.evaluate(test_loader)
            logging.info(f"Metrics: {metrics}")

            # Save results in json file
            logging.info("Saving results...")
            jsonname = (
                f"results_{llm_label}_rank{ner_model.rank}_{self.dataset_name}.json"
            )

            with open(jsonname, "w") as f:
                json.dump(
                    {
                        "hist": hist,
                        "metrics": metrics,
                        "version": self.version,
                    },
                    f,
                )
        else:
            logging.warning("No test loader found, skipping metrics computation")
            return


###############################  EVAL TASKS #########################


class EvalModel(Task):

    ner_model: Param[AttentionCNN_NER]
    """Model to evaluate"""

    eval_datasets: Param[List] = [None]
    """List of datasets to evaluate on"""

    data_folder: Meta[Path] = ""
    """Path to the data folder"""

    version: Constant[str] = "1.1"
    """Version of the task"""

    @torch.no_grad()
    def execute(self):
        """Called when this task is run"""

        ner_model = self.ner_model.cuda()
        logging.info(f"Loaded model {ner_model}")

        llm_name = ner_model.llm_name
        # Load model
        logging.info(f"Loading llm {llm_name}...")
        model = utils.load_llm(llm_name, to_hookedtransformer=False, cut_to_layer=ner_model.layer).eval()

        logging.info(f"Evaluating model on datasets {self.eval_datasets}")
        metrics = {}

        # Load datasets
        for dataset_name in self.eval_datasets:
            logging.info(f"Loading {dataset_name}")
            test_dataset = load_dataset_splits(
                dataset_name=dataset_name,
                model=model,
                splits=["test"],
                data_folder=self.data_folder,
                mode="last",
                max_ent_length=20,
                max_length=200,
            )["test"]

            metrics[dataset_name] = ner_model.evaluate(
                test_dataset.get_loader(batch_size=30), verbose=False
            )
            logging.info(f" --> Metrics on {dataset_name}: {metrics[dataset_name]}")

        # Save results in json file
        logging.info("Saving results...")

        with open("Eval.json", "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "version": self.version,
                },
                f,
            )
        logging.info(f"Results saved !")


### DEPRECATED TESTS ###

class LearnNERselfAttn(Task):
    """Main Task that learns a rank r Self Attention layer to perform NER from LLM representations"""

    model_name: Param[str]
    layer: Param[int]
    rank: Param[int]
    mode: Param[str] = "last"  # mode for NER patterns, full, last or block
    causal_mask: Param[bool] = True
    mask_bos: Param[bool] = True

    # Training
    epochs: Param[int] = 5
    batch_size: Param[int] = 32
    n_val: Param[int] = 100  # number of steps between validation and logging
    lr: Param[float] = 1e-2
    patience: Param[int] = 3  # lr scheduler patience
    pos_weight: Param[float] = 1.0
    dilate_entities: Param[List[int]] = [3]
    accumulation_steps: Param[int] = 2
    grad_clip: Param[float] = 1.0
    val_metric: Param[str] = "recall"  # metric to use for early stopping
    val_limit: Param[int] = 1000  # limit the validation set size

    # Data
    dataset_name: Param[str]
    max_length: Param[int] = 200
    max_ent_length: Param[int] = 20

    # Misc
    run: Param[int] = (
        0  # Run number, used if we want to run the same task multiple times
    )
    version: Constant[str] = (
        "1.0"  # Can change if code has been updated and need to recompute
    )

    # Meta params, not used to compute signature
    data_folder: Meta[str] = ""  # Folder where the data is stored, not a parameter

    def execute(self):
        """Called when this task is run"""
        ### Load model and dataset

        # Load model
        logging.info(
            f"Loading model {self.model_name} and dataset {self.dataset_name}..."
        )
        model = utils.load_llm(self.model_name)
        dim = model.QK.shape[-1]
        logging.debug(f"Done ! {self.model_name} dimension is {dim}")

        # Load dataset + Tokenize and compute token-level NER tags
        dataset = load_dataset_splits(
            dataset_name=self.dataset_name,
            data_folder=self.data_folder,
            model=model,
            mode=self.mode,
            val_limit=self.val_limit,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
        )

        train_dataset, val_dataset, test_dataset = (
            dataset["train"],
            dataset["dev"],
            dataset.get("test", None),
        )

        for param in model.parameters():  # freeze model
            param.requires_grad = False

        attn = SelfAttention(
            dim,
            k=self.rank,  # rank r attention layer
            causal_mask=self.causal_mask,
            mask_bos=self.mask_bos,
        ).cuda()

        logging.info("Building loaders...")
        train_loader = train_dataset.get_loader(batch_size=self.batch_size)
        val_loader = val_dataset.get_loader(
            batch_size=20
        )  # smaller batch size for validation to spare memory

        hist = attn.train(
            model,
            self.layer,
            train_loader,
            val_loader,
            epochs=self.epochs,
            lr=self.lr,
            grad_clip=self.grad_clip,
            pos_weight=self.pos_weight,
            accumulation_steps=self.accumulation_steps,
            patience=self.patience,
            n_val=self.n_val,
            val_metric=self.val_metric,
            dilate_entities=self.dilate_entities,
        )
        model_label = self.model_name.split("/")[-1]
        # save model
        ckptname = (
            f"Attn_{self.mode}_{model_label}_layer{self.layer}_rank{self.rank}.pth"
        )
        torch.save(attn.state_dict(), ckptname)

        del train_loader, val_loader, train_dataset, val_dataset

        # Compute metrics
        if test_dataset is None:
            logging.warning("No test loader found, skipping metrics computation")

        else:
            test_loader = test_dataset.get_loader(batch_size=30)
            metrics = compute_metrics(test_loader, model, attn, self.layer)
            logging.info(f"Metrics: {metrics}")

            # Save results in json file
            logging.info("Saving results...")
            jsonname = f"results_{model_label}_l{self.layer}_rank{self.rank}_{self.dataset_name}_{self.mode}.json"

            with open(jsonname, "w") as f:
                json.dump(
                    {
                        "hist": hist,
                        "metrics": metrics,
                        "layer": self.layer,
                        "rank": self.rank,
                        "version": self.version,
                    },
                    f,
                )


class Learn_AttentionCNN(Task):
    """Task that learns a NER model from LLM representations"""

    # model
    ner_model: Param[AttentionCNN_NER]
    """Model to train"""
    # Loss params
    pos_weight: Param[float] = 1.0
    lasso_reg: Param[float] = 0.0
    dilate_entities: Param[List[int]] = [3]
    # Training
    epochs: Param[int] = 5
    batch_size: Param[int] = 32
    lr: Param[float] = 1e-2
    patience: Param[int] = 3  # lr scheduler patience
    accumulation_steps: Param[int] = 2
    grad_clip: Param[float] = 1.0
    n_val: Param[int] = 3000  # number of steps between validation and logging
    val_metric: Param[str] = "recall"  # metric to use for early stopping

    # Data
    dataset_name: Param[str]
    val_limit: Meta[int] = 1000  # limit the validation set size, not a parameter
    max_length: Param[int] = 200
    max_ent_length: Param[int] = 20
    # Misc
    run: Param[int] = (
        0  # Run number, used if we want to run the same task multiple times
    )
    version: Constant[str] = (
        "1.0"  # Can change if code has been updated and need to recompute
    )

    # Meta params, not used to compute signature
    data_folder: Meta[str] = ""  # Folder where the data is stored, not a parameter
    """Path to the data folder"""
    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""

    def task_outputs(self, dep):
        return dep(
            AttentionCNN_NER.Loader(model=self.ner_model, parameters=self.parameters_path)
        )

    def execute(self):
        """Called when this task is run"""

        ### Load model and dataset
        ner_model = self.ner_model  # .instance() ?

        # Load model
        logging.info(
            f"Loading llm {ner_model.llm_name} and dataset {self.dataset_name}..."
        )
        model = utils.load_llm(ner_model.llm_name, to_hookedtransformer=False, cut_to_layer=ner_model.layer).eval()
        logging.debug(f"Done ! {ner_model.llm_name} dimension is {ner_model.dim}")

        # Load dataset + Tokenize and compute token-level NER tags
        dataset = load_dataset_splits(
            dataset_name=self.dataset_name,
            data_folder=self.data_folder,
            model=model,
            mode="first",
            val_limit=self.val_limit,
            max_length=self.max_length,
            max_ent_length=self.max_ent_length,
        )

        train_dataset, val_dataset, test_dataset = (
            dataset["train"],
            dataset["dev"],
            dataset.get("test", None),
        )

        # freeze model
        for param in model.parameters():
            param.requires_grad = False

        # Loaders
        logging.info("Building loaders...")
        train_loader = train_dataset.get_loader(batch_size=self.batch_size)
        val_loader = val_dataset.get_loader(batch_size=self.batch_size)

        hist = ner_model.train(
            train_loader,
            val_loader,
            layer=ner_model.layer,
            # train
            epochs=self.epochs,
            lr=self.lr,
            grad_clip=self.grad_clip,
            pos_weight=self.pos_weight,
            accumulation_steps=self.accumulation_steps,
            patience=self.patience,
            min_lr=5e-5,
            # loss
            lasso_reg=self.lasso_reg,
            dilate_entities=self.dilate_entities,
            # validation
            n_val=self.n_val,
            val_metric=self.val_metric,
        )

        model_label = ner_model.llm_name.split("/")[-1]

        # save model
        torch.save(ner_model.state_dict(), self.parameters_path)
        del train_loader, val_loader, train_dataset, val_dataset

        if test_dataset is not None:
            test_loader = test_dataset.get_loader(batch_size=30)
            # Compute metrics
            metrics = ner_model.evaluate(test_loader)
            logging.info(f"Metrics: {metrics}")

            # Save results in json file
            logging.info("Saving results...")
            jsonname = f"results_{model_label}_l{ner_model.layer}_rank{ner_model.rank}_{self.dataset_name}.json"

            with open(jsonname, "w") as f:
                json.dump(
                    {
                        "hist": hist,
                        "metrics": metrics,
                        "layer": ner_model.layer,
                        "rank": ner_model.rank,
                        "version": self.version,
                    },
                    f,
                )
        else:
            logging.warning("No test loader found, skipping metrics computation")
            return


