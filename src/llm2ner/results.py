import json, os, logging, time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

import llm2ner
from llm2ner import utils
from llm2ner.models.model import (
    NERmodel,
    count_perf,
    count_perf_tags,
)
from llm2ner.models.TokenMatching import (
    AttentionCNN_NER,
    TokenMatchingNER,
    CLQK_NER,
)
from experimaestro import Config, from_task_dir
from experimaestro.scheduler import JobState




def process_eval(eval_path: Path):
    """Process the Eval.json file in a given job path
    Compute the aggregated metrics and add them to the dataframe
    """

    # load json
    with open(eval_path, "r") as f:
        results = json.load(f)

    # build dataframe from result dict: one row per dataset = key
    results_df = pd.DataFrame(
        [
            {"dataset": data_name, **metrics}
            for data_name, metrics in results["metrics"].items()
        ]
    )
    # compute aggregated metrics if more than one dataset
    if len(results_df) > 1:
        results_df = add_agg_metrics(results_df)

    return results_df.round(4)


def add_agg_metrics(results_df, ignore_datasets: List[str] = []):
    """Add aggregated metrics to a results dataframe which rows are datasets
    The aggregated metrics are the weighted average of the metrics by the number of samples"""
    
    # separate ignored datasets
    if len(ignore_datasets):
        logging.info(f"Ignoring datasets: {ignore_datasets} for Mean and aggregation of metrics")
        ignored_df = results_df[results_df["dataset"].isin(ignore_datasets)]
        results_df = results_df[~results_df["dataset"].isin(ignore_datasets)]
    
    # get total samples
    tot = results_df["n_samples"].sum()

    for row in ["Aggregated", "Mean"]:
        #drop if exists
        results_df = results_df[results_df["dataset"] != row]

    # compute weighted average of f1, precision, recall
    if "Aggregated" not in results_df["dataset"].values:
        
        agg_row = {"dataset": "Aggregated", "n_samples": tot}
        for metric in ["precision", "recall"]:
            agg_row[metric] = (results_df[metric] * results_df["n_samples"] / tot).sum()
        agg_row["f1"] = (
            2
            * agg_row["precision"]
            * agg_row["recall"]
            / (agg_row["precision"] + agg_row["recall"])
        )
        # sum other columns
        for metric in results_df.columns:
            if metric not in agg_row.keys():
                agg_row[metric] = results_df[metric].sum()

        # Then add New row : raw mean of all columns

        mean_row = {"dataset": "Mean", "n_samples": tot}
        for metric in results_df.columns:
            if metric not in ["dataset", "n_samples"]:
                mean_row[metric] = results_df[metric].mean()
        results_df = pd.concat([results_df, pd.DataFrame([agg_row, mean_row])], ignore_index=True)
    else:
        logging.warning(
            "Results dataframe already contains an 'Aggregated' row, skipping aggregation"
        )
    ### Add total / total_spans performance on all rows if available
    if "total_spans" in results_df.columns and "total" in results_df.columns:
        def compute_predicted_fraction(row):
            if row["total_spans"] == 0 or row["total_spans"] is None:
                return 0.0
            return row["total"] / row["total_spans"]
        
        results_df["predicted_fraction"] = results_df.apply(compute_predicted_fraction, axis=1)
    else:
        logging.info(
            "Results dataframe does not contain 'total' and 'total_spans' columns, skipping predicted_fraction computation"
        )
    if len(ignore_datasets):
        results_df = pd.concat([results_df, ignored_df], ignore_index=True)
    return results_df


def compute_agg(df, n_samples_key="n_samples", eval_prefix="eval.", agg_prefix="agg."):
    """Compute Agg F1, recall, precision from all eval keys.
    The agg metrics are the mean of the eval metrics weighted by the number of samples.
    Thus, the dataframe must contain the number of samples for each eval key.

    Args:
        df: pd.DataFrame, dataframe containing the results
    """
    eval_keys = [key for key in df.keys() if key.startswith(eval_prefix)]
    print(f"Found {len(eval_keys)} eval keys: {eval_keys}")
    # compute the mean of each eval key

    def agg(row, key="recall"):

        tot = 0
        try:
            tot = np.sum(
                [
                    row[eval_key].get(n_samples_key, 0)
                    for eval_key in eval_keys
                    if row[eval_key]
                ]
            )
            if tot == 0:
                print(
                    f"Warning: cannot find n_samples for {row['hash']}, skipping agg computation."
                )
                return None
            agg_recall = (
                np.sum(
                    [
                        row[eval_key][n_samples_key] * row[eval_key][key]
                        for eval_key in eval_keys
                        if row[eval_key]
                    ]
                )
                / tot
            )
        except Exception as e:
            print(f"Error computing mean for row {row['hash']}: {e}")
            return None
        return agg_recall

    for m in ["precision", "recall"]:
        df[agg_prefix + m] = df.apply(lambda x: agg(x, m), axis=1)

    def agg_f1(row):
        """Compute the F1 score from precision and recall."""
        if row[agg_prefix + "precision"] is None or row[agg_prefix + "recall"] is None:
            return None
        p = row[agg_prefix + "precision"]
        r = row[agg_prefix + "recall"]
        if p + r == 0:
            return 0.0
        return (
            2 * (p * r) / (p + r + 1e-10)
        )  # Adding a small value to avoid division by zero

    df[agg_prefix + "f1"] = df.apply(lambda x: agg_f1(x), axis=1)


### EXPERIMAESTRO EXPERIMENTS UTILS ###

def get_config_dict(cfg: Config, flatten: bool = False) -> dict:
    """
    Recursively convert nested experimaestro Configs to dictionaries.
    Args:
        cfg (Config): The Config object to convert.
    Returns:
        dict: A dictionary representation of the Config object.
    """
    config = cfg.__xpm__  # ConfigInformation
    config_dict = {}

    for arg, value in config.xpmvalues():
        if isinstance(value, Config):
            # recursive call
            config_dict[arg.name] = get_config_dict(value)
        else:
            config_dict[arg.name] = value

    return flatten_dict(config_dict) if flatten else config_dict


def flatten_dict(d: dict, sep: str = ".", parent_key: str = "") -> dict:
    """
    Flatten a nested dictionary.
    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string for the current level.
        sep (str): The separator to use between keys.
    Returns:
        dict: A flattened dictionary.
    """
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, sep=sep, parent_key=k).items())
        else:
            items.append((new_key, v))
    return dict(items)


def LoadTasks(jobs_path, verbose=False) -> pd.DataFrame:
    """
    Load all tasks from a given path in a pandas dataframe indexed by the task identifier (hash)``
    It will flatten all task parameters, recursive attributes are named as "parent.child"
    """
    data = []
    if type(jobs_path) is not Path:
        try:
            jobs_path = Path(jobs_path)
        except Exception as e:
            print(f"Couldn't convert '{jobs_path}' to pathlib.Path: {e}")
            return None
    tasks = os.listdir(jobs_path)

    if verbose:
        print(f"Loading {len(tasks)} available jobs from: {str(jobs_path)}")

    for task_dir in tqdm(tasks, disable=not verbose):
        task_path = jobs_path / Path(task_dir)
        if not task_path.is_dir():
            if verbose:
                print(f"Path {task_path} is not a directory, skipping")
            continue

        try:
            params_path = task_path / "params.json"
            with open(params_path) as json_file:
                params = json.load(json_file)

            # get config dict
            cfg_dict = flatten_dict(
                params["objects"][-1]["fields"]
            )  # extract last object == the task
            cfg_dict.update(flatten_dict(params["tags"]))

            # update with hash, task_path
            cfg_dict["date"] = os.path.getmtime(params_path)
            cfg_dict["hr_date"] = time.strftime(
                "%b %d %Hh%M", time.localtime(cfg_dict["date"])
            )
            cfg_dict["hash"] = task_dir
            cfg_dict["path"] = task_path
            cfg_dict["objects"] = params["objects"]

            data.append(cfg_dict)

        except Exception as e:
            if verbose:
                print(f"Error loading task {task_dir}: {e}")
            continue

    if len(data) == 0:
        if verbose:
            print(f"No tasks found in {jobs_path}")
        return None

    df = pd.DataFrame(data)
    return df


def get_status(task_path: Path) -> str:
    """
    Get the status of a task from its path
    Args:
        task_path: path to the task
    Returns:
        status: str, status of the task
    """
    if type(task_path) is not Path:
        try:
            task_path = Path(task_path)
        except Exception as e:
            print(f"Couldn't convert '{task_path}' to pathlib.Path: {e}")
            return None

    if not task_path.is_dir():
        print(f"Path {task_path} is not a directory")
        return None

    files = os.listdir(task_path)
    # print(f"available files: {files}")
    # find .done file
    if len([f for f in files if ".err" in f]) == 0:  # not launched
        status = JobState.WAITING
    elif len([f for f in files if ".done" in f]) > 0:
        status = JobState.DONE
    else:
        # if len([f for f in files if ".failed" in f]) > 0:
        status = JobState.ERROR
    return status


def LoadAllTasks(xpm_path, verbose=False) -> dict:
    """
    Find and load all tasks from a given path
    Arguments:
        xpm_path: path to the directory containing the jobs
        verbose: if True, print progress
    Returns:
        a dictionary with the task name as key and a dataframe containing all tasks of this type
    """
    if type(xpm_path) is not Path:
        try:
            xpm_path = Path(xpm_path)
        except Exception as e:
            logging.info(f"Couldn't convert '{xpm_path}' to pathlib.Path: {e}")
            return None

    if not xpm_path.is_dir():
        logging.info(f"Path {xpm_path} is not a directory")
        return None

    if "jobs" in [path.parts[-1] for path in xpm_path.iterdir()]:
        xpm_path = xpm_path / Path("jobs")

    if "jobs" not in xpm_path.parts:
        logging.warning(f"Path {xpm_path} does not contain a jobs directory")
        return None

    job_paths = [path for path in xpm_path.iterdir() if path.is_dir()]

    if not len(job_paths):
        logging.info(f"No jobs found in {xpm_path}")
        return None
    if verbose:
        logging.info(f"Loading from all available jobs paths: {job_paths}")

    data = {}
    for job_path in job_paths:
        data[job_path.parts[-1]] = LoadTasks(job_path, verbose=verbose)

    return data


### SPECIFIC FOR LLM NER###
def loadResults(xp_path):
    """load all results from a given experimaestro experiment directory
    xp_path: pathlib.Path, path to the jobs to load.
    """
    if type(xp_path) is not Path:
        try:
            xp_path = Path(xp_path)
        except Exception as e:
            print(f"could not convert {xp_path} to pathlib.Path: {e}")
            return None

    jobs = os.listdir(xp_path)
    # print(f"available jobs: {jobs}")

    results = []
    for job in tqdm(jobs):
        jobPath = xp_path / job
        job_data = {
            "path": jobPath,
        }
        with open(jobPath / "params.json") as json_file:
            params = json.load(json_file)

        params = params["objects"][0]["fields"]
        # add params to job_data
        job_data.update(params)

        if "mode" not in job_data:
            job_data["mode"] = "full"  # default mode
        # params = params["params"]
        # print(job_data)
        hist_path = jobPath / "history.json"
        if not hist_path.exists():
            # print(f"missing history for {job}")
            pass
        else:
            try:
                with open(hist_path) as json_file:
                    history = json.load(json_file)
            except:
                print(f"error loading {hist_path}")
            job_data["history"] = history

        files = os.listdir(jobPath)

        # Find Evaluation files
        eval_files = sorted([f for f in files if "result" in f.lower()])
        # print("found eval files:", eval_files)
        if len(eval_files) == 0:
            job_data["Eval"] = None
        else:
            eval_f = str(jobPath / eval_files[-1])
            try:
                with open(eval_f) as json_file:
                    payload = json.load(json_file)
                    job_data["Eval"] = payload.get("metrics", None)
                    job_data["hist"] = payload["hist"]
            except Exception as e:
                print(f"error loading {eval_f}, {e}")
            job_data["date"] = os.path.getmtime(eval_f)

        # Find Inference files
        inf_files = [f for f in files if "inference" in f.lower()]

        if len(inf_files) == 0:
            job_data["inference"] = None
        else:
            # Sort files by modification time, most recent first
            inf_files.sort(key=lambda f: os.path.getmtime(jobPath / f), reverse=True)
            # Select the most recent file
            job_data["inference"] = jobPath / inf_files[0]

        results.append(job_data)

    return pd.DataFrame(results)


def loadResultsFromDirs(xp_paths):
    """load all results from a list of experimaestro experiment directories
    xp_paths: list of pathlib.Path, paths to the jobs to load.
    """
    results = []
    for xp_path in xp_paths:
        if type(xp_path) is not Path:
            try:
                xp_path = Path(xp_path)
            except Exception as e:
                print(f"could not convert {xp_path} to pathlib.Path: {e}")
                continue
        if not xp_path.exists():
            print(f"directory {xp_path} does not exist")
            continue
        logging.info(f"loading results from {xp_path}")
        res = loadResults(xp_path)
        if res is not None:
            # add path to results
            res["xp_path"] = xp_path
            results.append(res)
    if len(results) == 0:
        print("no results found")
        return None
    else:
        return pd.concat(results)


def getResults(df, hash) -> dict:
    """get results for a given hash
    df: pd.DataFrame, dataframe of results
    hash: str, hash of the job to get
    """
    row = df[df["path"].apply(lambda x: hash in str(x))]
    if len(row) == 0:
        print(f"no results found for {hash}")
        return None
    return row.iloc[0]


def loadCheckpoint(df, job_hash):
    """load a checkpoint from a dataframe of results
    df: pd.DataFrame, dataframe of results
    hash: str, hash of the checkpoint to load
    """
    row = getResults(df, job_hash)
    if row is None:
        return None
    path = row["path"]
    # find file with "pth" extension
    files = os.listdir(path)
    checkpoint = [f for f in files if ".pth" in f]
    if len(checkpoint) == 0:
        print(f"no checkpoint found for {job_hash}")
        return None
    checkpoint = checkpoint[0]
    return path / checkpoint


def load_from_job(jobpath, verbose=False):
    """Load a model from a job path
    Args:
        jobpath: path to the job
    """
    # load the model from the job path
    params = jobpath / "params.json"
    with open(params, "r") as f:
        params = json.load(f)
        # print(params)
    config = params["objects"][0]["fields"]
    if verbose:
        logging.info(f"Loading model from {jobpath} with fields: {config}")

    model_class_name = params["objects"][0]["type"]
    model_class = getattr(llm2ner, model_class_name)

    model_cfg: Config = model_class.C(**config)
    model = model_cfg.instance()

    # find ".pth" file in the jobpath
    files = list(jobpath.glob("*.pth"))
    if len(files) == 0:
        raise ValueError("No .pth file found in the job path")
    elif len(files) > 1:
        # sort files by modification time, most recent first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        logging.warning(
            f"Multiple .pth files found in the job path : {[model_file.name for model_file in files]} using most recent one"
        )

    model_file = files[0]
    if verbose:
        logging.info(f"loading params from {model_file}")
    # load the model state dict
    state_dict = torch.load(model_file)
    # load the model state dict into the model
    model.load_state_dict(state_dict)
    # print(model)

    return model, model_cfg

