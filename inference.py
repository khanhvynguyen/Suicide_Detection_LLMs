import os

os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"

import hydra
from omegaconf import OmegaConf
import pandas as pd
from transformers import (
    AutoTokenizer,
)
import torch
from typing import Dict
import time
from datetime import datetime
from utils import DummyLogger, prepare_to_submit
from process_data import prepare_data_loader
from fine_tune import get_torch_model, load_torch_model, predict, get_checkpoint, clear_gpu_mem
from prompting import run_prompting
from collections import Counter

label_to_int = {"cannot_parse": -1, "indicator": 0, "ideation": 1, "behavior": 2, "attempt": 3}
int_to_label = {v: k for k, v in label_to_int.items()}


def inference_fine_tuned_model(
    cfg, test_path: str, eval_batch_size: int, checkpoint_criteria: str
) -> pd.DataFrame:
    checkpoint_folder = cfg["checkpoint_path"]
    config = OmegaConf.load(cfg["model_config"])
    config.model = config

    cur_time = datetime.today().strftime("%B-%d-%Y-%H-%M-%S")
    path = f"logs/{config.name}--{cur_time}.log"
    logger = DummyLogger(path)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    if config.model_max_length is not None:
        tokenizer.model_max_length = config.model_max_length

    max_length = tokenizer.model_max_length

    logger.print_and_log(f"Tokenizer input max length: {max_length}")
    logger.print_and_log(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    logger.log_metrics({"model_max_length": max_length, "vocab_size": tokenizer.vocab_size})

    num_folds = 5
    test_loader = prepare_data_loader(
        test_path,
        tokenizer=tokenizer,
        shuffle=False,
        batch_size=eval_batch_size,
        workers=3,
        max_length=max_length,
        truncate_position=config.truncate_position,
        preprocess_fn=None,
        is_train=False,
        extra_data_paths=None,
    )

    preds_folds = []
    for fold_i in range(1, num_folds + 1):
        model = get_torch_model(config)
        checkpoint_name = get_checkpoint(checkpoint_criteria, f"{checkpoint_folder}/fold_{fold_i}")
        path_i = os.path.join(checkpoint_folder, f"fold_{fold_i}/{checkpoint_name}")

        model = load_torch_model(model_path=path_i, model=model)
        logger.print_and_log(f"----- Done loading model fold {fold_i} for INFERENCE! Start predicting...")

        y_pred_probs, _ = predict(model, test_loader)
        logger.print_and_log(f"----- Done predicting fold {fold_i}!")

        preds_folds.append(y_pred_probs)
        clear_gpu_mem(verbose=True)

    final_pred_probs_all = torch.stack(preds_folds)  ## (5, num_samples, num_classes)
    preds = prepare_to_submit(final_pred_probs_all, "", False)

    return preds


def inference_prompting_model(dataset_path: str, checkpoint_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    preds = run_prompting(df, checkpoint_path)
    return preds


def most_common(x):
    x = [v for v in x.values if v != -1]
    x = Counter(x)
    x = sorted(x.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    return x[0][0]


def get_prob(row):
    probs = [0, 0, 0, 0]
    c = Counter(row)
    for i in range(4):
        probs[i] = round(c.get(i, 0) / len(row), 2)
    return probs


def get_ensemble_df(weights: Dict, df: pd.DataFrame):
    df_new = pd.DataFrame()
    all_values = list(weights.values())
    if sum(all_values) == 0:
        return df
    for col, num_repeats in weights.items():
        if num_repeats == 0:
            continue
        for i in range(num_repeats):
            df_new[col + f"___{i+1}"] = df[col]
    return df_new


def get_ensemble_result(df):
    print("ensemble from models:", df.columns)

    ## check if prompting model was doing ok:
    count_minus_1 = df["prompting"].value_counts().get(-1, 0)
    if count_minus_1 > 0:
        print(f"============== [Warning]: {count_minus_1} samples are not predicted by prompting model!")
        print(f"============== [Warning]: Something may went wrong with prompting model!!!")

    # weights = {"llama3_8B_1": 1, "llama3_8B_2": 1, "gemma2_9b_it": 2, "llama3p1_8B": 1, "prompting": 3}
    weights = {"llama3_8B_1": 1, "llama3_8B_2": 1, "gemma2_9b_it": 1, "llama3p1_8B": 1, "prompting": 2}
    df_ensemble = get_ensemble_df(weights, df)
    df["ensemble"] = df_ensemble.apply(lambda x: most_common(x), axis=1)

    df_submit = pd.DataFrame()
    df_submit["index"] = list(range(len(df["ensemble"])))
    df_submit["suicide risk"] = df["ensemble"].map(int_to_label)
    df_submit["probability distribution"] = df_ensemble.apply(get_prob, axis=1)

    return df_submit


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def run_inference(cfg) -> pd.DataFrame:
    print(OmegaConf.to_yaml(cfg))

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/detailed_outputs", exist_ok=True)

    # Load models
    print(f"Total models to load: {len(cfg.models)}")
    df = pd.DataFrame()

    for model_idx, model_name in enumerate(cfg.models.keys(), 1):
        print(f"...Running inference for model {model_idx}/{len(cfg.models)}: {model_name}")

        if model_name == "prompting":
            preds = inference_prompting_model(
                cfg["dataset_path"],
                cfg["models"][model_name]["checkpoint_path"],
            )
        else:
            preds = inference_fine_tuned_model(
                cfg["models"][model_name],
                cfg["dataset_path"],
                eval_batch_size=cfg.eval_batch_size,
                checkpoint_criteria=cfg["checkpoint_criteria"],
            )

        ## store each model's preds to file for further analysis
        model_pred_path = f"{output_dir}/detailed_outputs/{model_name}.csv"
        preds.to_csv(model_pred_path, index=False)
        # print(f"------- saved {model_name}'s preds to {model_pred_path}!")

        ## collect answers from each model
        df[model_name] = preds["suicide risk"].map(label_to_int)

    model_pred_path = f"{output_dir}/detailed_outputs/raw_ensemble_df.csv"
    df.to_csv(model_pred_path, index=False)

    ## get final ensemble prediction
    ensemble_df = get_ensemble_result(df)

    ## save final_df to file
    submission_path = f"{output_dir}/final_ensemble.xlsx"
    ensemble_df.to_excel(submission_path, index=False)
    print(f"------------------ saved final_df to {submission_path}")
    print(f"------------------ Please use {submission_path} for evaluation!")

    return ensemble_df


if __name__ == "__main__":
    start = time.time()
    run_inference()
    end = time.time()
    print(f"------------------ Done inference!!! running in {round((end-start)/3600,1)} hours!")
