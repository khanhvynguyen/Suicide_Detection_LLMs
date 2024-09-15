import random
import yaml
import pandas as pd
import glob
from typing import Dict, List, Union, Optional
import functools
import time
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import numpy as np
import torch
from IPython.display import display
from collections import Counter
from timm.optim import AdamP, AdamW
from collections.abc import MutableMapping
import torch
import gc
from transformers import AutoTokenizer


class DummyLogger:
    def __init__(self, path):
        self.log = open(path, "w")

    def log_hyperparams(self, metrics):
        self.log_metrics(metrics)

    def log_metrics(self, metrics, step=None):
        my_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.print_and_log(my_str)

    def print_and_log(self, print_string):
        print("{}".format(print_string))
        self.log.write("{}\n".format(print_string))
        self.log.flush()

    def __getattr__(self, _):
        # Replace lambda with a named method
        return self._do_nothing

    def _do_nothing(self, *args, **kwargs):
        return -1


def get_gpu_mem_all() -> None:
    ## get all gpu available
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        free_gb = get_gpu_mem(cuda=f"cuda:{i}")
        print(f"\tdevice: {i+1}/{n_gpus}, avail mem: {free_gb}GB")


def get_gpu_mem(cuda="cuda:0") -> str:
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    return f"{round(free_gb, 2)}/{round(total_gb, 2)}"


def clear_gpu_mem(verbose: bool = False):
    if verbose:
        print(f"mem available before clearing:")
        get_gpu_mem_all()

    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"mem available after clearing:")
        get_gpu_mem_all()


def read_yaml(file_path):
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    return data


def dict2obj(my_dict: Dict):
    class Obj:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Obj(**my_dict)


def timer_func(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_stop = time.time()
        minutes = (t_stop - t_start) / 60
        print(f'Function "{func.__name__}" executed in {minutes:.4f} minutes')
        return result

    return wrap_func


def make_cv_data(train_with_labels, n_splits=5, shuffle=True, random_state=2024, label_col="label"):

    train_with_labels = pd.read_csv("data/raw_data/posts_with_labels.csv")
    label_to_int = {"indicator": 0, "ideation": 1, "behavior": 2, "attempt": 3}

    train_with_labels["label"] = train_with_labels["post_risk"].map(label_to_int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    y = train_with_labels[label_col]
    path = "data/cv_data/"
    os.makedirs(path, exist_ok=True)

    for fold_i, (train_index, val_index) in tqdm(enumerate(skf.split(train_with_labels, y))):
        print(f"\n-------------- making fold {fold_i+1}/{n_splits}...")
        train_set_cv = train_with_labels.iloc[train_index]
        val_set_cv = train_with_labels.iloc[val_index]

        train_path = f"{path}/train_{fold_i+1}.csv"
        val_path = f"{path}/val_{fold_i+1}.csv"
        # print(",".join(map(str, val_index)))
        train_set_cv.to_csv(train_path, index=False)
        val_set_cv.to_csv(val_path, index=False)
        print(f"wrote to {train_path} and {val_path}")
    print("done!")


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


def get_machine_name():
    import socket

    machine_name = socket.gethostname()
    return machine_name


def prepare_to_submit(
    pred_probs_all: Union[torch.Tensor, np.ndarray], file_path: str, write_to_files: bool = False
):
    int_to_label = {0: "indicator", 1: "ideation", 2: "behavior", 3: "attempt"}

    pred_probs = pred_probs_all.mean(dim=0)  ## avg by probability (num_samples, num_classes)
    df = pd.DataFrame()
    df["index"] = list(range(len(pred_probs)))
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()
    # pred_probs_all: 5, num_samples, num_classes
    pred_probs_vote = torch.argmax(pred_probs_all, dim=-1)  # (5, num_samples)
    pred_probs_vote_class = (pred_probs_vote).mode(dim=0).values  ## (num_samples)

    pred_probs = [[round(pi, 3) for pi in p] for p in pred_probs]
    pred_class = [np.argmax(probs) for probs in pred_probs]
    pred_class = [int_to_label[p] for p in pred_class]
    df["suicide risk"] = pred_class
    df["probability distribution"] = pred_probs

    if write_to_files:
        df.to_excel(file_path, index=False)
    model_i_pred_class = [[pi.item() for pi in p] for p in pred_probs_vote.T]

    df["suicide risk VOTE"] = model_i_pred_class
    df["suicide risk VOTE class"] = pred_probs_vote_class.detach().cpu().numpy()
    df["suicide risk VOTE class"] = df["suicide risk VOTE class"].map(int_to_label)
    if write_to_files:
        file_path_2 = file_path.replace(".xlsx", "VOTE.csv")
        df.to_csv(file_path_2, index=False)

    display(df.sample(5))
    val_counts = df["suicide risk"].value_counts() / len(df)
    val_counts = val_counts.reindex(["indicator", "ideation", "behavior", "attempt"])
    print("pred distribution:", val_counts)

    val_counts = df["suicide risk VOTE class"].value_counts() / len(df)
    val_counts = val_counts.reindex(["indicator", "ideation", "behavior", "attempt"])
    if write_to_files:
        print(f"wrote file to {file_path}!")
        print(f"wrote file to {file_path_2}!")
    return df


def show_cv_result(all_res: Dict):
    all_res = pd.DataFrame(all_res)
    display(all_res)

    ## get mean and std for each col
    res = {}
    num_folds = len(all_res)
    for col in all_res.columns:
        mean = all_res[col].mean()
        std = all_res[col].std()
        res[col] = (mean, std)
        print(f"avg {col} across {num_folds} folder: {mean:.4f}; std: {std:.4f}")
    return res


def make_my_optimizer(opt_name: str, model_params, cfg: dict):
    cfg = cfg.copy()
    if "weight_decay_end" in cfg:
        del cfg["weight_decay_end"]
    opt_name = opt_name.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model_params, **cfg)
    elif opt_name == "adam":
        # https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay
        # https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
        optimizer = torch.optim.Adam(model_params, **cfg)
    elif opt_name == "adamw":
        optimizer = AdamW(model_params, **cfg)
    elif opt_name == "adamp":
        optimizer = AdamP(model_params, **cfg)
    else:
        raise NotImplementedError(f"Not implemented optimizer: {opt_name}")

    return optimizer


def set_seeds(seed=2024):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def load_torch_model(model_path: str, model, replace_cls_head: bool = False):
    def is_update(k):
        if replace_cls_head:
            return "lora" in k
        else:
            return "lora" in k or "modules_to_save" in k

    keys = torch.load(model_path).keys()
    print(f'epoch of the checkpoint: {torch.load(model_path).get("epoch")}')
    if "state_dict" not in keys:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path)["state_dict"]
    new_state_dict = model.state_dict().copy()
    for k, v in state_dict.items():
        if is_update(k):
            if k.startswith("model."):
                k = k[6:]
            # print(f"-- added {k}")
            new_state_dict[k] = v  ## overwrite lora values
    model.load_state_dict(new_state_dict)
    print(f"----------+=+- load_torch_model(): loaded checkpoint from {model_path}!")

    del state_dict
    del new_state_dict
    return model


def store_df(final_pred_probs_unlabeled, num_classes, name: str):
    final_pred_probs = final_pred_probs_unlabeled.detach().cpu().numpy()
    y_pred = torch.argmax(final_pred_probs_unlabeled, 1)

    df = pd.DataFrame(final_pred_probs)
    df.columns = [f"pred_{i}" for i in range(num_classes)]
    df["pred"] = y_pred.detach().cpu().numpy()
    print(f"\nvalue counts for {name} predictions")
    val_counts = df["pred"].value_counts() / len(df)
    val_counts = val_counts.sort_index()
    print(val_counts)
    return df


def get_num_tokens(models: Optional[List[str]] = None):
    if models is None:
        models = []
        models.append("hf_checkpoints/gemma-2-9b-it")
        models.append("hf_checkpoints/Meta-Llama-3-8B")
        models.append("hf_checkpoints/Meta-Llama-3.1-8B-Instruct")

    res_all = {}
    for model in models:
        print(model)
        tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        m = tokenizer.model_max_length
        m = min(m, 2500)
        print("tokenizer.model_max_length", m)

        df_label = pd.read_csv("data/raw_data/posts_with_labels.csv")
        df_no_label = pd.read_csv("data/raw_data/posts_without_labels.csv")
        df = pd.concat([df_label[["post"]], df_no_label["post"]], axis=0)
        texts = df["post"].tolist()
        res = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=1000000)
            res.append(len(inputs["input_ids"][0]))
        print(Counter(res))
        print(f"Max length: {max(res)}")
        print(f"Min length: {min(res)}")

        print(f"num of samples that exceed max length: {sum([1 for i in res if i > m])}/{len(res)}")
        res_all[model] = res
    return res_all


def get_val_path(path: str):
    ## get all file path in "val_preds"
    if "--std" in path:
        path = path.split("--std")[-1].split("--")[1:]
        path = "--".join(path)
    if "VOTE.csv" in path:
        path = path.replace("VOTE.csv", "")
    if "xlsx" in path:
        path = path.replace("xlsx", "")

    print(f"path={path}")
    all_files = glob.glob("val_preds/**", recursive=True)

    ## filter to only get files
    all_files = [f for f in all_files if path in f]
    assert len(all_files) == 1

    return all_files[0]


def generate_dummy_prob_for_baseline(preds: Optional[List | str] = None, return_df=False, n_rounds=4):
    ## generate dummy prob for dummy baseline
    int_to_label = {0: "indicator", 1: "ideation", 2: "behavior", 3: "attempt"}
    label2int = {v: k for k, v in int_to_label.items()}
    if preds is None:
        ## generate dummy predictions
        print(f"generate dummy predictions!!!")
        preds = np.random.choice(4, 100, p=[0.258, 0.380, 0.280, 0.082])
    if isinstance(preds, str):
        if "VOTE" in preds:
            preds_df = pd.read_csv(preds)
        else:
            preds_df = pd.read_excel(preds)
        preds = preds_df["suicide risk"].map(label2int)
    else:
        preds_df = None

    if isinstance(preds, List) and preds[0] in int_to_label.values():
        preds = [label2int[x] for x in preds]
    probs = []
    for p in preds:
        # generate 4 probs number
        prob_nums = np.random.rand(4)
        prob_nums /= prob_nums.sum()
        ## re-order to make the index `p` is the highest
        highest_idx = np.argmax(prob_nums)
        if highest_idx != p:
            prob_nums[highest_idx], prob_nums[p] = prob_nums[p], prob_nums[highest_idx]
        prob_nums = [round(p, n_rounds) for p in prob_nums]
        probs.append(prob_nums)

    if return_df:
        assert preds_df is not None
        preds_df["probability distribution"] = probs
        return preds_df

    return probs
