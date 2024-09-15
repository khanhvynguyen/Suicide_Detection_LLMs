import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from dataset.suicide import SuicideDatasetV2
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from typing import Optional

preprocess = {None: None}


def register_preprocess_fn(name):
    def decorator(fn):
        preprocess[name] = fn
        return fn

    return decorator


def get_data():
    # read data
    label_to_int = {"indicator": 0, "ideation": 1, "behavior": 2, "attempt": 3}

    train_with_labels = pd.read_csv("data/raw_data/posts_with_labels.csv")
    train_with_labels["label"] = train_with_labels["post_risk"].map(label_to_int)
    train_without_labels = pd.read_csv("data/raw_data/posts_without_labels.csv")
    test_set = pd.read_csv("data/raw_data/test_set.csv")
    train_sets = {"train_with_labels": train_with_labels, "train_without_labels": train_without_labels}
    return train_sets, test_set


def collate_data(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    max_len = max([len(ids) for ids in input_ids])
    attention_masks = []
    encodeds = []
    for i in range(len(input_ids)):
        attention_mask = [1] * len(input_ids[i]) + [0] * (max_len - len(input_ids[i]))
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        attention_masks.append(attention_mask)
        encoded = input_ids[i] + [pad_token_id] * (max_len - len(input_ids[i]))
        encoded = torch.tensor(encoded, dtype=torch.long)
        encodeds.append(encoded)

    input_ids = torch.stack(encodeds)
    attention_mask = torch.stack(attention_masks)
    if "label" not in batch[0] and "labels" not in batch[0]:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        label_col = "label" if "label" in batch[0] else "labels"
        label = [item[label_col] for item in batch]
        label = torch.stack(label)
        return {"input_ids": input_ids, "attention_mask": attention_mask, label_col: label}


def light_clean_text(text):
    # remove all http links
    text = re.sub(r"http\S+", "url_link", text)
    # replace one or more "\n" with " "
    text = re.sub(r"\n+", " ", text)
    # replace one or more whitespaces with " "
    text = re.sub(r"\s+", " ", text)
    # replace more single quotes to 1 single quote
    text = re.sub(r"\'+", "'", text)
    # remove one or more double quotes
    text = re.sub(r"\"+", '"', text)
    return text


def prepare_data_loader(
    path: str,
    tokenizer,
    shuffle: bool,
    batch_size,
    max_length: int,
    truncate_position: str,
    preprocess_fn,
    is_train: bool,
    extra_data_paths,
    rename_to_labels=False,
    sample_extra_data: Optional[int] = None,
    workers: int = 3,
):
    dataset = SuicideDatasetV2(
        path,
        tokenizer,
        max_length=max_length,
        truncate_position=truncate_position,
        preprocess_fn=preprocess_fn,
        is_train=is_train,
        extra_data_paths=extra_data_paths,
        sample_extra_data=sample_extra_data,
        rename_to_labels=rename_to_labels,
    )
    collate_data_fn = lambda batch: collate_data(batch, tokenizer.pad_token_id)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_data_fn,
    )
    return data_loader
