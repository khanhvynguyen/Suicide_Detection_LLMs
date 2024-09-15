from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import random
from typing import Optional


class SuicideDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


class SuicideDatasetV2(Dataset):

    def __init__(
        self,
        csv_file,
        tokenizer,
        max_length,
        truncate_position,
        extra_data_paths=None,
        sample_extra_data: Optional[int] = None,
        preprocess_fn=None,
        rename_to_labels=False,
        is_train: bool = False,
    ):
        """
        truncate_position: str, one of "left", "right", "middle"
        preprocess_fn: None or callable, function to preprocess text before tokenization
        """
        self.data_df = pd.read_csv(csv_file)
        self.rename_to_labels = rename_to_labels
        if rename_to_labels:
            self.data_df = self.data_df.rename(columns={"label": "labels"})
        extra_data_df = None
        self.extra_data_list = []
        if extra_data_paths is not None and is_train:
            for path, trunctate_pos in extra_data_paths:
                extra_data = {}
                tmp_df = pd.read_csv(path)
                if rename_to_labels:
                    tmp_df["labels"] = tmp_df.rename(columns={"label": "labels"})
                extra_data["data"] = tmp_df
                extra_data["trunctate_pos"] = trunctate_pos
                self.extra_data_list.append(extra_data)

        self.max_length = max_length if max_length is not None else self._longest_encoded_len(tokenizer)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.encoded_texts = []
        encoded_texts_extra = []

        self.encoded_texts = self.get_encoded_texts(self.data_df, truncate_position)
        for extra_data in self.extra_data_list:
            extra_encoded_texts = self.get_encoded_texts(extra_data["data"], extra_data["trunctate_pos"])
            encoded_texts_extra.extend(extra_encoded_texts)
            if extra_data_df is not None:
                extra_data_df = pd.concat([extra_data_df, extra_data["data"]], ignore_index=True).reset_index(
                    drop=True
                )
            else:
                extra_data_df = extra_data["data"]

        ## sample only `sample_extra_data` of encoded_texts_extra_sample
        print(f"----sample_extra_data={sample_extra_data}")
        if sample_extra_data is not None and sample_extra_data < len(encoded_texts_extra):
            print(f"----sampled {sample_extra_data} from {len(encoded_texts_extra)}")
            idx = list(range(len(encoded_texts_extra)))
            sampled_idx = random.sample(idx, sample_extra_data)
            encoded_texts_extra = [encoded_texts_extra[i] for i in sampled_idx]
            extra_data_df = extra_data_df.iloc[sampled_idx]

        self.data_df = pd.concat([self.data_df, extra_data_df], ignore_index=True).reset_index(drop=True)
        self.encoded_texts.extend(encoded_texts_extra)
        print("SuicideDatasetV2: len(self.data_df)=", len(self.data_df))
        if "label" in self.data_df.columns:
            print(f"value counts: {self.data_df['label'].value_counts()}")

    def get_encoded_texts(self, data, truncate_position):
        encoded_texts = []
        for text in data["post"]:
            encoded = self.tokenize_and_truncate(self.tokenizer, text, truncate_position)
            encoded_texts.append(encoded)
        return encoded_texts

    def tokenize_and_truncate(self, tokenizer, text, truncate_position):
        encoded = tokenizer.encode(text)
        if len(encoded) > self.max_length:
            if truncate_position == "left":
                start_ind = len(encoded) - self.max_length
                encoded = encoded[start_ind:]
            elif truncate_position == "right":
                encoded = encoded[: self.max_length]
            elif truncate_position == "middle":
                first_half = self.max_length // 2
                rest = self.max_length - first_half
                start_ind = len(encoded) - rest + 1
                encoded = encoded[:first_half] + encoded[start_ind:]
            else:
                raise ValueError(f"Invalid value for truncate_position: {truncate_position}")
        return encoded

    def __getitem__(self, index):

        encoded = self.encoded_texts[index]
        # attention_mask = self.attention_masks[index]

        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if self.rename_to_labels:
            label_col = "labels"
        else:
            label_col = "label"
        if label_col in self.data_df.columns:
            label = self.data_df.iloc[index][label_col]
            label = torch.tensor(label, dtype=torch.long)
            out = {label_col: label, "input_ids": encoded}  # , "attention_mask": attention_mask}
        else:
            out = {"input_ids": encoded}  # , "attention_mask": attention_mask}

        return out

    def __len__(self):
        return len(self.data_df)

    def _longest_encoded_len(self, tokenizer):
        max_length = 0
        for text in self.data_df["post"]:
            encoded_length = len(tokenizer.encode(text))
            max_length = max(max_length, encoded_length)
        return max_length


from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-v1")
    max_length = 512
    dataset = SuicideDatasetV2(
        "data/cv_data/train_1.csv", tokenizer=tokenizer, max_length=max_length, truncate_position="right"
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch)
