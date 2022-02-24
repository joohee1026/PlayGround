import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

ORG = [
    "Lead", "Position", "Claim", "CounterClaim", "Rebuttal", "Evidence", "Concluding Statement"
]
LABELS = [
    'O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim',
    'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal',
    'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'
]
LABELS_WEIGHT = [
    .1, .4, .2, .6, .4, .8, .6, .8, .6, .9, .7, .6, .4, .4, .2
]
LABEL2ID = {v:k for k,v in enumerate(LABELS)}
ID2LABEL = {k:v for k,v in enumerate(LABELS)}


def get_data(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    aids = data.id.unique()
    data_prep = pd.read_json(os.path.join(data_dir, "train_prep_len.json"))
    return data, aids, data_prep


def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABEL2ID[word_labels[word_idx]])
    return label_ids


def tokenize(data, tokenizer, stride, max_length, labels=True):
    encoded_data = tokenizer(
        data["text_split"].tolist(),
        is_split_into_words=True,
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )

    if labels:
        encoded_data["labels"] = []

    encoded_data["word_ids"] = []
    n = len(encoded_data["overflow_to_sample_mapping"])
    for i in range(n):
        text_idx = encoded_data["overflow_to_sample_mapping"][i]
        word_ids = encoded_data.word_ids(i)
        if labels:
            word_labels = data["entities"].iloc[text_idx]
            label_ids = get_labels(word_ids, word_labels)
            encoded_data["labels"].append(label_ids)
        encoded_data["word_ids"].append([w if w is not None else -1 for w in word_ids])

    encoded_data = {k: torch.as_tensor(v) for k, v in encoded_data.items()}
    return encoded_data


def get_loader(data, model_dir, n_batch, max_len, stride, fold, n_fold=5):
    if isinstance(fold, int):
        train = data[data[f"f{n_fold}"] != fold].reset_index(drop=True)
        valid = data[data[f"f{n_fold}"] == fold].reset_index(drop=True)
    else:
        train = data[data.f5 != 0].reset_index(drop=True)
        valid = data[data.f5 == 0].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    train_tkn = tokenize(train, tokenizer, stride, max_len)
    valid_tkn = tokenize(valid, tokenizer, stride, max_len)

    strided_train_len = len(train_tkn["input_ids"])
    strided_valid_len = len(valid_tkn["input_ids"])
    print(f"** [Total / Train / Valid] ::: [{len(data)}/{len(train)}/{len(valid)}]")
    print(f"** [Total / Train / Valid] ::: >>> [{strided_train_len+strided_valid_len}/{strided_train_len}/{strided_valid_len}]")

    train_loader = DataLoader(
        FBPDataset(train_tkn), batch_size=n_batch, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        FBPDataset(valid_tkn), batch_size=n_batch*2, num_workers=4,
        shuffle=False, pin_memory=True, drop_last=False
    )
    return train_loader, valid_loader, (train, valid)


class FBPDataset(Dataset):
    def __init__(self, encoded_data):
        self.data = encoded_data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {k: self.data[k][idx] for k in self.data.keys()}
