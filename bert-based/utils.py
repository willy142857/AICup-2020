from datetime import datetime
import os

import pandas as pd
import torch
from torch.utils.data import Dataset


def log_model_args(epochs: int, batch_size: int, lr: float, max_length: int, pretrained_model: str, model_path: str):
    if not os.path.isfile('model_list.csv'):
        df = pd.DataFrame(columns=['epochs', 'batch_size', 'lr', 'max_length', 'pretrained_model', 'model_path'])
        df.to_csv('model_list.csv', index=False)
        
    df = pd.read_csv('model_list.csv')

    new_df = pd.DataFrame([[epochs, batch_size, lr, max_length, pretrained_model, model_path, None]], columns=df.columns)
    df = df.append(new_df)

    df.to_csv('model_list.csv', index=False)


def timestamp(fmt='%Y-%m-%d-%H:%M'):
    return datetime.now().strftime(fmt)


class TrainDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = [t for text in texts for t in text]
        self.targets = [t for target in targets for t in target]
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return int(len(self.texts)/self.max_len)

    def __getitem__(self, idx):
        start = self.max_len * idx
        end = self.max_len * (idx + 1)

        text = self.texts[start:end]
        targets = self.targets[start:end]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(targets, dtype=torch.long)
        }


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = [t for text in texts for t in text]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return int(len(self.texts)/self.max_len) + 1

    def __getitem__(self, idx):
        start = self.max_len * idx
        end = self.max_len * (idx + 1)

        text = self.texts[start:end]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def strQ2B(s):
    """把字串全形轉半形"""
    rstring = ""
    for uchar in s:
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        rstring += chr(u_code)
    return rstring
