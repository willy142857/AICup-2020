from collections import defaultdict
import pickle

from datasets import load_metric
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import transformers
from transformers import AdamW, AutoModelForTokenClassification, BertTokenizer, BertConfig
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from preproc import Dataset, Preprocess
from parser import get_training_args
from utils import log_model_args, timestamp, TrainDataset

transformers.logging.set_verbosity_error()

def train(model, dataloader: DataLoader, optimizer, device):
    model = model.train()

    losses = []
    acc_list = []

    with tqdm(dataloader) as t:
        t.set_description('Training')

        for data in t:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=2)
            correct_pred_count = torch.sum(preds == targets).double().item()
            acc = correct_pred_count / (targets.shape[0]*targets.shape[1])

            acc_list.append(acc)
            losses.append(loss.item())

            t.set_postfix(acc=np.mean(acc_list), loss=np.mean(losses))

            loss.backward()
            optimizer.step()

    return np.mean(acc_list), np.mean(losses)


def eval(model, dataloader: DataLoader, device, metric):
    model = model.eval()

    losses = []
    acc_list = []

    labels = pd.read_csv('../data/labels.txt', header=None)[0].values
    le = LabelEncoder()
    le.fit(labels)

    with torch.no_grad():
        with tqdm(dataloader) as t:
            t.set_description('Validation')

            for data in t:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                targets = data["targets"].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=targets)
                loss = outputs.loss
                logits = outputs.logits

                _, preds = torch.max(logits, dim=2)
                correct_pred_count = torch.sum(preds == targets).double().item()
                acc = correct_pred_count / (targets.shape[0]*targets.shape[1])

                acc_list.append(acc)
                losses.append(loss.item())
                metric.add_batch(predictions=[[l] for l in le.inverse_transform(preds.cpu().flatten())],
                                 references=[[l] for l in le.inverse_transform(targets.cpu().flatten())])

                t.set_postfix(acc=np.mean(acc_list), loss=np.mean(losses))

    final_score = metric.compute()

    return np.mean(acc_list), np.mean(losses), final_score

def get_train_and_val_data(train_path: str, val_path: str, label_path: str):
    train_data, _ = Dataset(train_path)
    val_data, _ = Dataset(val_path)

    x_train = [[pair[0] for pair in data] for data in train_data]
    x_val = [[pair[0] for pair in data] for data in val_data]
    y_train = Preprocess(train_data)
    y_val = Preprocess(val_data)

    labels = pd.read_csv(label_path, header=None)[0].values
    le = LabelEncoder()
    le.fit(labels)

    y_train = [le.transform(y) for y in y_train]
    y_val = [le.transform(y) for y in y_val]

    return x_train, x_val, y_train, y_val


if __name__ == '__main__':
    args = get_training_args()

    # prepare data
    x_train, x_val, y_train, y_val = get_train_and_val_data('../data/train_2.data', '../data/sample.data', args.labels_list)

    # generate training and validation dataloader
    max_len = args.max_length
    batch_size = args.batch_size
    num_workers = 8
    epochs = args.epochs
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    train_dataloader = DataLoader(TrainDataset(x_train, y_train, tokenizer=tokenizer, max_len=max_len),
                                  batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(TrainDataset(x_val, y_val, tokenizer=tokenizer, max_len=max_len),
                                batch_size=batch_size, num_workers=num_workers)

    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = pd.read_csv(args.labels_list, header=None)[0].values
    config = BertConfig.from_pretrained(args.pretrained_model, return_dict=True)
    config.num_labels = len(labels)
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model, config=config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    metric = load_metric('seqeval')

    history = defaultdict(list)
    best_accuracy = 0
    model_path = args.save_dir if args.save_dir else f"model/{timestamp()}.bin"

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_acc, train_loss = train(
            model, train_dataloader, optimizer, device)

        val_acc, val_loss, final_score = eval(model, val_dataloader, device, metric=metric)
        print(final_score)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_path)
            best_f1_score = final_score['overall_f1']

    # log_model_args(args.epochs, args.batch_size, args.lr,
    #                args.max_length, args.pretrained_model, model_path)
