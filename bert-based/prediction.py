import pandas as pd
from sklearn.preprocessing import LabelEncoder
import transformers
from transformers import AutoModelForTokenClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from parser import get_testing_args
from preproc import get_testing_data
from utils import TestDataset

transformers.logging.set_verbosity_error()


def get_predictions(model, dataloader: DataLoader, device):
    model = model.eval()

    predictions = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs.logits, dim=2)

            predictions.extend(preds)

    predictions = torch.stack(predictions).cpu()

    return predictions


if __name__ == '__main__':
    args = get_testing_args()

    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_model, return_dict=True, num_labels=37)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    data = get_testing_data(args.test_file)
    max_len = args.max_length
    batch_size = args.batch_size
    num_workers = 8
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    predictions = []
    for d in tqdm(data):
        test_dataloader = DataLoader(TestDataset(d, tokenizer=tokenizer, max_len=max_len),
                                     batch_size=batch_size, num_workers=num_workers)

        pred = get_predictions(model, test_dataloader, device)
        predictions.append(pred)


    labels = pd.read_csv(args.labels_list, header=None)[0].values
    le = LabelEncoder()
    le.fit(labels)

    predictions = [le.inverse_transform(pred.flatten())
                   for pred in predictions]

    # output data
    # the file must be formatted as *.tsv and has 5 columns
    # --------------------------------------------------------------------------
    # | article_id | start_position | end_position | entity_text | entity_type |
    # |            |                |              |             |             |
    # --------------------------------------------------------------------------
    output = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id in range(len(predictions)):
        pos = 0
        start_pos = None
        end_pos = None
        entity_text = None
        entity_type = None
        for pred_id in range(len(data[test_id])):
            if predictions[test_id][pred_id][0] == 'B':
                start_pos = pos
                entity_type = predictions[test_id][pred_id][2:]
            elif start_pos is not None and predictions[test_id][pred_id][0] == 'I' and predictions[test_id][pred_id+1][0] == 'O':
                end_pos = pos
                entity_text = ''.join([data[test_id][position][0]
                                       for position in range(start_pos, end_pos+1)])

                tokens = [str(test_id), str(start_pos), str(
                    end_pos+1), entity_text, entity_type]
                line = '\t'.join(tokens)
                output += line + '\n'

                start_pos = None
            pos += 1

    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write(output)
