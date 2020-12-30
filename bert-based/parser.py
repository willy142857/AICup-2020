from argparse import ArgumentParser


def get_training_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--pretrained_model', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--labels_list', type=str, default='labels.txt')
    parser.add_argument('--save_dir', type=str, default='model/best_model_state.bin')
    parser.add_argument('--gpu_id', type=str, default=None)

    return parser.parse_args()


def get_testing_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--pretrained_model', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--model_path', type=str, default='best_model_state.bin')
    parser.add_argument('--labels_list', type=str, default='labels.txt')
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--output_path', type=str, default='output.tsv')
    parser.add_argument('--gpu_id', type=str, default=None)

    return parser.parse_args()