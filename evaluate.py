import os
from pprint import pprint
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from embert import Accuracy, SimpleSembert, TopKAccuracy
from twemoji.twemoji_dataset import TwemojiData
import re
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def get_emoji_id_to_char():
    emoji_description_path = os.path.join(
        get_project_root(), "emoji_embedding/data/processed/emoji_descriptions.csv"
    )
    emoji_description = pd.read_csv(
        emoji_description_path, usecols=["emoji_id", "emoji_char"]
    )
    emoji_id_char = {
        k: v for k, v in zip(emoji_description.emoji_id, emoji_description.emoji_char)
    }
    return emoji_id_char


def print_samples(
    model,
    data,
    n_samples=32,
    seed=5,
):

    emoji_id_char = get_emoji_id_to_char()
    seed = np.random.seed(seed)
    i = np.random.randint(0, len(data) - n_samples)

    X = data[i : i + n_samples][0]
    y = data[i : i + n_samples][1]
    y_emojis = [[emoji_id_char[em] for em in row] for row in y]

    outputs = model(X, TEST_IDX)
    _, top10_emoji_ids = torch.topk(outputs, 10, dim=-1)
    top10_predictions = [
        [emoji_id_char[em.item()] for em in row] for row in top10_emoji_ids
    ]
    acc = Accuracy()
    score = acc(outputs, y)

    print(score)
    for i, sentence in enumerate(X):
        print(
            f"sentence:\n {sentence}:\n actual emojis: {y_emojis[i]}\n predicted top 10: {top10_predictions[i]}\n\n"
        )
        print("*" * 20)


def evaluate_on_dataset(model, data, k_ls=["1", "5", "10", "100"]):
    accuracy_dict = {k: 0 for k in k_ls}
    score_dict = {k: TopKAccuracy(int(k)) for k in k_ls}
    counter = 0
    with tqdm(enumerate(data)) as tbatch:
        for i, batch in tbatch:
            X = batch[0]
            y = batch[1]
            outputs = model(X, TEST_IDX)
            batch_accuracy = {k: score_dict[k](outputs, y) for k in k_ls}
            accuracy_dict = {
                k: accuracy_dict[k] + len(X) * v for k, v in batch_accuracy.items()
            }
            counter += len(X)
            running_accuracies = {k: v / counter for k, v in accuracy_dict.items()}

            tbatch.set_postfix(**running_accuracies)

    print(f"total accuracies are {running_accuracies}")
    return running_accuracies


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument(
        "model_name",
        help="Which model to use for evaluation",
    )
    argp.add_argument(
        "dataset_name",
        help="Which dataset to do evaluation on",
        choices=[
            "train",
            "train_v2",
            "train_v2_min_2",
            "valid",
            "valid_v2",
            "valid_v2_min_2",
            "test",
            "test_v2",
            "test_v2_min_2",
            "extra_zero",
            "extra_zero_v2",
            "extra_zero_v2_min_2",
            "balanced_test_v2",
        ],
    )
    argp.add_argument(
        "--run_name",
        help="Whether to only include tweets with one single emoji",
        default="main_run",
    )
    argp.add_argument(
        "--l1",
        help="Whether to only include tweets with one single emoji",
        default=False,
    )
    argp.add_argument(
        "--function",
        help="Whether to print example predictions or to do full on evaluation",
        choices=["evaluate", "samples", "compare"],
        default="evaluate",
    )
    argp.add_argument(
        "--nrows",
        help="Number of rows to load from the dataset",
        default=None,
    )
    argp.add_argument(
        "--k", help="K to specify top k prediction in evaluate", default=1
    )
    argp.add_argument(
        "--text_col", help="Column to use for evaluation", default="text_no_emojis"
    )
    argp.add_argument("--n_samples", help="n samples for samples function", default=32)
    argp.add_argument("--outputs_path", default=None)
    args = argp.parse_args()

    # load model
    model_name = args.model_name
    run_name = args.run_name
    dataset_name = args.dataset_name
    l1 = bool(args.l1) if isinstance(args.l1, str) else args.l1
    function = args.function
    nrows = int(args.nrows) if args.nrows is not None else args.nrows
    k = int(args.k)
    n_samples = int(args.n_samples)
    text_col = args.text_col

    pretrained_path = os.path.join(
        get_project_root(), f"trained_models/{run_name}/{model_name}.ckpt"
    )
    model = SimpleSembert()
    model = model.to(device)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        start_chunk = int(re.findall(r"\d+", pretrained_path.split("/")[-1])[0])
        print(f"loaded pretrained params from: {pretrained_path}")
    model.eval()
    # pprint(model)

    # load datasets
    dataset = TwemojiData(
        dataset_name, shuffle=False, batch_size=64, nrows=nrows, text_col=text_col
    )
    if l1:
        dataset.df = dataset.df.loc[dataset.df.emoji_ids.apply(len) == 1].reset_index(
            drop=True
        )

    # make calcualtions
    if function == "samples":
        print_samples(
            model, dataset, n_samples=n_samples, seed=np.random.randint(0, 200000)
        )
    elif function == "evaluate":
        total_accuracy = evaluate_on_dataset(model, dataset, k=k)

        save_path = os.path.join(get_project_root(), f"evaluation")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        s = f"Evaluation with model: {model_name} on dataset {dataset_name} with nrows: {nrows},  restricted to 1 emoji {l1}\n"
        s += f"Accuracy is {total_accuracy}\n\n"

        file_path = os.path.join(save_path, f"evaluation.txt")
        with open(file_path, "a+") as f:
            f.write(s)
    elif function == "compare":
        total_accuracy = evaluate_on_dataset(
            model, dataset, k_ls=["1", "5", "10", "100"]
        )

        save_path = os.path.join(get_project_root(), f"evaluation")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        s = f"Evaluation with model: {model_name} on dataset {dataset_name} with nrows: {nrows}, restricted to 1 emoji {l1}\n"
        s += f"Accuracy is {total_accuracy}\n\n"

        file_path = os.path.join(save_path, f"evaluation.txt")
        with open(file_path, "a+") as f:
            f.write(s)
