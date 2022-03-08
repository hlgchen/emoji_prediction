import os
from pprint import pprint
import numpy as np
import pandas as pd
import torch
import pandas as pd
from tqdm import tqdm

from embert import Accuracy, SimpleSembert, TopKAccuracy
from twemoji.twemoji_dataset import TwemojiData
import re
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))
prevalance_path = os.path.join(
    get_project_root(), "twemoji/data/twemoji_prevalence.csv"
)
TOP_EMOJIS = (
    pd.read_csv(prevalance_path)
    .sort_values(by="prevalence", ascending=False)
    .emoji_ids.tolist()
)


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


def get_outputs(model, X, restriction_type=None):
    outputs = model(X, TEST_IDX)
    if restriction_type is not None:
        if restriction_type > 0:
            excluded_emojis = TOP_EMOJIS[:restriction_type]
        else:
            excluded_emojis = TRAIN_IDX
        mask_idx = [int(i not in excluded_emojis) for i in TEST_IDX]
        mask_idx = torch.tensor([mask_idx for _ in range(len(X))])
        outputs = outputs * mask_idx
    return outputs


def print_samples(
    model,
    data,
    n_samples=32,
    seed=None,
    restricted_type=None,
):
    emoji_id_char = get_emoji_id_to_char()
    seed = np.random.seed(seed)
    i = np.random.randint(0, len(data) - n_samples)

    X = data[i : i + n_samples][0]
    y = data[i : i + n_samples][1]
    y_emojis = [[emoji_id_char[em] for em in row] for row in y]

    outputs = get_outputs(model, X, restricted_type)
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


def evaluate_on_dataset(
    model,
    data,
    k_ls=["1", "5", "10", "100"],
    restricted_type=None,
):
    accuracy_dict = {k: 0 for k in k_ls}
    score_dict = {k: TopKAccuracy(int(k)) for k in k_ls}
    counter = 0
    with tqdm(enumerate(data)) as tbatch:
        for _, batch in tbatch:
            X = batch[0]
            y = batch[1]
            outputs = get_outputs(model, X, restricted_type)
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
        choices=["samples", "compare"],
        default="compare",
    )
    argp.add_argument(
        "--nrows",
        help="Number of rows to load from the dataset",
        default=None,
    )
    argp.add_argument(
        "--text_col", help="Column to use for evaluation", default="text_no_emojis"
    )
    argp.add_argument(
        "--r",
        help="can be integer, specifies number of top emojis to disregard for evaluation. If -1 all training emojis are ignored.",
        default=None,
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
    n_samples = int(args.n_samples)
    text_col = args.text_col
    restricted_type = int(args.r) if args.r is not None else args.r

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
            model,
            dataset,
            n_samples=n_samples,
            seed=None,
            restricted_type=restricted_type,
        )
    elif function == "compare":
        total_accuracy = evaluate_on_dataset(
            model,
            dataset,
            k_ls=["1", "5", "10", "100"],
            restricted_type=restricted_type,
        )

        save_path = os.path.join(get_project_root(), f"evaluation")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        s = f"Evaluation with model: {model_name} on dataset {dataset_name} with nrows: {nrows}, restricted to 1 emoji {l1}\n"
        s += f"Accuracy is {total_accuracy}\n\n"

        file_path = os.path.join(save_path, f"evaluation.txt")
        with open(file_path, "a+") as f:
            f.write(s)
