import os
from pprint import pprint
import numpy as np
import torch
import pandas as pd

from embert import Accuracy, SimpleEmbert
from twemoji.twemoji_dataset import TwemojiData
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def print_outputs(model, data, seed=5):
    seed = np.random.seed(seed)
    i = np.random.randint(0, len(data) - 32)

    X = data[i : i + 32][0]
    y = data[i : i + 32][1]
    y_emojis = [[emoji_id_char[em] for em in row] for row in y]

    outputs = model(X, TEST_IDX)
    top10_probas, top10_emoji_ids = torch.topk(outputs, 10, dim=-1)
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


if __name__ == "__main__":
    pretrained_path = os.path.join(
        get_project_root(), "trained_models/run1/simple_embert_chunk13.ckpt"
    )
    model = SimpleEmbert()
    model = model.to(device)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        start_chunk = int(re.findall(r"\d+", pretrained_path.split("/")[-1])[0])
        print(f"loaded pretrained params from: {pretrained_path}")
    model.eval()

    pprint(model)

    valid_data = TwemojiData("valid", shuffle=False, batch_size=64, nrows=1000)
    zero_shot_data = TwemojiData(
        "extra_zero", shuffle=False, batch_size=64, nrows=10000
    )
    zero_shot_data.df = zero_shot_data.df.loc[
        zero_shot_data.df.emoji_ids.apply(len) == 1
    ].reset_index(drop=True)

    emoji_description_path = os.path.join(
        get_project_root(), "emoji_embedding/data/processed/emoji_descriptions.csv"
    )
    emoji_description = pd.read_csv(
        emoji_description_path, usecols=["emoji_id", "emoji_char"]
    )
    emoji_id_char = {
        k: v for k, v in zip(emoji_description.emoji_id, emoji_description.emoji_char)
    }

    print_outputs(model, valid_data, seed=1)
