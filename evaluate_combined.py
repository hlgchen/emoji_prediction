import os
import torch
import pandas as pd

from twemoji.twemoji_dataset import TwemojiData, TwemojiBalancedData, TwemojiDataChunks
from recommender import EREC
import pprint
from tqdm import tqdm


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_accuracy(p_emoji_ids, t_emoji_ids):
    accuracy = 0
    for i in range(len(p_emoji_ids)):
        y = set(t_emoji_ids[i])
        predicted_emojis = p_emoji_ids[i]
        accuracy += (1 / len(p_emoji_ids)) * (len(predicted_emojis.intersection(y)) > 0)
    return accuracy


if __name__ == "__main__":
    dataset = "test_v2"
    # dataset = "balanced_test_v2"
    # dataset = "valid_v2"

    config = {
        1: (1, 0),
        5: (3, 2),
        10: (7, 3),
        100: (90, 10),
    }

    data = TwemojiData(dataset, batch_size=32, nrows=128000)

    # description_path = os.path.join(
    #     get_project_root(), "emoji_embedding/data/processed/emoji_descriptions.csv"
    # )
    # df_des = pd.read_csv(description_path, usecols=["emoji_id", "emoji_char"])
    # mapping_dict = {k: v for k, v in zip(df_des.emoji_id, df_des.emoji_char)}
    model = EREC()

    sum_accuracies = {}
    for i in [1, 5, 10, 100]:
        sum_accuracies[f"e{i}"] = 0

    counter = 0

    with tqdm(enumerate(data)) as tbatch:
        for _, batch in tbatch:
            X = batch[0]
            y = batch[1]

            e_preds, l_preds = model(X)

            accuracy_dict = dict()
            for k, dist in config.items():
                preds = [set(l[: dist[1]]) for l in l_preds]
                for j, e in enumerate(e_preds):
                    e_copy = e.copy()
                    while len(preds[j]) < k:
                        preds[j].add(e_copy.pop(0))
                accuracy_dict[f"e{k}"] = get_accuracy(preds, y)

            sum_accuracies = {
                k: sum_accuracies[k] + len(X) * v for k, v in accuracy_dict.items()
            }
            counter += len(X)
            running_accuracies = {k: v / counter for k, v in sum_accuracies.items()}

            tbatch.set_postfix(**running_accuracies)

    pprint.pprint(running_accuracies)
    save_path = os.path.join(get_project_root(), f"evaluation")
    file_path = os.path.join(save_path, f"evaluation_combined.txt")
    with open(file_path, "a+") as f:
        f.write(dataset + "\n")
        f.write(pprint.pformat(config, indent=4) + "\n")
        f.write(pprint.pformat(running_accuracies, indent=4))
        f.write("\n\n\n")
