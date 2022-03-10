import os
import torch

from twemoji.twemoji_dataset import TwemojiData, TwemojiBalancedData, TwemojiDataChunks
from embert import SimpleSembert
from pprint import pprint
from tqdm import tqdm


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_model(balanced=False):
    model = SimpleSembert(dropout=0.2)
    model = model.to(device)
    if balanced:
        pretrained_path = os.path.join(
            get_project_root(),
            "trained_models/balanced_sembert_dropout/balanced_sembert_dropout_chunk106.ckpt",
        )
    else:
        pretrained_path = os.path.join(
            get_project_root(),
            "trained_models/sembert_dropout/sembert_dropout_chunk77.ckpt",
        )
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    return model


def get_prediction(prediction, topk):
    _, predcitions = torch.topk(prediction, topk, dim=-1)
    return predcitions


def get_combined_prediction(predictions1, predictions2, weighting, topk=None):
    if sum(weighting) == 1:
        predictions = weighting[0] * predictions1 + weighting[1] * predictions2
        _, combined_predictions = torch.topk(predictions, topk, dim=-1)
    else:
        _, p_emoji_ids1 = torch.topk(predictions1, weighting[0], dim=-1)
        _, p_emoji_ids2 = torch.topk(predictions2, weighting[1], dim=-1)

        combined_predictions = torch.cat([p_emoji_ids1, p_emoji_ids2], dim=1)
    return combined_predictions


def get_accuracy(p_emoji_ids, t_emoji_ids):
    accuracy = 0
    for i in range(len(p_emoji_ids)):
        y = set(t_emoji_ids[i])
        predicted_emojis = set(p_emoji_ids[i].tolist())
        accuracy += (1 / len(p_emoji_ids)) * (len(predicted_emojis.intersection(y)) > 0)
    return accuracy


if __name__ == "__main__":
    # dataset = "test_v2"
    dataset = "balanced_test_v2"

    config = {
        "weighting1": [0.5, 0.5],
        "weighting5": [3, 2],
        "weighting10": [5, 5],
        "weighting100": [50, 50],
    }

    data = TwemojiData(dataset)

    model1 = get_model()
    model2 = get_model(balanced=True)
    counter = 0

    with tqdm(enumerate(data)) as tbatch:
        for _, batch in tbatch:
            X = batch[0]
            y = batch[1]
            pred1 = model1(X, TEST_IDX)
            pred2 = model2(X, TEST_IDX)

            accuracy_dict = {}
            # average weighting
            for i in [1, 5, 10, 100]:
                p = get_combined_prediction(pred1, pred2, config["weighting1"], i)
                accuracy_dict[f"e{i}"] = get_accuracy(p, y)

                accuracy_dict[f"m1_{i}"] = get_accuracy(get_prediction(pred1, i), y)
                accuracy_dict[f"m2_{i}"] = get_accuracy(get_prediction(pred2, i), y)

            # weighting 5
            p = get_combined_prediction(pred1, pred2, config["weighting5"])
            accuracy_dict["c5"] = get_accuracy(p, y)

            # weighting 10
            p = get_combined_prediction(pred1, pred2, config["weighting10"])
            accuracy_dict["c10"] = get_accuracy(p, y)

            # weighting 100
            p = get_combined_prediction(pred1, pred2, config["weighting100"])
            accuracy_dict["c100"] = get_accuracy(p, y)

            accuracy_dict = {
                k: accuracy_dict[k] + len(X) * v for k, v in accuracy_dict.items()
            }
            counter += len(X)
            running_accuracies = {k: v / counter for k, v in accuracy_dict.items()}

            tbatch.set_postfix(**running_accuracies)

    pprint(running_accuracies)
