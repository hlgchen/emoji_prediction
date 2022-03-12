import os
import torch

from twemoji.twemoji_dataset import TwemojiData
from embert import Sembert, Baseline
import pprint
from tqdm import tqdm


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_model(balanced=False):
    model = Sembert(dropout=0.2)
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


def get_combined_prediction(prediction_ls, weighting, topk=None):
    predictions = sum([weighting[i] * pred for i, pred in enumerate(prediction_ls)])
    _, combined_predictions = torch.topk(predictions, topk, dim=-1)
    return combined_predictions


def get_accuracy(p_emoji_ids, t_emoji_ids):
    accuracy = 0
    for i in range(len(p_emoji_ids)):
        y = set(t_emoji_ids[i])
        predicted_emojis = set(p_emoji_ids[i].tolist())
        accuracy += (1 / len(p_emoji_ids)) * (len(predicted_emojis.intersection(y)) > 0)
    return accuracy


if __name__ == "__main__":
    dataset = "test_v2"
    # dataset = "balanced_test_v2"
    # dataset = "valid_v2"

    config = {"weighting1": [1 / 3, 1 / 3, 1 / 3]}

    data = TwemojiData(dataset, batch_size=16, nrows=128000)

    model1 = get_model()
    model2 = get_model(balanced=True)
    model3 = Baseline()
    counter = 0

    sum_accuracies = {}
    for i in [1, 5, 10, 100]:
        sum_accuracies[f"e{i}"] = 0
        sum_accuracies[f"m1_{i}"] = 0
        sum_accuracies[f"m2_{i}"] = 0
        sum_accuracies[f"m3_{i}"] = 0

    with tqdm(enumerate(data)) as tbatch:
        for _, batch in tbatch:
            X = batch[0]
            y = batch[1]
            pred1 = model1(X, TEST_IDX)
            pred2 = model2(X, TEST_IDX)
            pred3 = model3(X, TEST_IDX)
            pred_ls = [pred1, pred2, pred3]

            accuracy_dict = {}
            # average weighting
            for i in [1, 5, 10, 100]:
                p = get_combined_prediction(pred_ls, config["weighting1"], i)
                accuracy_dict[f"e{i}"] = get_accuracy(p, y)

                accuracy_dict[f"m1_{i}"] = get_accuracy(get_prediction(pred1, i), y)
                accuracy_dict[f"m2_{i}"] = get_accuracy(get_prediction(pred2, i), y)
                accuracy_dict[f"m3_{i}"] = get_accuracy(get_prediction(pred3, i), y)

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
