import pandas as pd
import torch
import os

from twemoji.twemoji_dataset import TwemojiData
from embert import Sembert, LiteralModel, Baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def get_model(balanced=False):
    model = Sembert(dropout=0.2)
    model = model.to(device)
    if balanced:
        pretrained_path = "trained_models/balanced_sembert_dropout/balanced_sembert_dropout_chunk106.ckpt"
    else:
        pretrained_path = "trained_models/sembert_dropout/sembert_dropout_chunk77.ckpt"
    pretrained_path = os.path.join(get_project_root(), pretrained_path)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    return model


def get_wrongly_classified(model, data):
    counter = 0
    k = 5
    counter_ls = []
    true_ls = []
    pred_ls = []
    for j, batch in enumerate(data):
        X = batch[0]
        y = batch[1]
        predictions = e_model1(X, TEST_IDX)
        _, top_pred = torch.topk(predictions, k)
        for i in range(len(X)):
            target = set(y[i])
            pred = set(top_pred[i].tolist())
            if len(target.intersection(pred)) == 0:
                target_emojis = [emoji_id_char[e] for e in target]
                pred_emojis = [emoji_id_char[e] for e in pred]
                counter_ls.append(counter)
                true_ls.append(target_emojis)
                pred_ls.append(pred_emojis)
            counter += 1
    return pd.DataFrame({"idx": counter_ls, "target": true_ls, "pred": pred_ls})


def get_topk_pred(model, X, k, batch_size=32):
    X_ls = [X[i : i + batch_size] for i in range(0, len(X), batch_size)]

    pred = []
    for i, X in enumerate(X_ls):
        predictions = model(X, TEST_IDX)
        _, topk_emoji_ids = torch.topk(predictions, k, dim=-1)

        pred += [set(p) for p in topk_emoji_ids.tolist()]
    return pred


if __name__ == "__main__":

    e_model1 = get_model(balanced=False)
    e_model2 = get_model(balanced=True)

    l_model1 = Baseline()
    l_model2 = LiteralModel()

    description_path = os.path.join(
        get_project_root(), "emoji_embedding/data/processed/emoji_descriptions.csv"
    )
    df_des = pd.read_csv(description_path)
    emoji_id_char = {k: v for k, v in zip(df_des.emoji_id, df_des.emoji_char)}

    data = TwemojiData("balanced_valid_v2")

    print("wrong_e_model1")
    wrong_e_model1 = get_wrongly_classified(e_model1, data)
    wrong_e_model1.to_pickle("wrong_e_model1.pkl")

    print("wrong_e_model2")
    wrong_e_model2 = get_wrongly_classified(e_model2, data)
    wrong_e_model2.to_pickle("wrong_e_model2.pkl")

    print("wrong_l_model1")
    wrong_l_model1 = get_wrongly_classified(l_model1, data)
    wrong_l_model1.to_pickle("wrong_l_model1.pkl")

    print("wrong_l_model2")
    wrong_l_model2 = get_wrongly_classified(l_model2, data)
    wrong_l_model2.to_pickle("wrong_l_model2.pkl")

    print("literal_df")
    literal_df = df_des[["emoji_id", "emjpd_emoji_name_og"]].copy()
    literal_df["like"] = "I really like " + df_des.emjpd_emoji_name_og
    literal_df["anger"] = "I am angered by " + df_des.emjpd_emoji_name_og
    literal_df["lit"] = df_des.emjpd_emoji_name_og + "is very lit, I wanna see it"

    for column in ["like", "anger", "lit"]:
        print(column)
        literal_df[f"e_model1_{column}"] = get_topk_pred(
            e_model1, literal_df[column].tolist(), 5
        )
        literal_df[f"e_model2_{column}"] = get_topk_pred(
            e_model2, literal_df[column].tolist(), 5
        )
        literal_df[f"l_model1_{column}"] = get_topk_pred(
            l_model1, literal_df[column].tolist(), 5
        )
        literal_df[f"l_model2_{column}"] = get_topk_pred(
            l_model2, literal_df[column].tolist(), 5
        )

    literal_df.to_pickle("literal_df.pkl")
