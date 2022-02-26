import os
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader

from cnn import Img2Vec
from emoji_image_dataset import EmojiImageDataset


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ************************ create vision embedding ***********************


def load_emimem(path="model/emoji_image_embedding/emimem.ckpt"):
    path = os.path.join(get_project_root(), "emoji_embedding", path)
    model = Img2Vec(200, path)
    model.eval()
    model.to(device)
    return model


def get_grouping_matrix(dataset):

    df = dataset.df
    same = df.emoji_name != df.emoji_name.shift(1)
    df["group_number"] = same.cumsum() - 1

    n_groups = df.emoji_name.nunique()
    m = torch.zeros(n_groups, len(df)).to(device)

    for i, j in zip(df.group_number, df.index):
        m[i][j] = 1

    index_emoji_id_mapping = {i: e for i, e in zip(df.group_number, df.emoji_id)}
    return m, index_emoji_id_mapping


def save_data(embeddings, mapping, dataset_name):
    path = os.path.join(
        get_project_root(), "emoji_embedding/data/processed/image_embeddings"
    )
    torch.save(embeddings, os.path.join(path, f"{dataset_name}.pt"))
    with open(os.path.join(path, f"{dataset_name}_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=4)


def create_embedding(model, dataset, dataset_name):
    ls = []
    m, mapping = get_grouping_matrix(dataset)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=64)

    with torch.no_grad():
        with tqdm(data_loader) as tbatch:
            for batch in tbatch:
                batch = batch.to(device)
                ls.append(model(batch))
    embeddings = m @ torch.cat(ls, dim=0)
    save_data(embeddings, mapping, dataset_name)
    return embeddings, mapping


def create_vision_embedding():

    total_image_ds = EmojiImageDataset()
    model = load_emimem()

    all_embeddings, all_embeddings_mapping = create_embedding(
        model, total_image_ds, "all_embeddings"
    )

    # subset for training
    train_df = (
        total_image_ds.df.loc[~total_image_ds.df.zero_shot][
            ["group_number", "emoji_id"]
        ]
        .drop_duplicates()
        .sort_values(by="group_number", ascending=True)
        .reset_index(drop=True)
    )
    train_idx = train_df.group_number.tolist()
    train_embeddings = all_embeddings[train_idx]
    train_embeddings_mapping = {i: e for i, e in zip(train_df.index, train_df.emoji_id)}
    save_data(train_embeddings, train_embeddings_mapping, "train_embeddings")


# ************************ create description embedding via word vectors***********************


if __name__ == "__main__":
    create_vision_embedding()
