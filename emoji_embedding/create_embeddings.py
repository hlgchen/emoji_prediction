import os
from tqdm import tqdm
import re
import ast

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import gensim.downloader as api

from ee_model import Img2Vec
from emoji_image_dataset import EmojiImageDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_data(embeddings, dataset_name):
    path = os.path.join(get_project_root(), "emoji_embedding/data/processed")
    path = os.path.join(path, f"{dataset_name}.pt")
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(embeddings, path)


def load_and_process_description():
    """Loads desciption emoji and processes emjpd_aliases column by concatenating the
    aliases to a string separated by whitespace"""
    path = "emoji_embedding/data/processed/emoji_descriptions.csv"
    path = os.path.join(get_project_root(), path)
    df = pd.read_csv(path)
    df.emjpd_aliases = df.emjpd_aliases.apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df.emjpd_aliases = df.emjpd_aliases.apply(lambda x: " ".join(x))
    return df


# ************************ create vision embedding ***********************


def load_emimem(path="model/emoji_image_embedding/emimem.ckpt"):
    path = os.path.join(get_project_root(), "emoji_embedding", path)
    model = Img2Vec(200, path)
    model.eval()
    model.to(device)
    return model


def get_grouping_matrix(dataset):

    """Creates a matrix (n_groups, n_images) that when multiplied
    with the embedding matrix of all images, averages the embeddings within one
    group.
    """

    df = dataset.df
    same = df.emoji_name != df.emoji_name.shift(1)
    df["group_number"] = same.cumsum() - 1

    n_groups = df.emoji_name.nunique()
    m = torch.zeros(n_groups, len(df)).to(device)

    for i, j in zip(df.group_number, df.index):
        m[i][j] = 1
    m = torch.divide(m, m.sum(dim=1).unsqueeze(1))
    return m


def create_embedding(model, dataset, dataset_name):
    """Iterates through the dataset in order and calculates the embeddings
    for each image. Averages the embeddings within each group.
    Saves embeddings in specified location.
    """
    ls = []
    m = get_grouping_matrix(dataset)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=64)

    with torch.no_grad():
        with tqdm(data_loader) as tbatch:
            for batch in tbatch:
                batch = batch.to(device)
                ls.append(model(batch))
    embeddings = m @ torch.cat(ls, dim=0)
    save_data(embeddings, dataset_name)
    return embeddings


def create_vision_embedding():
    """Creates image embeddings for all emojis in emoji_descriptions.
    This is saved on image_embeddings/all_embeddings.pt
    Another file is saved containing only the embeddings of emojis
    that are known during training of the vision model (not zeroshot set).

    The embeddings are in the same order as emojis in emoji_descriptions.csv or
    keys.csv. This means that the first embedding in embedding_files
    correspond to the first emoji in the csv files.
    """

    print("creating vision embeddings")

    total_image_ds = EmojiImageDataset()
    model = load_emimem()

    create_embedding(model, total_image_ds, "image_embeddings/all_embeddings")


# ************************ create description embedding via word vectors***********************


def load_embedding_model(model):
    """Load GloVe Vectors from Gensim
    Params:
        - model {str}: string specifying the model, possibilities include:
            - glove-wiki-gigaword-200
            - glove-twitter-200
            - word2vec-google-news-300
            - glove-wiki-gigaword-300
    Return:
        - wv_from_bin {gensim.models.keyedvectors.KeyedVectors}: Embeddings of all words
    """
    wv_from_bin = api.load(model)
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


def get_vector_wrapper(emb, emb_vocabulary, default):
    """returns get vector function with given variables"""

    def get_vector(text):
        """Given a string or a list of words (in lower case)
        the average word vector for all words that can be found
        in the embedding vocabulary is returned. Words that can't be found
        are not included in the average calculation. If no word is in the
        embedding vocaublary the default (vector of 0s) is returned.

        Returns: {np.array}
        """
        words = []
        if isinstance(text, str) and len(text) > 0:
            # Returns list of all words (and numbers) in a given text.
            # Special characters and punctuation are ignored.
            words = re.compile("\w+").findall(text)
        elif isinstance(text, list):
            words = text
        words = [w for w in words if w in emb_vocabulary]
        if len(words) > 0:
            embeddings = [emb.get_vector(w) for w in words]
            return np.mean(embeddings, axis=0)
        else:
            return default

    return get_vector


def get_word_vector_embedding(weighting, df, get_vector):
    """Returns average word embedding for emojis given weighting rule. For
    each column specified in weighting the average embedding will be calculated.
    The embeddings of the columns are averaged with the specified weights
    (weights don't have to sum to 1).

    Params:
        - weighting {dictionary}: dictionary containing weights for columns that are to
                    be considered in the weighted average calculation of word embeddings
        - df {pd.DataFrame}: Dataframe with different emojis in each row and their descriptions,
                    in particular it has the columns specified in weighting. The descriptions are taken
                    for the word embedding average calculation.
        - get_vector {callable}: function that return the embedding vector given a text

    Returns:
        - {pd.Series}: pandas Series of np.arrays that contain the word embeddings for the emojis
    """
    result = []
    total_weights = []
    for col, weight in weighting.items():
        vectors = df[col].apply(get_vector)
        addition = weight * vectors
        result.append(addition)

        indictaion = weight * (vectors.apply(sum) != 0)
        total_weights.append(indictaion)
    result = pd.concat(result, axis=1).sum(axis=1)
    total_weights = pd.concat(total_weights, axis=1).sum(axis=1)
    return result / total_weights


def create_word_vector_embeddings():
    """Creates average word embeddings for all emojis (based of their description)
    The order in the embeddings is the same as in keys.csv
    """
    print("creating word vector averages")

    df = load_and_process_description()
    model = "glove-twitter-200"
    emb = load_embedding_model(model)
    emb_vocabulary = set(emb.index_to_key)
    default = np.zeros(emb.get_vector("hello").shape, dtype=np.float32)
    get_vector = get_vector_wrapper(emb, emb_vocabulary, default)

    weighting = {
        "emjpd_emoji_name_og": 30,
        "emjpd_aliases": 15,
        "emjpd_description_main": 35,
        "emjpd_description_side": 5,
        "hemj_emoji_description": 15,
    }

    all_embeddings = get_word_vector_embedding(weighting, df, get_vector)
    train_embeddings = all_embeddings.loc[~df.zero_shot]

    all_embeddings = torch.stack([torch.from_numpy(x) for x in all_embeddings])
    train_embeddings = torch.stack([torch.from_numpy(x) for x in train_embeddings])
    save_data(all_embeddings, "word_vector_embeddings/all_embeddings")


# ************************ create description embedding via Sentence BERT***********************


def create_sbert_embeddings():

    """Creates Sentence bert embeddings for emojiname + aliases + main description
    Using model "all-mpnet-base-v1" with normalization of output embeddings.
    """

    print("creating sentence bert embeddings")

    df = load_and_process_description()
    filler = "\u25A1"
    s = df.emjpd_emoji_name_og.fillna("") + filler
    s += df.emjpd_aliases.fillna("") + filler
    s += df.emjpd_description_main.fillna("").str.replace("\n", filler) + filler
    s_ls = s.tolist()

    model_name = "all-mpnet-base-v1"
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512

    embeddings = model.encode(
        s_ls,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        normalize_embeddings=True,
    )

    save_data(embeddings, "bert_embeddings/all_embeddings")


if __name__ == "__main__":
    create_vision_embedding()
    create_word_vector_embeddings()
    create_sbert_embeddings()
