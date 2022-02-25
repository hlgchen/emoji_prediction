import os
import pandas as pd
import numpy as np
import json
from utils import get_project_root


def prepare_meta_data(must_have_zero_shot=[], zero_shot_config=100, seed=100):
    """
    Saves the image folder structure in a pandas Dataframe and splits
    the images in training and zeroshot data.

    For each emoji several images are saved in the same folder. For each image
    the path to it as well as the emoji type is saved in the pandas Dataframe.
    The path saved is relative to the project root.
    Subsequently the emoji types are split in training (for embedding) and zero-shot
    emojis, that are to be used during testing of zero shot capabilities.

    Params:
        - must_have_zero_shot {list}: list of emojis that have to be in zero shot set
        - zero_shot_config {int/list}: integer specifying number of emojis in zero-shot
                                        test set, or list of particular emojis that are
                                        to be put in the test set.
        - seed {int}: random seed fro split
    """
    img_path = "data/emojipedia/"
    out_train_path = "data/meta/img_meta.csv"
    out_zero_path = "data/meta/img_meta_zeroshot.csv"
    if os.getcwd().split("/")[-1] == "emoji_prediction":
        img_path = os.path.join("emoji_embedding", img_path)
        out_train_path = os.path.join("emoji_embedding", out_train_path)
        out_zero_path = os.path.join("emoji_embedding", out_zero_path)
    img_paths = []
    img_labels = []
    for path, subdirs, _ in os.walk(img_path):
        for sub in subdirs:
            for f in os.listdir(os.path.join(path, sub)):
                if f[-3:] == "jpg":
                    img_paths.append(os.path.join(path, sub, f))
                    img_labels.append(sub)
    df = pd.DataFrame({"path": img_paths, "label": img_labels})

    if isinstance(zero_shot_config, int):
        np.random.seed(seed)
        selection = (
            np.random.choice(
                df.label.unique(), zero_shot_config - len(must_have_zero_shot)
            ).tolist()
            + must_have_zero_shot
        )
    else:
        selection = zero_shot_config + must_have_zero_shot

    df_zero = df.loc[df.label.isin(selection)]
    df_train = df.loc[~df.label.isin(selection)]

    df_train.to_csv(out_train_path, index=False)
    df_zero.to_csv(out_zero_path, index=False)


def split_data(valid_num=1, test_num=0, seed=1):
    """
    Splits the (emojipedia embedding) training data (that is data exluding the zero shot data)
    into training, validation and test data. Saves each of those tables seperately.
    The split is specified via the number of images for each emoji type that are to be put
    in validation/test.

    Params:
        - valid_num {int}: number of images for each emoji in validation set
        - test_num {int}: number of images for each emoji in test set
        - seed {int}: random seed for reproducivility
    """
    np.random.seed(seed)

    paths = dict()
    paths["meta_path"] = "emoji_embedding/data/meta/img_meta.csv"
    paths["meta_train_path"] = "emoji_embedding/data/meta/img_meta_train.csv"
    paths["meta_valid_path"] = "emoji_embedding/data/meta/img_meta_valid.csv"
    paths["meta_test_path"] = "emoji_embedding/data/meta/img_meta_test.csv"
    paths["meta_emoji_idx"] = "emoji_embedding/data/meta/emoji_idx.json"
    paths["meta_idx_emoji"] = "emoji_embedding/data/meta/idx_emoji.json"

    for k, v in paths.items():
        paths[k] = os.path.join(get_project_root(), v)

    df_train = pd.read_csv(paths["meta_path"])
    emoji_idx = {k: v for v, k in enumerate(df_train.label.unique())}
    idx_emoji = {v: k for k, v in emoji_idx.items()}
    df_train["class"] = df_train.label.apply(lambda x: emoji_idx[x])
    print(f"number of unique classes: {df_train.label.nunique()}")
    with open(paths["meta_emoji_idx"], "w") as f:
        json.dump(emoji_idx, f)
    with open(paths["meta_idx_emoji"], "w") as f:
        json.dump(idx_emoji, f)

    if valid_num > 0:
        df_valid = df_train.groupby(by="label").sample(valid_num, replace=True)
        df_train = df_train.drop(df_valid.index)
        df_valid.to_csv(paths["meta_valid_path"], index=False)
    if test_num > 0:
        df_test = df_train.groupby(by="label").sample(test_num, replace=True)
        df_train = df_train.drop(df_test.index)
        df_valid.to_csv(paths["meta_test_path"], index=False)
    df_train.to_csv(paths["meta_train_path"], index=False)


if __name__ == "__main__":
    """Constructs metadata from folder structure for emojipedia images.
    Splits data into embedding training data and zero-shot data.
    Then splits embedding training data into training and validation data.
    """
    zero_shot_emoji_must_have = [
        "flushed_face",
        "backhand_index_pointing_right",
        "raising_hands",
        "male_sign",
        "sparkling_heart",
        "female_sign",
        "person_shrugging",
        "smiling_face",
        "flexed_biceps",
        "collision",
    ]

    prepare_meta_data(zero_shot_emoji_must_have)
    split_data()

    train_data = EmojiClassificationDataset("train")
    valid_data = EmojiClassificationDataset("valid")
