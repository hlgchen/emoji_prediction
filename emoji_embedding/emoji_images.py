import os
import torch
import pandas as pd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import numpy as np
import json


from utils import get_project_root


def prepare_meta_data(zero_shot_config=100, seed=100):
    """
    Saves the image folder structure in a pandas Dataframe and splits
    the images in training and zeroshot data.

    For each emoji several images are saved in the same folder. For each image
    the path to it as well as the emoji type is saved in the pandas Dataframe.
    The path saved is relative to the project root.
    Subsequently the emoji types are split in training (for embedding) and zero-shot
    emojis, that are to be used during testing of zero shot capabilities.

    Params:
        - zero_shot_config {int/list}: integer specifying number of emojis in zero-shot
                                        test set, or list of particular emojis that are
                                        to be put in the test set.
        - seed {int}: random seed fro split
    """
    img_path = "data/emojipedia/"
    out_train_path = "data/meta/img_meta.csv"
    out_zero_path = "data/meta/zero_shot_meta.csv"
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
        selection = np.random.choice(df.label.unique(), zero_shot_config)
    else:
        selection = zero_shot_config

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


class EmojiClassificationDataset(Dataset):
    def __init__(self, dataset_type="", img_size=224):
        """
        Creates emoji Dataset.

        Params:
            - dataset_type {str}: string that specifies the dataset type.
                                Can be ["", "train", "valid", "test"].
                                If "" the whole dataset is loaded (excluding zeroshot emojis).
            - img_size {int}: height and width of the images in the dataset
        """
        path_suffix = "_" + dataset_type
        meta_path = f"emoji_embedding/data/meta/img_meta{path_suffix}.csv"
        meta_path = os.path.join(get_project_root(), meta_path)
        self.project_root = get_project_root()
        self.df = pd.read_csv(meta_path)
        self.scale = Rescale(img_size)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx, 0]
        img_name = os.path.join(self.project_root, img_name)
        input_image = Image.open(img_name).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        label = self.df.iloc[idx, 2]  # corresponds to column class in dataframe
        label = torch.Tensor([label])
        return input_tensor, label

    def __len__(self):
        return len(self.df)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Params:
        output_size {tuple or int}: Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        # resize
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))[
            : self.output_size, : self.output_size
        ]
        return img


def plot(sample):
    """Plot sample from dataset"""
    fig = plt.figure()
    print(sample[0].shape, sample[1].shape)
    ax = plt.subplot()
    plt.tight_layout()
    ax.set_title(sample[1])
    ax.axis("off")

    plt.imshow(sample[0].permute(1, 2, 0))


if __name__ == "__main__":
    """Constructs metadata from folder structure for emojipedia images.
    Splits data into embedding training data and zero-shot data.
    Then splits embedding training data into training and validation data.
    """

    prepare_meta_data()
    split_data()

    train_data = EmojiClassificationDataset("train")
    valid_data = EmojiClassificationDataset("valid")
