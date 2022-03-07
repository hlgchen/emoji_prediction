import os
import numpy as np
import pandas as pd


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TwemojiData:
    def __init__(
        self,
        data,
        nrows=None,
        shuffle=False,
        batch_size=64,
        limit=None,
        text_col="text_no_emojis",
        seed=1,
    ):
        np.random.seed(seed)
        if isinstance(data, str):
            twemoji_path = os.path.join(
                get_project_root(), f"twemoji/data/twemoji_{data}.csv"
            )
            self.df = pd.read_csv(
                twemoji_path, usecols=[text_col, "emoji_ids"], nrows=nrows
            )
        else:
            self.df = data.copy()
        self.df.emoji_ids = (
            self.df.emoji_ids.str[1:-1]
            .str.split(",")
            .apply(lambda x: [int(y) for y in x])
            .tolist()
        )
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.limit = limit
        self.text_col = text_col

    def get_lists(self):
        labels = self.df.emoji_ids.tolist()
        text = self.df[self.text_col].tolist()
        return text, labels

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        text, labels = self.get_lists()

        limit = (
            min(len(self.df), self.limit) if self.limit is not None else len(self.df)
        )
        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, limit)
            yield text[start:end], labels[start:end]

    def __getitem__(self, idx):
        return self.df[self.text_col].tolist()[idx], self.df.emoji_ids.tolist()[idx]

    def __len__(self):
        return len(self.df)


class TwemojiDataChunks:
    def __init__(
        self,
        dataset_type,
        chunksize=64000,
        shuffle=False,
        batch_size=64,
        text_col="text_no_emojis",
        seed=1,
        balanced=False,
    ):
        print(f"random seed is: {seed}")
        np.random.seed(seed)
        twemoji_path = os.path.join(
            get_project_root(), f"twemoji/data/twemoji_{dataset_type}.csv"
        )
        df = pd.read_csv(
            twemoji_path,
            usecols=[text_col, "emoji_ids"],
        )
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        df_ls = [df[i : i + chunksize] for i in range(0, len(df), chunksize)]
        self.data_ls = [
            TwemojiData(
                df,
                shuffle=shuffle,
                batch_size=batch_size,
                text_col=text_col,
            )
            if not balanced
            else TwemojiBalancedData(
                df,
                batch_size=batch_size,
                text_col=text_col,
            )
            for df in df_ls
        ]
        self.n_chunks = len(self.data_ls)

    def __iter__(self):
        for twemoji_dataset in self.data_ls:
            yield twemoji_dataset

    def __getitem__(self, idx):
        return self.data_ls[idx]

    def __len__(self):
        return self.n_chunks


class TwemojiBalancedData:
    def __init__(
        self,
        data,
        nrows=None,
        batch_size=64,
        limit=None,
        text_col="text_no_emojis",
        seed=1,
    ):
        np.random.seed(seed)
        if isinstance(data, str):
            twemoji_path = os.path.join(
                get_project_root(), f"twemoji/data/twemoji_{data}.csv"
            )
            self.df = pd.read_csv(
                twemoji_path, usecols=[text_col, "emoji_ids"], nrows=nrows
            )
        else:
            self.df = data.copy()
        self.df.emoji_ids = (
            self.df.emoji_ids.str[1:-1]
            .str.split(",")
            .apply(lambda x: [int(y) for y in x])
            .tolist()
        )
        self.length_df = len(self.df)
        self.df["idx"] = self.df.emoji_ids
        self.df = self.df.explode(column="idx")
        self.df.index = self.df.idx
        self.all_emojis = self.df.idx.unique()

        self.batch_size = batch_size
        self.limit = limit
        self.text_col = text_col

    def __iter__(self):
        limit = (
            min(self.length_df, self.limit)
            if self.limit is not None
            else self.length_df
        )
        for _ in range(0, limit, self.batch_size):
            text = []
            labels = []
            for emoji in np.random.choice(self.all_emojis, self.batch_size):
                sample = self.df.loc[[emoji]].sample()
                text.append(sample[self.text_col].iloc[0])
                labels.append(sample.emoji_ids.iloc[0])
            yield text, labels

    def __len__(self):
        return self.length_df


if __name__ == "__main__":
    data = TwemojiBalancedData("valid_v2")

    for batch in data:
        b = batch
        break
