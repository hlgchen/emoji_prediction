import os
import pandas as pd


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TwemojiData:
    def __init__(self, dataset_type, nrows=None, shuffle=False, batch_size=32):
        twemoji_path = os.path.join(
            get_project_root(), f"twemoji/data/twemoji_{dataset_type}.csv"
        )
        self.df = pd.read_csv(
            twemoji_path, usecols=["text_no_emojis", "emoji_ids"], nrows=nrows
        )
        self.df.emoji_ids = (
            self.df.emoji_ids.str[1:-1]
            .str.split(",")
            .apply(lambda x: [int(y) for y in x])
            .tolist()
        )
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_samples = len(self.df)

    def get_lists(self):
        labels = self.df.emoji_ids.tolist()
        text = self.df.text_no_emojis.tolist()
        return text, labels

    def __iter__(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        text, labels = self.get_lists()

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            yield text[start:end], labels[start:end]

    def __getitem__(self, idx):
        return self.df.text_no_emojis.tolist()[idx], self.df.emoji_ids.tolist()[idx]
