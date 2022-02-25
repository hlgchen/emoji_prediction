import os
import pandas as pd
from torch.utils.data import Dataset

from pathlib import Path
from emoji import UNICODE_EMOJI


def get_project_root():
    """Returns absolute path of project root."""
    return Path(__file__).parent.resolve()


# ************* preprocess raw data *******************


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data


def extract_emojis(s):
    """Extracts all emojis from a given string and returns it as a string"""
    return "".join([c for c in s if c in UNICODE_EMOJI["en"]])


def preprocess(data):
    """
    Preprocessed the text data and returns a pandas dataframe.

    params:
        - data {list}: list of tweet_id + tweet, which are just one string
    returns:
        - df {pd.DataFrame}: processed file with the following rows:
            - id: tweet id (some tweetids have letters at the end, thus this is a string)
            - raw_text: raw text from raw data excluding tweet id and "\n" at the end
            - emojis: emojis that appear in the text. This is a string, no separation between emojis
            - text_no_emojis: raw_text where emojis are removed
            - text_replaced_emojis: raw_text where emojis are replaced by " xxxxxxxx "

    """
    split_data = [s.split(" ", maxsplit=1) for s in data]
    split_data = [list(i) for i in zip(*split_data)]
    ids = split_data[0]
    raw_texts = [s.strip() for s in split_data[1]]
    emojis = [extract_emojis(s) for s in raw_texts]

    df = pd.DataFrame({"id": ids, "raw_text": raw_texts, "emojis": emojis})
    df["text_no_emojis"] = df.apply(
        lambda x: x.raw_text.replace("|".join(x.emojis.split()), ""), axis=1
    )
    df["text_replaced_emojis"] = df.apply(
        lambda x: x.raw_text.replace("|".join(x.emojis.split()), " xxxxxxxx "), axis=1
    )
    return df


def tocsv(infile, outfile):
    """Reads filename infile (should be in same folder as this script.
    Writes the processed file to outfile."""
    infile = os.path.join(get_project_root(), infile)
    outfile = os.path.join(get_project_root(), outfile)

    print(f"loading {infile}")
    data = load_data(infile)
    print(f"finished reading, has {len(data)} lines")

    print("processing")
    df = preprocess(data)
    print("finished processing")

    df.to_csv(outfile, encoding="utf-8")
    print(f"wrote processed file to {outfile}\n")


# ************* finish preprocess raw data *******************


class twemoji_data(Dataset):

    """ """

    def __init__(self, dataset_type):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    tocsv("raw_train.txt", "twemoji_train.csv")
    tocsv("raw_valid.txt", "twemoji_valid.csv")
    tocsv("raw_test.txt", "twemoji_test.csv")
