import os
import pandas as pd
from time import time
import re

from emoji import UNICODE_EMOJI


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


keys_path = os.path.join(get_project_root(), "emoji_embedding/data/processed/keys.csv")
keys_df = pd.read_csv(keys_path)
keys_df["emoji"] = keys_df.emoji_char_ascii.apply(
    lambda x: x.encode("ASCII").decode("unicode-escape")
)
keys_df["emoji_simple"] = keys_df.emoji_char_ascii_beg.apply(
    lambda x: x.encode("ASCII").decode("unicode-escape")
)
keys_simple_df = keys_df.loc[keys_df.emoji != keys_df.emoji_simple]

emoji_set = set(keys_df.emoji)
emoji_simple_set = set(keys_simple_df.emoji_simple)

emoji_emoji_id = {k: v for k, v in zip(keys_df.emoji, keys_df.emoji_id)}
emoji_simple_emoji_id = {
    k: v for k, v in zip(keys_simple_df.emoji_simple, keys_simple_df.emoji_id)
}

zero_shot_emoji_id_set = set(keys_df.loc[keys_df.zero_shot].emoji_id)

# ************* preprocess raw data *******************


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data


def extract_emojis(s):
    """Extracts emojis from a string and returns them along with their emoji_id in a tuple.
    If the emoji is not known in emoji_description, no emoji_id is returned.
    """
    emoji_ls = [c for c in s if c in emoji_set]
    emoji_id_ls = [emoji_emoji_id[c] for c in emoji_ls]

    emoji_simple_ls = [c for c in s if c in emoji_simple_set]
    emoji_id_simple_ls = [emoji_simple_emoji_id[c] for c in emoji_simple_ls]

    emoji_official_ls = [c for c in s if c in UNICODE_EMOJI["en"]]

    emojis = set(emoji_ls + emoji_simple_ls + emoji_official_ls)
    emoji_ids = set(emoji_id_ls + emoji_id_simple_ls)
    return ["".join(emojis), emoji_ids]


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
            - emoji_ids: emoji_ids that correspond to emojis (if emojis are known to emoji_description)
            - text_no_emojis: raw_text where emojis are removed
            - text_replaced_emojis: raw_text where emojis are replaced by " xxxxxxxx "

    """
    split_data = [s.split(" ", maxsplit=1) for s in data]
    split_data = [list(i) for i in zip(*split_data)]
    ids = split_data[0]
    raw_texts = [s.strip() for s in split_data[1]]
    emojis_emoji_ids = [extract_emojis(s) for s in raw_texts]
    emojis = [e[0] for e in emojis_emoji_ids]
    emoji_ids = [e[1] for e in emojis_emoji_ids]

    df = pd.DataFrame(
        {"id": ids, "raw_text": raw_texts, "emojis": emojis, "emoji_ids": emoji_ids}
    )
    df["text_no_emojis"] = df.apply(
        lambda x: re.sub("|".join(list(x.emojis)), "", x.raw_text), axis=1
    )
    # df["text_replaced_emojis"] = df.apply(
    #     lambda x: re.sub("|".join(list(x.emojis)), " xxxxxxxx ", x.raw_text)), axis=1
    # )

    df.emojis = df.emojis.where(df.emojis != "")
    df.emoji_ids = df.emoji_ids.where(df.emoji_ids.apply(len) > 0)
    df.text_no_emojis = df.text_no_emojis.where(df.text_no_emojis != "")
    return df


def tocsv(infile, outfile):
    """Reads filename infile (should be in same folder as this script.
    Writes the processed file to outfile."""
    start = time()
    infile = os.path.join(get_project_root(), "twemoji/raw_data", infile)

    print(f"loading {infile}")
    data = load_data(infile)
    print(f"finished reading, has {len(data)} lines")

    print("processing")
    df = preprocess(data)
    print("finished processing")

    outfile = os.path.join(get_project_root(), "twemoji/unfiltered_processed", outfile)
    df.to_csv(outfile, encoding="utf-8", index=False)
    print(f"wrote processed file to {outfile}, took {time() - start}\n")

    return df


# ************* filter raw data *******************


def filter_data(df):
    """Return dataframe where all emojis are part of the training set and
    another datframe where there is at least one emoji that is in the zero shot set."""
    train_mask = df.emoji_ids.apply(
        lambda x: len(x.intersection(zero_shot_emoji_id_set)) == 0
    )
    print(f"filtered {len(df) - train_mask.sum()} from a total of {len(df)}")
    return df.loc[train_mask], df.loc[~train_mask]


# *************** finish filtering ***********************

if __name__ == "__main__":
    df_unfiltered_train = tocsv("raw_train.txt", "twemoji_train.csv")
    df_unfiltered_valid = tocsv("raw_valid.txt", "twemoji_valid.csv")
    df_unfiltered_test = tocsv("raw_test.txt", "twemoji_test.csv")

    print("train NA\n", df_unfiltered_train.isna().sum())
    print("valid NA\n", df_unfiltered_valid.isna().sum())
    print("test NA\n", df_unfiltered_test.isna().sum())

    df_train = df_unfiltered_train.dropna().reset_index(drop=True)
    df_valid = df_unfiltered_valid.dropna().reset_index(drop=True)
    df_test = df_unfiltered_test.dropna().reset_index(drop=True)

    print("dropped NAs")

    print("remove zero shot samples from train")
    df_train, df_train_zero = filter_data(df_train)
    print("remove zero shot samples from valid")
    df_valid, df_valid_zero = filter_data(df_valid)

    outpath = os.path.join(get_project_root(), "twemoji/data")
    df_train.to_csv(os.path.join(outpath, "twemoji_train.csv"), index=False)
    df_valid.to_csv(os.path.join(outpath, "twemoji_valid.csv"), index=False)
    df_test.to_csv(os.path.join(outpath, "twemoji_test.csv"), index=False)
    pd.concat([df_train_zero, df_valid_zero]).to_csv(
        os.path.join(outpath, "twemoji_extra_zero.csv"), index=False
    )
    print("saved files")
