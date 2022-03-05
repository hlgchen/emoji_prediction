import os
import pandas as pd
from time import time
import re
import preprocessor
import contractions
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")


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


def remove_extra_spaces(text):
    """
    Return :- string after removing extra whitespaces
    Input :- String
    Output :- String
    """
    space_pattern = r"\s+"
    without_space = re.sub(pattern=space_pattern, repl=" ", string=text)
    without_space = without_space.strip()
    return without_space


def lemmatization_wrapper(lemma):
    def lemmatization(text):
        """
        Result :- string after stemming
        Input :- String
        Output :- String
        """
        # word tokenization
        tokens = word_tokenize(text)

        for index in range(len(tokens)):
            # lemma word
            lemma_word = lemma.lemmatize(tokens[index])
            tokens[index] = lemma_word

        return " ".join(tokens)

    return lemmatization


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
            - text_no_emojis: raw_text where emojis are removed, white space is removed
            - text_no_emojis_clean: removed mentions and links, white space is removed
            - text_no_emojis_superclean: same as clean but with lemmatization and expansion of contractions
            - text_replaced_emojis: raw_text where emojis are replaced by " xxxxxxxx "

        - df_filtered {pd.DataFrame}: same as df except filtered based on minimum
                number of words per tweet in text_no_emojis_clean
            - len_text: text length excluding mentions and links

    """
    split_data = [s.split(" ", maxsplit=1) for s in data]
    split_data = [list(i) for i in zip(*split_data)]
    ids = split_data[0]

    # Preprocessing 1. remove white space front and back
    raw_texts = [s.strip() for s in split_data[1]]  # modified

    emojis_emoji_ids = [extract_emojis(s) for s in raw_texts]
    emojis = [e[0] for e in emojis_emoji_ids]
    emoji_ids = [e[1] for e in emojis_emoji_ids]

    df = pd.DataFrame(
        {"id": ids, "raw_text": raw_texts, "emojis": emojis, "emoji_ids": emoji_ids}
    )

    ##1. Dataset with minimal data cleaning to compare with original results of the paper
    df["text_no_emojis"] = df.apply(
        lambda x: re.sub("|".join(list(x.emojis)), "", x.raw_text), axis=1
    )
    # Remove extra spaces
    df.text_no_emojis = df.text_no_emojis.map(lambda a: remove_extra_spaces(a))

    ##2a. Dataset with some preprocessing
    # Remove mentions and links
    df["text_no_emojis_clean"] = df.text_no_emojis.map(lambda a: preprocessor.clean(a))
    # Remove "rt : " at the start of the sentence
    df.text_no_emojis_clean = df.text_no_emojis_clean.where(
        df.text_no_emojis_clean.str[:2] != ": ", df.text_no_emojis_clean.str[2:]
    )
    # Remove extra white space
    df.text_no_emojis_clean = df.text_no_emojis_clean.map(
        lambda a: remove_extra_spaces(a)
    )

    ## 3. Dataset with more preprocessing
    # Expansion of Contractions
    df["text_no_emojis_superclean"] = df.text_no_emojis_clean.map(
        lambda a: contractions.fix(a)
    )

    # Lemmatization
    lemmatization = lemmatization_wrapper(WordNetLemmatizer())
    df.text_no_emojis_superclean = df.text_no_emojis_superclean.map(
        lambda a: lemmatization(a)
    )

    # add number of words
    df["n_words"] = df.text_no_emojis_clean.apply(lambda x: len(x.split(" ")))

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
    df_unfiltered_train = tocsv("raw_train.txt", "twemoji_train_v2.csv")
    df_unfiltered_valid = tocsv("raw_valid.txt", "twemoji_valid_v2.csv")
    df_unfiltered_test = tocsv("raw_test.txt", "twemoji_test_v2.csv")

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
    df_train.to_csv(os.path.join(outpath, "twemoji_train_v2.csv"), index=False)
    df_valid.to_csv(os.path.join(outpath, "twemoji_valid_v2.csv"), index=False)
    df_test.to_csv(os.path.join(outpath, "twemoji_test_v2.csv"), index=False)
    extra_zero_df = pd.concat([df_train_zero, df_valid_zero])
    extra_zero_df.to_csv(
        os.path.join(outpath, "twemoji_extra_zero_v2.csv"), index=False
    )

    # min_num_words = 2
    # df_train.loc[df_train.n_words > min_num_words].to_csv(
    #     os.path.join(outpath, f"twemoji_train_v2_min_{min_num_words}.csv"), index=False
    # )
    # df_valid.loc[df_valid.n_words > min_num_words].to_csv(
    #     os.path.join(outpath, f"twemoji_valid_v2_min_{min_num_words}.csv"), index=False
    # )
    # df_test.loc[df_test.n_words > min_num_words].to_csv(
    #     os.path.join(outpath, f"twemoji_test_v2_min_{min_num_words}.csv"), index=False
    # )
    # extra_zero_df.loc[extra_zero_df.n_words > min_num_words].to_csv(
    #     os.path.join(outpath, f"twemoji_extra_zero_v2_min_{min_num_words}.csv"),
    #     index=False,
    # )
    print("saved files")
