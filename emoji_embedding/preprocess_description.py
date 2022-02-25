import json
import pandas as pd
import os

from utils import get_project_root

# ************************ preprocess hotemoji data ********************************
def decode(description):
    try:  # data was scrapped as a byte representation string, need to convert to utf-8
        description = description.encode("latin-1").decode("utf-8")
        if "📑" in description:
            description = description.split("📑")[0]
        return description
    except:
        return description


def preprocess_hotemoji_data(save_path=None):
    read_path = os.path.join(
        get_project_root(),
        "emoji_embedding",
        "data/hotemoji/hotemoji_description_raw.csv",
    )
    hemj_df = (
        pd.read_csv(read_path)
        .dropna(subset=["emoji_char"])
        .rename(
            columns={
                "emoji_char": "emoji_char_bytes",
                "emoji_name": "hemj_emoji_name_og",
                "emoji_description": "hemj_emoji_description",
            }
        )
    )

    hemj_df["emoji_char"] = hemj_df.emoji_char_bytes.apply(
        lambda x: x.encode("latin-1").decode(
            "utf-8"
        )  # data was scrapped as a byte representation string
    )
    hemj_df["emoji_char_ascii"] = hemj_df.emoji_char.apply(
        lambda x: x.encode("unicode-escape").decode("ASCII")
    )

    hemj_df["hemj_emoji_name"] = hemj_df.hemj_emoji_name_og.str.lower().str.replace(
        " ", "_"
    )

    hemj_df.hemj_emoji_description = hemj_df.hemj_emoji_description.apply(decode)
    hemj_df = hemj_df.drop_duplicates()

    if save_path is not None:
        save_path = os.path.join(get_project_root(), "emoji_embedding", save_path)
        hemj_df.to_csv(save_path, index=False)

    return hemj_df


# ************************ preprocess emojipedia data ********************************


def get_reference_emojis(description, emoji_set):
    """Function that is used in pandas apply for each row.
    Returns list of emojis that are referenced in a description."""
    return [ref for ref in emoji_set if ref in description]


def get_usage(description):
    """Function that is used in pandas apply for each row.
    Returns string with sentences that contain "used" but not "used by/to" """
    result = "\n".join(
        [
            line
            for line in description.split("\n")
            if ("used" in line) & ("used by" not in line) & ("used to" not in line)
        ]
    )
    if len(result) > 0:
        return result
    else:
        return None


def split_description(description):
    """Function that is used in pandas apply for each row.
    Splits description into main and meta information part.
    The "side_text" contains information such as introduction year of emoji"""
    description_ls = description.split("\n")
    empty_counter = 0
    for i, t in enumerate(description_ls[::-1]):
        if t == "":
            empty_counter += 1
        if empty_counter == 2:
            break
    main_text = "\n".join([t for t in description_ls[:-i] if t != ""])
    side_text = "\n".join([t for t in description_ls[-i:] if t != ""])
    return pd.Series([main_text, side_text])


def preprocess_emojipedia_data(save_path=None):
    read_path = os.path.join(
        get_project_root(),
        "emoji_embedding",
        "data/emojipedia/",
    )
    emjpd_dict = dict()
    for path, subdirs, _ in os.walk(read_path):
        for sub in subdirs:
            for f in os.listdir(os.path.join(path, sub)):
                if f[-4:] == "json":
                    with open(os.path.join(path, sub, f), "r") as f:
                        emjpd_dict[sub] = json.load(f)

    df = pd.DataFrame()
    df["emoji_char"] = [v["emoji_char"] for v in emjpd_dict.values()]
    df["emoji_char_ascii"] = df.emoji_char.apply(
        lambda x: x.encode("unicode-escape").decode("ASCII")
    )
    df["emjpd_emoji_name"] = emjpd_dict.keys()
    df["emjpd_emoji_name_og"] = [v["emoji"] for v in emjpd_dict.values()]
    df["emjpd_aliases"] = [v["aliases"] for v in emjpd_dict.values()]
    df["emjpd_full_description"] = [v["description"] for v in emjpd_dict.values()]
    df.emjpd_full_description = df.emjpd_full_description.str.replace(
        u"\xa0", u" "
    )  # remove nonbreaking space
    df["emjpd_shortcodes"] = [v.get("shortcodes", []) for v in emjpd_dict.values()]

    emoji_set = set(df.emoji_char.tolist())
    df["emjpd_description_ref_emj"] = df.emjpd_full_description.apply(
        lambda x: get_reference_emojis(x, emoji_set)
    )
    df["emjpd_usage_info"] = df.emjpd_full_description.apply(get_usage)
    df[
        ["emjpd_description_main", "emjpd_description_side"]
    ] = df.emjpd_full_description.apply(split_description)

    if save_path is not None:
        save_path = os.path.join(get_project_root(), "emoji_embedding", save_path)
        df.to_csv(save_path, index=False)

    return df


# ************************ merge datasets ********************************


def merge_emoji_datasets(df, hemj_df, save_path):
    mdf = df.merge(hemj_df, how="left", on=["emoji_char", "emoji_char_ascii"])

    mdf["emoji_name_og"] = mdf.emjpd_emoji_name_og.where(
        mdf.emjpd_emoji_name_og.notna(), mdf.hemj_emoji_name_og
    ).fillna("")

    mdf["emoji_char_ascii_beg"] = mdf.emoji_char_ascii.apply(
        lambda x: "\\" + [y for y in x.split("\\") if len(y) > 0][0]
    )

    for col in [
        "emoji_name_og",
        "emjpd_emoji_name_og",
        "hemj_emoji_name_og",
        "emjpd_full_description",
        "emjpd_description_main",
        "emjpd_description_side",
        "hemj_emoji_description",
        "emjpd_usage_info",
    ]:
        mdf[col] = mdf[col].str.lower()

    mdf["emjpd_aliases"] = mdf["emjpd_aliases"].apply(
        lambda x: [s.lower() for s in x] if isinstance(x, list) else []
    )

    cols = [
        "emoji_char",  # emoji-symbol/picture
        "emoji_name_og",  # emojipedia emoji name if available, otherwise hotemoji name, otherwise ""
        "emoji_char_ascii",  # emoji expressed in ascii
        "emoji_char_ascii_beg",  # the first emoji part expressed as ascii
        "emoji_char_bytes",  # emoji expressed as bytes (only for hotemoji)
        "emjpd_emoji_name",  # emojipedia emoji name (corresponds to folder in scrapped data)
        "emjpd_emoji_name_og",  # emojipedia emoji name as is in the heading on website
        "hemj_emoji_name",  # hotemoji emoji name (_ instead of spaces)
        "hemj_emoji_name_og",  # hotemoji emoji name
        "emjpd_aliases",  # emojipedia aliases for emoji
        "emjpd_shortcodes",  # emojipedia shortcuts for slack/github etc.
        "emjpd_full_description",  # emojipedia all description data scraped unprocessed in utf-8
        "emjpd_description_main",  # emojipedia main description without meta data at bottom
        "emjpd_description_side",  # emojipedia meta data about emoji (when was the emoji introduced)
        "hemj_emoji_description",  # hotemoji description scaraped utf-8
        "emjpd_usage_info",  # emojipedia sentences with how an emoji is used (contains "used" but not "used to/by")
        "emjpd_description_ref_emj",  # emojipedia emojis that are referrenced within the description in some way
    ]

    mdf = mdf[cols]

    save_path = os.path.join(get_project_root(), "emoji_embedding", save_path)
    mdf.to_csv(save_path, index=False)


if __name__ == "__main__":

    hemj_df = preprocess_hotemoji_data()
    df = preprocess_emojipedia_data()
    merge_emoji_datasets(df, hemj_df, "data/processed/emoji_descriptions.csv")
