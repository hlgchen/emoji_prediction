import os
import json
import pandas as pd
import numpy as np

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
    """
    Preprocessed scraped hotemoji data and returns it as pd.DataFrame.
    If a save_path is specified the data is saved.
    """
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
    """
    Preprocesses scraped emojipedia data and returns it as a dataframe.
    If a save_path is specified data is saved there.
    """
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
        "\xa0", " "
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
    """
    Does a left join on emojipedia and hotemoji data. The emojipedia data is on the left.
    Essentially only description information from hotemoji is added to the data of emojipedia.
    Saves the merged dataframe in the specified save_path.
    """
    mdf = df.merge(hemj_df, how="left", on=["emoji_char", "emoji_char_ascii"])
    mdf = mdf.rename(columns={"emjpd_emoji_name": "emoji_name"})
    mdf["emoji_char_ascii_beg"] = mdf.emoji_char_ascii.apply(
        lambda x: "\\" + [y for y in x.split("\\") if len(y) > 0][0]
    )

    for col in [
        "emoji_name",
        "emjpd_emoji_name_og",
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
    mdf["emoji_id"] = mdf.index

    cols = [
        "emoji_id",  # just the index of the emoji
        "emoji_char",  # emoji-symbol/picture
        "emoji_name",  # emojipedia emoji name processed
        "emoji_char_ascii",  # emoji expressed in ascii
        "emoji_char_ascii_beg",  # the first emoji part expressed as ascii
        "emjpd_emoji_name_og",  # emojipedia emoji name
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


# ************************ get_keys ********************************


def get_zeroshot_emojis():
    """Returns hard coded list of emojis that are in the embedding image zero shot set"""
    ls = [
        "muted_speaker",
        "palm_tree",
        "lipstick",
        "person_golfing",
        "waxing_crescent_moon",
        "movie_camera",
        "sports_medal",
        "skis",
        "speaking_head",
        "woman_facepalming",
        "eye_in_speech_bubble",
        "flag-_tokelau",
        "medical_symbol",
        "woman_detective",
        "flat_shoe",
        "sauropod",
        "flag-_burundi",
        "raising_hands",
        "smiling_face_with_hearts",
        "face_with_open_mouth",
        "sparkling_heart",
        "martial_arts_uniform",
        "family-_man,_woman,_boy",
        "backhand_index_pointing_right",
        "mage",
        "person_taking_bath",
        "passenger_ship",
        "male_sign",
        "telescope",
        "flag-_slovakia",
        "smiling_face",
        "man_mage",
        "computer_mouse",
        "woman_bowing",
        "woman_gesturing_no",
        "firefighter",
        "monorail",
        "trumpet",
        "person_in_manual_wheelchair",
        "pool_8_ball",
        "waffle",
        "victory_hand",
        "potable_water",
        "one-thirty",
        "keycap_digit_two",
        "flag-_bouvet_island",
        "flag-_timor-leste",
        "soon_arrow",
        "weary_face",
        "flag-_åland_islands",
        "woman_pouting",
        "minibus",
        "department_store",
        "wolf",
        "railway_car",
        "female_sign",
        "sheaf_of_rice",
        "flag-_china",
        "receipt",
        "horizontal_traffic_light",
        "flag-_british_indian_ocean_territory",
        "flexed_biceps",
        "1st_place_medal",
        "b_button_(blood_type)",
        "black_small_square",
        "cloud",
        "ferry",
        "fast-forward_button",
        "woman_supervillain",
        "train",
        "bikini",
        "broccoli",
        "left_luggage",
        "no_mobile_phones",
        "two_hearts",
        "flag-_panama",
        "person_shrugging",
        "family-_woman,_woman,_girl",
        "hedgehog",
        "flag-_egypt",
        "outbox_tray",
        "enraged_face",
        "bacon",
        "deaf_woman",
        "person_wearing_turban",
        "last_quarter_moon",
        "flushed_face",
        "flag-_cocos_(keeling)_islands",
        "lying_face",
        "flag-_isle_of_man",
        "safety_pin",
        "chart_increasing",
        "sparkle",
        "elf",
        "crescent_moon",
        "umbrella_with_rain_drops",
        "collision",
        "fish_cake_with_swirl",
        "nut_and_bolt",
    ]
    return ls


def get_keys(zero_shot_emojis, save_path):
    """
    Given specified zero_shot_emojis, a dataframe is contained that
    can serve as the "key" - mapping for other tables.

    Params:
        - zero_shot_emojis {list}: list of emojis that are specified as
                                    zero shot emojis. These emojis will not
                                    be seen by the vision model during training.
        - save_path {str}: string specifying the location to save the dataframe.

    The dataframe has the following columns:
        - emoji_id: integer emoji_id
        - emoji_name: emojisname as in emojipedia but with processing of space etc.
        - emoji_char_ascii: emoji ascii escape notation
        - emoji_char_ascii_beg: first emoji escape notation,
                            (emoji can be composed of many parts)
        - zero_shot: boolean specifying whether an emoji is part of the zero shot set
    """
    description_path = "data/processed/emoji_descriptions.csv"
    description_path = os.path.join(
        get_project_root(), "emoji_embedding", description_path
    )
    df = pd.read_csv(description_path)[
        ["emoji_id", "emoji_name", "emoji_char_ascii", "emoji_char_ascii_beg"]
    ]
    df["zero_shot"] = df.emoji_name.apply(lambda x: x in zero_shot_emojis)

    save_path = os.path.join(get_project_root(), "emoji_embedding", save_path)
    df.to_csv(save_path, index=False)

    return df


# ************************ split datasets ********************************


def prepare_meta_data(key_df, out_path, seed=1):

    """
    Creates dataframe with paths, emoji_name, zero_shot - flag and dataset_type
    for each image in the emojipedia dataset.
    dataset_type can be 'train', 'valid' or 'zero'.
    'zero' will be all images from the emojis that are part of the zero_shot test.
    All other emojis will be seen during training of the vision model.
    The validation and train test split is for the images only. For each emoji,
    one random image will be put in the validation set.
    """

    np.random.seed(seed)
    img_path = "data/emojipedia/"
    img_path = os.path.join(get_project_root(), "emoji_embedding", img_path)

    img_paths = []
    img_labels = []
    for path, subdirs, _ in os.walk(img_path):
        for sub in subdirs:
            for f in os.listdir(os.path.join(path, sub)):
                if f[-3:] == "jpg":
                    save_path = os.path.join(path, sub, f)
                    img_paths.append(save_path.split("emoji_prediction")[1])
                    img_labels.append(sub)
    df = pd.DataFrame({"path": img_paths, "emoji_name": img_labels})

    df = df.merge(key_df[["emoji_id", "emoji_name", "zero_shot"]])

    df["dataset_type"] = "train"
    valid_index = df.groupby("emoji_name").sample(1).index
    df.loc[valid_index, "dataset_type"] = "valid"
    df.dataset_type = df.dataset_type.where(df.zero_shot == False, "zero")

    out_path = os.path.join(get_project_root(), "emoji_embedding", out_path)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":

    hemj_df = preprocess_hotemoji_data()
    df = preprocess_emojipedia_data()
    merge_emoji_datasets(df, hemj_df, "data/processed/emoji_descriptions.csv")

    zero_shot_emojis = get_zeroshot_emojis()
    key_df = get_keys(zero_shot_emojis, "data/processed/keys.csv")

    prepare_meta_data(key_df, "data/processed/img_meta.csv")
