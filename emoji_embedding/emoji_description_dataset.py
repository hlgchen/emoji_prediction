import numpy as np
import pandas as pd


def swap(row):
    maximum_pos_swaps = max(min(row.n_cs, len(row.neg)), 1)
    number_swaps = max(1, np.random.randint(0, maximum_pos_swaps))
    swap_index = (
        np.random.choice(range(0, row.n_cs), number_swaps, replace=False)
        if row.n_cs > 0
        else []
    )
    swap = np.random.choice(row.neg, number_swaps, replace=False).tolist()

    result = []
    for i, pos in enumerate(row.pos_selection):
        if i in swap_index:
            result.append(swap.pop())
        else:
            result.append(pos)
    for _ in range(len(swap)):
        result.append(swap.pop())
    return ".".join(result)


def get_dataset(df, n_triplets):
    # create new dataframe
    des_df = df[
        ["emjpd_emoji_name_og", "emjpd_description_main", "hemj_emoji_description"]
    ].fillna("")

    # the anchor is just the emoji name + emojipedia main description
    des_df["anchor"] = (
        des_df.emjpd_emoji_name_og + "\u25A1" + des_df.emjpd_description_main
    ).tolist()
    des_df["n_a"] = [
        len([y for y in x.split(". ") if len(y) > 3]) for x in des_df["anchor"]
    ]

    # create list of correct emoji description sentences
    des_df["correct_selection"] = des_df.emjpd_description_main.apply(
        lambda x: x.split(". ")
    ) + des_df.hemj_emoji_description.fillna("").apply(lambda x: x.split(". "))
    des_df.correct_selection = des_df.correct_selection.apply(
        lambda x: [s.strip() for s in x if len(s) > 3]
    )
    des_df["n_cs"] = des_df.correct_selection.apply(len)

    # from correct selection, randomly sample some sentences and
    # concatenate to artificial description but with correct contents
    des_df["pos_selection"] = des_df.apply(
        lambda x: np.random.choice(
            x.correct_selection,
            min(x.n_a, x.n_cs),
            replace=False,
        ),
        axis=1,
    )
    pos = (
        des_df.emjpd_emoji_name_og
        + "\u25A1"
        + des_df.pos_selection.apply(lambda x: ".".join(x))
    ).tolist()

    # create negative samples
    negatives = []
    for _ in range(n_triplets):
        des_df["neg"] = (
            des_df.loc[des_df.n_cs > 1]
            .sample(len(des_df), replace=True)
            .correct_selection.tolist()
        )

        # swap at least one positive to one negative
        neg = (
            des_df.emjpd_emoji_name_og + "\u25A1" + des_df.apply(swap, axis=1)
        ).tolist()
        negatives += neg

    anchor = des_df["anchor"].tolist() * n_triplets
    positives = pos * n_triplets
    return anchor, positives, negatives


class EDDataset:
    def __init__(self, df, n_triplets, batch_size=64, seed=1):
        self.df = df
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        np.random.seed(seed)

    def __iter__(self):
        anchor, positives, negatives = get_dataset(self.df, self.n_triplets)

        for start in range(0, len(anchor), self.batch_size):
            end = min(start + self.batch_size, len(anchor))
            yield anchor[start:end], positives[start:end], negatives[start:end]

    def __len__(self):
        return len(self.df) * self.n_triplets
