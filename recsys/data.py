"""Load MovieLens-1M, split it by time, and build the user x item matrix.

We treat every rating as an *implicit* interaction ("this user engaged with
this movie"), ignoring the star value. That mirrors property search, where
the signals are views, saves and enquiries rather than explicit ratings.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """ratings.dat lines look like `UserID::MovieID::Rating::Timestamp`."""
    return pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user", "item", "rating", "ts"],
    )


def load_release_years(data_dir: Path) -> pd.Series:
    """Release year parsed from titles like `Toy Story (1995)`, indexed by item id."""
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["item", "title", "genres"],
        encoding="latin-1",
    )
    years = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    return pd.Series(years.to_numpy(), index=movies["item"].to_numpy())


def temporal_split(ratings: pd.DataFrame, val_q: float = 0.90, test_q: float = 0.95):
    """Global time split: everything before one cutoff is history, after is future.

    Cutoffs are quantiles of the timestamp, so train is the oldest 90% of
    interactions, validation the next 5% and test the newest 5%. No model ever
    sees an interaction that happened after the period it is evaluated on.
    """
    t_val = ratings["ts"].quantile(val_q)
    t_test = ratings["ts"].quantile(test_q)
    train = ratings[ratings["ts"] < t_val]
    val = ratings[(ratings["ts"] >= t_val) & (ratings["ts"] < t_test)]
    test = ratings[ratings["ts"] >= t_test]
    return train, val, test


@dataclass
class Fold:
    """One train-then-evaluate setup.

    X:       binary CSR matrix, rows = users, columns = items, built only from
             the `history` window. X[u, i] = 1 if user u interacted with item i.
    users:   original user id of each row.
    items:   original item id of each column.
    history: the history-window ratings, with row/col indices added as u/i.
    targets: row index -> column indices the user interacted with in the
             future window. These are what we try to recommend.
    """

    X: sp.csr_matrix
    users: np.ndarray
    items: np.ndarray
    history: pd.DataFrame
    targets: dict
    n_cold_item_rows: int  # future interactions on items the models never saw


def make_fold(history: pd.DataFrame, future: pd.DataFrame) -> Fold:
    users = np.sort(history["user"].unique())
    items = np.sort(history["item"].unique())
    u_index = pd.Series(np.arange(len(users)), index=users)
    i_index = pd.Series(np.arange(len(items)), index=items)

    history = history.assign(
        u=u_index.loc[history["user"]].to_numpy(),
        i=i_index.loc[history["item"]].to_numpy(),
    )
    X = sp.csr_matrix(
        (np.ones(len(history), dtype=np.float32), (history["u"], history["i"])),
        shape=(len(users), len(items)),
    )
    X.data[:] = 1.0  # collapse any duplicates to a plain 0/1 interaction

    # Only users with some history can get collaborative recommendations
    # (cold-start users are a separate problem), and only items in the
    # training catalogue can be scored by any of these models.
    warm = future[future["user"].isin(u_index.index)]
    known = warm["item"].isin(i_index.index)
    warm = warm[known]
    targets = {
        int(u_index[user]): i_index.loc[group["item"].unique()].to_numpy()
        for user, group in warm.groupby("user")
    }
    return Fold(X, users, items, history, targets, n_cold_item_rows=int((~known).sum()))
