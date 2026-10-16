"""Two-stage recommender: ALS proposes candidates, gradient-boosted trees re-rank them.

Stage 1 (recall): ALS scores the whole catalogue cheaply; keep each user's top
N as candidates.
Stage 2 (precision): for each (user, candidate) pair, build a feature row that
mixes signals no single CF model has (recent popularity, release year, average
rating, the item-kNN opinion, ...) and let a scikit-learn
HistGradientBoostingClassifier predict P(user interacts with item). Sort the
candidates by that probability and keep the top 10.

Training labels come from the validation window: models fitted on train
propose candidates, and a candidate is a positive if the user really did
interact with it during validation. At test time the same features are
computed from models refitted on train + validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Fold
from .evaluate import mask_seen, top_k

FEATURES = [
    "als_score",         # stage-1 score
    "als_rank",          # position in the user's ALS list (0 = best)
    "knn_score",         # second CF opinion from item-kNN
    "pop_share",         # item's share of all interactions in the history window
    "recent_pop_share",  # item's share of interactions in the window's last 30 days
    "item_mean_rating",  # average star rating the item received
    "release_year",      # from the movie title
    "user_activity",     # log(1 + number of interactions the user has)
]


def _item_table(fold: Fold, release_years: pd.Series, recent_days: int = 30) -> pd.DataFrame:
    """Per-item features. Popularity is a *share*, not a raw count, because
    activity in MovieLens collapses over time: the 30 days before the
    validation window hold 115,686 interactions, the 30 days before the test
    window only 4,728. Shares mean the same thing in both; counts would not."""
    h = fold.history
    n_items = len(fold.items)
    recent = h[h["ts"] >= h["ts"].max() - recent_days * 86400]
    return pd.DataFrame(
        {
            "pop_share": np.bincount(h["i"], minlength=n_items) / len(h),
            "recent_pop_share": np.bincount(recent["i"], minlength=n_items) / len(recent),
            "item_mean_rating": h.groupby("i")["rating"].mean().reindex(range(n_items)).to_numpy(),
            "release_year": release_years.reindex(fold.items).to_numpy(),
        }
    )


def build_candidates(fold, als, knn, users, release_years, n_candidates=100) -> pd.DataFrame:
    """One row per (user, candidate item) with features and a 0/1 label."""
    als_scores = mask_seen(als.score(users), fold, users)
    cand = top_k(als_scores, n_candidates)  # (n_users, n_candidates) item columns
    knn_scores = knn.score(users)

    rows = pd.DataFrame(
        {
            "user": np.repeat(users, n_candidates),
            "item": cand.ravel(),
            "als_score": np.take_along_axis(als_scores, cand, axis=1).ravel(),
            "als_rank": np.tile(np.arange(n_candidates), len(users)),
            "knn_score": np.take_along_axis(knn_scores, cand, axis=1).ravel(),
        }
    )
    items = _item_table(fold, release_years)
    rows = rows.join(items, on="item")
    activity = np.log1p(np.asarray(fold.X.sum(axis=1)).ravel())
    rows["user_activity"] = activity[rows["user"]]

    # Label: did this user interact with this item in the future window?
    n_items = len(fold.items)
    target_keys = np.concatenate([u * n_items + fold.targets[u] for u in users])
    rows["label"] = np.isin(rows["user"] * n_items + rows["item"], target_keys).astype(int)
    return rows


def train_reranker(train_rows: pd.DataFrame) -> HistGradientBoostingClassifier:
    # Fixed, conservative settings: we have no third time window left to tune
    # them on without touching the test set. early_stopping=False keeps the run
    # deterministic (early stopping would hold out a random 10% of rows).
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    return model.fit(train_rows[FEATURES], train_rows["label"])


def rerank(model, rows: pd.DataFrame, users: np.ndarray, k: int = 10) -> np.ndarray:
    """Sort each user's candidates by predicted probability; return top-k item columns."""
    prob = model.predict_proba(rows[FEATURES])[:, 1]
    prob = prob.reshape(len(users), -1)  # rows are grouped by user, in `users` order
    items = rows["item"].to_numpy().reshape(len(users), -1)
    return np.take_along_axis(items, top_k(prob, k), axis=1)
