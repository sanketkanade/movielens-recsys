"""Turn model scores into top-K lists and score them against the future window."""

import numpy as np

from .data import Fold
from .metrics import ndcg_at_k, recall_at_k


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k highest scores in each row, best first."""
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def mask_seen(scores: np.ndarray, fold: Fold, users: np.ndarray) -> np.ndarray:
    """Never recommend something the user already interacted with."""
    seen = fold.X[users].nonzero()
    scores[seen] = -np.inf
    return scores


def recommend(model, fold: Fold, users: np.ndarray, k: int = 10) -> np.ndarray:
    scores = mask_seen(model.score(users).astype(np.float64), fold, users)
    return top_k(scores, k)


def evaluate_lists(fold: Fold, users: np.ndarray, recs: np.ndarray, k: int = 10) -> dict:
    """Average the metrics over users. Keeps the per-user values for error bars."""
    ndcg = np.array([ndcg_at_k(r, fold.targets[u], k) for u, r in zip(users, recs)])
    recall = np.array([recall_at_k(r, fold.targets[u], k) for u, r in zip(users, recs)])
    return {
        f"ndcg@{k}": float(ndcg.mean()),
        f"recall@{k}": float(recall.mean()),
        # Share of the catalogue that appears in at least one user's list:
        # a crude measure of how much the model only pushes the same few hits.
        f"coverage@{k}": float(len(np.unique(recs)) / fold.X.shape[1]),
        "users": int(len(users)),
        "per_user": {"ndcg": ndcg, "recall": recall},
    }


def evaluate(model, fold: Fold, k: int = 10) -> dict:
    users = np.array(sorted(fold.targets))
    return evaluate_lists(fold, users, recommend(model, fold, users, k), k)
