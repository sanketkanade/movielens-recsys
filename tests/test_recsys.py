"""Small checks on the parts that are easy to get subtly wrong."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.data import make_fold, temporal_split
from recsys.evaluate import top_k
from recsys.metrics import ndcg_at_k, recall_at_k
from recsys.models import ALS, ItemKNN


def test_recall_counts_hits_over_relevant():
    assert recall_at_k([1, 2, 3], [3, 9], k=3) == 0.5
    assert recall_at_k([1, 2, 3], [3, 9], k=2) == 0.0


def test_ndcg_hand_computed():
    # One relevant item, found at position 2: DCG = 1/log2(3), IDCG = 1.
    assert ndcg_at_k([5, 7, 8], [7], k=3) == pytest.approx(1 / np.log2(3))
    # Perfect ranking scores 1, even with fewer relevant items than k.
    assert ndcg_at_k([7, 5, 8], [7, 5], k=3) == pytest.approx(1.0)


def test_top_k_is_sorted_best_first():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(scores, 3).tolist() == [[1, 3, 2]]


def test_temporal_split_has_no_leakage():
    ratings = pd.DataFrame({"user": 1, "item": range(100), "rating": 4, "ts": range(100)})
    train, val, test = temporal_split(ratings)
    assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()


def test_fold_drops_unknown_users_and_items():
    history = pd.DataFrame({"user": [1, 1, 2], "item": [10, 20, 10], "rating": 5, "ts": 0})
    future = pd.DataFrame({"user": [1, 1, 3], "item": [10, 30, 20], "rating": 5, "ts": 1})
    fold = make_fold(history, future)
    # User 3 has no history; item 30 was never seen, so it is counted as cold.
    assert list(fold.targets) == [0]
    assert fold.targets[0].tolist() == [0]
    assert fold.n_cold_item_rows == 1


def test_item_knn_prefers_co_occurring_items():
    # Items 0 and 1 always appear together; item 2 is on its own.
    X = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32))
    scores = ItemKNN(k=2).fit(X).score(np.array([3]))
    assert scores[0, 1] > scores[0, 2]


def test_als_loss_never_increases():
    X = sp.random(60, 40, density=0.1, format="csr", random_state=0)
    X.data[:] = 1.0
    model = ALS(factors=8, reg=1.0, alpha=5.0, iterations=8).fit(X, verbose=True)
    assert all(b <= a + 1e-6 for a, b in zip(model.losses, model.losses[1:]))
