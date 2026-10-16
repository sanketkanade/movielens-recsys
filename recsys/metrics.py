"""Top-K ranking metrics with binary relevance.

For one user, `ranked` is the model's ordered list of recommended item ids and
`relevant` is the set of items they actually interacted with in the future
window. Each function returns a number in [0, 1]; we average over users.
"""

import numpy as np


def recall_at_k(ranked, relevant, k: int = 10) -> float:
    """Share of the user's future items that appear in the top k."""
    hits = len(set(ranked[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(ranked, relevant, k: int = 10) -> float:
    """Normalised discounted cumulative gain.

    A hit at position p (1-based) earns 1 / log2(p + 1), so a hit at the top
    counts fully and a hit at position 10 counts ~0.29. We divide by the best
    possible score (all top slots filled with relevant items) so 1.0 means a
    perfect ranking.
    """
    relevant = set(relevant)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # positions 1..k
    dcg = sum(discounts[p] for p, item in enumerate(ranked[:k]) if item in relevant)
    idcg = discounts[: min(k, len(relevant))].sum()
    return dcg / idcg
