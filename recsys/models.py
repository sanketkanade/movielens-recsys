"""Three recommenders that share one interface.

    model.fit(X)          X is the binary users x items CSR matrix
    model.score(users)    dense (len(users), n_items) array; higher = better

Recommending is then the same for every model: take the highest-scoring
items the user has not already interacted with (see evaluate.py).
"""

import numpy as np
import scipy.sparse as sp


class Popularity:
    """Recommend the items with the most interactions, to everyone.

    No personalisation at all. It is the baseline every other model must beat;
    on many datasets it is surprisingly hard to beat by much.
    """

    def fit(self, X: sp.csr_matrix):
        self.counts = np.asarray(X.sum(axis=0)).ravel()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.counts, (len(users), 1))


class ItemKNN:
    """Item-based collaborative filtering ("people who liked X also liked Y").

    1. Similarity between two items = cosine of their user columns in X, i.e.
       co-occurrence count / sqrt(popularity_a * popularity_b). Dividing by
       popularity stops blockbusters from being "similar" to everything.
    2. For each item keep only its k most similar neighbours (removes noise
       from weak, incidental co-occurrences and is a hyperparameter).
    3. A user's score for item j = sum of similarities between j and the
       items the user already interacted with.
    """

    def __init__(self, k: int = 100):
        self.k = k

    def fit(self, X: sp.csr_matrix):
        X = X.astype(np.float32)
        norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel())  # binary X: sum = squared norm
        norms[norms == 0] = 1.0
        co = (X.T @ X).toarray()  # items x items co-occurrence counts
        sim = co / norms[:, None] / norms[None, :]
        np.fill_diagonal(sim, 0.0)  # an item is not its own neighbour

        # Column j holds the similarities of every item to target item j.
        # Zero all but the k largest in each column.
        if self.k < sim.shape[0]:
            drop = np.argpartition(sim, -self.k, axis=0)[: -self.k]
            np.put_along_axis(sim, drop, 0.0, axis=0)
        self.X = X
        self.sim = sim
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[users] @ self.sim)


class ALS:
    """Matrix factorisation for implicit feedback, solved by alternating least squares.

    Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit Feedback
    Datasets". Each user u gets a vector x_u and each item i a vector y_i
    (`factors` numbers each); the predicted preference is their dot product.

    Every cell of the matrix is used, not just the observed ones:
      preference p_ui = 1 if u interacted with i, else 0
      confidence c_ui = 1 + alpha if interacted, else 1
    and we minimise
      sum_ui c_ui (p_ui - x_u . y_i)^2 + reg * (|x_u|^2 + |y_i|^2).

    A missing interaction is weak evidence of disinterest (the user may simply
    not have seen the item), so it gets confidence 1, while an observed one
    gets 1 + alpha.

    With item vectors fixed, each user's problem is ordinary weighted ridge
    regression with a closed-form solution; then fix users and solve items.
    Each half-step can only lower the loss, so it converges in a few sweeps.
    """

    def __init__(self, factors=64, reg=1.0, alpha=10.0, iterations=15, seed=42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, X: sp.csr_matrix, verbose: bool = False):
        rng = np.random.default_rng(self.seed)
        X = X.tocsr()
        Xt = X.T.tocsr()
        self.user_factors = rng.normal(0, 0.01, (X.shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, (X.shape[1], self.factors))
        self.losses = []
        for _ in range(self.iterations):
            self.user_factors = self._solve(X, self.item_factors)
            self.item_factors = self._solve(Xt, self.user_factors)
            if verbose:
                self.losses.append(self.loss(X))
        return self

    def _solve(self, R: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        """Solve for every row of R given the other side's factors Y.

        Setting the gradient to zero gives, for one user u:
          (Y^T C_u Y + reg I) x_u = Y^T C_u p_u
        C_u differs from the identity only on u's observed items, so
          Y^T C_u Y = Y^T Y + alpha * Y_u^T Y_u     (Y_u = rows of u's items)
          Y^T C_u p_u = (1 + alpha) * sum of Y_u's rows
        Y^T Y is shared by every user, so each solve only touches that
        user's own items. That is what makes ALS cheap on sparse data.
        """
        base = Y.T @ Y + self.reg * np.eye(self.factors)
        out = np.zeros((R.shape[0], self.factors))
        for row in range(R.shape[0]):
            idx = R.indices[R.indptr[row] : R.indptr[row + 1]]
            if len(idx) == 0:
                continue
            Yu = Y[idx]
            A = base + self.alpha * (Yu.T @ Yu)
            b = (1.0 + self.alpha) * Yu.sum(axis=0)
            out[row] = np.linalg.solve(A, b)
        return out

    def loss(self, X: sp.csr_matrix) -> float:
        """The training objective, computed without building the dense matrix.

        sum over ALL cells of (x_u . y_i)^2 = trace(U^T U V^T V); then correct
        the observed cells, where the target is 1 and the weight is 1 + alpha.
        """
        U, V = self.user_factors, self.item_factors
        rows, cols = X.nonzero()
        pred = np.einsum("ij,ij->i", U[rows], V[cols])
        total = np.trace((U.T @ U) @ (V.T @ V))
        total += np.sum((1 + self.alpha) * (1 - pred) ** 2 - pred**2)
        total += self.reg * (np.sum(U**2) + np.sum(V**2))
        return float(total)

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors[users] @ self.item_factors.T
