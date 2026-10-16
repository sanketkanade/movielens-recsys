# MovieLens-1M recommender: popularity vs item-kNN vs ALS, plus a GBM re-ranker

A small, reproducible top-10 recommender study on MovieLens-1M. It compares a
popularity baseline, item-based collaborative filtering, implicit-feedback
matrix factorisation (ALS, implemented from scratch in NumPy) and a two-stage
ALS + scikit-learn gradient-boosted re-ranker. Evaluation uses a time-based
split, so every model is trained on the past and tested on the future.

`docker build` and `docker run` reproduce every number below from a fresh
download of the data (the run quoted here took 330 s on a 4-CPU Docker VM on
an Apple M4).

## Results

Test window: 20 Jun 2001 to 28 Feb 2003, 809 users with prior history.
Hyperparameters were chosen on the validation window only; the test window
was scored once per model. Copied from [`results/results.md`](results/results.md),
which `run.py` generates.

| Model | Chosen on validation | NDCG@10 | Recall@10 | Coverage@10 |
|---|---|---|---|---|
| Popularity | - | 0.2187 | 0.0434 | 0.043 |
| Item-kNN | k=1600 | 0.2271 | 0.0443 | 0.086 |
| **ALS** | factors=128, reg=400, alpha=10 | **0.2394** | 0.0525 | 0.122 |
| ALS + GBM re-ranker | 100 candidates, 8 features | 0.2226 | **0.0543** | **0.140** |

Is the difference real? Paired bootstrap over test users (2,000 resamples):

| Comparison | Δ NDCG@10 | 95% CI |
|---|---|---|
| Item-kNN vs Popularity | +0.0085 | [-0.0020, +0.0197] |
| ALS vs Popularity | +0.0208 | [+0.0105, +0.0316] |
| ALS vs Item-kNN | +0.0123 | [+0.0029, +0.0217] |
| ALS + GBM re-ranker vs ALS | -0.0168 | [-0.0308, -0.0027] |

The same re-ranker, trained and evaluated inside the validation window
(2-fold split by user, 922 users), does help:

| Validation window | NDCG@10 | Recall@10 |
|---|---|---|
| ALS | 0.1800 | 0.0512 |
| ALS + GBM re-ranker | 0.2192 | 0.0774 |

Δ NDCG@10 = +0.0392, 95% CI [+0.0265, +0.0518].

In short:

- ALS is the best model on the future window: NDCG@10 is 9.5% higher than
  popularity (Recall@10 21.0% higher) and it reaches 2.8x as much of the
  catalogue. The 95% CI of the NDCG gain excludes zero, over both popularity
  and item-kNN.
- Item-kNN beats popularity on paper, but the confidence interval includes
  zero, so on this data I can't claim it is better.
- The re-ranker improves NDCG@10 by 21.8% when trained and tested in the same
  period, but loses 7.0% when applied to the following period. What it learned
  about the first half of 2001 (which films and release years were in demand) did not
  carry over to 2001-2003. The [lessons](#what-i-learned) section covers what
  that means.

## Run it

With Docker:

```bash
docker build -t movielens-recsys .
docker run --rm -v "$PWD/results:/app/results" movielens-recsys
```

The build runs the unit tests and fails if they fail. The run downloads
MovieLens-1M (checked against GroupLens's published MD5), tunes, evaluates,
and writes `results/results.md`, `results/results.json` and
`results/tuning_validation.csv`.

Without Docker (Python 3.12):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python run.py
```

The pipeline is deterministic (fixed seeds, no random early stopping). The
macOS run and the Linux container produced identical results to four decimal
places; only the runtime line differs.

## The problem

Given everything a user has interacted with so far, rank the items they are
most likely to interact with next and show the top 10. This is top-N
recommendation from implicit feedback: the model learns from which items a user
touched, not from how much they liked them. Every rating counts as one
interaction and the star value is ignored. That matches property search, where
the signals are views, saves and enquiries rather than ratings.

## Data

[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/): 1,000,209 ratings
from 6,040 users on 3,706 movies, collected from April 2000 to February 2003.
Every user has at least 20 ratings. [`download_data.py`](download_data.py)
fetches it at run time. The licence does not allow redistribution, so the data
is never committed or baked into the image.

## Split: global temporal, not leave-last-out

All interactions are sorted by timestamp and cut at two global points:

| Window | Share of interactions | Dates | Used for |
|---|---|---|---|
| Train | oldest 90% (900,188) | 25 Apr 2000 to 29 Dec 2000 | fitting during tuning |
| Validation | next 5% (50,010) | 29 Dec 2000 to 20 Jun 2001 | choosing hyperparameters; re-ranker labels |
| Test | newest 5% (50,011) | 20 Jun 2001 to 28 Feb 2003 | final evaluation, after refitting on train + validation |

Why this split and not per-user leave-last-out:

1. **No leakage from the future.** With leave-last-out, user A's held-out item
   can be from 2000 while user B's training data runs to 2003, so the model is
   trained on interactions that happened after the moment it is evaluated on.
   A global cutoff makes that impossible, and a unit test checks it.
2. **It mirrors production.** A deployed recommender is trained on everything
   up to yesterday and serves tomorrow. A global cutoff evaluates exactly that,
   including drift in what people want, and that drift turned out to be the
   most important finding here.
3. **Several future items per user.** Each test user has a median of 29 future
   items, so Recall@10 measures something more useful than a single hit or miss.

The cutoffs are quantiles of interaction counts, not calendar dates, because
the activity is very uneven: 90.5% of all ratings were made in 2000. The cost
of this split is that only users who come back after the cutoff can be
evaluated. That is 809 of the 813 users active in the test window; the other 4
had no history and are a cold-start case these models cannot handle. Of the
50,011 test interactions, 25 involve movies never seen in training. Those are
dropped from the targets because no collaborative model can score them, and
they are counted in the results.

## Models

All three share one interface ([`recsys/models.py`](recsys/models.py)):
`fit(X)` on a binary users x items sparse matrix, and `score(users)`, which
returns a score for every item. [`recsys/evaluate.py`](recsys/evaluate.py) then
removes items the user has already seen and keeps the top 10.

**Popularity.** Score = number of interactions with the item. Everyone gets the
same list, minus what they have already seen. Any personalised model has to
beat it, and under a temporal split it is hard to beat by much.

**Item-kNN.** Cosine similarity between item columns:
`co-occurrences(a, b) / sqrt(popularity(a) * popularity(b))`. Dividing by
popularity stops blockbusters from looking similar to everything. Each item
keeps only its k most similar neighbours, and a user's score for item j is the
sum of j's similarities to the items they already have. k was tuned over
{50, 100, 200, 400, 800, 1600}; validation NDCG rose steadily with k (0.1417 at
k=50 to 0.1711 at k=1600). Larger k makes the scores lean more on popularity,
and coverage falls as it does.

**ALS (implicit-feedback matrix factorisation).** From Hu, Koren & Volinsky
(2008). Each user and item gets a vector, and the predicted preference is their
dot product. Unlike rating prediction, every cell of the matrix is a training
example: preference is 1 if the user interacted and 0 otherwise, weighted by a
confidence of `1 + alpha` for observed cells and `1` for the rest. A missing
interaction is weak evidence of disinterest (the user may never have seen the
item), so it is down-weighted rather than ignored. The objective is

```
sum over all (u, i) of  c_ui * (p_ui - x_u . y_i)^2   +   reg * (|x_u|^2 + |y_i|^2)
```

With item vectors fixed, each user's vector has a closed-form ridge-regression
solution, and vice versa, so training alternates between the two.
[`ALS._solve`](recsys/models.py) uses the standard speed-up:
`Y^T C_u Y = Y^T Y + alpha * Y_u^T Y_u`, where `Y^T Y` is shared by every user,
so each solve only touches that user's own items. The training loss is computed
without building the dense matrix, and a unit test checks it never increases.
The grid was factors {32, 64, 128} x reg {100, 200, 400} x alpha {2, 5, 10}.
Validation NDCG ranged from 0.1534 to 0.1800. `reg` and `alpha` trade off
against each other: a larger `alpha` needed a larger `reg`, and the best
settings sit along that diagonal of the grid. Model size hardly mattered: the best result at 32 factors was 0.1796, at 128
factors 0.1800.

**Two-stage: ALS candidates + gradient-boosted re-ranker**
([`recsys/reranker.py`](recsys/reranker.py)). This is the shape most production
recommenders take. A cheap model narrows the catalogue to ~100 candidates per
user, then a richer model re-orders them. The re-ranker is scikit-learn's
`HistGradientBoostingClassifier`, which predicts P(user interacts with item)
for each candidate from 8 features: ALS score and rank, item-kNN score, overall
and last-30-days popularity, average star rating, release year, and user
activity. Popularity features are shares rather than counts, because activity
collapses over time: the 30 days before the validation window hold 115,686
interactions, the 30 days before the test window only 4,728. To train
it, models fitted on the train window propose candidates for validation users,
and a candidate is labelled positive if the user really did interact with it
during validation. At test time the same features come from models refitted on
train + validation. Its settings are fixed and conservative, because there is
no third time window to tune them on without touching the test set. On
average only 27% of a test user's future items make it into ALS's top 100
(candidate recall@100 = 0.2696), which caps what any re-ranker could recover.

## Evaluation

For each test user, the model ranks every item they have not seen, and the
top 10 is compared with what they actually interacted with in the test window.

- **Recall@10** = (future items in the top 10) / (number of future items). Averaged over users.
- **NDCG@10** gives a hit at position p a weight of `1 / log2(p + 1)` and divides
  by the best achievable score, so 1.0 is a perfect ordering. It rewards putting
  hits near the top, where users actually look.
- **Coverage@10** = share of the catalogue that appears in at least one user's
  top 10. It is a rough check on whether a model just repeats the same hits.
- **Paired bootstrap CI.** Per-user NDCG differences between two models,
  resampled 2,000 times. Pairing matters because a user who is easy for one
  model is usually easy for the other.

Metric values from different windows are not comparable with each other. Test
users have more future items (median 29) than validation users, which raises
every model's test NDCG. Compare models within a window.

## What I learned

- **Popularity is a strong baseline when you evaluate honestly.** People who come
  back months later lean towards what is popular at the time. Item-kNN's gain
  over popularity was within noise, and with 809 users a 0.0085 NDCG gap is not
  enough to call. Reporting confidence intervals stopped me from claiming a win
  I didn't have.
- **For ALS, regularisation mattered far more than model size.** Too little
  regularisation for the chosen `alpha` and ALS chases the long tail. Too much
  and it collapses into a popularity model. Coverage in the validation grid
  shows both: averaged over model sizes it ranged from 0.276 (reg=100,
  alpha=10) down to 0.041 (reg=400, alpha=2), close to popularity's 0.040. The
  worst setting by NDCG (0.1534) was the long-tail one, with 0.285 coverage.
- **The evaluation design can flip the conclusion.** The re-ranker was a clear
  win (+21.8% NDCG) when trained and tested on the same period and a clear loss
  (-7.0%) on the next period. A random or leave-last-out split would have
  reported the win. In permutation importance on the test candidates, shuffling
  `als_score` or `pop_share` slightly *improved* AUC (-0.0027 and -0.0055), so
  the patterns the model learned from those features did not carry over to the
  later period. `als_rank` was the most useful feature (+0.0386 AUC). In
  production, that points to retraining the re-ranker often on the most recent
  window and preferring features whose meaning is stable over time.
- **Writing ALS by hand** (deriving the update, the `Y^T Y` trick, checking the
  loss falls every sweep) made the hyperparameters make sense. `alpha` is how
  much more an observed interaction counts than a missing one, and `reg` is how
  far the vectors are pulled towards zero.

## From movies to property listings

How this maps to recommending listings from views, saves and enquiries:

- **Implicit signals of different strength.** ALS already has a slot for this:
  the confidence `c_ui`. Instead of `1 + alpha` for any interaction, it could
  be built as a weighted sum, with an enquiry worth more than a save and a save
  worth more than a view. The weights would be tuned on a validation window
  against the outcome the business cares about, most likely enquiries.
- **Temporal evaluation and retraining.** Property demand shifts with
  seasons, interest rates and suburbs heating up. The re-ranker result here is
  a small version of that problem. I would evaluate on future windows, retrain
  on a fixed cadence, and monitor feature distributions for drift.
- **Cold-start listings are the main difference.** In this test window only 25
  of 50,011 interactions (0.05%) were on movies the models had never seen, so
  MovieLens hides the problem. On a property site a listing is live for weeks,
  so a large share of the inventory is always new, and collaborative filtering
  cannot score an item with no interactions. Ways to handle it, roughly in the
  order I would try them:
  1. **Content features in the re-ranker**: suburb, price band, bedrooms,
     property type, and text or photo embeddings. They exist the moment a
     listing goes live, just as `release_year` exists for a movie here.
  2. **Content-based retrieval**: match a new listing to users whose
     viewed listings have similar attributes, as a second candidate source next
     to ALS.
  3. **ALS fold-in** once the first few views arrive: a new listing's vector is
     one ridge-regression solve against the existing user vectors (the item
     half of `ALS._solve`), with no retraining.
  4. **Exploration**: a small, controlled boost for new listings so they
     collect the interactions the other methods need.
- **Cold-start users** (a first visit, no history) can start from
  popularity within their search filters, then switch to session-based
  recommendations as they browse.
- **Other differences.** A listing is a one-off item that leaves the
  market when it sells. Hard filters (location, price, property type) shrink
  the candidate set before any ranking happens. And recommending more of the
  catalogue (the coverage column) matters more when every listing has a vendor
  and an agent waiting on it.

## Limitations

- One dataset and one split. These numbers are not comparable with papers that
  use leave-one-out or random splits.
- A movie rating is a stronger signal than a property view, and MovieLens
  users rated in bursts when they signed up, so the interaction process
  differs from browsing.
- The re-ranker is pointwise binary classification. A learning-to-rank objective
  (LambdaMART, for example LightGBM's `lambdarank`) is the natural next step. Its
  hyperparameters were not tuned.
- Only users with history are evaluated: 4 users new in the test window are
  excluded.
- The ALS solver loops over users in Python. That is fine for 6,000 users; at
  millions of users I would use a library such as `implicit` or Spark ALS.

## Layout

```
download_data.py        fetch + checksum MovieLens-1M into data/ (git-ignored)
run.py                  the whole experiment: split, tune, test, re-rank, write results/
recsys/data.py          loading, temporal split, sparse user x item matrix
recsys/models.py        Popularity, ItemKNN, ALS
recsys/reranker.py      candidate features + HistGradientBoostingClassifier re-ranker
recsys/metrics.py       Recall@K, NDCG@K
recsys/evaluate.py      top-K with seen items masked, per-user metrics
tests/test_recsys.py    metric, split, kNN and ALS checks (run during docker build)
results/                output of the Docker run quoted above
```

Data: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent Systems 5(4).
