"""End-to-end pipeline: download -> split -> tune on validation -> evaluate on test.

    python run.py           # writes results/results.md, results.json, tuning_validation.csv

Protocol (the test window is touched exactly once per model):
  1. Tune: fit each hyperparameter setting on TRAIN, score it on VALIDATION,
     keep the setting with the best NDCG@10.
  2. Refit the chosen setting on TRAIN + VALIDATION and evaluate on TEST.
  3. Re-ranker: learn from VALIDATION labels using stage-1 models fitted on
     TRAIN, then apply it to TEST candidates from the refitted models.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from download_data import download
from recsys.data import load_ratings, load_release_years, make_fold, temporal_split
from recsys.evaluate import evaluate, evaluate_lists
from recsys.models import ALS, ItemKNN, Popularity
from recsys.reranker import FEATURES, build_candidates, rerank, train_reranker

MODELS = {"Popularity": Popularity, "Item-kNN": ItemKNN, "ALS": ALS}
GRIDS = {
    "Popularity": [{}],
    "Item-kNN": [{"k": k} for k in (50, 100, 200, 400, 800, 1600)],
    "ALS": [
        {"factors": f, "reg": reg, "alpha": a}
        for f in (32, 64, 128)
        for reg in (100, 200, 400)
        for a in (2, 5, 10)
    ],
}
K = 10
RESULTS = Path("results")


def tune(name, fold, log):
    """Grid search on the validation window; returns the best params by NDCG@10."""
    best_params, best_ndcg = None, -1.0
    for params in GRIDS[name]:
        res = evaluate(MODELS[name](**params).fit(fold.X), fold, K)
        log.append(
            {
                "model": name,
                **params,
                "val_ndcg@10": res["ndcg@10"],
                "val_recall@10": res["recall@10"],
                "val_coverage@10": res["coverage@10"],
            }
        )
        print(f"  [tune] {name:10s} {params}  val NDCG@10={res['ndcg@10']:.4f}", flush=True)
        if res["ndcg@10"] > best_ndcg:
            best_params, best_ndcg = params, res["ndcg@10"]
    return best_params


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Mean of (a - b) over users, with a 95% CI from resampling users.

    Both arrays are per-user scores for the same users, so the comparison is
    paired: a user who is easy for one model is usually easy for the other.
    """
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def describe(df):
    span = pd.to_datetime(df["ts"], unit="s")
    return f"{len(df):,} interactions, {span.min():%Y-%m-%d} to {span.max():%Y-%m-%d}"


def main():
    start = time.time()
    data_dir = download()
    ratings = load_ratings(data_dir)
    years = load_release_years(data_dir)

    train, val, test = temporal_split(ratings)
    val_fold = make_fold(train, val)
    test_fold = make_fold(pd.concat([train, val]), test)
    split_info = {
        "train": describe(train),
        "validation": describe(val),
        "test": describe(test),
        "val_eval_users": len(val_fold.targets),
        "test_eval_users": len(test_fold.targets),
        "test_cold_item_interactions": test_fold.n_cold_item_rows,
    }
    print("[split]", *(f"{k}: {v}" for k, v in split_info.items()), sep="\n  ")

    # 1. Tune every model on validation.
    log = []
    best = {name: tune(name, val_fold, log) for name in MODELS}

    # 2. Refit on train + validation, evaluate once on test.
    fitted, results = {}, {}
    for name, cls in MODELS.items():
        fitted[name] = cls(**best[name]).fit(test_fold.X)
        results[name] = evaluate(fitted[name], test_fold, K)
        print(f"[test] {name:10s} NDCG@10={results[name]['ndcg@10']:.4f} Recall@10={results[name]['recall@10']:.4f}")

    # 3. Two-stage: ALS candidates re-ranked by gradient-boosted trees.
    val_users = np.array(sorted(val_fold.targets))
    als_val = ALS(**best["ALS"]).fit(val_fold.X)
    knn_val = ItemKNN(**best["Item-kNN"]).fit(val_fold.X)
    train_rows = build_candidates(val_fold, als_val, knn_val, val_users, years)
    gbm = train_reranker(train_rows)
    test_users = np.array(sorted(test_fold.targets))
    test_rows = build_candidates(test_fold, fitted["ALS"], fitted["Item-kNN"], test_users, years)
    name = "ALS + GBM re-ranker"
    results[name] = evaluate_lists(test_fold, test_users, rerank(gbm, test_rows, test_users, K), K)
    best[name] = {"candidates": 100, "features": len(FEATURES)}
    print(f"[test] {name} NDCG@10={results[name]['ndcg@10']:.4f} Recall@10={results[name]['recall@10']:.4f}")

    # How many of each user's future items made it into the 100 candidates?
    # This caps what any re-ranker could achieve.
    hits = test_rows.groupby("user")["label"].sum()
    sizes = pd.Series({u: len(t) for u, t in test_fold.targets.items()})
    candidate_recall = float((hits / sizes.loc[hits.index]).mean())

    # Post-hoc analysis only (no decisions are made from it): how much does the
    # re-ranker's ability to separate positives (ROC AUC) drop when one feature
    # is shuffled?
    imp = permutation_importance(
        gbm, test_rows[FEATURES], test_rows["label"], scoring="roc_auc", n_repeats=5, random_state=0
    )
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    # Significance: paired bootstrap over test users.
    comparisons = {}
    for name in results:
        if name != "Popularity":
            comparisons[f"{name} vs Popularity"] = paired_bootstrap(
                results[name]["per_user"]["ndcg"], results["Popularity"]["per_user"]["ndcg"]
            )
    comparisons["ALS + GBM re-ranker vs ALS"] = paired_bootstrap(
        results["ALS + GBM re-ranker"]["per_user"]["ndcg"], results["ALS"]["per_user"]["ndcg"]
    )
    comparisons["ALS vs Item-kNN"] = paired_bootstrap(
        results["ALS"]["per_user"]["ndcg"], results["Item-kNN"]["per_user"]["ndcg"]
    )

    same_window = same_window_check(val_fold, val_users, train_rows, als_val)

    write_results(
        split_info, best, results, comparisons, candidate_recall, importance, same_window, log, time.time() - start
    )


def same_window_check(fold, users, rows, als):
    """Diagnostic: does the re-ranker help when it is trained and evaluated in
    the SAME time window (validation), on different users?

    2-fold split of validation users: train on one half, re-rank the other,
    then swap, so every user is scored by a re-ranker that never saw them.
    Comparing this with the test-window result separates "the method does not
    work" from "what it learned does not carry over to a later period".
    """
    halves = np.array_split(np.random.default_rng(0).permutation(users), 2)
    reranked = {}
    for fit_half, eval_half in [(halves[0], halves[1]), (halves[1], halves[0])]:
        eval_half = np.sort(eval_half)
        model = train_reranker(rows[rows["user"].isin(fit_half)])
        recs = rerank(model, rows[rows["user"].isin(eval_half)], eval_half, K)
        reranked.update(zip(eval_half, recs))
    rr = evaluate_lists(fold, users, np.array([reranked[u] for u in users]), K)
    base = evaluate(als, fold, K)
    return {
        "users": int(len(users)),
        "als_ndcg@10": base["ndcg@10"],
        "reranked_ndcg@10": rr["ndcg@10"],
        "als_recall@10": base["recall@10"],
        "reranked_recall@10": rr["recall@10"],
        "ndcg_difference_95ci": paired_bootstrap(rr["per_user"]["ndcg"], base["per_user"]["ndcg"]),
    }


def write_results(split_info, best, results, comparisons, candidate_recall, importance, same_window, log, seconds):
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(RESULTS / "tuning_validation.csv", index=False)

    summary = {
        "split": split_info,
        "chosen_params": best,
        "test": {n: {k: v for k, v in r.items() if k != "per_user"} for n, r in results.items()},
        "ndcg_difference_95ci": {n: {"mean": m, "lo": lo, "hi": hi} for n, (m, lo, hi) in comparisons.items()},
        "reranker_candidate_recall@100": candidate_recall,
        "reranker_permutation_importance_auc": importance,
        "reranker_same_window_check": same_window,
        "runtime_seconds": round(seconds, 1),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (RESULTS / "results.json").write_text(json.dumps(summary, indent=2))

    def fmt_params(p):
        return ", ".join(f"{k}={v}" for k, v in p.items()) or "-"

    lines = [
        "# Results (MovieLens-1M, temporal split, test window)",
        "",
        f"Evaluated on {split_info['test_eval_users']} test users. Generated by `run.py`; do not edit by hand.",
        "",
        "| Model | Chosen on validation | NDCG@10 | Recall@10 | Coverage@10 |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {fmt_params(best[name])} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {r['coverage@10']:.3f} |"
        )
    lines += [
        "",
        "NDCG@10 differences, mean over test users with a paired-bootstrap 95% CI:",
        "",
        "| Comparison | Δ NDCG@10 | 95% CI |",
        "|---|---|---|",
    ]
    for name, (m, lo, hi) in comparisons.items():
        lines.append(f"| {name} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += [
        "",
        f"Re-ranker candidate recall@100 (ceiling for stage 2): {candidate_recall:.4f}",
        "",
        "Re-ranker trained and evaluated in the same window (validation, 2-fold split by user, "
        f"{same_window['users']} users):",
        "",
        "| | NDCG@10 | Recall@10 |",
        "|---|---|---|",
        f"| ALS | {same_window['als_ndcg@10']:.4f} | {same_window['als_recall@10']:.4f} |",
        f"| ALS + GBM re-ranker | {same_window['reranked_ndcg@10']:.4f} | {same_window['reranked_recall@10']:.4f} |",
        "",
        "Δ NDCG@10 = {:+.4f}, 95% CI [{:+.4f}, {:+.4f}]".format(*same_window["ndcg_difference_95ci"]),
        "",
        "Re-ranker permutation importance (drop in ROC AUC on test candidates when the feature is shuffled):",
        "",
        "| Feature | Δ AUC |",
        "|---|---|",
        *[f"| {f} | {v:.4f} |" for f, v in importance.items()],
        "",
        "Split:",
        "",
        *[f"- {k}: {v}" for k, v in split_info.items()],
        "",
        f"Runtime: {seconds:.0f} s on {platform.platform()} (Python {platform.python_version()}, "
        f"NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}).",
        "",
    ]
    (RESULTS / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
