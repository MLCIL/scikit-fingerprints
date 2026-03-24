"""
Hypeparameter tuning execution time and gain
scikit-fingerprints vs scikit-learn benchmark.
"""

import csv
import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarking.utils.timing import measure_time
from skfp.datasets.moleculenet import load_hiv
from skfp.fingerprints import RDKitFingerprint
from skfp.model_selection import FingerprintEstimatorGridSearch
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

N_REPEATS = 5
STEP = 500
DATASET_CUTOFF = 10000
RANDOM_STATE = 0

OUTPUTS_DIR = os.path.join("benchmark_times", "benchmark_times_saved")
PLOTS_DIR = os.path.join("benchmark_times", "benchmark_times_plotted")

USE_PDF = True  # If True, save plot as PDF, otherwise save as PNG
USE_ERROR_BARS = False  # If True, use error bars instead of shaded fill_between

CSV_FILENAME = "hp_tuning_benchmark.csv"
PLOT_TIME_FILENAME = "hp_tuning_benchmark_time"
PLOT_SCORE_FILENAME = "hp_tuning_benchmark_score"
PLOT_ABS_GAIN_FILENAME = "hp_tuning_benchmark_abs_gain"
PLOT_REL_GAIN_FILENAME = "hp_tuning_benchmark_rel_gain"

RESULT_CSV_PATH = os.path.join(OUTPUTS_DIR, CSV_FILENAME)
RESULT_PLOT_TIME_PATH = os.path.join(
    PLOTS_DIR, f"{PLOT_TIME_FILENAME}.pdf" if USE_PDF else f"{PLOT_TIME_FILENAME}.png"
)
RESULT_PLOT_SCORE_PATH = os.path.join(
    PLOTS_DIR, f"{PLOT_SCORE_FILENAME}.pdf" if USE_PDF else f"{PLOT_SCORE_FILENAME}.png"
)
RESULT_PLOT_ABS_GAIN_PATH = os.path.join(
    PLOTS_DIR,
    f"{PLOT_ABS_GAIN_FILENAME}.pdf" if USE_PDF else f"{PLOT_ABS_GAIN_FILENAME}.png",
)
RESULT_PLOT_REL_GAIN_PATH = os.path.join(
    PLOTS_DIR,
    f"{PLOT_REL_GAIN_FILENAME}.pdf" if USE_PDF else f"{PLOT_REL_GAIN_FILENAME}.png",
)

CSV_HEADER = [
    "n",
    "baseline_score",
    "small_skfp_time",
    "small_skfp_time_std",
    "small_skfp_score",
    "small_skfp_abs_gain",
    "small_skfp_rel_gain",
    "small_sklearn_time",
    "small_sklearn_time_std",
    "small_sklearn_score",
    "small_sklearn_abs_gain",
    "small_sklearn_rel_gain",
    "large_skfp_time",
    "large_skfp_time_std",
    "large_skfp_score",
    "large_skfp_abs_gain",
    "large_skfp_rel_gain",
    "large_sklearn_time",
    "large_sklearn_time_std",
    "large_sklearn_score",
    "large_sklearn_abs_gain",
    "large_sklearn_rel_gain",
]

PLOT_CONFIGS = [
    # (col_prefix, label, marker, color)
    ("small_skfp", "skfp small grid", "o", "orange"),
    ("small_sklearn", "sklearn small grid", "s", "red"),
    ("large_skfp", "skfp large grid", "o", "limegreen"),
    ("large_sklearn", "sklearn large grid", "s", "darkgreen"),
]


def main():
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    run_benchmark()
    plot_results()


def measure_skfp(
    smiles: list[str],
    labels: np.ndarray,
    fingerprint: BaseEstimator,
    classifier: BaseEstimator,
    fp_params: dict[str, list[Any]],
    clf_params: dict[str, list[Any]],
) -> tuple[float, float]:
    """
    Perform hyperparameter tuning using scikit-fingerprints.
    """
    fp = clone(fingerprint)
    clf = clone(classifier)

    clf_cv = GridSearchCV(
        clf,
        clf_params,
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        scoring="roc_auc",
        n_jobs=-1,
    )
    fp_cv = FingerprintEstimatorGridSearch(
        fingerprint=fp,
        fp_param_grid=fp_params,
        estimator_cv=clf_cv,
        cache_best_fp_array=True,
    )

    skfp_mean, skfp_std = measure_time(
        lambda args: fp_cv.fit(*args),
        (smiles, labels),
        "scikit-fingerprints FingerprintEstimatorGridSearch",
        iterations=N_REPEATS,
    )

    return skfp_mean, skfp_std, fp_cv.best_score_


def measure_sklearn(
    smiles: list[str],
    labels: np.ndarray,
    fingerprint: BaseEstimator,
    classifier: BaseEstimator,
    fp_params: dict[str, list[Any]],
    clf_params: dict[str, list[Any]],
) -> tuple[float, float]:
    """
    Perform hyperparameter tuning using scikit-learn.
    """
    pipeline_param_grid = {
        **{f"fp__{k}": v for k, v in fp_params.items()},
        **{f"clf__{k}": v for k, v in clf_params.items()},
    }
    pipeline = Pipeline(
        [
            ("fp", clone(fingerprint)),
            ("clf", clone(classifier)),
        ]
    )
    sklearn_cv = GridSearchCV(
        pipeline,
        pipeline_param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        scoring="roc_auc",
        n_jobs=-1,
    )
    elapsed, std = measure_time(
        lambda args: sklearn_cv.fit(*args),
        (smiles, labels),
        "sklearn Pipeline GridSearchCV",
    )
    return elapsed, std, sklearn_cv.best_score_


def measure_tuning(
    smiles: list[str],
    labels: np.ndarray,
    fingerprint: BaseEstimator,
    classifier: BaseEstimator,
    fp_params: dict[str, list[Any]],
    clf_params: dict[str, list[Any]],
    n_repeats: int,
) -> dict[str, float]:
    """
    Evaluate hyperparameter tuning performance and timing.
    """
    baseline_scores = []
    skfp_times = []
    skfp_time_stds = []
    skfp_scores = []
    sklearn_times = []
    sklearn_time_stds = []
    sklearn_scores = []

    for _ in range(n_repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        fp_baseline = clone(fingerprint)
        x_fp = fp_baseline.transform(smiles)
        baseline_clf = clone(classifier)
        baseline_score = cross_val_score(
            baseline_clf, x_fp, labels, cv=cv, scoring="roc_auc"
        ).mean()
        baseline_scores.append(baseline_score)

        skfp_time, skfp_time_std, skfp_score = measure_skfp(
            smiles, labels, fingerprint, classifier, fp_params, clf_params
        )
        skfp_times.append(skfp_time)
        skfp_time_stds.append(skfp_time_std)
        skfp_scores.append(skfp_score)

        sklearn_time, sklearn_time_std, sklearn_score = measure_sklearn(
            smiles, labels, fingerprint, classifier, fp_params, clf_params
        )
        sklearn_times.append(sklearn_time)
        sklearn_time_stds.append(sklearn_time_std)
        sklearn_scores.append(sklearn_score)

    baseline_scores = np.array(baseline_scores)
    skfp_scores = np.array(skfp_scores)
    sklearn_scores = np.array(sklearn_scores)

    skfp_abs_gains = skfp_scores - baseline_scores
    skfp_rel_gains = skfp_abs_gains / baseline_scores
    sklearn_abs_gains = sklearn_scores - baseline_scores
    sklearn_rel_gains = sklearn_abs_gains / baseline_scores

    return {
        "baseline_score": np.mean(baseline_scores),
        "skfp_time": np.mean(skfp_times),
        "skfp_time_std": np.mean(skfp_time_stds),
        "skfp_score": np.mean(skfp_scores),
        "skfp_abs_gain": np.mean(skfp_abs_gains),
        "skfp_rel_gain": np.mean(skfp_rel_gains),
        "sklearn_time": np.mean(sklearn_times),
        "sklearn_time_std": np.mean(sklearn_time_stds),
        "sklearn_score": np.mean(sklearn_scores),
        "sklearn_abs_gain": np.mean(sklearn_abs_gains),
        "sklearn_rel_gain": np.mean(sklearn_rel_gains),
    }


def run_benchmark() -> None:
    """
    Run hyperparameter tuning benchmarks for scikit-fingerprints and scikit-learn.

    Steps:
    1. Load the HIV dataset from MoleculeNet (subset if DATASET_CUTOFF is set).
    2. Shuffle the dataset to ensure a balanced class distribution.
    3. Incrementally increase the number of molecules and perform
       hyperparameter tuning using:
       - scikit-fingerprints (FingerprintEstimatorGridSearch),
       - scikit-learn (Pipeline with GridSearchCV).
    4. Evaluate two search spaces:
       - a small grid (fewer parameter combinations),
       - a large grid.
    5. Compute baseline model performance (without tuning) and compare it
       to tuned models using ROC-AUC.
    6. Measure tuning time, score, and calculate absolute and relative gains.
    7. Save all results into a CSV file.

    Keep in mind that grid search running time scales with the number of parameter combinations and cross-validation folds, which can result in long execution times for this benchmark.
    """
    smiles, labels = load_hiv()

    small_fp_param_grid = {
        "count": [False, True],
        "fp_size": [1024, 2048],
    }
    small_clf_param_grid = {
        "min_samples_split": [2, 5, 10],
    }

    large_fp_param_grid = {
        "count": [False, True],
        "fp_size": [512, 1024, 2048],
    }
    large_clf_param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    fingerprint = RDKitFingerprint(n_jobs=-1)
    classifier = RandomForestClassifier(n_jobs=-1, random_state=0)

    # balance class distribution
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(smiles))
    smiles = np.asarray(smiles)[idx].tolist()
    labels = np.asarray(labels)[idx]

    length = min(len(smiles), DATASET_CUTOFF) if DATASET_CUTOFF else len(smiles)

    # top off the dataset in case the number of molecules isn't a multiple of STEP
    steps = list(range(STEP, length, STEP))
    if not steps or steps[-1] != length:
        steps.append(length)

    with open(RESULT_CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

    for n in steps:
        print(f"Benchmarking with {n} molecules (small grid)...")
        small_results = measure_tuning(
            smiles[:n],
            labels[:n],
            fingerprint,
            classifier,
            small_fp_param_grid,
            small_clf_param_grid,
            n_repeats=N_REPEATS,
        )
        print(f"Benchmarking with {n} molecules (large grid)...")
        large_results = measure_tuning(
            smiles[:n],
            labels[:n],
            fingerprint,
            classifier,
            large_fp_param_grid,
            large_clf_param_grid,
            n_repeats=N_REPEATS,
        )
        with open(RESULT_CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    n,
                    small_results["baseline_score"],
                    small_results["skfp_time"],
                    small_results["skfp_time_std"],
                    small_results["skfp_score"],
                    small_results["skfp_abs_gain"],
                    small_results["skfp_rel_gain"],
                    small_results["sklearn_time"],
                    small_results["sklearn_time_std"],
                    small_results["sklearn_score"],
                    small_results["sklearn_abs_gain"],
                    small_results["sklearn_rel_gain"],
                    large_results["skfp_time"],
                    large_results["skfp_time_std"],
                    large_results["skfp_score"],
                    large_results["skfp_abs_gain"],
                    large_results["skfp_rel_gain"],
                    large_results["sklearn_time"],
                    large_results["sklearn_time_std"],
                    large_results["sklearn_score"],
                    large_results["sklearn_abs_gain"],
                    large_results["sklearn_rel_gain"],
                ]
            )

    print(f"Benchmark finished. Results saved to {RESULT_CSV_PATH}")


def plot_results() -> None:
    """
    Plot timing results for scikit-fingerprints vs scikit-learn and save as PNG or PDF.
    """
    try:
        df = pd.read_csv(RESULT_CSV_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {RESULT_CSV_PATH}")

    fig_time, ax_time = plt.subplots(figsize=(10, 6))
    if USE_ERROR_BARS:
        for col, label, marker, color in PLOT_CONFIGS:
            ax_time.errorbar(
                df["n"],
                df[f"{col}_time"],
                yerr=df[f"{col}_time_std"],
                marker=marker,
                color=color,
                label=label,
                capsize=3,
            )
    else:
        for col, label, marker, color in PLOT_CONFIGS:
            ax_time.plot(
                df["n"],
                df[f"{col}_time"],
                marker=marker,
                color=color,
                label=label,
            )
            ax_time.fill_between(
                df["n"],
                df[f"{col}_time"] - df[f"{col}_time_std"],
                df[f"{col}_time"] + df[f"{col}_time_std"],
                color=color,
                alpha=0.3,
            )
    ax_time.set_xlabel("Number of molecules")
    ax_time.set_ylabel("Time [s]")
    ax_time.set_title("Hyperparameter tuning time vs dataset size")
    ax_time.grid(True, linestyle="--", alpha=0.7)
    ax_time.legend()
    fig_time.tight_layout()
    fig_time.savefig(RESULT_PLOT_TIME_PATH)
    plt.close(fig_time)
    print(f"Plot saved to {RESULT_PLOT_TIME_PATH}")

    fig_score, ax_score = plt.subplots(figsize=(6, 5))
    ax_score.plot(
        df["n"],
        df["baseline_score"],
        marker="^",
        color="steelblue",
        label="Baseline",
    )
    for col, label, marker, color in PLOT_CONFIGS:
        ax_score.plot(
            df["n"],
            df[f"{col}_score"],
            marker=marker,
            color=color,
            label=label,
        )
    ax_score.set_xlabel("Number of molecules")
    ax_score.set_ylabel("ROC-AUC")
    ax_score.set_title("ROC-AUC: Baseline vs Tuned models")
    ax_score.grid(True, linestyle="--", alpha=0.7)
    ax_score.legend()
    fig_score.tight_layout()
    fig_score.savefig(RESULT_PLOT_SCORE_PATH)
    plt.close(fig_score)
    print(f"Plot saved to {RESULT_PLOT_SCORE_PATH}")

    fig_abs, ax_abs = plt.subplots(figsize=(6, 5))
    for col, label, marker, color in PLOT_CONFIGS:
        ax_abs.plot(
            df["n"],
            df[f"{col}_abs_gain"],
            marker=marker,
            color=color,
            label=label,
        )
    ax_abs.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_abs.set_xlabel("Number of molecules")
    ax_abs.set_ylabel("Absolute gain (ROC-AUC)")
    ax_abs.set_title("Absolute gain vs dataset size")
    ax_abs.grid(True, linestyle="--", alpha=0.7)
    ax_abs.legend()
    fig_abs.tight_layout()
    fig_abs.savefig(RESULT_PLOT_ABS_GAIN_PATH)
    plt.close(fig_abs)
    print(f"Plot saved to {RESULT_PLOT_ABS_GAIN_PATH}")

    fig_rel, ax_rel = plt.subplots(figsize=(6, 5))
    for col, label, marker, color in PLOT_CONFIGS:
        ax_rel.plot(
            df["n"],
            df[f"{col}_rel_gain"] * 100,
            marker=marker,
            color=color,
            label=label,
        )
    ax_rel.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_rel.set_xlabel("Number of molecules")
    ax_rel.set_ylabel("Relative gain [%]")
    ax_rel.set_title("Relative gain vs dataset size")
    ax_rel.grid(True, linestyle="--", alpha=0.7)
    ax_rel.legend()
    fig_rel.tight_layout()
    fig_rel.savefig(RESULT_PLOT_REL_GAIN_PATH)
    plt.close(fig_rel)
    print(f"Plot saved to {RESULT_PLOT_REL_GAIN_PATH}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
