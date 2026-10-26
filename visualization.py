"""
Publication-oriented figures summarizing the nested cross-validation
and candidate prioritization results. Every function saves its
figure to `output_path` and closes it (no interactive display), so
the pipeline runs unattended in headless / server environments.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import TOP_N_CANDIDATES  # noqa: E402


def plot_oof_predictions(
    audit_df: pd.DataFrame,
    group_a_label: str,
    group_b_label: str,
    model_name: str,
    output_path: Path,
) -> None:
    """Out-of-fold predicted probability of the positive class, by true group."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    mask_a = audit_df["true_class"] == group_a_label
    mask_b = audit_df["true_class"] == group_b_label

    ax.scatter(
        np.zeros(mask_a.sum()), audit_df.loc[mask_a, "P_positive_class"], s=90, edgecolors="black"
    )
    ax.scatter(
        np.ones(mask_b.sum()), audit_df.loc[mask_b, "P_positive_class"], s=90, edgecolors="black"
    )
    ax.axhline(0.5, linestyle="--", linewidth=1.2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_a_label, group_b_label])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Out-of-fold predicted probability of the positive class")
    ax.set_title(f"Strict Nested-CV Predictions\n{model_name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_permutation_null(
    null_best_aucs: np.ndarray,
    true_best_auc: float,
    null_95th_percentile: float,
    empirical_p: float,
    output_path: Path,
) -> None:
    """Histogram of the permutation null distribution of best-model AUC."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(null_best_aucs, bins=25, edgecolor="white", alpha=0.85)
    ax.axvline(true_best_auc, linewidth=2.5, label=f"Observed best AUC = {true_best_auc:.3f}")
    ax.axvline(
        null_95th_percentile,
        linestyle="--",
        linewidth=1.8,
        label=f"Null 95th percentile = {null_95th_percentile:.3f}",
    )

    ax.set_xlabel("Best nested-CV AUC across candidate models")
    ax.set_ylabel("Permutation count")
    ax.set_title(f"Strict Model-Selection-Aware Permutation Test\nP = {empirical_p:.4g}")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_stability(
    feature_stability: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of the top-N most stably selected features."""
    top_stability = feature_stability.head(top_n).sort_values("selection_frequency", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_stability["feature_id"], top_stability["selection_frequency"])

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Selection frequency across strict outer LOO-CV folds")
    ax.set_title("Strict Feature-Selection Stability")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prioritization_scatter(
    candidate_pool: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Scatter of diagnostic score vs. ML selection frequency, with
    marker size scaled to |logFC| and the selected candidates
    highlighted as stars.
    """
    plot_df = candidate_pool.head(top_n).copy()

    effect_min = plot_df["abs_logFC"].min()
    effect_max = plot_df["abs_logFC"].max()

    if effect_max > effect_min:
        plot_df["bubble_size"] = 100 + 600 * (plot_df["abs_logFC"] - effect_min) / (
            effect_max - effect_min
        )
    else:
        plot_df["bubble_size"] = 300

    fig, ax = plt.subplots(figsize=(9, 7))

    other = plot_df[~plot_df["selected_for_validation"]]
    selected = plot_df[plot_df["selected_for_validation"]]

    ax.scatter(
        other["diagnostic_score"],
        other["selection_frequency"],
        s=other["bubble_size"],
        alpha=0.35,
        edgecolors="black",
        linewidths=0.6,
        label="Other candidates",
    )
    ax.scatter(
        selected["diagnostic_score"],
        selected["selection_frequency"],
        s=selected["bubble_size"],
        marker="*",
        edgecolors="black",
        linewidths=1.4,
        label="Top candidates",
    )

    for _, row in selected.iterrows():
        ax.annotate(
            row["feature_id"],
            (row["diagnostic_score"], row["selection_frequency"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Diagnostic ranking score")
    ax.set_ylabel("Strict ML selection frequency")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("Integrated ML-Diagnostic Prioritization")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_integrated_ranking(
    candidate_pool: pd.DataFrame,
    output_path: Path,
    top_n: int = TOP_N_CANDIDATES,
    display_n: int = 15,
) -> None:
    """Dot plot of the integrated prioritization score, with a line marking the top-N boundary."""
    rank_plot = candidate_pool.head(display_n).sort_values("integrated_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    selected_mask = rank_plot["selected_for_validation"]

    ax.scatter(
        rank_plot.loc[~selected_mask, "integrated_score"],
        rank_plot.loc[~selected_mask, "feature_id"],
        s=100,
        alpha=0.45,
        edgecolors="black",
    )
    ax.scatter(
        rank_plot.loc[selected_mask, "integrated_score"],
        rank_plot.loc[selected_mask, "feature_id"],
        s=220,
        marker="*",
        edgecolors="black",
        linewidths=1.3,
        label="Top candidates",
    )

    for _, row in rank_plot.iterrows():
        ax.text(
            row["integrated_score"] + 0.005,
            row["feature_id"],
            f"{row['integrated_score']:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold" if row["selected_for_validation"] else "normal",
        )

    if len(candidate_pool) > top_n:
        boundary_score = candidate_pool.iloc[top_n - 1]["integrated_score"]
        next_score = candidate_pool.iloc[top_n]["integrated_score"]
        boundary = (boundary_score + next_score) / 2
        ax.axvline(boundary, linestyle="--", linewidth=1.2, alpha=0.7, label="Top-N boundary")

    ax.set_xlabel("Integrated Prioritization Score")
    ax.set_ylabel("")
    ax.set_title("Final Integrated Ranking of Candidates")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
