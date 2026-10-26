"""
Integration of strict ML feature-selection stability with an
externally computed differential-feature ranking, to prioritize
candidates for downstream validation (e.g. targeted qPCR).

The ranking file's diagnostic scores are never recomputed here; they
are only combined with the ML stability metric produced by the
nested cross-validation.
"""

from collections import Counter

import numpy as np
import pandas as pd

from config import MIN_SELECTION_FREQUENCY, TOP_N_CANDIDATES


def compute_feature_stability(selected_feature_lists: list, n_outer_folds: int) -> pd.DataFrame:
    """
    Compute how often each feature was selected across the outer
    LOO-CV folds of the best-performing model's full audit run.
    """
    counts = Counter(selected_feature_lists)

    stability = pd.DataFrame(counts.items(), columns=["feature_id", "selected_folds"])
    stability["selection_frequency"] = stability["selected_folds"] / n_outer_folds

    stability = stability.sort_values(
        ["selected_folds", "feature_id"], ascending=[False, True]
    ).reset_index(drop=True)

    return stability


def prioritize_candidates(
    feature_stability: pd.DataFrame,
    ranking: pd.DataFrame,
    min_selection_frequency: float = MIN_SELECTION_FREQUENCY,
    top_n: int = TOP_N_CANDIDATES,
) -> tuple:
    """
    Combine ML feature-selection stability with the external
    diagnostic ranking into a single integrated prioritization.

    The integrated score is the geometric mean of the ML selection
    frequency and the diagnostic ranking score (both already scaled
    to [0, 1]):

        integrated_score = sqrt(selection_frequency * diagnostic_score)

    A geometric mean requires a candidate to perform reasonably well
    on BOTH components; a weak component penalizes the combined
    score.

    Only features with `selection_frequency >= min_selection_frequency`
    are eligible (a transparent stability threshold, not a
    differential-expression filter).

    Returns (candidate_pool, top_candidates), where `candidate_pool`
    is ranked by integrated score and `top_candidates` is its first
    `top_n` rows.
    """
    merged = feature_stability.merge(ranking, on="feature_id", how="inner").copy()

    merged["integrated_score"] = np.sqrt(
        merged["selection_frequency"] * merged["diagnostic_score"]
    )

    pool = merged[merged["selection_frequency"] >= min_selection_frequency].copy()
    pool = pool.sort_values(
        ["integrated_score", "selection_frequency", "diagnostic_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    pool["integrated_rank"] = np.arange(1, len(pool) + 1)
    pool["selected_for_validation"] = pool["integrated_rank"] <= top_n

    top_candidates = pool.head(top_n).copy()

    return pool, top_candidates
