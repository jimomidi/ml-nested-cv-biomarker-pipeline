"""
Central configuration for the nested cross-validation biomarker
selection and prioritization pipeline.

All paths are relative to the project root, so the pipeline runs
unmodified on any machine. Edit the values below to match your own
dataset and study design; no other file needs to change.
"""

from pathlib import Path

# ----------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------
# Group labels
#
# GROUP_A_LABEL / GROUP_B_LABEL must exactly match the values used
# in the "group" column of the sample metadata file.
#
# POSITIVE_LABEL identifies the class whose predicted probability is
# tracked throughout the pipeline (e.g. the "P_positive_class"
# column and the out-of-fold prediction plot).
# ----------------------------------------------------------------
GROUP_A_LABEL = "group_a"
GROUP_B_LABEL = "group_b"
POSITIVE_LABEL = GROUP_B_LABEL

# ----------------------------------------------------------------
# Nested cross-validation
# ----------------------------------------------------------------
# Candidate values for the number of selected features (SelectKBest)
# evaluated by the inner grid search.
K_GRID = [10, 20, 30]

# Number of parallel jobs used by the inner grid search.
N_JOBS_INNER = -1

# ----------------------------------------------------------------
# Model-selection-aware permutation test
# ----------------------------------------------------------------
# Number of label permutations. A small value is convenient for
# quick iteration; use a larger value (e.g. 1000) for a
# publication-quality estimate of the null distribution.
N_PERMUTATIONS = 50

# ----------------------------------------------------------------
# Expression prefilter (fit on the training fold only, never on the
# full dataset, to avoid information leakage from held-out samples)
# ----------------------------------------------------------------
MIN_DETECTED_FRACTION = 0.30
MIN_MEAN_EXPRESSION = 1.0

# ----------------------------------------------------------------
# Candidate prioritization
# ----------------------------------------------------------------
# Minimum outer-fold selection frequency required for a feature to
# be eligible for the integrated prioritization ranking.
MIN_SELECTION_FREQUENCY = 0.50

# Number of top-ranked candidates flagged for downstream validation
# (e.g. targeted qPCR).
TOP_N_CANDIDATES = 4

# ----------------------------------------------------------------
# Paths (all relative to the project root)
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Tab-separated expression matrix: an "ID" column plus one
# "<sample_id>_TPM" column per sample.
EXPRESSION_FILE = DATA_DIR / "expression_data.tsv"

# CSV file with two columns: "sample_id" and "group".
METADATA_FILE = DATA_DIR / "sample_metadata.csv"

# Externally computed differential-feature ranking (e.g. from a
# differential expression pipeline). Not recalculated here.
RANKING_FILE = DATA_DIR / "feature_ranking.xlsx"

RESULTS_FILE = OUTPUT_DIR / "integrated_analysis_results.xlsx"
