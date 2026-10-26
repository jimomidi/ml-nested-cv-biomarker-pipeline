# ML Nested-CV Biomarker Pipeline

A strict, leakage-safe machine learning pipeline for binary classification of high-dimensional
expression data (e.g. miRNA-seq or RNA-seq derived abundance values), with
model-selection-aware significance testing and an integrated ranking that
combines cross-validated feature stability with an externally computed
differential-expression ranking to prioritize candidates for downstream
validation (e.g. targeted qPCR).

## Purpose

Given an expression matrix and a two-group sample assignment, the pipeline:

1. Compares several classifiers (logistic regression, SVM, random forest, and
   optionally XGBoost) using **strict nested leave-one-out cross-validation**,
   where expression prefiltering, feature selection, and hyperparameter
   tuning are all performed only on the training fold of each outer split.
2. Audits the best-performing model: out-of-fold predictions, accuracy, MCC,
   confusion matrix, and how often each feature was selected across outer
   folds ("feature-selection stability").
3. Runs a **model-selection-aware permutation test**: the entire nested-CV
   procedure (including the choice among candidate models) is repeated on
   label-shuffled data, producing an empirical null distribution and p-value
   that correctly account for model selection.
4. Loads an externally computed differential-feature ranking (produced
   upstream, e.g. by a differential expression pipeline) and combines it with
   the ML selection frequency into a single **integrated prioritization
   score**, used to flag a small number of top candidates for downstream
   validation.
5. Generates summary figures and saves all intermediate and final tables to
   an Excel workbook.

The differential-expression ranking values (log fold-change, p-values, FDR,
diagnostic score) are **not recalculated** by this pipeline; they are read
from an existing ranking file and only combined with the ML results.

## Workflow

```
sample metadata --+
                   +--> nested LOO-CV model comparison --> best model
expression data ---+                                           |
                                                                 v
                                               full audit (predictions,
                                               accuracy, MCC, feature
                                               stability) + permutation test
                                                                 |
                                                                 v
                            feature ranking --> integrated prioritization
                                                                 |
                                                                 v
                                                 figures + Excel results
```

## Repository structure

```
config.py            Central configuration: paths, group labels, and all
                      tunable parameters (random seed, CV grid, thresholds).
data_loading.py       Loading and validation of the expression matrix,
                      sample metadata, and feature ranking files.
modeling.py           Expression prefilter, nested LOO-CV, model comparison,
                      and the permutation test.
prioritization.py     Feature-selection stability and integrated candidate
                      ranking.
visualization.py      Figure generation (matplotlib, saved to disk).
run_pipeline.py        Orchestrates the full pipeline end-to-end.
```

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies (numpy, pandas,
  scikit-learn, matplotlib, openpyxl). XGBoost is optional; if it is not
  installed, the pipeline automatically skips it and continues with the
  remaining models.

## Installation

```bash
git clone <this-repository-url>
cd ml-nested-cv-biomarker-pipeline
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Expected input format

Create a `data/` directory in the project root containing the following
three files (see `config.py` to change file names or locations):

**`expression_data.tsv`** — tab-separated expression matrix with:
- an `ID` column containing feature identifiers (e.g. gene, miRNA, or
  protein IDs)
- one column per sample, named `<sample_id>_TPM` (values in transcripts per
  million, or an equivalent normalized abundance unit)

**`sample_metadata.csv`** — comma-separated sample-to-group assignment with
exactly two columns:
- `sample_id`: must match the `<sample_id>` used in the expression file's
  column names
- `group`: must be exactly `group_a` or `group_b` (or whatever values you set
  `GROUP_A_LABEL` / `GROUP_B_LABEL` to in `config.py`)

**`feature_ranking.xlsx`** — a precomputed differential-feature ranking table
(e.g. produced by a differential expression pipeline) with at least the
following columns:

| Column | Description |
|---|---|
| `diagnostic_rank` | Rank assigned by the upstream ranking pipeline |
| `feature_id` | Must match the feature identifiers in the expression file |
| `direction` | Direction of change (e.g. up/down) |
| `logFC` | Log fold-change |
| `abs_logFC` | Absolute log fold-change |
| `pval` | p-value |
| `FDR` | False discovery rate |
| `mean_expression_group_a` | Mean expression in group A |
| `mean_expression_group_b` | Mean expression in group B |
| `diagnostic_score` | A 0-1 diagnostic score from the upstream ranking pipeline |

If your group labels differ from the defaults, rename the two
`mean_expression_*` columns to match `GROUP_A_LABEL` / `GROUP_B_LABEL` in
`config.py`.

## Usage

```bash
python run_pipeline.py
```

All parameters (random seed, number of permutations, feature-selection
grid, expression prefilter thresholds, candidate prioritization thresholds,
group labels, and file paths) are centralized in `config.py`; no other file
needs to be edited for a standard run.

## Outputs

Running the pipeline creates a `results/` directory containing:

- `integrated_analysis_results.xlsx`, with sheets:
  - `OOF_Audit` — per-sample out-of-fold predictions for the best model
  - `Feature_Stability` — selection frequency of every feature selected in
    at least one outer fold
  - `Permutation_Models` — permutation-test summary per candidate model
  - `Candidate_Prioritization` — the full integrated ranking
  - `Top_Candidates` — the top-N candidates flagged for validation
- `figures/`, with:
  - `oof_predictions.png` — out-of-fold predicted probabilities by group
  - `permutation_null_distribution.png` — permutation null distribution
  - `feature_stability.png` — top stable features
  - `prioritization_scatter.png` — diagnostic score vs. ML stability
  - `integrated_ranking.png` — final integrated candidate ranking

## Reproducibility

- A fixed random seed (`RANDOM_SEED` in `config.py`, default `42`) controls
  model initialization, cross-validation splits, and the permutation test.
- The expression prefilter, feature selection, and hyperparameter search are
  refit on each outer training fold only, so held-out samples never
  influence any preprocessing decision.
- `N_PERMUTATIONS` defaults to a small value (`50`) for fast iteration.
  Increase it (e.g. to `1000`) for a publication-quality estimate of the
  permutation p-value; results will take proportionally longer to compute.
- All paths are relative to the project root, so the pipeline runs
  identically on any machine.

## Customization

- **Classifiers and hyperparameter grids**: edit `get_models()` in
  `modeling.py`.
- **Feature-selection grid** (`K_GRID`), **expression prefilter thresholds**
  (`MIN_DETECTED_FRACTION`, `MIN_MEAN_EXPRESSION`), **candidate
  prioritization thresholds** (`MIN_SELECTION_FREQUENCY`,
  `TOP_N_CANDIDATES`), and **group labels** are all set in `config.py`.
- **Figures**: each plotting function in `visualization.py` accepts an
  explicit output path and can be called independently.

## Notes

- This repository contains the analysis pipeline only. No input data,
  patient-level information, or study-specific identifiers are included or
  required to use the code; you provide your own files matching the format
  described above.
