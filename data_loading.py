"""
Loading and validation of the pipeline's three input files: the
expression matrix, the sample metadata, and the external feature
ranking.

No filtering or transformation of expression values happens here;
that is performed strictly inside the cross-validation loop (see
`modeling.ExpressionFilterLog2`) so held-out samples never influence
preprocessing decisions.
"""

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_sample_metadata(metadata_file: Path, group_a: str, group_b: str) -> pd.DataFrame:
    """
    Load the sample-to-group mapping.

    Expected columns:
      - sample_id: must match the sample columns in the expression file
      - group: must be exactly `group_a` or `group_b`
    """
    metadata = pd.read_csv(metadata_file)

    required_columns = {"sample_id", "group"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata file '{metadata_file}' is missing columns: {sorted(missing_columns)}"
        )

    valid_groups = {group_a, group_b}
    invalid_groups = sorted(set(metadata["group"]) - valid_groups)
    if invalid_groups:
        raise ValueError(
            f"Metadata file '{metadata_file}' contains group labels other than "
            f"{sorted(valid_groups)}: {invalid_groups}"
        )

    return metadata


def load_expression_matrix(expression_file: Path, sample_ids: Sequence[str]) -> pd.DataFrame:
    """
    Load a tab-separated expression matrix and return it as a
    samples x features matrix restricted to `sample_ids`.

    The input file is expected to contain:
      - an "ID" column with feature identifiers (e.g. gene, miRNA,
        or protein IDs)
      - one column per sample, named "<sample_id>_TPM"

    Returns the raw (untransformed) matrix; no expression filtering
    is applied here.
    """
    raw = pd.read_csv(expression_file, sep="\t")

    expression_columns = [c for c in raw.columns if c.endswith("_TPM")]
    if not expression_columns:
        raise ValueError(f"No '*_TPM' expression columns found in '{expression_file}'")

    matrix = raw[["ID"] + expression_columns].copy()
    matrix.columns = ["feature_id"] + [c.replace("_TPM", "") for c in expression_columns]

    available_samples = [s for s in sample_ids if s in matrix.columns]
    missing_samples = [s for s in sample_ids if s not in matrix.columns]
    if missing_samples:
        raise ValueError(
            f"Samples listed in the metadata file but missing from the expression "
            f"file: {missing_samples}"
        )

    for sample in available_samples:
        matrix[sample] = pd.to_numeric(matrix[sample], errors="coerce").fillna(0.0)

    full_matrix = matrix.set_index("feature_id")[available_samples].T.copy()
    full_matrix = full_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return full_matrix


def build_labels(full_matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.Series:
    """Map each sample in `full_matrix` to its group label from `metadata`."""
    group_by_sample = metadata.set_index("sample_id")["group"]
    labels = group_by_sample.reindex(full_matrix.index)

    if labels.isna().any():
        missing = labels[labels.isna()].index.tolist()
        raise ValueError(f"No metadata group label found for samples: {missing}")

    return labels


def load_feature_ranking(ranking_file: Path, group_a: str, group_b: str) -> pd.DataFrame:
    """
    Load a precomputed differential-feature ranking table.

    This table is produced upstream (e.g. by a differential
    expression pipeline) and is NOT recalculated here. It must
    contain at least the following columns:

      diagnostic_rank, feature_id, direction, logFC, abs_logFC,
      pval, FDR, mean_expression_<group_a>, mean_expression_<group_b>,
      diagnostic_score
    """
    ranking = pd.read_excel(ranking_file)

    required_columns = [
        "diagnostic_rank",
        "feature_id",
        "direction",
        "logFC",
        "abs_logFC",
        "pval",
        "FDR",
        f"mean_expression_{group_a}",
        f"mean_expression_{group_b}",
        "diagnostic_score",
    ]

    missing_columns = [c for c in required_columns if c not in ranking.columns]
    if missing_columns:
        raise ValueError(f"Ranking file '{ranking_file}' is missing columns: {missing_columns}")

    return ranking[required_columns].copy()
