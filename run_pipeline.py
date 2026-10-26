"""
End-to-end execution of the nested cross-validation biomarker
selection and integrated prioritization pipeline.

Usage:
    python run_pipeline.py

See README.md for the expected input file formats, configurable
parameters, and produced outputs.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

import config
import visualization as viz
from data_loading import build_labels, load_expression_matrix, load_feature_ranking, load_sample_metadata
from modeling import get_models, run_nested_cv, run_permutation_test
from prioritization import compute_feature_stability, prioritize_candidates

warnings.filterwarnings("ignore")


def _section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    np.random.seed(config.RANDOM_SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------
    _section("LOADING SAMPLE METADATA AND EXPRESSION DATA")

    metadata = load_sample_metadata(config.METADATA_FILE, config.GROUP_A_LABEL, config.GROUP_B_LABEL)
    full_matrix = load_expression_matrix(config.EXPRESSION_FILE, metadata["sample_id"].tolist())
    labels = build_labels(full_matrix, metadata)

    print("Expression matrix:", full_matrix.shape)
    print("\nGroup sizes:")
    print(labels.value_counts())

    label_encoder = LabelEncoder()
    label_encoder.fit(np.array([config.GROUP_A_LABEL, config.GROUP_B_LABEL]))
    y = label_encoder.transform(labels)
    X = full_matrix.copy()

    print("\nClass encoding:")
    for code, cls in enumerate(label_encoder.classes_):
        print(code, "=", cls)

    # ------------------------------------------------------------
    # 2. Strict nested LOO-CV model comparison
    # ------------------------------------------------------------
    _section("STRICT NESTED-CV -- MODEL COMPARISON")

    models = get_models()
    true_results = {}
    for model_name, (estimator, param_grid) in models.items():
        print(f"Running {model_name}...")
        result = run_nested_cv(
            X_raw=X,
            y=y,
            estimator=estimator,
            param_grid=param_grid,
            label_encoder=label_encoder,
            positive_label=config.POSITIVE_LABEL,
            collect_audit=False,
        )
        true_results[model_name] = result
        print(f"  AUC = {result['auc']:.4f}")

    true_auc = {name: result["auc"] for name, result in true_results.items()}
    best_model_name = max(true_auc, key=true_auc.get)
    true_best_auc = true_auc[best_model_name]

    print("\nBest model:", best_model_name)
    print("Best AUC:", f"{true_best_auc:.4f}")

    # ------------------------------------------------------------
    # 3. Full audit of the best model
    # ------------------------------------------------------------
    _section("FULL AUDIT OF THE BEST MODEL")

    best_estimator, best_grid = models[best_model_name]
    best_result = run_nested_cv(
        X_raw=X,
        y=y,
        estimator=best_estimator,
        param_grid=best_grid,
        label_encoder=label_encoder,
        positive_label=config.POSITIVE_LABEL,
        collect_audit=True,
    )
    audit_df = best_result["audit_df"]
    oof_true, oof_pred = best_result["y_true"], best_result["y_pred"]

    accuracy = accuracy_score(oof_true, oof_pred)
    mcc = matthews_corrcoef(oof_true, oof_pred)
    cm = confusion_matrix(oof_true, oof_pred)

    print(audit_df.to_string(index=False))
    print("\nAUC:", f"{true_best_auc:.6f}")
    print("Accuracy:", f"{accuracy:.6f}")
    print("MCC:", f"{mcc:.6f}")
    print("\nConfusion matrix:")
    print(cm)

    # ------------------------------------------------------------
    # 4. Feature-selection stability
    # ------------------------------------------------------------
    _section("FEATURE-SELECTION STABILITY")

    feature_stability = compute_feature_stability(
        best_result["selected_feature_lists"], n_outer_folds=len(X)
    )
    print(feature_stability.head(30).to_string(index=False))

    # ------------------------------------------------------------
    # 5. Model-selection-aware permutation test
    # ------------------------------------------------------------
    _section(f"STRICT PERMUTATION TEST -- {config.N_PERMUTATIONS} permutations")

    perm_result = run_permutation_test(
        X=X,
        y=y,
        models=models,
        label_encoder=label_encoder,
        positive_label=config.POSITIVE_LABEL,
        n_permutations=config.N_PERMUTATIONS,
    )
    null_best_aucs = perm_result["null_best_aucs"]

    empirical_p = (1 + np.sum(null_best_aucs >= true_best_auc)) / (1 + len(null_best_aucs))
    null_mean = np.mean(null_best_aucs)
    null_median = np.median(null_best_aucs)
    null_95 = np.percentile(null_best_aucs, 95)
    null_99 = np.percentile(null_best_aucs, 99)

    print("\nNull distribution (best model per permutation):")
    print("Mean:", f"{null_mean:.4f}")
    print("Median:", f"{null_median:.4f}")
    print("95th percentile:", f"{null_95:.4f}")
    print("99th percentile:", f"{null_99:.4f}")
    print("Empirical P:", f"{empirical_p:.6f}")
    print("\nWinning models under shuffled labels:")
    print(pd.Series(perm_result["winning_models"]).value_counts())

    model_perm_rows = []
    for model_name in models:
        values = perm_result["null_model_aucs"][model_name]
        true_auc_value = true_results[model_name]["auc"]
        model_p = (1 + np.sum(values >= true_auc_value)) / (1 + len(values))
        model_perm_rows.append(
            {
                "model": model_name,
                "true_auc": true_auc_value,
                "null_auc_mean": values.mean(),
                "null_auc_95th": np.percentile(values, 95),
                "empirical_p": model_p,
            }
        )
    model_perm_df = pd.DataFrame(model_perm_rows)

    print("\nMODEL-SPECIFIC PERMUTATION RESULTS")
    print(model_perm_df.to_string(index=False))

    # ------------------------------------------------------------
    # 6. Integrate with external feature ranking
    # ------------------------------------------------------------
    _section("LOADING EXTERNAL FEATURE RANKING")

    ranking = load_feature_ranking(config.RANKING_FILE, config.GROUP_A_LABEL, config.GROUP_B_LABEL)
    print("Ranking file shape:", ranking.shape)

    candidate_pool, top_candidates = prioritize_candidates(
        feature_stability,
        ranking,
        min_selection_frequency=config.MIN_SELECTION_FREQUENCY,
        top_n=config.TOP_N_CANDIDATES,
    )

    _section("INTEGRATED CANDIDATE PRIORITIZATION")
    display_columns = [
        "integrated_rank",
        "feature_id",
        "selected_folds",
        "selection_frequency",
        "diagnostic_rank",
        "diagnostic_score",
        "pval",
        "FDR",
        "abs_logFC",
        "direction",
        "integrated_score",
        "selected_for_validation",
    ]
    print(candidate_pool[display_columns].head(20).to_string(index=False))

    _section(f"TOP {config.TOP_N_CANDIDATES} CANDIDATES")
    top_display_columns = [
        "integrated_rank",
        "feature_id",
        "selection_frequency",
        "diagnostic_rank",
        "diagnostic_score",
        "pval",
        "FDR",
        "abs_logFC",
        "integrated_score",
    ]
    print(top_candidates[top_display_columns].to_string(index=False))

    # ------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------
    _section("GENERATING FIGURES")

    viz.plot_oof_predictions(
        audit_df,
        config.GROUP_A_LABEL,
        config.GROUP_B_LABEL,
        best_model_name,
        config.FIGURE_DIR / "oof_predictions.png",
    )
    viz.plot_permutation_null(
        null_best_aucs,
        true_best_auc,
        null_95,
        empirical_p,
        config.FIGURE_DIR / "permutation_null_distribution.png",
    )
    viz.plot_feature_stability(feature_stability, config.FIGURE_DIR / "feature_stability.png")
    viz.plot_prioritization_scatter(candidate_pool, config.FIGURE_DIR / "prioritization_scatter.png")
    viz.plot_integrated_ranking(
        candidate_pool,
        config.FIGURE_DIR / "integrated_ranking.png",
        top_n=config.TOP_N_CANDIDATES,
    )

    print("Figures saved to:", config.FIGURE_DIR)

    # ------------------------------------------------------------
    # 8. Save tables
    # ------------------------------------------------------------
    _section("SAVING RESULTS")

    with pd.ExcelWriter(config.RESULTS_FILE) as writer:
        audit_df.to_excel(writer, sheet_name="OOF_Audit", index=False)
        feature_stability.to_excel(writer, sheet_name="Feature_Stability", index=False)
        model_perm_df.to_excel(writer, sheet_name="Permutation_Models", index=False)
        candidate_pool.to_excel(writer, sheet_name="Candidate_Prioritization", index=False)
        top_candidates.to_excel(writer, sheet_name="Top_Candidates", index=False)

    # ------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------
    _section("FINAL SUMMARY")

    print("Best strict model:", best_model_name)
    print("Observed nested-CV AUC:", f"{true_best_auc:.4f}")
    print("Strict permutation P:", f"{empirical_p:.6f}")
    print("Null 95th percentile:", f"{null_95:.4f}")

    print("\nSelected candidates:")
    for _, row in top_candidates.iterrows():
        print(
            f"{int(row['integrated_rank'])}. {row['feature_id']}  |  "
            f"ML stability={row['selection_frequency']:.3f}  |  "
            f"diagnostic rank={int(row['diagnostic_rank'])}  |  "
            f"diagnostic score={row['diagnostic_score']:.3f}  |  "
            f"FDR={row['FDR']:.4g}  |  "
            f"|logFC|={row['abs_logFC']:.3f}  |  "
            f"integrated score={row['integrated_score']:.3f}"
        )

    print("\nSaved:", config.RESULTS_FILE)


if __name__ == "__main__":
    main()
