import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import accuracy_score, f1_score

from resources import (
    FUSION_DIR,
    ENSEMBLE_DIR,
    PLOTS_DIR
)

import warnings
warnings.filterwarnings("ignore")

TARGETS = ["category", "specificity"]

# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dictionaries to hold the final predictions
    final_submission_dict_a = {}
    final_submission_dict_b = {}
    
    for target in TARGETS:
        print(f"\n=======================================================")
        print(f"🏆 FINAL BLIND A EVALUATION: {target.upper()}")
        print(f"=======================================================")
        
        base_file = FUSION_DIR / f"meta_features_blinda_{target}.parquet"
        ens_file = ENSEMBLE_DIR / f"ensemble_predictions_blinda_{target}.parquet"
        
        if not base_file.exists() or not ens_file.exists():
            print(f"⚠️ Missing evaluation files for {target}. Ensure fusion scripts have run.")
            continue
            
        # 1. Load and merge the datasets
        df_base = pl.read_parquet(base_file)
        df_ens = pl.read_parquet(ens_file)
        
        # Merge on session_id to get all predictions in one place
        df_ens = df_ens.drop([c for c in df_ens.columns if c.startswith("gt_")], strict=False)
        df_master = df_base.join(df_ens, on="session_id", how="inner")
        
        gt_col = f"gt_{target}"
        
        # Filter out the "unknown" rows (dummy 'u' or nulls) for evaluation
        is_unknown = (pl.col(gt_col) == "unknown") | (pl.col(gt_col).is_null())
        df_eval = df_master.filter(~is_unknown)
        df_unknown = df_master.filter(is_unknown)
        
        num_total = len(df_master)
        num_eval = len(df_eval)
        num_unknown = len(df_unknown)
        
        print(f"ℹ️  Dataset Split: {num_total} total rows.")
        print(f"    -> Ignored {num_unknown} rows with unknown ('u') ground truth.")
        print(f"    -> Evaluating metrics on the {num_eval} known samples.")
        print("-" * 102)
        
        if num_eval == 0:
            print(f"⚠️ No known labels found to evaluate for {target}. Skipping metrics...")
            continue
            
        y_true = df_eval[gt_col].to_numpy()
        total_rows = len(y_true)
        
        # 2. Extract and compute metrics for every model
        pred_cols = [c for c in df_eval.columns if c.startswith("pred_")]
        
        results = []
        for col in pred_cols:
            model_name = col.replace("pred_", "")
            
            # Make sure we pull predictions ONLY for the evaluated rows to calculate metrics
            y_pred = df_eval[col].to_numpy()
            
            # Calculate metrics
            correct_count = np.sum(y_true == y_pred)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            
            if model_name.startswith("G1_") or model_name.startswith("G2_") or model_name.startswith("G3_"):
                m_type = "Ensemble Fusion"
            else:
                m_type = "Base Model"
                
            results.append({
                "Model": model_name,
                "Correct": correct_count,
                "Total": total_rows,
                "Accuracy": acc,
                "Macro_F1": f1,
                "Type": m_type
            })
            
        df_results = pd.DataFrame(results).sort_values(by="Macro_F1", ascending=False)
        df_results.reset_index(drop=True, inplace=True)
        
        # 3. Terminal Printout
        print(f"\n{'Model Rank':<45} | {'Type':<16} | {'Correct':<10} | {'Accuracy':<10} | {'Macro F1':<10}")
        print("-" * 102)
        for _, row in df_results.iterrows():
            correct_str = f"{row['Correct']}/{row['Total']}"
            print(f"{row['Model']:<45} | {row['Type']:<16} | {correct_str:<10} | {row['Accuracy']:>9.4%} | {row['Macro_F1']:>9.4%}")
            
        # 4. Extract Top Model Predictions for ALL rows (including unknowns)
        best_overall_name = df_results.iloc[0]["Model"]
        best_overall_col = f"pred_{best_overall_name}"
        
        # Pull from df_master (all 80 rows) instead of df_eval
        session_ids = df_master["session_id"].to_list()
        best_preds = df_master[best_overall_col].to_list()
        
        print(f"\n🎯 Extracting '{best_overall_name}' predictions for all {num_total} samples...")
        
        for sid, pred in zip(session_ids, best_preds):
            if sid not in final_submission_dict_a:
                final_submission_dict_a[sid] = {}
            final_submission_dict_a[sid][target] = pred
            
        # 5. Calculate "Ensemble Lift"
        best_base_f1 = df_results[df_results["Type"] == "Base Model"]["Macro_F1"].max()
        best_base_name = df_results[df_results["Macro_F1"] == best_base_f1].iloc[0]["Model"]
        best_overall_f1 = df_results["Macro_F1"].max()
        lift = best_overall_f1 - best_base_f1
        
        print(f"🚀 ENSEMBLE LIFT (Macro F1): +{lift:.4f} over best base model ({best_base_name})")
        
        # 6. Generate the Visualizations
        plt.figure(figsize=(14, 10))
        palette = {"Ensemble Fusion": "#ff7f0e", "Base Model": "#1f77b4"}
        plot_df = df_results.head(30)
        
        ax = sns.barplot(
            data=plot_df, 
            x="Macro_F1", 
            y="Model", 
            hue="Type", 
            palette=palette, 
            dodge=False
        )
        
        plt.axvline(x=best_base_f1, color='red', linestyle='--', linewidth=2, label=f'Best Base Model ({best_base_f1:.4f})')
        plt.title(f"Final Blind A Performance Leaderboard: {target.upper()}\n(Sorted by F1-Macro on {num_eval} known samples)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Macro F1 Score", fontsize=12)
        plt.ylabel("Model / Strategy", fontsize=12)
        
        for p in ax.patches:
            width = p.get_width()
            if width > 0:
                center_y = p.get_y() + p.get_height() / 2.
                row_idx = int(round(center_y))
                if row_idx < len(plot_df):
                    correct = plot_df.iloc[row_idx]["Correct"]
                    total = plot_df.iloc[row_idx]["Total"]
                    label_text = f'{width:.4f}  ({correct}/{total})'
                    plt.text(width + 0.002, center_y, label_text, ha="left", va="center", fontsize=10)
                
        plt.legend(loc="lower right", fontsize=12)
        if plot_df["Macro_F1"].min() != plot_df["Macro_F1"].max():
            plt.xlim(plot_df["Macro_F1"].min() * 0.95, plot_df["Macro_F1"].max() * 1.05)
        
        plt.tight_layout()
        plot_path = PLOTS_DIR / f"final_blind_a_evaluation_{target}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"📊 Saved Leaderboard plot to: {plot_path}")

        # Determine the Best Model based on Blind A
        best_overall_name = df_results.iloc[0]["Model"]
        best_overall_col = f"pred_{best_overall_name}"
        
        print(f"\n🥇 BEST MODEL FOUND: '{best_overall_name}' (Macro F1: {df_results.iloc[0]['Macro_F1']:.4f})")

        # --- BLIND B (Extraction/Imputation) ---
        print(f"🎯 Applying '{best_overall_name}' to Blind B dataset...")
        base_file_b = FUSION_DIR / f"meta_features_blindb_{target}.parquet"
        ens_file_b = ENSEMBLE_DIR / f"ensemble_predictions_blindb_{target}.parquet"
        
        if base_file_b.exists() and ens_file_b.exists():
            df_base_b = pl.read_parquet(base_file_b)
            df_ens_b = pl.read_parquet(ens_file_b)
            
            # Merge Blind B
            df_ens_b = df_ens_b.drop([c for c in df_ens_b.columns if c.startswith("gt_")], strict=False)
            df_master_b = df_base_b.join(df_ens_b, on="session_id", how="inner")
            
            # Extract predictions using the winning model from Blind A
            session_ids_b = df_master_b["session_id"].to_list()
            best_preds_b = df_master_b[best_overall_col].to_list()
            
            for sid, pred in zip(session_ids_b, best_preds_b):
                if sid not in final_submission_dict_b:
                    final_submission_dict_b[sid] = {}
                final_submission_dict_b[sid][target] = pred
            
            print(f"   -> Successfully extracted {len(session_ids_b)} predictions for Blind B.")
        else:
            print(f"⚠️ Missing Blind B files for {target}. Expected:")
            print(f"   {base_file_b.name}")
            print(f"   {ens_file_b.name}")

    # ==============================================================================
    # EXPORT FINAL DICTIONARIES
    # ==============================================================================
    print("\n=======================================================")
    print(f"✅ EXPORT COMPLETE")
    
    # Save Blind A
    if final_submission_dict_a:
        out_json_path_a = ENSEMBLE_DIR / "final_blind_a_predictions.json"
        with open(out_json_path_a, "w") as f:
            json.dump(final_submission_dict_a, f, indent=4)
        print(f"-> Blind A: Saved {len(final_submission_dict_a)} sessions to {out_json_path_a.name}")
        
    # Save Blind B
    if final_submission_dict_b:
        out_json_path_b = ENSEMBLE_DIR / "final_blind_b_predictions.json"
        with open(out_json_path_b, "w") as f:
            json.dump(final_submission_dict_b, f, indent=4)
        print(f"-> Blind B: Saved {len(final_submission_dict_b)} imputed sessions to {out_json_path_b.name}")
        
    print("=======================================================")

if __name__ == "__main__":
    main()