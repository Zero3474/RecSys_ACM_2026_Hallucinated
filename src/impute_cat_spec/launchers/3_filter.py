import json
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

from resources import (
    BASELINE_DIR,
    PLOTS_DIR,
    MODELS_DIR
)

TARGETS = ["category", "specificity"]

# Filtering parameters
TOP_K_COMPETENT = 30          # Used for the heatmap visualization limit
MAX_ENSEMBLE_SIZE = 25        # How many models to greedily select
COMPETENCE_THRESHOLD = 0.85   # Models must have at least 85% of the Best Model's F1 to be considered

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def compute_performance(df: pl.DataFrame, target: str) -> pd.DataFrame:
    gt_col = f"gt_{target}"
    y_true = df[gt_col].to_numpy()
    
    results = []
    model_cols = [c for c in df.columns if c.startswith(f"{target}__")]
    
    for col in model_cols:
        y_pred = df[col].to_numpy()
        results.append({
            "model_id": col.replace(f"{target}__", ""),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred)
        })
        
    return pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)

def plot_leaderboard(perf_df: pd.DataFrame, target: str):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=perf_df.head(30), x="f1_macro", y="model_id", hue="model_id", palette="viridis", legend=False)
    plt.title(f"Top 30 Individual Models by F1-Macro ({target.upper()})")
    plt.xlabel("F1-Macro")
    plt.ylabel("Model Combination")
    plt.axvline(x=perf_df["f1_macro"].max() * COMPETENCE_THRESHOLD, color='r', linestyle='--', label=f"{COMPETENCE_THRESHOLD*100}% of Max F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"3_{target}_competence_leaderboard.png")
    plt.close()

def plot_diversity_clustermap(df: pl.DataFrame, top_models: list, target: str):
    """Upgraded: Uses Hierarchical Clustering to group similar models together visually."""
    cols_to_compare = [f"{target}__{m}" for m in top_models]
    
    n = len(cols_to_compare)
    kappa_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            y1 = df[cols_to_compare[i]].to_numpy()
            y2 = df[cols_to_compare[j]].to_numpy()
            kappa_matrix[i, j] = cohen_kappa_score(y1, y2)
            
    kappa_df = pd.DataFrame(kappa_matrix, index=top_models, columns=top_models)
    
    # Generate Clustermap
    g = sns.clustermap(
        kappa_df, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f",
        vmin=0.5, 
        vmax=1.0,
        figsize=(14, 12),
        annot_kws={"size": 8},
        cbar_pos=(0.02, 0.8, 0.05, 0.18)
    )
    g.fig.suptitle(f"Hierarchical Diversity Clustermap (Cohen's Kappa) - {target.upper()}\nBlocks of red indicate models that behave identically.", y=1.05)
    plt.savefig(PLOTS_DIR / f"3_{target}_diversity_clustermap.png", bbox_inches='tight')
    plt.close()
    
    return kappa_df

def plot_oracle_growth(history: list, target: str):
    steps = [h["step"] for h in history]
    scores = [h["oracle_acc"] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scores, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(f"Ensemble Potential: Maximum Reachable Accuracy ({target.upper()})\n(Competence-Gated Oracle Selection)")
    plt.xlabel("Number of Models in Ensemble")
    plt.ylabel("Maximum Reachable Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(steps)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"3_{target}_oracle_growth_curve.png")
    plt.close()

def greedy_oracle_selection(df: pl.DataFrame, perf_df: pd.DataFrame, target: str, max_size: int, threshold: float) -> list:
    gt_col = f"gt_{target}"
    y_true = df[gt_col].to_numpy()
    
    # 1. Competence Gating: Only allow models within X% of the top model
    best_f1 = perf_df["f1_macro"].max()
    min_f1 = best_f1 * threshold
    competent_models = perf_df[perf_df["f1_macro"] >= min_f1]["model_id"].tolist()
    
    print(f"\n[Filter] Retained {len(competent_models)} / {len(perf_df)} models that meet the {threshold*100}% F1 threshold.")
    
    selected_models = []
    history = []
    oracle_correct_mask = np.zeros(len(y_true), dtype=bool)
    
    print("\n*** ENSEMBLE POTENTIAL (ORACLE UPPER BOUND) ***")
    print(f"{'Step':<5} | {'Model Added':<40} | {'Indiv Acc':<10} | {'Max Reachable Acc (Oracle)'}")
    print("-" * 90)
    
    for step in range(1, max_size + 1):
        best_candidate = None
        best_new_oracle_acc = -1
        best_candidate_acc = 0
        best_new_mask = None
        
        # 2. Search only within the competent pool
        for model in competent_models:
            if model in selected_models:
                continue
                
            col_name = f"{target}__{model}"
            y_pred = df[col_name].to_numpy()
            
            model_correct_mask = (y_pred == y_true)
            new_oracle_mask = oracle_correct_mask | model_correct_mask
            new_oracle_acc = new_oracle_mask.mean()
            
            if new_oracle_acc > best_new_oracle_acc:
                best_new_oracle_acc = new_oracle_acc
                best_candidate = model
                best_candidate_acc = model_correct_mask.mean()
                best_new_mask = new_oracle_mask
                
        if best_candidate is None or best_new_oracle_acc == oracle_correct_mask.mean():
            print(f"\n[!] Stopping early at step {step-1}: No remaining *competent* models can improve the maximum score.")
            break
            
        selected_models.append(best_candidate)
        oracle_correct_mask = best_new_mask
        
        history.append({"step": step, "model": best_candidate, "oracle_acc": best_new_oracle_acc})
        print(f"{step:<5} | {best_candidate:<40} | {best_candidate_acc:.4%} | {best_new_oracle_acc:.4%}")
        
    return selected_models, history

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading and merging sweep predictions from cluster...")
    pred_files = list(BASELINE_DIR.glob("sweep_hard_predictions_*.parquet"))
    
    if not pred_files:
        raise FileNotFoundError(f"Could not find any parquet files in {BASELINE_DIR}. Check your cluster output.")
    
    print(f"Found {len(pred_files)} prediction files. Merging...")
    preds_df = pl.read_parquet(pred_files[0])
    
    for f in pred_files[1:]:
        temp_df = pl.read_parquet(f)
        temp_df = temp_df.drop([c for c in temp_df.columns if c.startswith("gt_")], strict=False)
        preds_df = preds_df.join(temp_df, on="session_id", how="inner")
        
    final_selections = {}

    for target in TARGETS:
        print(f"\n=======================================================")
        print(f"ANALYZING TARGET: {target.upper()}")
        print(f"=======================================================")
        
        # 1. Competence Leaderboard
        perf_df = compute_performance(preds_df, target)
        plot_leaderboard(perf_df, target)
        
        # 2. Clustered Diversity Heatmap
        top_models = perf_df.head(TOP_K_COMPETENT)["model_id"].tolist()
        plot_diversity_clustermap(preds_df, top_models, target)
        
        # 3. Competence-Gated Oracle Selection
        selected_ensemble, history = greedy_oracle_selection(
            preds_df, 
            perf_df, 
            target, 
            max_size=MAX_ENSEMBLE_SIZE, 
            threshold=COMPETENCE_THRESHOLD
        )
        
        # 4. Plot the Growth Curve
        plot_oracle_growth(history, target)
            
        final_selections[target] = selected_ensemble
        
    # Save the final recommendations
    out_file = MODELS_DIR / "recommended_ensembles.json"
    with open(out_file, "w") as f:
        json.dump(final_selections, f, indent=4)
    print(f"\nSaved optimal diverse models for tuning to {out_file}")

if __name__ == "__main__":
    main()