import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import mode

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from resources import (
    FUSION_DIR,
    ENSEMBLE_DIR
)

import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
TARGETS = ["category", "specificity"]
RANDOM_STATE = 42

# ==============================================================================
# DATA PREPARATION HELPER
# ==============================================================================
def parse_meta_features(df: pl.DataFrame, target: str, encoder=None):
    """
    Extracts ground truth (if available), hard predictions, and soft probability matrices.
    """
    gt_col = f"gt_{target}"
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    
    # If no encoder is provided, build it (usually done on training/test meta-features)
    if encoder == None:
        unique_classes = sorted(list(set(df[gt_col].unique().to_list() + 
                                         [val for c in pred_cols for val in df[c].unique().to_list()])))
        encoder = LabelEncoder().fit(unique_classes)
        
    y = encoder.transform(df[gt_col].to_numpy()) if gt_col in df.columns else None
    
    # 2D Array of hard predictions (encoded as integers)
    hard_preds = np.zeros((len(df), len(pred_cols)), dtype=np.int32)
    for idx, col in enumerate(pred_cols):
        hard_preds[:, idx] = encoder.transform(df[col].to_numpy())
        
    # 3D Array of soft probabilities: [n_models, n_samples, n_classes]
    n_classes = len(encoder.classes_)
    soft_probs = np.zeros((len(prob_cols), len(df), n_classes), dtype=np.float32)
    for idx, col in enumerate(prob_cols):
        soft_probs[idx, :, :] = np.vstack(df[col].to_numpy())
        
    return y, hard_preds, soft_probs, encoder

def flatten_meta_features(soft_probs, add_rank_transform=False):
    """Flattens 3D array [models, samples, classes] into 2D [samples, models * classes]"""
    n_models, n_samples, n_classes = soft_probs.shape
    flattened = np.hstack([soft_probs[m] for m in range(n_models)])
    
    if add_rank_transform:
        ranks = np.zeros_like(soft_probs, dtype=np.float32)
        for m in range(n_models):
            ranks[m] = np.argsort(np.argsort(-soft_probs[m], axis=1), axis=1)
        flattened_ranks = np.hstack([ranks[m] for m in range(n_models)])
        flattened = np.hstack([flattened, flattened_ranks])
        
    return flattened

# ==============================================================================
# FUSION MODEL RUNNERS (Returning Predictions & Probabilities)
# ==============================================================================
def run_hard_voting(hard_preds):
    modes, _ = mode(hard_preds, axis=1, keepdims=False)
    return modes, None

def run_soft_voting(soft_probs):
    mean_probs = np.mean(soft_probs, axis=0)
    preds = np.argmax(mean_probs, axis=1)
    return preds, mean_probs

def run_geometric_mean(soft_probs, eps=1e-15):
    log_probs = np.log(soft_probs + eps)
    sum_log_probs = np.sum(log_probs, axis=0)
    geom_probs = np.exp(sum_log_probs / soft_probs.shape[0])
    geom_probs /= np.sum(geom_probs, axis=1, keepdims=True)
    preds = np.argmax(geom_probs, axis=1)
    return preds, geom_probs

def run_borda_count(soft_probs):
    ranks = np.zeros_like(soft_probs, dtype=np.int32)
    for m in range(soft_probs.shape[0]):
        ranks[m] = np.argsort(np.argsort(-soft_probs[m], axis=1), axis=1)
    avg_ranks = np.mean(ranks, axis=0)
    preds = np.argmin(avg_ranks, axis=1)
    pseudo_probs = 1.0 / (avg_ranks + 1.0)
    pseudo_probs /= np.sum(pseudo_probs, axis=1, keepdims=True)
    return preds, pseudo_probs

def fit_caruana_ensemble_selection(soft_train, y_train, iterations=50):
    n_models, n_samples, n_classes = soft_train.shape
    selected_models = []
    current_train_sum = np.zeros((n_samples, n_classes))
    
    for i in range(iterations):
        best_metric = -1
        best_model_idx = -1
        for m in range(n_models):
            temp_sum = current_train_sum + soft_train[m]
            metric = f1_score(y_train, np.argmax(temp_sum, axis=1), average="macro")
            if metric > best_metric:
                best_metric = metric
                best_model_idx = m
        selected_models.append(best_model_idx)
        current_train_sum += soft_train[best_model_idx]
    return selected_models

def predict_caruana(soft_eval, selected_models):
    val_sum = np.zeros((soft_eval.shape[1], soft_eval.shape[2]))
    for m in selected_models:
        val_sum += soft_eval[m]
    probs = val_sum / len(selected_models)
    preds = np.argmax(probs, axis=1)
    return preds, probs

def fit_scipy_optimization(soft_train, y_train):
    n_models = soft_train.shape[0]
    def loss_func(weights):
        w = weights / np.sum(weights)
        blend = np.tensordot(w, soft_train, axes=(0, 0))
        return -f1_score(y_train, np.argmax(blend, axis=1), average="macro")
    
    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x / np.sum(res.x)

def predict_scipy(soft_eval, weights):
    probs = np.tensordot(weights, soft_eval, axes=(0, 0))
    preds = np.argmax(probs, axis=1)
    return preds, probs

# ==============================================================================
# MAIN ENGINE EXECUTION
# ==============================================================================
def main():
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    
    for target in TARGETS:
        print(f"\n=======================================================")
        print(f"🔥 TRAINING FUSION & GENERATING PREDICTIONS FOR: {target.upper()}")
        print(f"=======================================================")
        
        # We use TEST as the training set for our Meta-Learners
        meta_train_file = FUSION_DIR / f"meta_features_test_{target}.parquet" 
        blinda_file = FUSION_DIR / f"meta_features_blinda_{target}.parquet"
        blindb_file = FUSION_DIR / f"meta_features_blindb_{target}.parquet"
        
        if not all(p.exists() for p in [meta_train_file, blinda_file, blindb_file]):
            print(f"⚠️ Missing required parquet splits for {target}. Skipping.")
            continue
            
        df_meta_train = pl.read_parquet(meta_train_file)
        df_a = pl.read_parquet(blinda_file)
        df_b = pl.read_parquet(blindb_file).drop(f"gt_{target}", strict=False)
        
        # Parse datasets: Fit encoder on the Meta-Train (Test) set, apply to Blind sets
        y_train, _, soft_train, le = parse_meta_features(df_meta_train, target, encoder=None)
        y_a, hard_a, soft_a, _ = parse_meta_features(df_a, target, encoder=le)
        y_b, hard_b, soft_b, _ = parse_meta_features(df_b, target, encoder=le)
        
        # Prepare DataFrames to export final predictions
        out_dfs = {
            "blind_a": df_a.select(["session_id"]),
            "blind_b": df_b.select(["session_id"])
        }
        
        # Track performance on Blind A for the validation leaderboard
        leaderboard = {}
        
        # ----------------------------------------------------------------------
        # EXECUTE GROUP 1: MATH HEURISTICS
        # ----------------------------------------------------------------------
        heuristics = {
            "G1_Hard_Voting": lambda s, h: run_hard_voting(h),
            "G1_Soft_Voting": lambda s, h: run_soft_voting(s),
            "G1_Geometric_Mean": lambda s, h: run_geometric_mean(s),
            "G1_Borda_Count": lambda s, h: run_borda_count(s)
        }
        
        for name, func in heuristics.items():
            # Blind A
            preds_a, probs_a = func(soft_a, hard_a)
            leaderboard[name] = (accuracy_score(y_a, preds_a), f1_score(y_a, preds_a, average="macro"))
            out_dfs["blind_a"] = out_dfs["blind_a"].with_columns([
                pl.Series(f"pred_{name}", le.inverse_transform(preds_a)),
                pl.Series(f"prob_{name}", probs_a.tolist() if probs_a is not None else [[]]*len(preds_a))
            ])
            # Blind B
            preds_b, probs_b = func(soft_b, hard_b)
            out_dfs["blind_b"] = out_dfs["blind_b"].with_columns([
                pl.Series(f"pred_{name}", le.inverse_transform(preds_b)),
                pl.Series(f"prob_{name}", probs_b.tolist() if probs_b is not None else [[]]*len(preds_b))
            ])

        # ----------------------------------------------------------------------
        # EXECUTE GROUP 2: OPTIMIZERS
        # ----------------------------------------------------------------------
        # Caruana Selection
        caruana_models = fit_caruana_ensemble_selection(soft_train, y_train)
        preds_ca, probs_ca = predict_caruana(soft_a, caruana_models)
        preds_cb, probs_cb = predict_caruana(soft_b, caruana_models)
        
        leaderboard["G2_Caruana_Selection"] = (accuracy_score(y_a, preds_ca), f1_score(y_a, preds_ca, average="macro"))
        for split, p_idx, p_arr in [("blind_a", preds_ca, probs_ca), ("blind_b", preds_cb, probs_cb)]:
            out_dfs[split] = out_dfs[split].with_columns([
                pl.Series("pred_G2_Caruana", le.inverse_transform(p_idx)),
                pl.Series("prob_G2_Caruana", p_arr.tolist())
            ])
            
        # SciPy Weight Minimization
        scipy_weights = fit_scipy_optimization(soft_train, y_train)
        preds_sa, probs_sa = predict_scipy(soft_a, scipy_weights)
        preds_sb, probs_sb = predict_scipy(soft_b, scipy_weights)
        
        leaderboard["G2_SciPy_Optimization"] = (accuracy_score(y_a, preds_sa), f1_score(y_a, preds_sa, average="macro"))
        for split, p_idx, p_arr in [("blind_a", preds_sa, probs_sa), ("blind_b", preds_sb, probs_sb)]:
            out_dfs[split] = out_dfs[split].with_columns([
                pl.Series("pred_G2_SciPy", le.inverse_transform(p_idx)),
                pl.Series("prob_G2_SciPy", p_arr.tolist())
            ])

        # ----------------------------------------------------------------------
        # EXECUTE GROUP 3: META-LEARNERS (Standard & Enhanced)
        # ----------------------------------------------------------------------
        for enhanced in [False, True]:
            suffix = "_Enhanced" if enhanced else "_Standard"
            
            X_train = flatten_meta_features(soft_train, add_rank_transform=enhanced)
            X_a_meta = flatten_meta_features(soft_a, add_rank_transform=enhanced)
            X_b_meta = flatten_meta_features(soft_b, add_rank_transform=enhanced)
            
            stackers = {
                f"G3_Logistic{suffix}": LogisticRegression(max_iter=3000, C=0.5, random_state=RANDOM_STATE),
                f"G3_MLP{suffix}": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
                f"G3_XGBoost{suffix}": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=RANDOM_STATE, eval_metric="mlogloss")
            }
            
            for name, clf in stackers.items():
                clf.fit(X_train, y_train)
                
                # Predict on Blind A
                preds_la = clf.predict(X_a_meta)
                probs_la = clf.predict_proba(X_a_meta)
                leaderboard[name] = (accuracy_score(y_a, preds_la), f1_score(y_a, preds_la, average="macro"))
                
                # Predict on Blind B
                preds_lb = clf.predict(X_b_meta)
                probs_lb = clf.predict_proba(X_b_meta)
                
                # Append arrays out to splits
                for split, p_idx, p_arr in [("blind_a", preds_la, probs_la), ("blind_b", preds_lb, probs_lb)]:
                    out_dfs[split] = out_dfs[split].with_columns([
                        pl.Series(f"pred_{name}", le.inverse_transform(p_idx)),
                        pl.Series(f"prob_{name}", p_arr.tolist())
                    ])

        # ----------------------------------------------------------------------
        # SAVE PERFORMANCE & SHOW LEADERBOARD
        # ----------------------------------------------------------------------
        out_dfs["blind_a"].write_parquet(ENSEMBLE_DIR / f"ensemble_predictions_blinda_{target}.parquet")
        out_dfs["blind_b"].write_parquet(ENSEMBLE_DIR / f"ensemble_predictions_blindb_{target}.parquet")
        print(f"💾 Saved comprehensive ensemble prediction files to {ENSEMBLE_DIR.name}/")

        print(f"\n{'Ensemble Fusion Strategy (Leaderboard on Blind A)':<45} | {'Accuracy':<10} | {'Macro F1':<10}")
        print("-" * 72)
        sorted_leaderboard = sorted(leaderboard.items(), key=lambda item: item[1][1], reverse=True)
        for strategy, (acc, f1) in sorted_leaderboard:
            print(f"{strategy:<45} | {acc:>9.4%} | {f1:>9.4%}")

if __name__ == "__main__":
    main()