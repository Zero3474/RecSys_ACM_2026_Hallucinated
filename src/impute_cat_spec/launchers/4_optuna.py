import json
import argparse
from pathlib import Path
from typing import Dict, Any

import polars as pl
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from resources import (
    TRAIN,
    TEST,
    EMBEDDINGS_DIR,
    MODELS_DIR
)

import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "random_state": 42,
    "n_trials": 100,
    "cv_folds": 5,
    
    "search_spaces": {
        "logistic": {
            # Widened bounds slightly to ensure Optuna explores extreme regularization
            "C": ("loguniform", 1e-5, 1e3),
            "max_iter": ("categorical", [1000]),
            "class_weight": ("categorical", ["balanced", None])
        },
        "svc": {
            # Widened bounds here as well
            "C": ("loguniform", 1e-5, 1e3),
            "max_iter": ("categorical", [2000]),
            "class_weight": ("categorical", ["balanced", None])
        },
        "mlp": {
            # Kept exactly as you had it. 100 trials is perfect for this space.
            "hidden_layer_sizes": ("categorical", ["128", "256,128", "128,64", "64", "256,128,64"]),
            "activation": ("categorical", ["relu", "tanh"]),
            "alpha": ("loguniform", 1e-5, 1e-1),
            "learning_rate_init": ("loguniform", 1e-4, 1e-2),
            "max_iter": ("categorical", [500])
        },
        "knn": {
            # Expanded max neighbors to 50. High-dimensional embeddings often benefit 
            # from looking at a wider neighborhood to smooth out noise.
            "n_neighbors": ("int", 3, 50),
            "weights": ("categorical", ["uniform", "distance"]),
            "p": ("categorical", [1, 2])
        },
        "lgbm": {
            "n_estimators": ("int", 100, 500),
            "learning_rate": ("loguniform", 1e-3, 0.1),
            "num_leaves": ("int", 20, 150),
            "max_depth": ("int", 3, 12),
            "subsample": ("uniform", 0.5, 1.0),            # Row sampling
            "colsample_bytree": ("uniform", 0.5, 1.0),     # Feature sampling
            "min_child_samples": ("int", 10, 50)           # Prevents overfitting small leaves
        },
        "xgb": {
            "n_estimators": ("int", 100, 500),
            "learning_rate": ("loguniform", 1e-3, 0.1),
            "max_depth": ("int", 3, 12),
            "subsample": ("uniform", 0.5, 1.0),            # Row sampling
            "colsample_bytree": ("uniform", 0.5, 1.0),     # Feature sampling
            "min_child_weight": ("int", 1, 10)             # Prevents overfitting small leaves
        }
    }
}

# DATA LOADING FUNCTIONS
def load_labels(file_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(file_path, columns=["session_id", "conversation_goal"])
    return df.with_columns(
        pl.col("conversation_goal").struct.field("category").alias("category"),
        pl.col("conversation_goal").struct.field("specificity").alias("specificity")
    ).drop("conversation_goal").sort("session_id")

def load_single_embedding(model_name: str, method: str, split: str) -> pl.DataFrame:
    """Loads and aggregates a single embedding type on demand to save memory."""
    file_path = EMBEDDINGS_DIR / f"{split}_{model_name}_embeddings.parquet"
    df = pl.read_parquet(file_path)
    
    if method == "first":
        agg_df = df.sort(["session_id", "turn_number"]).group_by("session_id").first().select(["session_id", "embedding"])
    else:
        emb_dim = df["embedding"].list.len().max()
        field_names = [f"f{i}" for i in range(emb_dim)]
        df = df.with_columns(pl.col("embedding").list.to_struct(fields=field_names))
        
        if method == "mean":
            exprs = [pl.col("embedding").struct.field(f).mean().alias(f) for f in field_names]
        elif method == "std":
            exprs = [pl.col("embedding").struct.field(f).std().fill_null(0.0).alias(f) for f in field_names]
        else:
            raise ValueError(f"Unknown aggregation: {method}")
            
        agg_df = df.group_by("session_id").agg(exprs)
        agg_df = agg_df.with_columns(pl.concat_list([pl.col(f) for f in field_names]).alias("embedding")).select(["session_id", "embedding"])
        
    return agg_df.sort("session_id")

# MODEL INSTANTIATION FUNCTION
def instantiate_model(trial: optuna.Trial, model_type: str, search_space: Dict[str, Any]):
    params = {}
    for param_name, config in search_space.items():
        param_type = config[0]
        if param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config[1])
        elif param_type == "int":
            params[param_name] = trial.suggest_int(param_name, config[1], config[2])
        elif param_type == "loguniform":
            params[param_name] = trial.suggest_float(param_name, config[1], config[2], log=True)
        elif param_type == "uniform":
            params[param_name] = trial.suggest_float(param_name, config[1], config[2])
            
    if model_type == "logistic":
        return LogisticRegression(**params, random_state=CONFIG["random_state"])
    elif model_type == "svc":
        return LinearSVC(**params, random_state=CONFIG["random_state"])
    elif model_type == "mlp":
        hl_str = params["hidden_layer_sizes"]
        params["hidden_layer_sizes"] = tuple(int(x) for x in hl_str.split(","))
        return MLPClassifier(**params, random_state=CONFIG["random_state"], early_stopping=True)
    elif model_type == "knn":
        return KNeighborsClassifier(**params)
    elif model_type == "lgbm":
        return LGBMClassifier(**params, random_state=CONFIG["random_state"], n_jobs=-1, verbose=-1)
    elif model_type == "xgb":
        return XGBClassifier(**params, random_state=CONFIG["random_state"], n_jobs=-1, eval_metric="mlogloss")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Tune a specific pipeline via Optuna with 5-Fold CV.")
    parser.add_argument("--target", type=str, required=True, choices=["category", "specificity"])
    parser.add_argument("--emb_model", type=str, required=True, help="e.g., qwen3-embed-4b")
    parser.add_argument("--agg_method", type=str, required=True, choices=["first", "mean", "std", "max", "min", "sum"])
    parser.add_argument("--clf_model", type=str, required=True, choices=list(CONFIG["search_spaces"].keys()))
    args = parser.parse_args()
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    pipeline_str = f"{args.emb_model}__{args.agg_method}__{args.clf_model}"
    print(f"\n===================================================")
    print(f"🚀 INITIALIZING CV WORKER FOR: {pipeline_str}")
    print(f"🎯 TARGET: {args.target.upper()}")
    print(f"===================================================")
    
    db_path = MODELS_DIR / f"optuna_{args.target}.db"
    storage_url = f"sqlite:///{db_path.absolute()}"
    
    # We still load TEST labels ONLY to ensure the LabelEncoder learns every possible 
    # category in the challenge, but we DO NOT load or evaluate on the test embeddings.
    train_labels = load_labels(TRAIN)
    test_labels = load_labels(TEST)

    le = LabelEncoder()
    all_targets = pl.concat([train_labels.select(args.target), test_labels.select(args.target)])
    le.fit(all_targets[args.target].to_list())
    
    y_train = le.transform(train_labels[args.target].to_list())
    
    # Free test labels from memory just to be absolutely safe
    del test_labels, all_targets
    
    # Load ONLY Training Embedding Data
    print(f"Loading training matrices for {args.emb_model} ({args.agg_method})...")
    train_emb = load_single_embedding(args.emb_model, args.agg_method, "train")
    train_ds = train_labels.join(train_emb, on="session_id", how="inner").sort("session_id")
    X_train = np.array(train_ds["embedding"].to_list(), dtype=np.float32)
    
    # Define CV Objective
    def objective(trial: optuna.Trial):
        # Initialize K-Fold
        skf = StratifiedKFold(n_splits=CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"])
        fold_accuracies = []
        
        # Loop through the 5 splits
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            clf = instantiate_model(trial, args.clf_model, CONFIG["search_spaces"][args.clf_model])
            clf.fit(X_fold_train, y_fold_train)
            
            preds = clf.predict(X_fold_val)
            fold_accuracies.append(accuracy_score(y_fold_val, preds))
            
        # Optuna evaluates based on the average performance across all 5 folds
        return np.mean(fold_accuracies)
        
    study = optuna.create_study(
        study_name=pipeline_str, 
        storage=storage_url, 
        load_if_exists=True, 
        direction="maximize",
        sampler=TPESampler(seed=CONFIG["random_state"])
    )
    
    study.optimize(objective, n_trials=CONFIG["n_trials"], show_progress_bar=True)
    
    print(f"\n✅ Optimization Complete: {pipeline_str}")
    print(f"🏆 Best 5-Fold CV Accuracy: {study.best_value:.4f}")
    
    result_dict = {
        "pipeline_id": pipeline_str,
        "target": args.target,
        "emb_model": args.emb_model,
        "agg_method": args.agg_method,
        "clf_model": args.clf_model,
        "cv_accuracy": study.best_value,
        "best_params": study.best_params
    }
    
    out_json = MODELS_DIR / f"tuned_{args.target}__{pipeline_str}.json"
    with open(out_json, "w") as f:
        json.dump(result_dict, f, indent=4)
        
    print(f"💾 Saved worker parameters to: {out_json}")

if __name__ == "__main__":
    main()