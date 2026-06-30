import argparse
from pathlib import Path
from typing import Any
import polars as pl
import numpy as np
from tqdm import tqdm
import warnings

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
    BASELINE_DIR
)

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

EMBEDDINGS = ["bge-large", "e5-large", "mpnet", "mxbai", "sbert-minilm", "tfidf", "qwen3-embed-0.6b", "qwen3-embed-4b"]
AGGREGATIONS = ["first", "mean", "std"]
ALL_MODELS = ["logistic", "knn", "mlp", "svc", "lgbm", "xgb"]
TARGETS = ["category", "specificity"]

RANDOM_STATE = 42

def get_model(model_name: str, use_gpu: bool = False) -> Any:
    """Returns a scikit-learn classifier with default/stable baselines."""
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "svc":
        return LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == "mlp":
        return MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
    elif model_name == "knn":
        return KNeighborsClassifier()
    elif model_name == "lgbm":
        kwargs = {"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
        if use_gpu:
            kwargs["device"] = "gpu"
        return LGBMClassifier(**kwargs)
    elif model_name == "xgb":
        kwargs = {"random_state": RANDOM_STATE, "n_jobs": -1, "eval_metric": "mlogloss"}
        if use_gpu:
            kwargs["tree_method"] = "hist"
            kwargs["device"] = "cuda"
        return XGBClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

# ==========================================
# 2. DATA PROCESSING AND AGGREGATION
# ==========================================
def load_labels(file_path: Path) -> pl.DataFrame:
    """Extracts ground truth structure."""
    df = pl.read_parquet(file_path, columns=["session_id", "conversation_goal"])
    return df.with_columns(
        pl.col("conversation_goal").struct.field("category").alias("category"),
        pl.col("conversation_goal").struct.field("specificity").alias("specificity")
    ).drop("conversation_goal")

def aggregate_embeddings(df: pl.DataFrame, method: str) -> pl.DataFrame:
    """Session-level aggregator processing first, mean, and standard deviation."""
    if method == "first":
        return df.sort(["session_id", "turn_number"]).group_by("session_id").first().select(["session_id", "embedding"])
    
    emb_dim = df["embedding"].list.len().max()
    field_names = [f"f{i}" for i in range(emb_dim)]
    df = df.with_columns(pl.col("embedding").list.to_struct(fields=field_names))
    
    if method == "mean":
        exprs = [pl.col("embedding").struct.field(f).mean().alias(f) for f in field_names]
    elif method == "std":
        exprs = [pl.col("embedding").struct.field(f).std().fill_null(0.0).alias(f) for f in field_names]
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")
    
    agg_df = df.group_by("session_id").agg(exprs)
    return agg_df.with_columns(
        pl.concat_list([pl.col(f) for f in field_names]).alias("embedding")
    ).select(["session_id", "embedding"])

def main():
    # --- SETUP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Run embedding baselines.")
    parser.add_argument("--model", type=str, choices=ALL_MODELS + ["all"], default="all", 
                        help="Which model to run. Use 'all' to run sequentially.")
    parser.add_argument("--use_gpu", action="store_true", 
                        help="Enable GPU acceleration for LGBM and XGBoost.")
    args = parser.parse_args()

    active_models = ALL_MODELS if args.model == "all" else [args.model]
    
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Labels
    train_labels = load_labels(TRAIN).sort("session_id")
    test_labels = load_labels(TEST).sort("session_id")
    
    pred_matrix = test_labels.select("session_id")
    proba_matrix = test_labels.select("session_id")
    
    # 2. Iterate through targets independently
    for target in TARGETS:
        print(f"\n==================== PROCESSING TARGET: {target.upper()} ====================")
        
        le = LabelEncoder()
        all_targets = pl.concat([train_labels.select(target), test_labels.select(target)])
        le.fit(all_targets[target].to_list())
        
        y_train = le.transform(train_labels[target].to_list())
        y_test = le.transform(test_labels[target].to_list())
        
        pred_matrix = pred_matrix.with_columns(pl.Series(f"gt_{target}", y_test))
        
        # 3. Loop through combinations
        for emb_model in tqdm(EMBEDDINGS, desc=f"Embedding Models for {target}"):
            train_emb_path = EMBEDDINGS_DIR / f"train_{emb_model}_embeddings.parquet"
            test_emb_path = EMBEDDINGS_DIR / f"test_{emb_model}_embeddings.parquet"
            
            if not train_emb_path.exists() or not test_emb_path.exists():
                print(f"Skipping missing embedding files for: {emb_model}")
                continue
                
            print(f"-> Loading turn arrays for: {emb_model}")
            raw_train_emb = pl.read_parquet(train_emb_path)
            raw_test_emb = pl.read_parquet(test_emb_path)
            
            for agg in AGGREGATIONS:
                print(f"   -> Aggregating via strategy: {agg}")
                agg_train = aggregate_embeddings(raw_train_emb, agg).rename({"embedding": "final_embedding"})
                agg_test = aggregate_embeddings(raw_test_emb, agg).rename({"embedding": "final_embedding"})
                
                # Join AND sort to guarantee row order alignment
                train_ds = train_labels.join(agg_train, on="session_id", how="inner").sort("session_id")
                test_ds = test_labels.join(agg_test, on="session_id", how="inner").sort("session_id")
                
                # Extract X and y from the SAME synced dataframe
                X_train = np.array(train_ds["final_embedding"].to_list(), dtype=np.float32)
                y_train_synced = le.transform(train_ds[target].to_list())
                
                X_test = np.array(test_ds["final_embedding"].to_list(), dtype=np.float32)
                
                for model_name in active_models:
                    col_identifier = f"{target}__{emb_model}__{agg}__{model_name}"
                    print(f"      -> Training: {model_name} (GPU: {args.use_gpu})")
                    
                    try:
                        clf = get_model(model_name, use_gpu=args.use_gpu)
                        clf.fit(X_train, y_train_synced)
                        
                        preds = clf.predict(X_test)
                        pred_matrix = pred_matrix.with_columns(pl.Series(col_identifier, preds))
                        
                        if hasattr(clf, "predict_proba"):
                            probas = clf.predict_proba(X_test)
                            if probas.shape[1] == 2:
                                probas = probas[:, 1]
                            else:
                                probas = probas[np.arange(len(preds)), preds]
                        else:
                            probas = clf.decision_function(X_test)
                            
                        proba_matrix = proba_matrix.with_columns(pl.Series(col_identifier, probas))
                        
                    except Exception as e:
                        print(f"      [ERROR] Failed on {col_identifier}: {str(e)}")
                        
    # 4. Save matrices with model-specific filenames to prevent cluster write collisions
    print(f"\n--- Model '{args.model}' executed. Saving prediction matrix data... ---")
    
    file_suffix = args.model
    pred_output = BASELINE_DIR / f"sweep_hard_predictions_{file_suffix}.parquet"
    proba_output = BASELINE_DIR / f"sweep_probabilities_{file_suffix}.parquet"
    
    pred_matrix.write_parquet(pred_output)
    proba_matrix.write_parquet(proba_output)
    
    print(f"Saved hard predictions matrix to: {pred_output}")
    print(f"Saved certainty/probability matrix to: {proba_output}")

if __name__ == "__main__":
    main()