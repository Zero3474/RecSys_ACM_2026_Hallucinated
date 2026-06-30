import json
from typing import Dict, Any

import polars as pl
import numpy as np
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from resources import (
    TRAIN,
    TEST,
    BLIND_A,
    BLIND_B,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    FUSION_DIR
)

import warnings
warnings.filterwarnings("ignore")

TARGETS = ["category", "specificity"]
RANDOM_STATE = 42

# ==========================================
# 2. DATA LOADING & PREP
# ==========================================
def load_and_standardize_labels(split: str) -> pl.DataFrame:
    """Loads datasets and extracts the target labels."""
    print(f"Loading labels for {split.upper()}...")
    if split in ["train", "test", "blind_a"]:
        paths = {"train": TRAIN, "test": TEST, "blind_a": BLIND_A}
        df = pl.read_parquet(paths[split], columns=["session_id", "conversation_goal"])
        df = df.with_columns(
            pl.col("conversation_goal").struct.field("category").alias("category"),
            pl.col("conversation_goal").struct.field("specificity").alias("specificity")
        ).drop("conversation_goal")
        
    elif split == "blind_b":
        df_b = pl.read_parquet(BLIND_B, columns=["session_id", "conversation_goal"])
        goal_dtype = df_b.schema["conversation_goal"]
        
        # If it is a struct AND contains the 'category' field
        if isinstance(goal_dtype, pl.Struct) and "category" in [f.name for f in goal_dtype.fields]:
            df_b = (
                df_b
                .unnest("conversation_goal")
                .with_columns(
                    pl.col("category").fill_null("unknown"),
                    pl.col("specificity").fill_null("unknown")
                )
            )
        else:
            # If it's pl.Null or missing the fields entirely, just create the columns manually
            df_b = df_b.with_columns(
                category=pl.lit("unknown"),
                specificity=pl.lit("unknown")
            ).drop("conversation_goal")
        
        df = df_b.select(["session_id", "category", "specificity"])
            
    return df.sort("session_id")

def load_single_embedding(model_name: str, method: str, split: str) -> pl.DataFrame:
    """Loads and aggregates a single embedding."""
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
            
        agg_df = df.group_by("session_id").agg(exprs)
        agg_df = agg_df.with_columns(pl.concat_list([pl.col(f) for f in field_names]).alias("embedding")).select(["session_id", "embedding"])
        
    return agg_df.sort("session_id")

# ==========================================
# 3. MODEL INSTANTIATION
# ==========================================
def instantiate_best_model(model_type: str, params: Dict[str, Any]):
    """Instantiates the model. LinearSVC is wrapped to generate Probabilities."""
    if model_type == "logistic":
        return LogisticRegression(**params, random_state=RANDOM_STATE)
    elif model_type == "svc":
        # Wrap SVC in CalibratedClassifierCV so it outputs proper .predict_proba() arrays!
        base_svc = LinearSVC(**params, random_state=RANDOM_STATE)
        return CalibratedClassifierCV(base_svc, cv=5)
    elif model_type == "mlp":
        if isinstance(params.get("hidden_layer_sizes"), str):
            params["hidden_layer_sizes"] = tuple(int(x) for x in params["hidden_layer_sizes"].split(","))
        return MLPClassifier(**params, random_state=RANDOM_STATE, early_stopping=True)
    elif model_type == "knn":
        return KNeighborsClassifier(**params)
    elif model_type == "lgbm":
        return LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    elif model_type == "xgb":
        return XGBClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mlogloss")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# ==========================================
# 4. EXECUTION
# ==========================================
def main():
    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    
    rec_ens_path = MODELS_DIR / "recommended_ensembles.json"
    if not rec_ens_path.exists():
        raise FileNotFoundError(f"Could not find {rec_ens_path}. Please run 3_filter.py first to generate the recommended models for ensembles.")
    
    with open(rec_ens_path, "r") as f:
        ensemble_data = json.load(f)
        
    df_train = load_and_standardize_labels("train")
    df_test = load_and_standardize_labels("test")
    df_blind_a = load_and_standardize_labels("blind_a")
    df_blind_b = load_and_standardize_labels("blind_b")

    for target in TARGETS:
        print(f"\n=======================================================")
        print(f"GENERATING META-FEATURES FOR TARGET: {target.upper()}")
        print(f"=======================================================")
        
        pipelines = ensemble_data.get(target, [])
        if not pipelines:
            continue
            
        # 1. Label Encoding
        le = LabelEncoder()
        all_targets = pl.concat([df_train.select(target), df_test.select(target)])
        le.fit(all_targets[target].to_list())
        
        y_train = le.transform(df_train[target].to_list())
        y_test = le.transform(df_test[target].to_list())
        
        # Initialize Fusion DataFrames with Ground Truth
        meta_test = df_test.select(["session_id", target]).rename({target: f"gt_{target}"})
        meta_blind_a = df_blind_a.select(["session_id", target]).rename({target: f"gt_{target}"})
        meta_blind_b = df_blind_b.select(["session_id", target]).rename({target: f"gt_{target}"})

        # Connect to SQLite to grab the best params safely
        db_path = MODELS_DIR / f"optuna_{target}.db"
        storage_url = f"sqlite:///{db_path.absolute()}"

        for pipeline_str in pipelines:
            emb_model, agg_method, clf_model = pipeline_str.split("__")
            print(f"\n-> Processing Pipeline: {pipeline_str}")
            
            # 2. Extract Best Params
            try:
                study = optuna.load_study(study_name=pipeline_str, storage=storage_url)
                best_params = study.best_params
                print(f"   [Optuna] Loaded best params (Score: {study.best_value:.4f})")
            except Exception as e:
                print(f"   [Warning] Could not load study for {pipeline_str}. Using default params. Error: {e}")
                best_params = {}
                
            # 3. Load Embeddings
            train_emb = load_single_embedding(emb_model, agg_method, "train")
            test_emb = load_single_embedding(emb_model, agg_method, "test")
            a_emb = load_single_embedding(emb_model, agg_method, "blind_a")
            b_emb = load_single_embedding(emb_model, agg_method, "blind_b")
            
            # Align and extract X
            X_train = np.array(df_train.join(train_emb, on="session_id", how="inner").sort("session_id")["embedding"].to_list(), dtype=np.float32)
            X_test = np.array(meta_test.join(test_emb, on="session_id", how="inner").sort("session_id")["embedding"].to_list(), dtype=np.float32)
            X_a = np.array(meta_blind_a.join(a_emb, on="session_id", how="inner").sort("session_id")["embedding"].to_list(), dtype=np.float32)
            X_b = np.array(meta_blind_b.join(b_emb, on="session_id", how="inner").sort("session_id")["embedding"].to_list(), dtype=np.float32)
            
            # Combine Train + Test for the final mega-models
            X_full = np.vstack((X_train, X_test))
            y_full = np.concatenate((y_train, y_test))
            
            # ==========================================
            # PHASE A: Train on Train (For Test Meta-Features)
            # ==========================================
            clf_meta = instantiate_best_model(clf_model, best_params)
            clf_meta.fit(X_train, y_train)
            
            preds_int = clf_meta.predict(X_test)
            preds_str = le.inverse_transform(preds_int)
            probs = clf_meta.predict_proba(X_test)
            
            temp_test = pl.DataFrame({
                "session_id": meta_test["session_id"], 
                f"pred_{pipeline_str}": preds_str,
                f"prob_{pipeline_str}": probs.tolist()
            })
            meta_test = meta_test.join(temp_test, on="session_id")
            
            # ==========================================
            # PHASE B: Train on Train+Test (For Blind Predictions)
            # ==========================================
            clf_final = instantiate_best_model(clf_model, best_params)
            clf_final.fit(X_full, y_full)
            
            for split_name, X_mat, result_df in [
                ("blind_a", X_a, meta_blind_a), 
                ("blind_b", X_b, meta_blind_b)
            ]:
                preds_int = clf_final.predict(X_mat)
                preds_str = le.inverse_transform(preds_int)
                probs = clf_final.predict_proba(X_mat)
                
                temp_df = pl.DataFrame({
                    "session_id": result_df["session_id"], 
                    f"pred_{pipeline_str}": preds_str,
                    f"prob_{pipeline_str}": probs.tolist() 
                })
                
                if split_name == "blind_a": meta_blind_a = meta_blind_a.join(temp_df, on="session_id")
                elif split_name == "blind_b": meta_blind_b = meta_blind_b.join(temp_df, on="session_id")

        # 6. Save Meta-Feature Matrices
        out_test = FUSION_DIR / f"meta_features_test_{target}.parquet"
        out_a = FUSION_DIR / f"meta_features_blinda_{target}.parquet"
        out_b = FUSION_DIR / f"meta_features_blindb_{target}.parquet"
        
        meta_test.write_parquet(out_test)
        meta_blind_a.write_parquet(out_a)
        meta_blind_b.write_parquet(out_b)
        
        print(f"\n✅ Saved Fusion Matrices for {target.upper()} to {FUSION_DIR.name}/")

if __name__ == "__main__":
    main()