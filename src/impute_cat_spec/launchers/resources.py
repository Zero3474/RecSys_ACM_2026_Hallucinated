from pathlib import Path

# Root data directory
DATA = Path("../../data")

# TalkPlay Challenge dataset paths
TRAIN = DATA / "talkpl-ai/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TEST = DATA / "talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
BLIND_A = DATA / "talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
BLIND_B = DATA / "talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"

# Imputation data directory
DATA_IMPUTATIONS = DATA / "impute_values"

# Subdirectories for pipeline execution
EMBEDDINGS_DIR = DATA_IMPUTATIONS / "embeddings"

BASELINE_DIR = DATA_IMPUTATIONS / "baseline_predictions"

PLOTS_DIR = DATA_IMPUTATIONS / "plots"

MODELS_DIR = DATA_IMPUTATIONS / "models"

FUSION_DIR = DATA_IMPUTATIONS / "fusion_features"

ENSEMBLE_DIR = DATA_IMPUTATIONS / "ensemble_outputs"

IMPUTATIONS = ENSEMBLE_DIR / "final_blind_b_predictions.json"

OUPUT_DIR = DATA_IMPUTATIONS / "outputs"
