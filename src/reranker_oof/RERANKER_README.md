# How to run the reranker pipeline

**Requirements:** 
- having produced the candidate generators output `models/CG_crossvalidation/<CG name>/datasets/`.
- having the Qwen embeddings at `models/retrieval_text_towers`

```bash
cd src/reranker_oof

```

**Assemble the dataset:**
```bash
uv run python -m launchers_overfit_blind_b.s03_assemble_dataset --config configs/blind_no_filter/dataset.yaml
```

To disable the test_tracks filtering just put to `false` the following in `src/reranker_oof/configs/blind_no_filter/dataset.yaml`:
```yaml
test_tracks:
  enabled: false
  path: data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/test_tracks-00000-of-00001.parquet
```



**Retrain reranker and make inferece:**
```bash
uv run python -u -m launchers_overfit_blind_b.s06_retrain_submit --config configs/blind_no_filter/xgb_v5.yaml
```
Note: The optuna database is required at `models/reranker_oof/optuna/blind_b/no_filter_v5`

**Find your outputs:**
You can find your outputs in the `models/reranker_oof/blind_b_retrain/no_filter_v5/v2_blind_last/` folder.


## COMPLETE TUNING PIPELINE

Given the above prerequisites, you can run the complete tuning pipeline as follows:
*(you can point the config files also at your own paths, if you have a different setup)*

```bash
# Tuning of weighted rrf aggregation
uv run python -m launchers_overfit_blind_b.s01_tune_rrf --config configs/blind_no_filter/dataset.yaml

# Export the best rrf weights to a yaml file
uv run python -m launchers_overfit_blind_b.s02_export_best_rrf --config configs/blind_no_filter/dataset.yaml

# Assemble the dataset for reranker training
uv run python -m launchers_overfit_blind_b.s03_assemble_dataset --config configs/blind_no_filter/dataset.yaml

# Tune the reranker hyperparameters with optuna
uv run python -u -m launchers_overfit_blind_b.s05_tune_reranker --config configs/blind_no_filter/xgb_v5.yaml

# Retrain the reranker with the best hyperparameters and make inference on blind-B
uv run python -u -m launchers_overfit_blind_b.s06_retrain_submit --config configs/blind_no_filter/xgb_v5.yaml
```
