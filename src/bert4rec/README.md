# Candidate Generators

Six candidate-generation models (the `feature_bert4rec` family) and the
`retrain_and_export` pipeline, with the per-model hyperparameters under `configs/`.

## Layout

```
src/basic_candidate_generators/
  launchers_crossvalidation/
    retrain_and_export.py     # model retrain + export (checkpoint + submission)
    _cv_utils.py              # shared utilities (data loaders, inference, scoring)
  src/
    BaseRecommender.py
    recommenders/             # the 6 models + the inheritance chain they import
  configs/                    # hyperparameters per (model, objective)
```

## Requirements

- Python via **uv** (`uv run --no-sync`).
- The official challenge track embeddings + track metadata + the cross-validation
  splits under `data/splitK/`, at the paths in each config's `fixed_params`.
- A GPU for training.

## How to run

From inside the package, one model at a time, passing its config via `--config`:

```bash
cd src/basic_candidate_generators
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model <MODEL_KEY> --urm_mode session \
    --config configs/<YAML> \
    --objective <ndcg|recall> --objective_k <20|200> \
    --storage_dir models/retrain --top_k 500
```

### Model → config → objective

| MODEL_KEY | --config | --objective | --objective_k |
|---|---|---|---|
| `query_full_multibehav_alltext_softor` | `configs/query_full_multibehav_alltext_softor_ndcg.yaml` | ndcg | 20 |
| `query_full_multibehav_alltext_softor` | `configs/query_full_multibehav_alltext_softor_recall.yaml` | recall | 200 |
| `split_hidim_xattn_hardneg_query_full_dif` | `configs/split_hidim_xattn_hardneg_query_full_dif_ndcg.yaml` | ndcg | 20 |
| `split_hidim_xattn_hardneg_query_full_nova` | `configs/split_hidim_xattn_hardneg_query_full_nova_ndcg.yaml` | ndcg | 20 |
| `split_hidim_xattn_hardneg_query_full_noiserobust` | `configs/split_hidim_xattn_hardneg_query_full_noiserobust_recall.yaml` | recall | 200 |
| `split_hidim_xattn_hardneg` | `configs/split_hidim_xattn_hardneg_ndcg.yaml` | ndcg | 20 |

`--model` must match the `model:` field inside the config (it sets the output
folder `<MODEL_KEY>_session/`); `class` and `module` are read from the config.

## Output

Under `--storage_dir`/`<MODEL_KEY>_session/`:

- `checkpoints/full.pkl` — the trained model
- `submission/blind_A_<MODEL_KEY>_session.json` — the Blind-A predictions

Add `--skip_datasets --skip_holdout_candidates` to write only these.

## Blind-B

No retraining required: reuse the trained `full.pkl` and run inference only on
the Blind-B set with `--predict_blindB`.

```bash
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model <MODEL_KEY> --urm_mode session --config configs/<YAML> \
    --objective <ndcg|recall> --objective_k <20|200> --storage_dir models/retrain \
    --predict_blindB --blindB_path <blind_b_parquet> --blindB_query_split <split>
```

This writes `submission/blind_B_<MODEL_KEY>_session.json`. For query-aware models,
`--blindB_query_split` must name a query-cache split present under the
checkpoint's `query_emb_dir`.

Each run is deterministic for a given seed (default 42).