# Regenerate CG datasets for the reranker (inference)

Reproduce the per-CG candidate parquets that the reranker consumes,
from the delivered **Optuna DBs** + **best_params YAMLs**. No re-tuning — each CG
reloads frozen params and re-runs inference across every split.

Output goes to `models/CG_crossvalidation/<cg>/datasets/`

| dataset.yaml split | filename |
|---|---|
| `train`   | `fold_{0..4}_oof_cg_val.parquet` |
| `val`     | `fold_{0..4}_oof_reranker_val.parquet` |
| `holdout` | `holdout_candidates.parquet` |
| `blind_a` | `blind_a_all_turns_candidates.parquet` |
| `blind_b` | `blind_b_candidates.parquet` |

The 8 reranker CGs: `bm25_colisten_oneshot`, `tower_a_oneshot`,
`emb_item_knn_8b_session_dro`, `hybrid_all_qwen_session_dro`,
`tower_ensemble_session_dro`, `tower_cf_ensemble_session_dro`,
`heuristic_v2_hybrid_session_dro`, `heuristic_v3_session_dro`.

## Prerequisites

- `data/splitK/` present (fold + holdout splits).
- Qwen query caches present: `dense_*` (splitK / holdout / blind-A) **and**
  `dense_blindb_all_*` (blind-B) under each Qwen folder. Text/tower CGs glob
  these at `fit()` time; without the blind-B caches no `blind_b_candidates` get
  written.
- GPU node (towers + ensembles retrain on the full dataset).
- Delivered artifacts in place under `models/CG_crossvalidation/<cg>/`:
  - oneshot CGs: `optuna_<cg>.db` (params read from study `<cg>_cvar70`).
  - DRO CGs + rrf: `best_params_<cg>_cvar70.yaml`.

```bash
cd src/basic_candidate_generators
```

The commands below are **ordered** — fusion CGs read their upstreams' on-disk
parquets, so run top to bottom.

## Stage 0 — oneshot components (all splits)

13 components fused by `rrf_oneshot` (incl. `bm25_colisten`, `tower_a`, two of
the 8). Each: first command = folds + holdout + blind-A; `--blind_b_only` = blind-B.

```bash
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_8b
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_8b --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_4b
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_4b --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_0p6b
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_0p6b --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tfidf
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tfidf --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model bm25
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model bm25 --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model bm25f
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model bm25f --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model char_ngram
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model char_ngram --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model entity_match
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model entity_match --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model bm25_colisten
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model bm25_colisten --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_a
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_a --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_b
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_b --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_c
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_c --blind_b_only
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_audioclap
uv run python -u -m launchers_dro_oneshot.export_oneshot_candidates --model tower_audioclap --blind_b_only
```

## Stage 1 — rrf_oneshot fusion (all splits)

Fuses the stage-0 components (incl. blind-B) with frozen RRF weights.

```bash
uv run python -u -m launchers_dro_oneshot.recreate_rrf_datasets \
  --best models/CG_crossvalidation/rrf_oneshot/best_params_rrf_oneshot_cvar70.yaml
```

## Stage 2 — session-DRO base CGs (all splits)

First command = folds + holdout + blind-A; second = blind-B.

```bash
uv run python -u -m launchers_dro.retrain_and_export_dro --model emb_item_knn_8b --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model emb_item_knn_8b
uv run python -u -m launchers_dro.retrain_and_export_dro --model hybrid_all_qwen --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model hybrid_all_qwen
uv run python -u -m launchers_dro.retrain_and_export_dro --model tower_ensemble --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model tower_ensemble
uv run python -u -m launchers_dro.retrain_and_export_dro --model tower_cf_ensemble --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model tower_cf_ensemble
```

## Stage 3 — session-DRO fusion heuristics (all splits, AFTER stages 1+2)

`heuristic_v2_hybrid` ← `hybrid_all_qwen`; `heuristic_v3` ← `tower_cf_ensemble` + `rrf_oneshot`.

```bash
uv run python -u -m launchers_dro.retrain_and_export_dro --model heuristic_v2_hybrid --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model heuristic_v2_hybrid
uv run python -u -m launchers_dro.retrain_and_export_dro --model heuristic_v3 --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model heuristic_v3
```

## Stage 4 — blind-A all-turns assembly (the 8 reranker CGs)

`blind_a_all_turns` = blind-A history turns (from fold OOF parquets) ∪ `blind_candidates`.

```bash
uv run python -u -m launchers_crossvalidation.assemble_blind_a --only \
  bm25_colisten_oneshot tower_a_oneshot \
  emb_item_knn_8b_session_dro hybrid_all_qwen_session_dro \
  tower_ensemble_session_dro tower_cf_ensemble_session_dro \
  heuristic_v2_hybrid_session_dro heuristic_v3_session_dro
```
