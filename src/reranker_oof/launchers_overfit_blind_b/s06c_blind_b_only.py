"""Assemble ONLY the Blind-B dataset chunk and resubmit from saved boosters —
no retrain, no Blind-A, no folds/holdout rebuild.

For when CG Blind-B candidates were (re)exported and you just want a fresh
blind_b scoring off an already-tuned/trained model, without paying for the
full ``s03_assemble_dataset`` (all splits) + ``s06_retrain_submit`` (retrain)
pipeline.

Two steps, same two config files as the full pipeline (RERANKER_README.md):
  1. ``s03_assemble_dataset --splits blind_b`` — builds
     ``models/reranker_oof/datasets/<name>/blind_b/chunk_*.parquet``.
     Note: still fits the per-CG isotonic calibrators from the "train" (fold
     OOF) CG parquets if ``cg_calibration.enabled`` — that's inherent to s03,
     unrelated to blind_b re-export, and those files must already exist.
  2. ``s06b_submit_from_boosters`` scoped to ``kinds=("blind_b",)`` — loads
     ``<out_dir>/boosters/booster_*.json``, scores blind_b, rewrites
     ``submissions/``, ``scored_blind_b*.parquet``, ``metrics_blind_b*.csv``,
     ``candidates/cand_..._blind_b*.parquet`` in place. Blind-A outputs are
     left untouched.

Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s06c_blind_b_only \\
        --dataset_config configs/blind_no_filter/dataset.yaml \\
        --config configs/blind_no_filter/xgb_v5.yaml \\
        --variants v2_blind_last
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import yaml  # noqa: E402

from launchers_overfit_blind_b import s03_assemble_dataset  # noqa: E402
from launchers_overfit_blind_b.s06b_submit_from_boosters import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset_config", type=Path,
                    default=Path("configs/blind_no_filter/dataset.yaml"),
                    help="same config s03_assemble_dataset takes")
    ap.add_argument("--config", type=Path, required=True,
                    help="xgb config (same one s06/s06b take)")
    ap.add_argument("--variants", nargs="+", choices=["v1_blind_all", "v2_blind_last"],
                    default=["v2_blind_last"])
    ap.add_argument("--force", action="store_true",
                    help="reassemble blind_b chunks even if already present")
    args = ap.parse_args()

    with open(args.config) as f:
        xgb_cfg = yaml.safe_load(f)

    # ── step 1: assemble ONLY the blind_b chunk (s03, scoped down) ──────────
    s03_argv = ["s03_assemble_dataset", "--config", str(args.dataset_config),
               "--splits", "blind_b"]
    if args.force:
        s03_argv.append("--force")
    old_argv = sys.argv
    sys.argv = s03_argv
    try:
        s03_assemble_dataset.main()
    finally:
        sys.argv = old_argv

    # ── step 2: score blind_b only from the saved boosters ──────────────────
    return run(xgb_cfg, variants_sel=args.variants, kinds=("blind_b",))


if __name__ == "__main__":
    raise SystemExit(main())
