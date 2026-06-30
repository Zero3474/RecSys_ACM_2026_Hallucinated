"""Swap one turn's predictions from a source submission into a base submission.

Use case: a base submission (good for turns 2..8) gets its turn-1 records'
`predicted_track_ids` replaced by those from a oneshot model tuned for turn 1.

Both files are JSON lists of records:
    {session_id, user_id, turn_number, predicted_track_ids, predicted_response}

Records are matched on (session_id, turn_number). Only base records whose
turn_number == --turn are touched; their `predicted_track_ids` (and optionally
`predicted_response`) are overwritten from the source. Base record order and
every other turn are preserved verbatim.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.swap_turn1_submission \
        --base   models/prediction.json \
        --source models/CG_crossvalidation/tower_cf_ensemble_session_dro/submission/blind_A_tower_cf_ensemble_session_dro.json \
        --out    models/prediction_t1swap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from launchers_crossvalidation._cv_utils import repo_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="Submission to patch.")
    p.add_argument("--source", required=True,
                   help="Submission supplying the replacement turn predictions.")
    p.add_argument("--out", required=True, help="Output path for the patched submission.")
    p.add_argument("--turn", type=int, default=1, help="turn_number to replace (default 1).")
    p.add_argument("--replace-response", action="store_true",
                   help="Also overwrite predicted_response (default: keep base's).")
    return p.parse_args()


def _load(path: str) -> list[dict]:
    data = json.loads(repo_path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of records, got {type(data).__name__}")
    return data


def main() -> None:
    args = parse_args()
    base = _load(args.base)
    source = _load(args.source)

    # Index source by (session_id, turn_number) for the target turn only.
    src_by_key = {
        (r["session_id"], r["turn_number"]): r
        for r in source if r["turn_number"] == args.turn
    }

    replaced = 0
    missing: list[str] = []
    for rec in base:
        if rec["turn_number"] != args.turn:
            continue
        key = (rec["session_id"], rec["turn_number"])
        src = src_by_key.get(key)
        if src is None:
            missing.append(rec["session_id"])
            continue
        rec["predicted_track_ids"] = src["predicted_track_ids"]
        if args.replace_response:
            rec["predicted_response"] = src.get("predicted_response", rec.get("predicted_response"))
        replaced += 1

    n_target = sum(1 for r in base if r["turn_number"] == args.turn)
    print(f"[swap] base records:        {len(base)}")
    print(f"[swap] turn {args.turn} records in base: {n_target}")
    print(f"[swap] replaced:            {replaced}")
    if missing:
        print(f"[swap] WARNING: {len(missing)} base turn-{args.turn} sessions "
              f"absent from source (left unchanged), e.g. {missing[:3]}")

    out_path = repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, indent=2))
    print(f"[swap] wrote {out_path}")


if __name__ == "__main__":
    main()
