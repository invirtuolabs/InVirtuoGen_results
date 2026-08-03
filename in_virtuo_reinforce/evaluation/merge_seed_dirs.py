"""Merge per-seed eval dirs into one dir so results_table.py --include_std can average them.

Runs submitted as parallel per-seed identifiers land in separate directories (one row each).
This concatenates the matching results_<oracle>.csv across seed dirs into <out>, giving one
csv per oracle with one row per seed - exactly what results_table.py expects for std.

Usage:
  python -m in_virtuo_reinforce.evaluation.merge_seed_dirs \
      --seed_dirs DIR_SEED0 DIR_SEED1 DIR_SEED2 --out MERGED_DIR
"""
import argparse
import glob
import os
import pandas as pd

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", required=True, help="Per-seed eval dirs to merge")
    p.add_argument("--out", required=True, help="Output (merged) dir to write")
    args = p.parse_args()

    oracles = set()
    for d in args.seed_dirs:
        for f in glob.glob(os.path.join(d, "*", "results_*.csv")):
            oracles.add(os.path.basename(os.path.dirname(f)))

    os.makedirs(args.out, exist_ok=True)
    merged = 0
    for o in sorted(oracles):
        frames = []
        for d in args.seed_dirs:
            f = os.path.join(d, o, f"results_{o}.csv")
            if os.path.exists(f):
                frames.append(pd.read_csv(f))
        if not frames:
            continue
        os.makedirs(os.path.join(args.out, o), exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(args.out, o, f"results_{o}.csv"), index=False
        )
        merged += 1
    print(f"merged {merged} oracles from {len(args.seed_dirs)} seed dirs -> {args.out}")
