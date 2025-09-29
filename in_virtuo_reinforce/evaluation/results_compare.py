import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results_root", nargs="+", default=["results_root"])
parser.add_argument("--out_dir", type=str, default="target_property")
parser.add_argument("--reference_csv", type=str, help="CSV with baseline scores per task")
parser.add_argument("--reference_col", type=str, default="GenMol", help="Baseline column to use")
args = parser.parse_args()

# collect auc_top10 per task per run
run_names = [os.path.basename(os.path.normpath(r)) for r in args.results_root]
task_scores = {}

for run, run_name in zip(args.results_root, run_names):
    for task in sorted(os.listdir(run)):
        task_dir = os.path.join(run, task)
        if not os.path.isdir(task_dir):
            continue
        totals = []
        for file in os.listdir(task_dir):
            if file.startswith("results_") and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(task_dir, file))
                if "auc_top10" in df.columns:
                    totals.append(df["auc_top10"].mean())
        if totals:
            task_scores.setdefault(task, {})[run_name] = float(np.mean(totals))

task_df = pd.DataFrame.from_dict(task_scores, orient="index")
task_df = task_df.reindex(columns=run_names)
task_df = task_df.sort_index()

# add baseline from reference CSV if provided
if args.reference_csv and os.path.isfile(args.reference_csv):
    ref = pd.read_csv(args.reference_csv).rename(columns={"Oracle": "task"})
    # normalize to have a 'task' column and the requested reference column
        # assume first column holds task names when header is malformed
    ref = ref[["task", args.reference_col]].dropna()
    ref["task"] = ref["task"].astype(str).str.strip()
    ref = ref.set_index("task")[args.reference_col].astype(float)
    # align to tasks present in task_df
    ref.index = task_df.index
    task_df[args.reference_col] = ref.values
else:
    if args.reference_csv:
        raise FileNotFoundError(f"Reference CSV not found: {args.reference_csv}")

# per run cumulative sum down the task index
cum_part = task_df[run_names].fillna(0).cumsum()
cum_part.columns = [f"{c}_cum" for c in cum_part.columns]

# baseline cumulative if present
if args.reference_col in task_df.columns:
    task_df[f"{args.reference_col}_cum"] = task_df[args.reference_col].fillna(0).cumsum()

out_df = pd.concat([task_df, cum_part], axis=1)

os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, "auc_top10_by_task_with_cumsum.csv")
print(out_df)
out_df.to_csv(out_path)
print("Wrote:", out_path)