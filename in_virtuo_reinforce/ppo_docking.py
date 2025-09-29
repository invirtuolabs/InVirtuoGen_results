import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_root", type=str, default="lead_optimization")
parser.add_argument("--reference_table", type=str, default="references/reference_pmo.csv")
parser.add_argument("--exclude_prescreen", action="store_true", help="Exclude GenMol and f-RAG from the table")
parser.add_argument("--include_std", action="store_true", help="Include standard deviations in the table")
# New arguments for ablation study mode
parser.add_argument("--ablation_mode", action="store_true", help="Run in ablation study mode (compare different runs without references)")
parser.add_argument("--results_paths", nargs='+', type=str, help="List of results directories to compare (for ablation mode)")
parser.add_argument("--model_names", nargs='+', type=str, help="List of model names corresponding to each results path")
args = parser.parse_args()

if args.ablation_mode:
    if not args.results_paths or not args.model_names:
        raise ValueError("For ablation mode, both --results_paths and --model_names must be provided")
    if len(args.results_paths) != len(args.model_names):
        raise ValueError("Number of results_paths must match number of model_names")

def collect_results_from_path(results_root, single_run=False):
    """Collect results from a single results directory"""
    rows = []
    all_runs_data = {}
    for task in sorted(os.listdir(results_root)):
        task_dir = os.path.join(results_root, task)
        if not os.path.isdir(task_dir):
            continue
        csv_path = None
        for file in os.listdir(task_dir):
            if file.endswith(".csv") and file.startswith("results_"):
                csv_path = os.path.join(task_dir, file)
                break
        if not csv_path or not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        all_runs_data[task] = df["auc_top10"].values

        # For ablation mode, use single run (first value) instead of mean
        if single_run:
            mean10 = df["auc_top10"][0]
            std10 = 0  # No std for single run
        else:
            mean10 = df["auc_top10"].mean() if args.include_std else df["auc_top10"][0]
            std10 = df["auc_top10"].std(ddof=0) if args.include_std else 0

        rows.append((task, mean10, std10))

    return rows, all_runs_data

def calculate_sum_stats(all_runs_data, single_run=False):
    """Calculate sum statistics"""
    task_means = {task: np.mean(values) for task, values in all_runs_data.items()}
    n_runs = 1 if single_run else max((len(v) for v in all_runs_data.values()), default=3)

    sum_per_run = []
    for run_idx in range(n_runs):
        run_sum = 0
        for task, values in all_runs_data.items():
            if single_run:
                run_sum += values[0] if len(values) > 0 else task_means[task]
            else:
                run_sum += values[run_idx] if run_idx < len(values) else task_means[task]
        sum_per_run.append(run_sum)
        if single_run or not args.include_std:
            break

    sum_mean = float(np.mean(sum_per_run))
    sum_std = 0.0 if single_run else float(np.std(sum_per_run, ddof=0))

    return sum_mean, sum_std

if args.ablation_mode:
    # Ablation study mode - compare different runs without references
    all_model_data = {}
    all_sum_stats = {}

    # Collect data from all models
    for i, (results_path, model_name) in enumerate(zip(args.results_paths, args.model_names)):
        rows, all_runs_data = collect_results_from_path(results_path, single_run=True)
        summary = pd.DataFrame(rows, columns=["Oracle", f"{model_name}_mean", f"{model_name}_std"])
        summary["Oracle"] = summary["Oracle"].str.replace("_", " ")
        all_model_data[model_name] = summary

        # Calculate sum statistics
        sum_mean, sum_std = calculate_sum_stats(all_runs_data, single_run=True)
        all_sum_stats[model_name] = (sum_mean, sum_std)

    # Merge all model data
    merged = all_model_data[args.model_names[0]][["Oracle"]].copy()
    for model_name in args.model_names:
        merged = pd.merge(merged, all_model_data[model_name], on="Oracle", how="outer")

    # Set up methods and headers for ablation mode
    methods = [f"{name}_mean" for name in args.model_names]
    latex_headers = args.model_names

    # Generate LaTeX table for ablation study
    latex = []
    latex.append(r"\begin{sidewaystable}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Ablation study results on the PMO benchmark. We report the AUC-top10 scores from single runs. Best results are highlighted in bold.}")
    latex.append(r"\label{tab:ablation}")
    latex.append(r"\begin{tabularx}{\linewidth}{l|" + ("Y " * len(methods)) + "}")
    latex.append(r"\toprule")
    latex.append("Oracle & " + " & ".join(latex_headers) + r" \\")
    latex.append(r"\midrule")

    for _, row in merged.iterrows():
        values = []
        for m in methods:
            v = row[m] if pd.notna(row[m]) else 0.0
            values.append(float(v))

        max_val = max(values)
        max_idx = values.index(max_val)

        row_fmt = []
        for i, v in enumerate(values):
            def fmt(x): return f"{x:.3f}"
            # In ablation mode, don't use mathbf, just highlight best with bold
            if i == max_idx:
                row_fmt.append(f"\\textbf{{{fmt(v)}}}")
            else:
                row_fmt.append(f"{fmt(v)}")

        latex.append(r"\small{" + f"{row['Oracle']}" + "} & " + " & ".join(row_fmt) + r" \\")

    latex.append(r"\midrule")

    # Sums for ablation mode
    sums = [all_sum_stats[name][0] for name in args.model_names]
    max_sum = max(sums)
    max_sum_idx = sums.index(max_sum)

    sum_fmt = []
    for i, s in enumerate(sums):
        def fmt(x): return f"{x:.3f}"
        if i == max_sum_idx:
            sum_fmt.append(f"\\textbf{{{fmt(s)}}}")
        else:
            sum_fmt.append(f"{fmt(s)}")

    latex.append(f"\\textbf{{Sum}} & " + " & ".join(sum_fmt) + r" \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabularx}")
    latex.append(r"\end{sidewaystable}")
    output_file = "ablation_study_table.tex"

else:
    # Original mode with references
    rows, all_runs_data = collect_results_from_path(args.results_root)
    summary = pd.DataFrame(rows, columns=["Oracle", "mean10", "std10"])
    summary["Oracle"] = summary["Oracle"].str.replace("_", " ")

    # Sum statistics
    invirtuo_sum_mean, invirtuo_sum_std = calculate_sum_stats(all_runs_data)

    # Reference
    reference = pd.read_csv(args.reference_table)
    reference["Oracle"] = reference["Oracle"].str.replace("_", " ")

    if args.exclude_prescreen:
        ref_methods = ["Genetic GFN", "Mol GA", "REINVENT", "Graph GA"]
    else:
        ref_methods = ["GenMol", "f-RAG"]

    # Create missing std columns as NaN
    for method in ref_methods:
        std_col = f"{method}_std"
        reference[std_col] = np.nan

    # Create missing sum std columns as NaN
    for method in ref_methods:
        sum_std_col = f"{method}_sum_std"
        if sum_std_col not in reference.columns:
            reference[sum_std_col] = np.nan

    merged = pd.merge(summary, reference, on="Oracle", how="left")
    if args.exclude_prescreen:
        methods = ["mean10", "Genetic GFN", "Mol GA", "REINVENT", "Graph GA"]
        std_methods = ["std10", "Genetic GFN_std", "Mol GA_std", "REINVENT_std", "Graph GA_std"]
        sum_std_methods = ["invirtuo_sum_std", "Genetic GFN_sum_std", "Mol GA_sum_std", "REINVENT_sum_std", "Graph GA_sum_std"]
        latex_headers = ["InVirtuoGen (no prescreen)", "Gen. GFN", "Mol GA", "REINVENT", "Graph GA"]
    else:
        methods = ["mean10", "GenMol", "f-RAG"]
        std_methods = ["std10", "GenMol_std", "f-RAG_std"]
        sum_std_methods = ["invirtuo_sum_std", "GenMol_sum_std", "f-RAG_sum_std"]
        latex_headers = ["InVirtuoGen", "GenMol", "f-RAG"]

    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    if args.exclude_prescreen:
        caption = r"\caption{The results of the best performing models on the PMO benchmark, where we quote the AUC-top10 averaged over 3 runs"
        if args.include_std:
            caption += r" with standard deviations"
        caption += r". The best results are highlighted in bold. Values within one standard deviation of the best are also marked in bold. The results for Genetic GFN \citep{kim2024geneticguidedgflownetssampleefficient} and Mol GA \citep{tripp2023geneticalgorithmsstrongbaselines} are taken from the respective papers. The other results are taken from the original PMO benchmark paper by \citep{gao2022sampleefficiencymattersbenchmark}.}"
    else:
        caption = r"\caption{Comparison of models on the PMO benchmark that screen ZINC250k before initialization. We report the AUC-top10 scores, averaged over three runs"
        if args.include_std:
            caption += r" with standard deviations"
        caption += r". Best results and those within one standard deviation of the best are indicated in bold. The scores for $f$-RAG \citep{lee2024moleculegenerationfragmentretrieval} and GenMol \cite{genmol} are taken from the respective publications.}"
    latex.append(caption)
    latex.append(r"\label{tab:no_prescreen} " if not args.exclude_prescreen else r"\label{tab:prescreen}")
    latex.append(r"\begin{tabularx}{\linewidth}{l|>{\columncolor{gray!20}}p{2.2cm} "+("Y " * (len(methods)-1))+"}") if not args.ablation_mode else     latex.append(r"\begin{tabularx}{\linewidth}{l|p{2.2cm} "+("Y " * (len(methods)-1))+"}")
    latex.append(r"\toprule")
    latex.append("Oracle & " + " & ".join(latex_headers) + r" \\")
    latex.append(r"\midrule")

    for _, row in merged.iterrows():
        values, stds = [], []
        for i, m in enumerate(methods):
            v = row[m] if pd.notna(row[m]) else 0.0
            values.append(float(v))
            if args.include_std and i < len(std_methods):
                sc = std_methods[i]
                s = row[sc] if (sc in row and pd.notna(row[sc])) else None
                stds.append(float(s) if s is not None else None)
            else:
                stds.append(None)

        max_val = max(values)
        max_idx = values.index(max_val)
        max_std = stds[max_idx]  # may be None

        row_fmt = []
        for i, (v, s) in enumerate(zip(values, stds)):
            def fmt(x): return f"{x:.3f}"
            max_val=np.round(max_val, 3)
            v=np.round(v, 3)
            is_within = args.include_std and (max_std is not None) and (v >= max_val - max_std) and (i != max_idx)
            if i == max_idx:
                row_fmt.append("$\mathbf{"+ fmt(v) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(s)}"  + ")}" if args.include_std and s is not None else f"$\\mathbf{{{fmt(v)}}}$")
            elif is_within:
                row_fmt.append("$\mathbf{"+ fmt(v) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(s)}"  + ")}" if args.include_std and s is not None else f"$\\mathbf{{{fmt(v)}}}$")
            else:
                row_fmt.append(f"${fmt(v)}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(s)}"  + ")}" if args.include_std and s is not None else f"{fmt(v)}")
        latex.append(r"\small{" + f"{row['Oracle']}" + "} & " + " & ".join(row_fmt) + r" \\")

    latex.append(r"\midrule")

    # Sums
    sums = [invirtuo_sum_mean]
    sum_stds = [invirtuo_sum_std if args.include_std else None]

    for i in range(1, len(methods)):
        m = methods[i]
        sums.append(float(merged[m].sum()))
        if args.include_std:
            ssc = sum_std_methods[i]
            if ssc in reference.columns and not reference[ssc].isna().all():
                sum_stds.append(float(reference[ssc].iloc[0]))
            else:
                sum_stds.append(None)
        else:
            sum_stds.append(None)

    max_sum = max(sums)
    max_sum_idx = sums.index(max_sum)
    max_sum_std = sum_stds[max_sum_idx]

    sum_fmt = []
    for i, (s, sd) in enumerate(zip(sums, sum_stds)):
        def fmt(x): return f"{x:.3f}"
        if i == max_sum_idx:
            sum_fmt.append("$\mathbf{"+ fmt(s) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(sd)}"  + ")}" if args.include_std and sd is not None else f"$\\mathbf{{{fmt(s)}}}$")
        else:
            within = args.include_std and (max_sum_std is not None) and (s >= max_sum - max_sum_std) and (s != max_sum)
            if within:
                sum_fmt.append("$\mathbf{"+ fmt(s) + "}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(sd)}"  + ")}" if args.include_std and sd is not None else f"$\\mathbf{{{fmt(s)}}}$")
            else:
                sum_fmt.append(f"${{{fmt(s)}}}$" + " {\\tiny (" + "$\\pm$ " + f"{fmt(sd)}"  + ")}" if args.include_std and sd is not None else f"{fmt(s)}")

    latex.append(f"\\textbf{{Sum}} & " + " & ".join(sum_fmt) + r" \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabularx}")
    latex.append(r"\end{table}")

    output_file = "pmo_comparison_table"
    if args.include_std:
        output_file += "_with_std"
    if args.exclude_prescreen:
        output_file += "_no_prescreen"
    output_file += ".tex"

print("\n".join(latex))

with open(output_file, "w") as f:
    f.write("\n".join(latex))
print(f"\nTable saved to {output_file}")