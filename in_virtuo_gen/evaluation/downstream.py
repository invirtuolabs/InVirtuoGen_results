import argparse
import ast
import os
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from ..models.invirtuofm import InVirtuoFM
from ..preprocess.generate_fragments import process_smiles
from ..train_utils.metrics import evaluate_smiles
from ..utils.fragments import bridge_smiles_fragments, remove_stereochemistry
from .plot_results import visualize_generated_molecules
import datamol as dm
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from typing import Optional
import random
from rdkit import Chem

import re

def enumerate_attach_points(smi: str) -> str:
    counter = {"i": 0}
    def repl(_):
        counter["i"] += 1
        return f"[{counter['i']}*]"
    return re.sub(r"\[\*\]", repl, smi)


torch.set_float32_matmul_precision("high")
def list_individual_attach_points(mol: dm.Mol, depth: Optional[int] = None):
    """List all individual attachement points.

    We do not allow multiple attachment points per substitution position.

    Args:
        mol: molecule for which we need to open the attachment points

    """
    ATTACHING_RXN = ReactionFromSmarts("[*;h;!$([*][#0]):1]>>[*:1][*]")
    mols = [mol]
    curated_prods = set()
    num_attachs = len(mol.GetSubstructMatches(dm.from_smarts("[*;h:1]"), uniquify=True))
    depth = depth or 1
    depth = min(max(depth, 1), num_attachs)
    while depth > 0:
        prods = set()
        for mol in mols:
            mol = dm.to_mol(mol)
            for p in ATTACHING_RXN.RunReactants((mol,)):
                    m = dm.sanitize_mol(p[0])
                    sm = dm.to_smiles(m, canonical=True)
                    sm = dm.reactions.add_brackets_to_attachment_points(sm)
                    prods.add(dm.reactions.convert_attach_to_isotope(sm, as_smiles=True))

        curated_prods.update(prods)
        mols = prods
        depth -= 1
    return list(curated_prods)

def write_latex_tables(
    results: dict,
    outpath: str = "comparison_tables.tex",
    include_std: bool = True,
    invirtuogen_seed_avg_lists: dict | None = None,
):
    """
    Generates a LaTeX table comparing models across tasks.
    Includes average performance over all tasks.
    Std shows variation across seeds. Average uses error propagation for baselines,
    and std of per-seed task-averages for InVirtuoGen.
    Bold marks any model within 1 std of the best model.
    """
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    with open(outpath, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("  \\centering\n")


        # Determine metrics to include
        all_metrics = set()
        for task_data in results.values():
            for method_data in task_data.values():
                all_metrics.update(method_data["metrics"].keys())
        metrics = sorted(m for m in all_metrics if m not in ["fcd", "qed", "sa", 'distance'])

        # Table setup
        f.write("  \\caption{Performance across five fragment‐constrained generation tasks, averaged over three random seeds: motif extension, linker design, superstructure generation, scaffold morphing, and scaffold decoration. In our setup, similar as for GenMol, scaffold decoration is identical to linker design, so results are shared. }\n")
        f.write("  \\small\n")
        f.write("  \\setlength{\\tabcolsep}{4pt} % reduce column padding\n")
        f.write("  \\renewcommand{\\arraystretch}{1.2} % row height\n")
        f.write("  \\begin{tabularx}{\\linewidth}{l l *{4}{>{\\centering\\arraybackslash}X}}\n")
        f.write("    \\toprule\n")
        header = ["Task", "Method"] + [m.replace("_", "\\_").replace("div", "Div").replace("quality", "Quality").replace("validity", "Validity").replace("uniqueness", "Uniqueness") for m in metrics]
        f.write("    " + " & ".join(header) + " \\\\\n")
        f.write("    \\midrule\n")

        # Prepare for computing averages with error propagation
        all_methods = set()
        method_metric_values = defaultdict(lambda: defaultdict(list))
        method_metric_stds = defaultdict(lambda: defaultdict(list))
        display_map = {
            "linker": "Linker Design",
            "morphing": "Scaffold Morphing",
            "motif": "Motif Extension",
            "decoration": "Scaffold Decoration",
            "superstructure": "Superstructure Design",
        }
        # Render each task
        for task, methods in results.items():
            task_display = display_map.get(task, task)

            method_names = [m for m in methods if m != "Best Previous Model"]
            methods = {m.replace("InVirtuoFM", "InVirtuoGen"): v for m, v in methods.items()}
            method_names = [m.replace("InVirtuoFM", "InVirtuoGen") for m in method_names]
            all_methods.update(method_names)


            # Find best value and its std for each metric
            best_per_metric = {}
            best_std_per_metric = {}
            for metric in metrics:
                best_val = float("-inf")
                best_std = 0
                for m in method_names:
                    val = methods[m]["metrics"].get(metric, float("-inf"))
                    if val > best_val:
                        best_val = val
                        best_std = methods[m].get("std", {}).get(metric, 0)
                best_per_metric[metric] = best_val
                best_std_per_metric[metric] = best_std

            for i, method in enumerate(method_names):
                row = []
                if i == 0:
                    row.append(f"\\multirow[c]{{{len(method_names)}}}{{*}}{{{task_display}}}")
                else:
                    row.append("")


                row.append(method)
                if method == "InVirtuoGen":
                    row[1]="\\rowcolor{gray!20}"+row[1]
                for metric in metrics:
                    val = methods[method]["metrics"].get(metric, float("nan"))
                    std_val = methods[method].get("std", {}).get(metric, 0)

                    # Track for weighted averages with error propagation
                    if not np.isnan(val):
                        method_metric_values[method][metric].append(val)
                        method_metric_stds[method][metric].append(std_val)

                    # Format with or without std
                    if include_std and std_val > 0:
                        fmt = f"${val:.2f} \\pm {std_val:.3f}$"
                    else:
                        fmt = f"${val:.2f}$"

                    # Bold if this model's mean is within the best model's std
                    best_val = best_per_metric[metric]
                    best_std = best_std_per_metric[metric]
                    # Bold if: (1) this is the best, or (2) this value is within [best - best_std, best + best_std]
                    if val == best_val or (best_std > 0 and abs(val - best_val) <= best_std):
                        fmt = f"$\\mathbf{{{fmt[1:-1]}}}$"

                    row.append(fmt)

                f.write("    " + " & ".join(row) + " \\\\\n")
            f.write("    \\midrule\n")

        method_names_sorted = sorted(all_methods)
        avg_values = {}
        avg_stds = {}

        # Methods that should use error propagation across tasks
        PROPAGATE_METHODS = {"SAFE-GPT", "GenMol"}

        for method in method_names_sorted:
            avg_values[method] = {}
            avg_stds[method] = {}

            for metric in metrics:
                vals = method_metric_values[method][metric]      # per-task means
                stds = method_metric_stds[method][metric]        # per-task stds

                # drop NaNs consistently
                cleaned = [(v, s) for v, s in zip(vals, stds) if v is not None and not np.isnan(v)]
                if not cleaned:
                    avg_values[method][metric] = float("nan")
                    avg_stds[method][metric] = float("nan")
                    continue

                vals_clean = np.array([v for v, _ in cleaned], dtype=float)
                stds_clean = np.array([s if s is not None else 0.0 for _, s in cleaned], dtype=float)
                n = len(vals_clean)


                mean_across_tasks = float(np.mean(vals_clean))
                avg_values[method][metric] = mean_across_tasks

                if method in PROPAGATE_METHODS:
                    # error propagation for baselines
                    propagated = float(np.sqrt(np.sum(stds_clean**2)) / n) if n > 0 else float("nan")
                    avg_stds[method][metric] = propagated
                elif method == "InVirtuoGen" and invirtuogen_seed_avg_lists and metric in invirtuogen_seed_avg_lists:
                    # For InVirtuoGen: std of the per-seed task averages
                    seed_avgs = np.array(invirtuogen_seed_avg_lists[metric], dtype=float)
                    avg_stds[method][metric] = float(np.std(seed_avgs, ddof=1)) if seed_avgs.size > 1 else 0.0
                else:
                    avg_stds[method][metric] = float("nan")

                # Find best average and its propagated std per metric
        best_avg_metric = {}
        best_avg_std = {}
        for metric in metrics:
            best_val = float("-inf")
            best_std = 0
            for m in method_names_sorted:
                val = avg_values[m][metric]
                if not np.isnan(val) and val > best_val:
                    best_val = val
                    best_std = avg_stds[m][metric]
            best_avg_metric[metric] = best_val
            best_avg_std[metric] = best_std

        # Write average rows
        for i, method in enumerate([ "SAFE-GPT", "GenMol", "InVirtuoGen"]):
            row = []
            if i == 0:
                row.append(f"\\multirow[c]{{{len(method_names_sorted)}}}{{*}}{{\\textbf{{Average}}}}")
            else:
                row.append("")
            row.append(f"{{{method}}}")
            if method == "InVirtuoGen":
                row[1]="\\rowcolor{gray!20}"+row[1]

            for metric in metrics:
                val = avg_values[method][metric]
                std_val = avg_stds[method][metric]

                # Format with or without std
                if include_std and std_val > 0:
                    fmt = f"${val:.2f} \\pm {std_val:.3f}$"
                else:
                    fmt = f"${val:.2f}$"

                # Bold if within the best model's propagated std
                if metric in best_avg_metric:
                    best_val = best_avg_metric[metric]
                    best_std = best_avg_std[metric]
                    if val == best_val or (best_std > 0 and abs(val - best_val) <= best_std):
                        fmt = f"$\\mathbf{{{fmt[1:-1]}}}$"

                row.append(fmt)

            f.write("    " + " & ".join(row) + " \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabularx}\n")
        f.write("  \\label{tab:task_model_comparison}\n")
        f.write("\\end{table}\n")

    print(f"Wrote LaTeX table to {outpath}")


def process_fragments(fragments):
    """Process fragments from string or list format."""
    if isinstance(fragments, str):
        try:
            frag_list = ast.literal_eval(fragments)
        except (ValueError, SyntaxError):
            frag_list = fragments.split()
    else:
        frag_list = fragments

    if isinstance(frag_list, list):
        return [remove_stereochemistry(frag) for frag in frag_list]
    return frag_list


def load_reference_metrics(reference_metrics_path="references/reference_metrics.csv"):
    """Load reference metrics (means and stds) from CSV into nested dict.

    Expected columns (examples):
      category,method,validity,validity_std,uniqueness,uniqueness_std,quality,quality_std,diversity,diversity_std,distance,distance_std
    Rows with missing stds are handled (std omitted).
    """
    if not os.path.exists(reference_metrics_path):
        print(f"Warning: Reference metrics file not found at {reference_metrics_path}")
        return {}

    df = pd.read_csv(reference_metrics_path)

    # Normalize column names to lower-case for robustness
    df.columns = [c.strip() for c in df.columns]

    results_nested = {}
    for _, row in df.iterrows():
        category = row["category"]
        method = row["method"]

        metrics = {}
        stds = {}
        for col in row.index:
            if col in ("category", "method"):
                continue
            val = row[col]
            if pd.isna(val):
                continue

            if col.endswith("_std"):
                base = col[:-4]
                # store std only if a numeric value
                try:
                    stds[base] = float(val)
                except Exception:
                    pass
            else:
                # mean value
                try:
                    metrics[col] = float(val)
                except Exception:
                    pass

        # attach
        results_nested.setdefault(category, {})[method] = {"metrics": metrics}
        if stds:
            results_nested[category][method]["std"] = stds

    return results_nested


class MolecularDesign:
    """Optimized molecular design class with better batching."""

    def __init__(self, model, tokenizer, max_length=150):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")

    def prompt_model(self, prompt_ids: list, params: dict, n=None, device="cuda", num_samples=100, superstructure=False):
        """Generate samples with optimized batching."""
        ids = []
        self.model = self.model.to(device)

        # Parse device type for autocast
        device_type = "cuda" if "cuda" in str(device) else "cpu"

        with torch.autocast(device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            # Process in larger batches for efficiency
            batch_size = params.get('gen_batch_size', 100)
            if not superstructure:
                for i, p in enumerate(prompt_ids):
                    # Generate all samples for this prompt at once if possible
                    remaining_samples = num_samples
                    prompt_samples = []

                    while remaining_samples > 0:
                        current_batch = min(batch_size, remaining_samples)
                        samples = self.model.sample(
                            prompt=p.to(self.device),
                            num_samples=current_batch,
                            n=n[i] if n is not None else None,
                            force_prompt=True,
                            **params
                        )
                        prompt_samples.extend(samples)
                        remaining_samples -= current_batch

                    ids.extend(prompt_samples[:num_samples])  # Ensure exactly num_samples
            else:
                for i in range(len(prompt_ids)//batch_size+1):
                    if len(prompt_ids[i*batch_size:(i+1)*batch_size])==0:
                        break
                    ids.extend(self.model.sample(
                        prompt=torch.nn.utils.rnn.pad_sequence(  prompt_ids[i*batch_size:(i+1)*batch_size], batch_first=True, padding_value=self.model.pad_token_id).to(self.device),
                        num_samples=batch_size,
                        n=n[i] if n is not None else None,
                        force_prompt=True,
                        **params
                    ))

        return ids

    def tokenize_and_flatten(self, examples, tokenizer, max_length):
        """Tokenize with proper tensor handling."""
        flattened_tokens = []
        for frag in examples:
            token_ids = tokenizer.encode(frag)
            token_ids = torch.tensor(token_ids, dtype=torch.long)[:max_length]
            flattened_tokens.append(token_ids)
        return flattened_tokens


def merge_seed_results(seed_results_list, tasks):
    """Merge results from multiple seeds by averaging metrics."""
    merged = {t: {} for t in tasks}

    for task in tasks:
        methods = set()
        for sr in seed_results_list:
            methods.update(sr.get(task, {}).keys())

        for method in methods:
            per_seed_metrics = []
            first_smiles = []

            for idx, sr in enumerate(seed_results_list):
                md = sr.get(task, {}).get(method)
                if md and "metrics" in md:
                    per_seed_metrics.append(md["metrics"])
                    if idx == 0:
                        first_smiles = md.get("smiles", [])

            # Average metrics across seeds
            all_keys = set().union(*(d.keys() for d in per_seed_metrics)) if per_seed_metrics else set()
            mean_metrics = {}
            std_metrics = {}

            for k in all_keys:
                vals = [float(d[k]) for d in per_seed_metrics if k in d and d[k] is not None]
                if vals:
                    mean_metrics[k] = float(np.mean(vals))
                    std_metrics[k] = float(np.std(vals))
                else:
                    mean_metrics[k] = float("nan")
                    std_metrics[k] = float("nan")

            merged[task][method] = {
                "metrics": mean_metrics,
                "std": std_metrics,
                "smiles": first_smiles
            }

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, default="invirtuo_gen.ckpt", required=True)
    parser.add_argument("--motif", action="store_true")
    parser.add_argument("--linker", action="store_true")
    parser.add_argument("--superstructure", action="store_true")
    parser.add_argument("--decoration", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Samples per prompt ")
    parser.add_argument("--no_exclude_salts", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--temperature_scaling", action="store_true")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--oracle_length", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--noise", default=0, type=float)
    parser.add_argument("--unmasking_noise", default=4.5, type=float)
    parser.add_argument("--start_t", default=0.0, type=float)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Device setup
    device = f"cuda:{args.device}" if len(args.device)==1 else args.device
    device = torch.device(device)

    # Adjust settings for fast mode
    args.num_samples_eval = 100
    gen_batch_size = 500

    os.makedirs("plots/downstream/", exist_ok=True)
    samples_per_prompt = args.num_samples_eval
    exclude_salts = not args.no_exclude_salts

    # Load data
    data_file = "references/frags_downstream.csv"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    df.columns = ["original_smiles", "motif", "linker", "decoration", "superstructure"]

    # Task selection
# default list includes morphing
    tasks = ["motif", "linker", "superstructure", "decoration"]
    if not args.all:
        tasks = [t for t in tasks if getattr(args, t)]

    # Preprocess fragments
    for task in tasks:
        df[task] = df[task].apply(process_fragments)
        if task == "superstructure":
            superstructure_frags = []
            temp_df = pd.read_csv("references/fragments.csv")
            for i in range(10):
                frags= list_individual_attach_points(Chem.MolFromSmiles(remove_stereochemistry(temp_df["superstructure_generation"].values[i])), depth=2)
                for _ in range(100):
                    superstructure_frags.append(enumerate_attach_points(random.choice(frags)))

    orig_frags = df["original_smiles"].apply(process_smiles).values.tolist()

    # Optimized sampling parameters
    params = {
        "noise": args.noise,
        "temperature": args.temperature,
        "dt": args.dt,
        "gen_batch_size": gen_batch_size,  # Larger batch for efficiency
        "batch_size": 500,  # Processing batch size
        "eta": args.eta,
        "temperature_scaling": args.temperature_scaling,
        "min_length": True,
        "start_t": args.start_t,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    print("Loading model...")
    model = InVirtuoFM.load_from_checkpoint(
        args.model_paths,
        tokenizer_path="tokenizer/smiles_new.json",
        map_location="cpu"
    ).to(device)
    model.eval()


    design = MolecularDesign(model, model.tokenizer)
    valid_indices_dict = {}
    frags_dict = {}
    smiles_dict = {}
    # Pre-tokenize oracle lengths if needed (do once)
    frags_n = None
    if args.oracle_length:
        frags_n = design.tokenize_and_flatten([x[0] for x in orig_frags], model.tokenizer, 150)
        frags_n = [len(i) for i in frags_n]

    # Pre-tokenize all task prompts (do once)
    task_prompts = {}
    for task in tasks:
        if task == "superstructure":
            frags_list = superstructure_frags
            task_prompts[task] = design.tokenize_and_flatten(frags_list, model.tokenizer, 150)
            frags_dict[task] = temp_df["superstructure_generation"].values.tolist()
        else:
            frags_list = df[task].apply(lambda x: " ".join(x) + " ").values.tolist()
            frags_dict[task] = frags_list
            task_prompts[task] = design.tokenize_and_flatten(frags_list, model.tokenizer, 150)

    # Run seed experiments
    results_per_seed = []
    seed_task_metric_lists = [defaultdict(list) for _ in range(args.num_seeds)]
    for seed in range(args.num_seeds):
        print(f"\nSeed {seed}/{args.num_seeds}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        res_seed = {task: {} for task in tasks}

        for task in tasks:
            valid_indices = []

            print(f"  Processing {task}...")

            # Generate samples (using pre-tokenized prompts)
            ids = design.prompt_model(
                prompt_ids=task_prompts[task],
                params=params.copy(),
                n=frags_n,
                device=device,
                num_samples=samples_per_prompt,
                superstructure=task == "superstructure"
            )

            # Evaluate in parallel where possible
            metrics_list = []
            smiles_list = []
            for i in range(len(task_prompts[task])):

                start_idx = i * samples_per_prompt
                end_idx = (i + 1) * samples_per_prompt
                if task == "superstructure":
                    if start_idx>=len(ids):
                        break
                indices, smile, metric = evaluate_smiles(
                    ids[start_idx:end_idx],
                    model.tokenizer,
                    return_values=True,
                    exclude_salts=exclude_salts,
                )


                metrics_list.append(metric)
                smiles_list.append(smile)
                if seed == 0:
                    valid_indices.append(indices)
            results_per_seed.append(res_seed)
            # Aggregate metrics
            mdf = pd.DataFrame(metrics_list)
            for drop_col in ("fcd", "sa", "qed"):
                if drop_col in mdf.columns:
                    mdf = mdf.drop(columns=[drop_col])
            avg_metrics = mdf.mean(numeric_only=True).to_dict()
            for k, v in avg_metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    seed_task_metric_lists[seed][k].append(float(v))
            if task == "linker" and "morphing" in tasks:
                for k, v in avg_metrics.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        seed_task_metric_lists[seed][k].append(float(v))
            res_seed[task]["InVirtuoGen"] = {"metrics": avg_metrics}
            if seed == 0:

                valid_indices_dict[task] = valid_indices
                smiles_dict[task] = smiles_list
                for smiles,inds in zip(smiles_list, valid_indices):
                    for i in inds:
                        assert smiles[i] is not None and smiles[i] != "INVALID"

    # Merge results across seeds
    merged = merge_seed_results(results_per_seed, tasks)

    # Format final results
    merged_results = {
        t: {
            "InVirtuoGen": {
                "metrics": merged[t]["InVirtuoGen"]["metrics"],
                "std": merged[t]["InVirtuoGen"].get("std", {}),
                "smiles": merged[t]["InVirtuoGen"].get("smiles", [])
            }
        }
        for t in tasks
    }
    if "linker" in merged_results and "InVirtuoGen" in merged_results["linker"]:
        ivg_linker = merged_results["linker"]["InVirtuoGen"]
        merged_results["morphing"] = {
            "InVirtuoGen": {
                "metrics": dict(ivg_linker["metrics"]),
                "std": dict(ivg_linker.get("std", {})),
                "smiles": ivg_linker.get("smiles", []),
            }
        }
    if args.all:
        tasks = ["motif", "linker", "morphing", "superstructure", "decoration"]
    # Load and merge with reference metrics
    results_old = load_reference_metrics()
    results_avg = {}

    for task in tasks:
        out = dict(results_old.get(task, {}))
        ivg_block = merged_results.get(task, {}).get("InVirtuoGen")
        if ivg_block is not None:
            out["InVirtuoGen"] = ivg_block
        results_avg[task] = out

    # Round metrics for display
    for task, methods in results_avg.items():
        for m in methods:
            for metric, v in list(methods[m]["metrics"].items()):
                if isinstance(v, (int, float)) and np.isfinite(v):
                    methods[m]["metrics"][metric] = round(float(v), 2)

    # Save results
    os.makedirs("results/fragment_constrained", exist_ok=True)

    # Save JSON with full details including std
    postfix = "_temperature_scaling" if args.temperature_scaling else ""
    postfix += "_big" if args.model_paths.find("big") != -1 else ""
    postfix += f"_dt_{args.dt}"
    postfix+= f"_eta_{args.eta}"
    save_path = f"results/fragment_constrained/results_with_std{postfix}.json"
    with open(save_path, "w") as f:
        json.dump(merged_results, f, indent=4)

    invirtuogen_seed_avg_lists = {}
    all_metrics_seen = set().union(*[set(d.keys()) for d in seed_task_metric_lists]) if seed_task_metric_lists else set()

    for metric in all_metrics_seen:
        per_seed_means = []
        for d in seed_task_metric_lists:
            vals = [x for x in d.get(metric, []) if np.isfinite(x)]
            if vals:
                per_seed_means.append(float(np.mean(vals)))
        if per_seed_means:
            invirtuogen_seed_avg_lists[metric] = per_seed_means

    write_latex_tables(results_avg, outpath=f"results/fragment_constrained/comparison_tables{postfix}.tex",
                    include_std=True, invirtuogen_seed_avg_lists=invirtuogen_seed_avg_lists)
    write_latex_tables(results_avg, outpath=f"results/fragment_constrained/comparison_tables_no_std{postfix}.tex",
                    include_std=False, invirtuogen_seed_avg_lists=invirtuogen_seed_avg_lists)

    print(f"\nCompleted evaluation with {args.num_seeds} seeds")
    print(f"Results saved to results/fragment_constrained/")
    for task in tasks:
        if task == "morphing":
            continue
        visualize_generated_molecules(
        frags=frags_dict[task],
        generated_smiles_dict={"InVirtuoGen": smiles_dict[task]},
        num_samples_per_frag=5,  # e.g. if you generated 100 samples per motif
        num_random_frags=min(1, len(frags_dict[task])) if task != "denovo" else 3,
        num_generated=5,
        title=f"{task.replace('superstructure', 'Superstructure Design').replace('decoration', 'Scaffold Decoration').replace('linker', 'Linker Design').replace('motif', 'Motif Extension')}", # Samples
        outpath=f"plots/downstream/{task}_samples.pdf",
        format="pdf",
        valid_indices=valid_indices_dict[task],
            )
