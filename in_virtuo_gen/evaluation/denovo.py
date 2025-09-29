#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import lightning.pytorch as pl

# external project imports
from in_virtuo_gen.train_utils.metrics import evaluate_smiles
from in_virtuo_gen.models.invirtuofm import InVirtuoFM
from ..preprocess.preprocess_tokenize import custom_decode_sequence

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================= plotting =================

def plot_multi_model_quality_vs_diversity(
    mean_results_list,
    std_results_list,
    pairs_dict,
    model_names,
    colors=None,
    save_path=None,
    logger=None,
):
    """
    Plot mean±std for each (T, r) with InVirtuoGen styling and reference curves.
    Each entry in mean_results_list/std_results_list corresponds to one model name.
    """
    if colors is None:
        colors = {
            "dt 0.1":  "#8483c9",
            "dt 0.01": "#524fd1",
            "dt 0.02": "#002854",
            "dt 0.001": "#002854",
        }
    label_map = {
        "dt 0.1":  "InVirtuoGen (h=0.1)",
        "dt 0.01": "InVirtuoGen (h=0.01)",
        "dt 0.02": "InVirtuoGen (h=0.02)",
        "dt 0.001":"InVirtuoGen (h=0.001)",
    }
    marker_map = {"dt 0.001": "o", "dt 0.01": "x", "dt 0.02": "x", "dt 0.1": "."}

    fig, ax = plt.subplots(figsize=(8, 6))
    # before your loop
    label_offsets = {
        (1, 0): (8, 6),
        (1, 1): (8, 0),
        (1, 2): (-30, -10),
        (2, 4): (4, -6),
        (3, 4): (6, -12),
        (3, 5): (6, -6),
        (3, 6): (6, 4),
    }

    for mean_res, std_res, name in zip(mean_results_list, std_results_list, model_names):
        pairs = sorted(pairs_dict[name], key=lambda pr: mean_res[pr]["diversity"])
        μds = [mean_res[p]["diversity"] for p in pairs]
        μqs = [mean_res[p]["quality"]   for p in pairs]
        σds = [std_res[p]["diversity"]  for p in pairs]
        σqs = [std_res[p]["quality"]    for p in pairs]
        for i, p in enumerate(pairs):
            # if label_map.get(name, name) == "InVirtuoGen (h=0.001)":
            #     annotation_text = f"     ({p[0]}, {p[1]})"
            #     ax.annotate(annotation_text, (μds[i], μqs[i]), textcoords="offset points", xytext=(1, 1), fontsize=9, color=colors.get(name))
            if label_map.get(name, name) == "InVirtuoGen (h=0.001)":
                p_tuple = tuple(p)  # assuming p like (k, n)
                dx, dy = label_offsets.get(p_tuple, (6, 6))
                annotation_text = f"({p[0]}, {p[1]})"
                ax.annotate(
                    annotation_text,
                    xy=(μds[i], μqs[i]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    color=colors.get(name),
                    clip_on=True,
                    zorder=5,
                )
        c = colors.get(name, "#524fd1")
        lbl = label_map.get(name, name)
        ax.errorbar(
            μds, μqs, xerr=σds, yerr=σqs,
            fmt=marker_map.get(name, "o")+"-" if name!="SAFE-GPT" else "o",
            color=c, ecolor=c, elinewidth=1.5, capsize=3,
            markersize=6, linewidth=2, label=lbl
        )

    # references
    Gen_q  = np.array([84.6, 83.8, 75.0, 63.0, 39.7])
    Gen_d  = np.array([0.818, 0.832, 0.858, 0.882, 0.911])
    Safe_q = np.array([54.7])
    Safe_d = np.array([0.879])

    ax.plot(Gen_d, Gen_q, "o--", color="#76b900", markersize=6, linewidth=1.5, label="GenMol")

    ax.plot(Safe_d, Safe_q, "o", color="black", markersize=6, linewidth=1.5, label="SAFE-GPT")

    ax.set_xlabel("Diversity", fontsize=18)
    ax.set_ylabel("Quality", fontsize=18)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlim(0.75, 0.935)
    ax.set_ylim(30, 100)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.legend(fontsize=16, loc="best")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="pdf", dpi=300)
        if logger:
            logger.log_image("quality_vs_diversity", [fig])
    plt.close(fig)


# ================= scanning and aggregation =================

def scan_quality_vs_diversity_one_seed(
    model,
    pairs,
    dt=0.01,
    num_samples=1000,
    temperature_scaling=False,
    model_name="default",
    eta=1.0,
    already_smiles=False,
    data_outpath=None,
):
    """
    Evaluate one model checkpoint for one seed over (T, r) pairs.
    Returns dict {(T, r): {"quality": float, "diversity": float}}.
    """
    results = {}
    device = model.device
    print(f"Sampling for {model_name}, device {device}")
    with torch.autocast(device.type, dtype=torch.float16, enabled = device.type == "cuda"):
        with torch.no_grad():
            for T, r in tqdm(pairs, desc=f"{model_name} sampling"):
                import time
                start_time = time.time()
                samples = model.sample(
                    num_samples=num_samples,
                    dt=dt,
                    noise=r,
                    temperature=T,
                    T_min=0.25,
                    temperature_scaling=temperature_scaling,
                    p=2,
                    top_p=0,
                    meta=False,
                    purity=False,
                    eta=eta,
                    already_smiles=already_smiles,

                )
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
                valid_indices, smiles, metrics = evaluate_smiles(
                    samples, model.tokenizer,
                    return_values=True,
                    exclude_salts=True,
                )

                results[(T, r)] = {"quality": float(metrics["quality"]), "diversity": float(metrics["diversity"])}
                if data_outpath:
                    with open(os.path.join(data_outpath, f"{T}_{r}.smiles"), "w") as f:

                        for i, sm in enumerate(smiles):
                            if i in valid_indices:
                                f.write(sm + "\n")
                    with open(os.path.join(data_outpath, f"{T}_{r}.seqs"), "w") as f:

                        for i, seq in enumerate(samples):
                            if i in valid_indices:
                                seq= custom_decode_sequence(model.tokenizer, seq)
                                f.write(seq + "\n")
                    with open(os.path.join(data_outpath, f"{T}_{r}.invalid"), "w") as f:
                        for i, sm in enumerate(smiles):
                            if i not in valid_indices:
                                f.write(sm + "\n")
    return results


def aggregate_across_seeds(seed_results):
    """
    seed_results: list of dicts, each {(T, r): {"quality": q, "diversity": d}}
    Returns mean_res, std_res keyed by (T, r).
    """
    all_pairs = set().union(*[set(d.keys()) for d in seed_results])
    mean_res, std_res = {}, {}
    for pair in sorted(all_pairs):
        qs, ds = [], []
        for res in seed_results:
            if pair in res:
                qs.append(res[pair]["quality"])
                ds.append(res[pair]["diversity"])
        qs = np.array(qs, dtype=float)
        ds = np.array(ds, dtype=float)
        mean_res[pair] = {"quality": float(qs.mean()), "diversity": float(ds.mean())}
        std_res[pair]  = {"quality": float(qs.std(ddof=1)) if qs.size > 1 else 0.0,
                          "diversity": float(ds.std(ddof=1)) if ds.size > 1 else 0.0}
    return mean_res, std_res


# ================= I/O helpers for plot_only =================

def load_aggregates(json_root, dt):
    """
    Load mean and std for a given dt from json_root/dt_{dt}/results_mean.json and results_std.json
    Fallback to json_root/results_mean.json when single dt was saved previously.
    """
    dt_dir = Path(json_root) / f"dt_{dt}"
    mean_fp = dt_dir / "results_mean.json"
    std_fp  = dt_dir / "results_std.json"
    if not mean_fp.exists():
        mean_fp = Path(json_root) / "results_mean.json"
    if not std_fp.exists():
        std_fp = Path(json_root) / "results_std.json"
    if not mean_fp.exists() or not std_fp.exists():
        raise FileNotFoundError(f"Missing JSONs for dt={dt} in {json_root}")
    mean_raw = json.load(open(mean_fp))
    std_raw  = json.load(open(std_fp))
    # keys were saved as strings like "(T, r)"; eval safely
    def parse_key(k):
        if isinstance(k, tuple):
            return k
        return tuple(json.loads(k) if (k.startswith('[') or k.startswith('{')) else eval(k))
    mean_res = {parse_key(k): v for k, v in mean_raw.items()}
    std_res  = {parse_key(k): v for k, v in std_raw.items()}
    return mean_res, std_res


# ================= main =================

def main():
    parser = argparse.ArgumentParser(description="Scan (T, r) over multiple dt values, aggregate mean±std, plot.")
    parser.add_argument("--checkpoint_paths", type=str, required=False, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dt", nargs="+", type=float, default=[0.01], help="One or more dt values")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--no_temperature_scaling", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--eta", type=float, default=999.0)
    parser.add_argument("--safe", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="plots/denovo")
    parser.add_argument("--results_dir", type=str, default="results/denovo_results")
    parser.add_argument("--tag", type=str, default="dt_scan")
    parser.add_argument("--no_save_json", action="store_true")
    parser.add_argument("--plot_only", action="store_true", help="Read saved JSONs and only plot")
    parser.add_argument("--json_root", type=str, default=None, help="Root folder with saved JSONs ")
    parser.add_argument("--postfix", type=str, default="", help="Postfix for the output directory")
    args = parser.parse_args()
    device = f"cuda:{args.device}" if len(args.device) == 1 else args.device
    device = torch.device(device)
    temperature_scaling = not args.no_temperature_scaling
    pairs = [(1, 0), (1, 1), (1, 2), (2, 4),  (3, 4), (3,5), (3,6)]
    out_pdf = os.path.join(args.plot_dir, f"quality_vs_diversity{args.postfix}_eta_{args.eta}.pdf")
    json_root = args.json_root or os.path.join(args.results_dir, f"json{args.postfix}_{args.eta}")
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    mean_results_list = []
    std_results_list  = []
    pairs_dict        = {}
    model_names       = []

    if args.plot_only:
        for dt in args.dt:
            mean_res, std_res = load_aggregates(json_root, dt)
            key = f"dt {dt}"
            mean_results_list.append(mean_res)
            std_results_list.append(std_res)
            pairs_dict[key] = pairs
            model_names.append(key)
        plot_multi_model_quality_vs_diversity(
            mean_results_list, std_results_list, pairs_dict, model_names, save_path=out_pdf, logger=None
        )
        print(f"Saved plot to {out_pdf}")
        return

    # sampling path
    if not args.checkpoint_paths:
        raise ValueError("checkpoint_paths is required unless --plot_only is set")



    for dt in args.dt:
        seed_results = []
        for seed in range(args.num_seeds):
            pl.seed_everything(seed)
            ckpt = args.checkpoint_paths
            model = InVirtuoFM.load_from_checkpoint(
                checkpoint_path=ckpt, map_location="cpu", gen_batch_size=500,
                tokenizer_path="tokenizer/safe.json" if args.safe else "tokenizer/smiles_new.json"
            )
            model = model.to(device)

            seed_dir = os.path.join(args.results_dir, f"dt_{dt}_eta_{args.eta}{args.postfix}")
            os.makedirs(seed_dir, exist_ok=True)

            res = scan_quality_vs_diversity_one_seed(
                model,
                pairs=pairs,
                dt=dt,
                num_samples=args.num_samples,
                temperature_scaling=temperature_scaling,
                model_name="InVirtuoGen",
                eta=args.eta,
                already_smiles=False,
                data_outpath=seed_dir,
            )
            seed_results.append(res)

        mean_res, std_res = aggregate_across_seeds(seed_results)

        mean_results_list.append(mean_res)
        std_results_list.append(std_res)
        key = f"dt {dt}"
        pairs_dict[key] = pairs
        model_names.append(key)

        if not args.no_save_json:
            dt_dir = Path(json_root) / f"dt_{dt}"
            dt_dir.mkdir(parents=True, exist_ok=True)
            # save per seed and aggregated
            for i, res in enumerate(seed_results):
                with open(dt_dir / f"results_seed_{i}.json", "w") as f:
                    json.dump({str(k): v for k, v in res.items()}, f)
            with open(dt_dir / "results_mean.json", "w") as f:
                json.dump({str(k): v for k, v in mean_res.items()}, f)
            with open(dt_dir / "results_std.json", "w") as f:
                json.dump({str(k): v for k, v in std_res.items()}, f)

    # one PDF with all curves
    Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
    plot_multi_model_quality_vs_diversity(
        mean_results_list, std_results_list, pairs_dict, model_names, save_path=out_pdf, logger=None
    )
    print(f"Saved plot to {out_pdf}")


if __name__ == "__main__":
    main()