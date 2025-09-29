import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw


def create_overall_radar_plot(results, output_dir, colors, outreach=False):
    """
    Create radar plots showing the performance of each model across all tasks,
    and also create an overall plot that averages the metrics across tasks.

    Args:
        results (dict): Nested dictionary containing metrics for each task and model.
                        Structure is assumed to be:
                        {task: {model_name: {'metrics': {metric_name: value, ...}}, ...}, ...}
        output_dir (str): Directory to save the plots.
        colors (dict): Colors assigned to each model for plotting.
    """
    metrics_order = ["validity", "uniqueness", "quality", "diversity", "distance"]

    os.makedirs(output_dir, exist_ok=True)
    # --- Create per-task radar plots ---
    for task, models in results.items():

        N = len(metrics_order)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2) # type: ignore[attr-defined]
        ax.set_theta_direction(-1) # type: ignore[attr-defined]

        plt.xticks(angles[:-1], [metric.capitalize() for metric in metrics_order], fontsize=12)
        ax.set_rlabel_position(0) # type: ignore[attr-defined]
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
        plt.ylim(0, 100)

        model_list = list(models.keys())
        # Ensure our models are plotted last
        for model in model_list:
            if model not in ["GenMol", "SAFE-GPT", "GenMol (NVIDIA)"]:
                model_list.remove(model)
                model_list.append(model)
        for model_name in model_list:
            data = models[model_name]
            model_name = model_name.replace("_masked", "")
            if outreach:
                if model_name in ["GenMol", "SAFE-GPT", "Best Previous Model"]:
                    continue
            else:
                if model_name == "GenMol (NVIDIA)":
                    continue
            metrics = data["metrics"]
            # For 'diversity' and 'distance' multiply by 100; others remain as is.
            values = [metrics.get(metric, 0) * (100 if metric in ["diversity", "distance"] else 1) for metric in metrics_order]
            values += values[:1]

            ax.plot(angles, values, linewidth=2, linestyle="-", label=model_name.replace("Gen",''), color=colors.get(model_name, "gray"))
            ax.fill(angles, values, alpha=0.25, color=colors.get(model_name, "gray"))

        plt.title(f"{task.capitalize()} Task Performance", size=15, y=1.1)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

        radar_file = os.path.join(output_dir, f"{task}_radar_plot.pdf")
        plt.savefig(radar_file, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)

    # --- Compute overall averages across tasks ---
    overall_metrics = {}
    num_tasks = len(results)
    for task, models in results.items():
        for model_name, data in models.items():

            metrics = data["metrics"]
            if model_name not in overall_metrics:
                overall_metrics[model_name] = {metric: 0 for metric in metrics_order}
            for metric in metrics_order:
                # Apply scaling for diversity and distance, then sum up.
                val = metrics.get(metric, 0) * (100 if metric in ["diversity", "distance"] else 1)
                overall_metrics[model_name][metric] += val if task != "linker" else 2 * val  # linker desing is weighted twice because it is the same task

    # Average the accumulated metrics over the number of tasks.
    for model_name in overall_metrics:
        for metric in metrics_order:
            overall_metrics[model_name][metric] /= num_tasks + 1

    # --- Create overall radar plot using the averaged metrics ---
    N = len(metrics_order)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2) # type: ignore[attr-defined]
    ax.set_theta_direction(-1) # type: ignore[attr-defined]

    plt.xticks(angles[:-1], [metric.capitalize() for metric in metrics_order], fontsize=12)
    ax.set_rlabel_position(0) # type: ignore[attr-defined]
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
    plt.ylim(0, 100)

    for model_name in model_list:
        metrics = overall_metrics[model_name]
        model_name = model_name.replace("_masked", "")
        model_name = model_name.replace("GenMol", "GenMol  (NVIDIA)")
        if outreach:
            if model_name in ["GenMol", "SAFE-GPT", "Best Previous Model"]:
                # for model_name, metrics in overall_metrics.items():
                #     if outreach:
                #         model_name = model_name.replace("_masked","")
                # if model_name in ["GenMol", "SAFE-GPT"]:
                print("skipping", model_name)
                continue
        else:
            if model_name == "Best Previous Model":
                continue
        values = [metrics.get(metric, 0) for metric in metrics_order]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle="-", label=model_name.replace("Gen",''), color=colors.get(model_name, "grey"))
        ax.fill(angles, values, alpha=0.25, color=colors.get(model_name, "gray"))

    plt.title("Fragment Constrained Generation", size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))

    overall_radar_file = os.path.join(output_dir, "overall_radar_plot.pdf")
    plt.savefig(overall_radar_file, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)



def combine_images_horizontally(img1, img2):
    """
    Combines two PIL images side-by-side.
    """
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    combined_img = Image.new("RGB", (total_width, max_height), color="white")
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))
    return combined_img


def visualize_generated_molecules(
    frags,
    generated_smiles_dict,  # dict with keys=model names, values = list of generated SMILES for that model.
    num_samples_per_frag: int,
    num_random_frags: int = 3,
    num_generated: int = 5,
    seed: int = 42,
    title: str = "Motif Extension Samples",
    outpath: str = "motif_samples.pdf",
    format="pdf",
    linker: bool = False,
    valid_indices=None,
):
    """
    Plots a grid of molecules for each randomly selected fragment:
    - First column: the original input (fragment).
    - Then for each model in generated_smiles_dict, plot num_generated randomly selected generated molecules.

    Args:
        frags (list[str]): A list of motif SMILES (length = #fragments).
        num_samples_per_frag (int): How many samples were generated per motif.
        num_random_frags (int): How many random motifs to display (default=3).
        num_generated (int): How many generated molecules to plot per model (default=5).
        seed (int): Random seed for reproducible selection (default=42).
        title (str): Title for the entire figure.
        outpath (str): Where to save the final figure.
        linker (bool): If True, assumes linker design (applies minor formatting).
        valid_indices: Optional list of lists. For each fragment, a list of valid indices to filter the generated samples.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    random.seed(seed)

    if frags is not None:
        total_frags = len(frags)
        chosen_indices = random.sample(range(total_frags), k=min(num_random_frags, total_frags))
        denovo = False
    else:
        denovo = True
        chosen_indices = range(3)

    # Number of models provided.
    model_names = list(generated_smiles_dict.keys())
    num_models = len(model_names)

    # Total columns: 1 for original + num_generated columns per model.
    ncols = int(frags is not None) + num_generated * num_models

    fig, axes = plt.subplots(
        nrows=len(chosen_indices),
        ncols=ncols,
        figsize=(3 * ncols, 3 * len(chosen_indices))
    )

    # If only 1 row, make 'axes' a 2D array so we can iterate uniformly
    if len(chosen_indices) == 1:
        axes = [axes]

    # For each chosen motif, pick random samples from each model
    for row_idx, frag_idx in enumerate(chosen_indices):

        # (a) Get the original fragment if not de novo
        if not denovo:
            plot_frag = frags[frag_idx].replace(" ", ".").rstrip(".")
            orig_mol = Chem.MolFromSmiles(plot_frag)

            # Plot the original fragment in the first column
            ax_orig = axes[row_idx][0]
            if orig_mol:
                img = Draw.MolToImage(orig_mol, size=(200, 200))
                ax_orig.imshow(img)
            else:
                ax_orig.text(0.5, 0.5, "Invalid\nFragment",
                           horizontalalignment="center",
                           verticalalignment="center",
                           fontsize=12, color="red")

            if row_idx == 0:
                ax_orig.set_title("Prompt", fontsize=14, fontweight='bold')
            ax_orig.axis("off")

        # (b) For each model, get the generated SMILES and plot them
        col_offset = int(frags is not None)  # starting column for generated molecules

        for model_idx, model in enumerate(model_names):
            smiles_list = generated_smiles_dict[model]

            # Handle different input formats for smiles_list
            if isinstance(smiles_list[0], list):
                # smiles_list is a list of lists (one list per fragment)
                if frag_idx < len(smiles_list):
                    block = smiles_list[frag_idx]
                else:
                    block = []
            else:
                # smiles_list is a flat list
                start = frag_idx * num_samples_per_frag
                end = start + num_samples_per_frag
                block = smiles_list[start:end]

            # If valid_indices is provided, filter to only valid molecules
            if valid_indices is not None and not denovo:
                if frag_idx < len(valid_indices) and valid_indices[frag_idx] is not None:
                    indices = valid_indices[frag_idx]
                    # Filter block to only include valid indices
                    valid_block = []
                    for idx in indices:
                        if idx < len(block) and block[idx] is not None:
                            valid_block.append(block[idx])
                    block = valid_block

            # Filter out None values and "INVALID" strings
            block = [smi for smi in block if smi is not None and smi != "INVALID"]

            # Randomly sample from the block if enough samples exist
            if len(block) >= num_generated:
                selected = random.sample(block, k=num_generated)
            else:
                # If not enough valid molecules, pad with empty cells
                selected = block + [None] * (num_generated - len(block))

            # (c) Plot the generated molecules for this model
            for i, smi in enumerate(selected):
                ax = axes[row_idx][col_offset + i]

                if smi is None:
                    # Empty cell
                    ax.text(0.5, 0.5, "No valid\nmolecule",
                           horizontalalignment="center",
                           verticalalignment="center",
                           fontsize=10, color="gray")
                else:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        ax.text(0.5, 0.5, "Invalid\nSMILES",
                               horizontalalignment="center",
                               verticalalignment="center",
                               fontsize=10, color="red")
                    else:
                        img = Draw.MolToImage(mol, size=(200, 200))
                        ax.imshow(img)

                # Add model name as title for first row
                if row_idx == 0 and i == num_generated // 2:  # Center the model name
                    ax.set_title(model, fontsize=14, fontweight='bold')

                ax.axis("off")

            col_offset += num_generated

    # Add main title

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", format=format, dpi=300)
    plt.close()

    print(f"Saved visualization to {outpath}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def autolabel(ax, rects, df, metric_names):
    """
    Annotate each bar with its absolute value.

    Args:
        ax: Matplotlib axis.
        rects: List of BarContainer objects (one per model).
        df: Original DataFrame with absolute metric values.
        metric_names: List of metric names (column names).
    """
    models = df.index.tolist()
    for i, rect_group in enumerate(rects):
        model_name = models[i]
        for j, rect in enumerate(rect_group):
            height = rect.get_height()
            abs_val = df.loc[model_name, metric_names[j]]
            ax.annotate(f"{abs_val:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=11)


def plot_results(results, colors, ax=None, title=None, savepath=None, outreach=False):
    """
    Plot the results for a single task as a bar chart. This function
    assumes that 'results' is a dictionary with model names as keys and
    their corresponding metric dictionaries as values.

    Args:
        results (dict): Dictionary of metrics for one task. For example:
            {
              "ModelA": {"validity": 95, "uniqueness": 70, "quality": 20, "diversity": 0.6, "distance": 0.65},
              "ModelB": {...},
              ...
            }
        colors (dict): Mapping from model names to colors.
        ax (matplotlib.axes.Axes, optional): Axis on which to plot. If None, a new figure is created.
        title (str, optional): Title for the plot.
        savepath (str, optional): If provided, the figure is saved to this path.

    Returns:
        fig, ax: The matplotlib figure and axis objects.
    """
    # Convert results to DataFrame.
    df = pd.DataFrame()
    for model in results.keys():
        df[model] = pd.Series(results[model]["metrics"])
    df = df.T

    # Process DataFrame:
    # - Scale 'distance' and 'diversity' (if present) by 100.
    if "distance" in df.columns:
        df["distance"] *= 100
    if "diversity" in df.columns:
        df["diversity"] *= 100
    # - Drop unwanted columns (e.g. "fcd") if present.
    if "fcd" in df.columns:
        df = df.drop(columns=["fcd"])
    # - Rename columns for prettier plotting.
    rename_map = {"validity": "Validity", "uniqueness": "Uniqueness", "quality": "Quality", "diversity": "Diversity", "distance": "Distance"}
    df = df.rename(columns=rename_map)
    metric_names = df.columns.tolist()

    # Normalize the DataFrame (so each metric column has a max value of 1).
    df_norm = df.div(df.max(axis=0))

    # Create an axis if one was not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 8))
    else:
        fig = ax.get_figure()

    # Create the bar chart.
    x = np.arange(len(metric_names)) * 1.5  # x positions for groups of bars
    bar_width = 0.25
    available_models = df.index.tolist()
    rects_list = []
    for i, model in enumerate(available_models):

        if outreach and model in ["SAFE-GPT", "GenMol"]:
            continue
        elif not outreach and model.lower() == "best previous model":
            continue

        rects = ax.bar(x + (i - len(available_models) // 2) * bar_width, df_norm.loc[model], bar_width, label=model.replace("Gen",''), color=colors.get(model, "gray"), edgecolor="black")
        rects_list.append(rects)
    # Set plot aesthetics as in your original style.
    ax.set_title(title if title is not None else "add_title", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right", fontsize=13)
    ax.set_ylabel("Normalized Metric Value", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_yticklabels([])

    # Annotate bars with absolute values.
    autolabel(ax, rects_list, df, metric_names)
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))

    # Final layout and overall title.
    fig.suptitle("Comparison of Generated Molecules vs. Reference Models", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0., 0., 1., 0.95]) # type: ignore[attr-defined]

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", format="pdf")
    return fig, ax
