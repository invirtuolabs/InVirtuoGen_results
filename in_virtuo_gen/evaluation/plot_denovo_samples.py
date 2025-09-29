# save as plot_smiles_grid.py
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import argparse
import os
import math
import random

RDLogger.DisableLog("rdApp.*")

def load_smiles(path):
    """Load SMILES from file, one per line with optional labels."""
    smiles_list = []
    labels = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):  # allow comments/blank lines
                continue

            # allow "SMILES [whitespace] label" format
            parts = s.split()
            smi = parts[0]
            lbl = " ".join(parts[1:]) if len(parts) > 1 else ""

            # Test if SMILES is valid
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    smiles_list.append(smi)
                    labels.append(lbl if lbl else smi)
                else:
                    print(f"Warning: Invalid SMILES on line {line_num}: {smi}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    return smiles_list, labels

def plot_molecules_grid(smiles_list, labels, args):
    """Create a grid plot of molecules using matplotlib."""
    # Remove duplicates while preserving order
    seen = set()
    unique_smiles = []
    unique_labels = []

    for smi, lbl in zip(smiles_list, labels):
        if smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)
            unique_labels.append(lbl)

    print(f"Removed {len(smiles_list) - len(unique_smiles)} duplicate SMILES")
    smiles_list = unique_smiles
    labels = unique_labels

    n_mols = min(len(smiles_list), args.max_mols)
    if n_mols == 0:
        raise ValueError("No valid molecules to plot")

    # Sample if needed
    if len(smiles_list) > args.max_mols:

        indices = random.sample(range(len(smiles_list)), args.max_mols)
        smiles_list = [smiles_list[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"Randomly sampled {args.max_mols} unique molecules from {len(unique_smiles)} total")

        n_mols = len(smiles_list)

    n_cols = args.cols
    n_rows = math.ceil(n_mols / n_cols)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot molecules
    for idx in range(n_mols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx][col_idx]

        smi = smiles_list[idx]
        label = labels[idx]

        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                ax.text(0.5, 0.5, "Invalid\nSMILES",
                       horizontalalignment="center", verticalalignment="center",
                       fontsize=12, color="red", transform=ax.transAxes)
            else:
                # Generate molecule image - this is the same approach as your working code
                img = Draw.MolToImage(mol, size=(args.w, args.h))
                ax.imshow(img)

        except Exception as e:
            # Handle failed molecule rendering
            ax.text(0.5, 0.5, f"Error\nrendering",
                   horizontalalignment="center", verticalalignment="center",
                   fontsize=10, color="red", transform=ax.transAxes)

        # Add title if requested
        if args.titles:
            # Truncate long labels
            display_label = label if len(label) <= 25 else label[:22] + "..."
            ax.set_title(display_label, fontsize=9, pad=5)

        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_mols, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx][col_idx].axis("off")

    # Adjust layout
    plt.tight_layout(pad=1.0)

    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot a grid of molecules from a SMILES file.")
    parser.add_argument("smiles", help="Path to SMILES file (one per line)")
    parser.add_argument("--out", default="plots/smiles_grid.png", help="Output image path")
    parser.add_argument("--cols", type=int, default=6, help="Molecules per row")
    parser.add_argument("--w", type=int, default=300, help="Molecule image width")
    parser.add_argument("--h", type=int, default=250, help="Molecule image height")
    parser.add_argument("--titles", action="store_true", help="Show molecule labels/SMILES")
    parser.add_argument("--max-mols", type=int, default=36, help="Maximum molecules to plot")

    args = parser.parse_args()

    # Load molecules
    print(f"Loading SMILES from {args.smiles}...")
    smiles_list, labels = load_smiles(args.smiles)

    if not smiles_list:
        raise SystemExit("No valid SMILES found in input file.")

    print(f"Loaded {len(smiles_list)} valid molecules.")

    # Create plot
    print(f"Creating {math.ceil(min(len(smiles_list), args.max_mols) / args.cols)} x {args.cols} grid...")
    fig = plot_molecules_grid(smiles_list, labels, args)

    # Save plot
    output_path = args.out
    if not any(output_path.endswith(ext) for ext in ['.png', '.pdf', '.svg', '.jpg']):
        output_path += '.png'

    print(f"Saving to {output_path}...")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Successfully saved grid to {output_path}")

if __name__ == "__main__":
    main()