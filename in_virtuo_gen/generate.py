# predict.py

import os

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch import Trainer

from in_virtuo_gen.models.invirtuofm import InVirtuoFM


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run predictions using a Lightning checkpoint.")
    parser.add_argument("--ckpt_path", type=str, default="invirtuo_gen.ckpt", help="Path to the checkpoint file.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Batch size for prediction.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices (GPUs).")
    parser.add_argument("--accelerator", type=str, default="cuda", help="Accelerator type.")

    args = parser.parse_args()
    # Verify checkpoint exists
    ckpt = torch.load(args.ckpt_path)
    if args.accelerator == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system, but GPU selected for generation, change to --accelerator cpu")
    # Load the model from checkpoint

    # Load model and data module

    model = InVirtuoFM.load_from_checkpoint(args.ckpt_path)

    model.eval()

    model.to(args.accelerator)

    gen_kwargs = {"num_samples": args.num_samples}
    model.gen_batch_size = args.batch_size
    # Run prediction

    smiles = model.generate_smiles(gen_kwargs=gen_kwargs)
    print(smiles)
    df = pd.DataFrame({"smiles": smiles})
    # Ensure all entries are individual strings, not lists
    df = df.explode("smiles", ignore_index=True)
    print(df)

    df.to_parquet("smiles.parquet")


if __name__ == "__main__":
    main()
