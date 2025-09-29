#!/usr/bin/env python
"""
Processes tokenized sequence data from .npy files or Hugging Face dataset (Arrow files), organizing sequences into buckets based on their lengths. The script:

1. Calculates the distribution of sequence lengths and saves it as a PyTorch file (.pt).
2. Generates and saves a histogram plot illustrating the sequence length distribution.
3. Creates bucketized memmapped (.npy) files, grouping sequences by specified length intervals for optimized data loading during model training.

Example usage:
    python create_mmap_buckets.py --input_pattern data/*.npy --output_folder mmap_buckets --chunk_size 50000 --bucket_step 10 --max_length 150

"""
import argparse
import gc
import glob
import os

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bucketize and save memmapped numpy arrays from input .npy files or a dataset (Arrow files)."
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="/media/Data/00Dataset/np/high_diversity/*0.npy",
        help="Glob pattern for input .npy files or path to a dataset directory."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./mmap_buckets",
        help="Folder to save the bucketized memmapped .npy files."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Chunk size for processing input files or dataset."
    )
    parser.add_argument(
        "--bucket_step",
        type=int,
        default=10,
        help="Step size for bucket quantization."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
        help="Maximum bucket length."
    )
    parser.add_argument(
        "--distribution_output",
        type=str,
        default="configs/distribution.pt",
        help="Path to save the distribution as a .pt file."
    )
    parser.add_argument(
        "--plots_output",
        type=str,
        default="plots/distribution.pt",
        help="Path to save the distribution plot (will be saved as .png)."
    )
    parser.add_argument(
        "--dataset_column",
        type=str,
        default="input_ids",
        help="Name of the column in the dataset containing sequences (only used when --use_datasets is set)."
    )
    parser.add_argument(
        "--use_datasets",
        action="store_true",
        help="Load input as a dataset (Arrow files) via Hugging Face datasets instead of .npy files."
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Flag to indicate if the dataset is validation data (affects bucket file naming)."
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Flag to indicate if the dataset is safe (affects bucket file naming)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.safe:
        tokenizer = Tokenizer.from_file("tokenizer/safe.json")
        pad_token_id = 3
        end_token_id = 2
        bos_token_id = 1
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/smiles_new.json")
        pad_token_id = tokenizer.encode("[PAD]")[0]
        end_token_id = tokenizer.encode("[EOS]")[0]
        bos_token_id = tokenizer.encode("[BOS]")[0]
    output_folder = args.output_folder
    chunk_size = args.chunk_size
    bucket_step = args.bucket_step
    max_length = args.max_length
    sum_count=0
    # Ensure output folder exists.
    os.makedirs(output_folder, exist_ok=True)

    # === FIRST PASS: Compute bucket sizes for all samples ===
    bucket_sizes = {}  # Maps bucket (e.g. 10,20,...,190) to total sample count.
    all_lengths = []  # FIXED: Collect all lengths for proper distribution calculation

    if args.use_datasets:
        from datasets import load_from_disk
        print("Loading dataset from", args.input_pattern)
        dataset = load_from_disk(args.input_pattern)
        total_samples = len(dataset)
        print(f"Dataset loaded with {total_samples} samples.")
        print("Calculating bucket sizes (using dataset)...")
        for chunk_start in tqdm(range(0, total_samples, chunk_size), desc="Counting in dataset"):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            # Extract the column of interest and convert to a numpy array.
            batch = dataset[chunk_start:chunk_end][args.dataset_column]
            chunk = np.array(batch).reshape(-1, max_length)
            # Compute effective length (number of tokens not equal to padding token 3).
            lengths = np.count_nonzero((chunk != pad_token_id) & (chunk != end_token_id) & (chunk != bos_token_id), axis=1)
            valid = lengths > 0
            if not np.any(valid):
                continue
            chunk = chunk[valid]
            lengths = lengths[valid]
            all_lengths.extend(lengths.tolist())  # FIXED: Collect all lengths
            # Determine the bucket for each sample.
            sample_buckets = ((lengths + bucket_step - 1) // bucket_step) * bucket_step
            sample_buckets = np.clip(sample_buckets, bucket_step, max_length)
            unique, counts = np.unique(sample_buckets, return_counts=True)
            for bucket, count in zip(unique, counts):
                if bucket == 0:
                    continue
                bucket_sizes[bucket] = bucket_sizes.get(bucket, 0) + count
            gc.collect()
    else:
        input_files = sorted(glob.glob(args.input_pattern))

        input_files = list(dict.fromkeys(input_files))  # Preserves order while removing duplicates

        print("Input files:", input_files)
        print("Calculating bucket sizes (using .npy files)...")
        for input_file in input_files:
            is_val = args.val
            data = np.memmap(input_file, mode="r", dtype=np.uint8 if not args.safe else np.uint16).reshape(-1, max_length)
            sum_count+=len(data)
            for chunk_start in tqdm(range(0, len(data), chunk_size),
                                    desc=f"Counting in {os.path.basename(input_file)}"):
                chunk_end = min(chunk_start + chunk_size, len(data))
                chunk = data[chunk_start:chunk_end]
                lengths = np.count_nonzero((chunk != pad_token_id) & (chunk != end_token_id) & (chunk != bos_token_id), axis=1)
                valid = lengths > 0
                if not np.any(valid):
                    continue
                chunk = chunk[valid]
                lengths = lengths[valid]
                all_lengths.extend(lengths.tolist())  # FIXED: Collect all lengths
                sample_buckets = ((lengths + bucket_step - 1) // bucket_step) * bucket_step
                sample_buckets = np.clip(sample_buckets, bucket_step, max_length)
                unique, counts = np.unique(sample_buckets, return_counts=True)
                for bucket, count in zip(unique, counts):
                    if bucket == 0:
                        continue
                    bucket_sizes[bucket] = bucket_sizes.get(bucket, 0) + count
            gc.collect()
    print(f"Total samples: {sum_count}")
    all_lengths = np.array(all_lengths)
    unique_lengths, counts = np.unique(all_lengths, return_counts=True)
    total_samples_bucket = sum(counts)
    probabilities = [c / total_samples_bucket for c in counts]

    distribution = {
        "unique_lengths": torch.tensor(unique_lengths, dtype=torch.int32),
        "probs": torch.tensor(probabilities, dtype=torch.float32)
    }

    # Save distribution.
    torch.save(distribution, args.distribution_output)
    print(f"\nSaved distribution as {args.distribution_output}")

    # Create and save distribution plot.
    plt.figure(figsize=(8, 6))
    plt.bar(unique_lengths, probabilities, color="skyblue", edgecolor="black")
    plt.xlabel("Sequence Length (Bucket)")
    plt.ylabel("Probability")
    plt.yscale("log")
    plt.title("Distribution of Sequence Lengths")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plot_path = args.plots_output.replace(".pt", "") + "_" +args.output_folder.split("/")[-1] + ".pdf"

    plt.savefig(plot_path)
    plt.close()
    print(f"Saved distribution plot as {plot_path}")

    # === Pre-allocate bucket files ===
    print("\nPre-allocating bucket files...")
    for bucket, total_size in bucket_sizes.items():
        bucket_label = f"{bucket - bucket_step}-{bucket}"
        # When using datasets, rely on --val flag; otherwise, check filename.
        is_val = args.val
        out_file = os.path.join(
            output_folder,
            f"bucket_{bucket_label}.npy" if not is_val else f"bucket_{bucket_label}_val.npy",
        )
        if os.path.exists(out_file):
            os.remove(out_file)
        # Create a memmap file with the correct shape.
        fp = np.memmap(out_file, mode="w+", shape=(total_size, bucket), dtype=np.uint8 if not args.safe else np.uint16)
        fp.flush()
        del fp

    # Initialize a dictionary to track the current write offset per bucket.
    bucket_positions = {bucket: 0 for bucket in bucket_sizes}

    # === SECOND PASS: Fill bucket files ===
    print("\nFilling bucket files...")
    if args.use_datasets:
        bucket_memmaps = {}
        for bucket, total_size in bucket_sizes.items():
            bucket_label = f"{bucket - bucket_step}-{bucket}"
            is_val = args.val
            out_file = os.path.join(
                output_folder,
                f"bucket_{bucket_label}.npy" if not is_val else f"bucket_{bucket_label}_val.npy",
            )
            bucket_memmaps[bucket] = np.memmap(
                out_file, mode="r+", dtype=np.uint8 if not args.safe else np.uint16,
                shape=(total_size, bucket)
            )

        print("Processing dataset to fill bucket files...")
        for chunk_start in tqdm(range(0, total_samples, chunk_size), desc="Processing dataset"):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            batch = dataset[chunk_start:chunk_end][args.dataset_column]
            chunk = np.array(batch).reshape(-1, max_length)
            lengths = np.count_nonzero((chunk != pad_token_id) & (chunk != end_token_id) & (chunk != bos_token_id), axis=1)
            valid = lengths > 0
            if not np.any(valid):
                continue
            chunk = chunk[valid]
            lengths = lengths[valid]
            sample_buckets = ((lengths + bucket_step - 1) // bucket_step) * bucket_step
            sample_buckets = np.clip(sample_buckets, bucket_step, max_length)
            for bucket in np.unique(sample_buckets):
                if bucket == 0:
                    continue
                indices = np.where(sample_buckets == bucket)[0]
                if indices.size == 0:
                    continue
                # Truncate each sample to the bucket length.
                bucket_data = chunk[indices, :bucket]
                current_pos = bucket_positions[bucket]
                num_samples = bucket_data.shape[0]

                bucket_memmaps[bucket][current_pos: current_pos + num_samples] = bucket_data
                bucket_positions[bucket] += num_samples

        # Clean up memmaps
        for bucket_memmap in bucket_memmaps.values():
            del bucket_memmap

    else:
        for input_file in input_files:
            print(f"\nProcessing input file: {input_file}")
            # Open bucket files for writing.
            bucket_files = {}
            is_val = args.val
            for bucket, total_size in bucket_sizes.items():
                bucket_label = f"{bucket - bucket_step}-{bucket}"
                out_file = os.path.join(
                    output_folder,
                    f"bucket_{bucket_label}.npy" if not is_val else f"bucket_{bucket_label}_val.npy",
                )
                bucket_files[bucket] = np.memmap(
                    out_file, mode="r+", dtype=np.uint8 if not args.safe else np.uint16, shape=(total_size, bucket)
                )
            data = np.memmap(input_file, mode="r", dtype=np.uint8 if not args.safe else np.uint16).reshape(-1, max_length)
            for chunk_start in tqdm(range(0, len(data), chunk_size),
                                    desc=f"Processing {os.path.basename(input_file)}"):
                chunk_end = min(chunk_start + chunk_size, len(data))
                chunk = data[chunk_start:chunk_end]
                lengths = np.count_nonzero((chunk != pad_token_id) & (chunk != end_token_id) & (chunk != bos_token_id), axis=1)
                valid = lengths > 0
                if not np.any(valid):
                    continue
                chunk = chunk[valid]
                lengths = lengths[valid]
                sample_buckets = ((lengths + bucket_step - 1) // bucket_step) * bucket_step
                sample_buckets = np.clip(sample_buckets, bucket_step, max_length)
                for bucket in np.unique(sample_buckets):
                    if bucket == 0:
                        continue
                    indices = np.where(sample_buckets == bucket)[0]
                    if indices.size == 0:
                        continue
                    bucket_data = chunk[indices, :bucket]
                    current_pos = bucket_positions[bucket]
                    num_samples = bucket_data.shape[0]
                    bucket_files[bucket][current_pos: current_pos + num_samples] = bucket_data
                    bucket_positions[bucket] += num_samples
            del data
            del bucket_files
            print(f"Finished processing {input_file}.")

    print("\nFinished saving all buckets as memmapped numpy arrays")
    print("\nFinal bucket statistics:")
    sum_count=0
    for bucket, count in bucket_positions.items():
        bucket_label = f"{bucket - bucket_step}-{bucket}"
        print(f"Bucket {bucket_label}: {count} samples")
        sum_count+=count
    print(f"Total samples: {sum_count}")
if __name__ == "__main__":
    main()