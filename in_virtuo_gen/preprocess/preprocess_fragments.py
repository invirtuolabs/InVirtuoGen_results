#!/usr/bin/env python3
import argparse
import traceback

from datasets import load_dataset
from rdkit import Chem, RDLogger

from ..utils.fragments import (
    bridge_smiles_fragments,
    num_frags,
    randomize_compatible_fragments,
    remove_stereochemistry,
    smiles2frags,
    order_fragments_by_attachment_points
)
from ..utils.mol import remove_salt
import random
RDLogger.DisableLog("rdApp.*")


def process_batch(examples, verbose=True, num_max_fragments=7, order_fragments=False):
    """
    Processes a batch of SMILES strings by removing salts and stereochemistry,
    fragmenting molecules, and randomizing fragments if applicable.

    Parameters:
        examples (dict): A dictionary with a 'smiles' key containing a SMILES string.

    Returns:
        dict: A dictionary with keys 'fragments' (list of fragment strings) and
              'num_fragments' (list of fragment counts).
    """
    results = {"fragments": [], "num_fragments": []}

    try:
        # Preprocess SMILES: remove salts and stereochemistry
        smiles = examples["smiles"]
        smiles = remove_salt(smiles)
        smiles = remove_stereochemistry(smiles)
        original_smiles = smiles

        # Determine maximum number of fragments available
        max_fragments = num_frags(smiles)
        if max_fragments > 1:
            num_fragments = min(max_fragments, num_max_fragments) #

            # Attempt up to 1000 times to get a valid fragmentation
            found_match = False
            for _ in range(1000):
                fragments = smiles2frags(smiles, num_fragments)
                random.shuffle(fragments)  # Randomize fragment order
                if order_fragments:
                    fragments = order_fragments_by_attachment_points(fragments)
                new_smiles = Chem.CanonSmiles(bridge_smiles_fragments(fragments))
                if smiles == new_smiles:
                    found_match = True
                    canonical_smiles = Chem.CanonSmiles(smiles)
                    all_frags = ' '.join(fragments)
                    if all_frags not in results["fragments"]:
                        results["fragments"].append(all_frags)
                        results["num_fragments"].append(num_fragments)
                    break

            if not found_match and verbose:
                print(f"Mismatch in SMILES: {original_smiles} != {new_smiles}")
        else:
            # For molecules that do not fragment, use the canonical SMILES directly.
            num_fragments = 1
            canonical_smiles = Chem.CanonSmiles(smiles)
            results["fragments"].append(canonical_smiles)
            results["num_fragments"].append(num_fragments)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if verbose:
            print(f"Error processing SMILES {original_smiles}: {str(e)}")

    return results


def process_smiles_dataset(input_path: str, output_path: str, batch_size: int = 10000, num_proc: int = 4, csv: bool = False, num_max_fragments: int = 7, order_fragments: bool = True):
    """
    Loads a dataset of SMILES strings, processes each SMILES entry by fragmentation,
    and saves the processed dataset to disk.

    Parameters:
        input_path (str): Path to the input dataset file containing SMILES strings.
        output_path (str): Path to save the processed dataset.
        batch_size (int, optional): Number of examples to process per batch. Default is 10000.
        num_proc (int, optional): Number of parallel processes to use. Default is 4.
    """
    print(f"Processing dataset from {input_path} to {output_path}")
    if csv:
        dataset = load_dataset("csv", data_files=input_path, split="train")
    else:
        dataset = load_dataset("text", data_files=input_path, split="train")
        dataset = dataset.rename_column("text", "smiles")
    print(f"Loaded dataset with {len(dataset)} examples")

    processed_dataset = dataset.map(
        lambda x: process_batch(x, num_max_fragments=num_max_fragments, order_fragments=order_fragments),
        batched=False,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns="smiles",
        keep_in_memory=True,
        load_from_cache_file=False,

    )

    print(f"Processed dataset with {len(processed_dataset)} examples")

    # Filter out entries where processing failed (fragments is None)
    processed_dataset = processed_dataset.filter(lambda x: x["fragments"] is not None)
    print(f"Filtered dataset with {len(processed_dataset)} examples")

    processed_dataset.save_to_disk(output_path, num_shards=1)
    print(f"Saved processed dataset to {output_path}")



def main():
    """
    Parses command-line arguments and initiates the processing of the SMILES dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input SMILES dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for processing.")
    parser.add_argument("--num_proc", type=int, default=40, help="Number of parallel processes.")
    parser.add_argument('--csv', action='store_true', help='load from csv')
    parser.add_argument('--num_max_fragments', type=int, default=7, help='Maximum number of fragments to use.')
    parser.add_argument('--order_fragments', action='store_true', help='Order fragments by attachment points.')
    args = parser.parse_args()
    config = vars(args)
    config["output_path"] = config["output_path"] + f"_max_frags_{config['num_max_fragments']}"
    if config["order_fragments"]:
        config["output_path"] += "_ordered"
    process_smiles_dataset(**config)


if __name__ == "__main__":
    main()
