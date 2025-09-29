import argparse
import random
import sys
import time
from functools import partial
from multiprocessing import Pool

import atomInSmiles
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from ..utils.fragments import bridge_smiles_fragments, num_frags, randomize_compatible_fragments, remove_stereochemistry, smiles2frags
from ..utils.mol import remove_salt

# Configure standard output for line buffering

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Disable RDKit logging
RDLogger.DisableLog("rdApp.*")


def process_smiles_map(smiles, augmentation=1, timeout=30, max_frags=7):
    """
    Processes a single SMILES string, fragmenting and augmenting it as needed.

    Args:
        Smiles (str): SMILES string
        augmentation (int): the number of augmentations to generate.
        timeout (int): Timeout for processing a single SMILES string (default: 30 seconds).

    Returns:
        list: A list of fragmented SMILES strings or the canonical SMILES for single-fragment molecules.
    """
    or_smiles = smiles
    results = []

    try:
        # Preprocess the SMILES string
        smiles = remove_salt(smiles)
        smiles = remove_stereochemistry(smiles)
        n_frags = num_frags(smiles)

        if n_frags > 1:
            num_fragments = min(n_frags, max_frags=7)

            try:
                attempt_count = 0

                while len(results) < augmentation and attempt_count < 1000:
                    fragments = smiles2frags(smiles, num_fragments)
                    newSMILES = Chem.CanonSmiles(bridge_smiles_fragments(fragments))
                    if smiles == newSMILES:
                        smiles = Chem.CanonSmiles(smiles)
                        fragments = randomize_compatible_fragments(fragments)
                        all_frags = " ".join(fragments)
                        if all_frags not in results:
                            results.append(all_frags)
                    else:
                        print(f"Mismatch in SMILES: {newSMILES} != {smiles}")
                    attempt_count += 1
            except Exception as e:
                print(f"Error processing fragment for SMILES {or_smiles}: {str(e)}")
        else:
            # Handle single-fragment molecules
            try:
                canonical_smiles = Chem.CanonSmiles(smiles)
                results.append(canonical_smiles)
            except Exception as e:
                print(f"Error canonicalizing SMILES {smiles}: {str(e)}")
    except Exception as e:
        print(f"Error with SMILES: {or_smiles} Error: {str(e)}")

    return results


def process_smiles(smiles,augmentation=1, timeout=30, verbose =False,max_frags=7):
    """
    Processes a single SMILES string, fragmenting and augmenting it as needed.

    Args:
        args (tuple): A tuple containing the SMILES string and the number of augmentations to generate.
        timeout (int): Timeout for processing a single SMILES string (default: 30 seconds).

    Returns:
        list: A list of fragmented SMILES strings or the canonical SMILES for single-fragment molecules.
    """
    # smiles, augmentation = args
    or_smiles = smiles
    results = []

    try:
        # Preprocess the SMILES string
        smiles = remove_salt(smiles)
        smiles = remove_stereochemistry(smiles)
        n_frags = num_frags(smiles)

        if n_frags > 1:
            num_fragments = min(n_frags, max_frags)
            try:
                attempt_count = 0
                found_match = False

                while len(results) < augmentation and attempt_count < (1+augmentation):
                    fragments = smiles2frags(smiles, num_fragments)
                    # Randomize the order of the fragments
                    random.shuffle(fragments)
                    newSMILES = Chem.CanonSmiles(bridge_smiles_fragments(fragments))
                    if smiles == newSMILES:
                        found_match = True
                        smiles = Chem.CanonSmiles(smiles)
                        fragments = randomize_compatible_fragments(fragments)
                        all_frags = " ".join(fragments)
                        if all_frags not in results:
                            results.append(all_frags)
                    attempt_count += 1

                if not found_match and verbose:
                    print(f"Mismatch in SMILES: {or_smiles} != {newSMILES}")
            except Exception as e:
                if verbose:
                    print(f"Error processing fragment for SMILES {or_smiles}: {str(e)}")
        else:
            # For single fragment molecules, use the original SMILES
            try:
                canonical_smiles = Chem.CanonSmiles(smiles)
                results.append(canonical_smiles)
            except Exception as e:
                if verbose:
                    print(f"Error canonicalizing SMILES {smiles}: {str(e)}")
    except Exception as e:
        if verbose:
            print(f'Error with SMILES: {or_smiles} Error: {str(e)}')

    return results


def process_chunk_with_timeout(chunk, augmentation, num_workers, timeout=30):
    """
    Processes a chunk of SMILES strings in parallel with a timeout.

    Args:
        chunk (list): A list of SMILES strings to process.
        augmentation (int): Number of augmentations to generate for each SMILES.
        num_workers (int): Number of worker processes to use.
        timeout (int): Timeout for processing each SMILES string (default: 30 seconds).

    Returns:
        list: A list of lists containing fragmented SMILES strings for each input SMILES.
    """
    with Pool(num_workers) as pool:
        process_func = partial(process_smiles, verbose=verbose, timeout=timeout)
        results = list(tqdm(pool.imap(process_func, [(smiles, augmentation) for smiles in chunk]),
                            total=len(chunk), desc="Processing molecules", unit="mol"))
    return results


if __name__ == "__main__":
    """
    Main function to process an input file of SMILES strings and generate fragmented SMILES strings.

    Command-line arguments:
        --input_path (-i): Path to the input containing SMILES strings.
        --output_path (-o): Path to the output file for writing fragmented SMILES.
        --num_workers: Number of worker processes to use (default: 24).
        --augmentation: Number of augmented SMILES to generate per input (default: 1).
        --chunk_size: Number of rows to read per chunk (default: 1,000,000).
    """
    parser = argparse.ArgumentParser(description="Process SMILES with optional AIS notation.")
    parser.add_argument("--input_path", "-i", required=True, help=".smi or .sdf file of molecules to be fragmented")
    parser.add_argument("--output_path", "-o", required=True, help="path of the output fragments file")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of worker processes")
    parser.add_argument("--augmentation", default=1, type=int, help="Number of augmented SMILES to generate, default is 1, i.e. no augmentation")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Number of rows per chunk to read")
    opt = parser.parse_args()

    # Process the input file in chunks
    with pd.read_csv(opt.input_path, header=None, names=["smiles"], chunksize=opt.chunk_size) as reader, open(opt.output_path, "w") as output_file:
        for chunk_num, chunk in enumerate(reader, 1):
            print(f"\nProcessing chunk {chunk_num}")
            smiles_chunk = chunk["smiles"].tolist()
            results = process_chunk_with_timeout(smiles_chunk, opt.augmentation, opt.num_workers, timeout=30)

            # Flatten results and remove duplicates
            flattened_results = []
            seen = set()
            for result_set in results:
                for result in result_set:
                    if result is not None and result not in seen:
                        seen.add(result)
                        flattened_results.append(result)

            # Write deduplicated results to the output file
            for result in flattened_results:
                output_file.write(result + "\n")

    print("Done")