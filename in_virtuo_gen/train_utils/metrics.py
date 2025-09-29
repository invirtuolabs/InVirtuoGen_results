import numpy as np
import pandas as pd
import os
from ..preprocess.preprocess_tokenize import custom_decode_sequence
from ..utils.fragments import bridge_smiles_fragments_fix, bridge_smiles_fragments
from ..utils.mol import calculate_average_tanimoto, compute_properties, is_drug_like_and_synthesizable, is_valid_smiles, get_canonical, novelty_score
import random
from collections import Counter
import numpy as np
import torch.distributed as dist
from rdkit import Chem
import re


def ids_to_smiles(
    generated_ids,
    print_flag=True,
    exclude_salts=True,
    tokenizer=None,
    already_smiles=False,
    return_frags=False,
):
    all_smiles = []
    valid_smiles = []
    invalid_molecules = []
    valid_indices = []
    invalid_tokens = []
    invalid_decoded = []
    valid_unique_indices = []
    frags_list = []
    # Ensure generated_string is defined outside the try/except block
    num_samples = len(generated_ids)
    for i, ids in enumerate(generated_ids):
        try:
            smiles = None
            decoded = custom_decode_sequence(tokenizer, ids) if not already_smiles else "".join(tokenizer.decode(ids).split())  # type: ignore[attr-defined]
            if not already_smiles:
                frags = [frag for frag in decoded.split() if frag]
                if return_frags:
                    frags_list.append(frags)
                # If you need to do more processing, consider renaming the variable for clarity.
                smiles = bridge_smiles_fragments(frags, print_flag=False)
                if smiles is not None and smiles.find(".") != -1:
                    smiles = bridge_smiles_fragments_fix(frags, print_flag=False)
                canonical = is_valid_smiles(smiles, exclude_salts)
                pattern = r"\[\d+\*\]"
                # Check if it exists
                if re.search(pattern, smiles):  # type: ignore[attr-defined]
                    raise ValueError(f"Found a structure like [i*] in {smiles}")
            else:
                smiles = decoded
                canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
            if canonical:
                if smiles not in valid_smiles:
                    valid_unique_indices.append(i)
                valid_smiles.append(smiles)
                valid_indices.append(i)
            else:
                # Use a placeholder error message if needed
                invalid_tokens.append(ids)
                invalid_decoded.append(decoded)
                invalid_molecules.append(decoded if smiles == "" or smiles is None else smiles)
            all_smiles.append(smiles)
        except Exception as e:
            invalid_molecules.append((decoded if smiles == "" or smiles is None else smiles))
            all_smiles.append("INVALID")  # pad with placeholder in case of error
            invalid_tokens.append(ids)
            invalid_decoded.append(decoded)

        if len(all_smiles) >= num_samples:
            break
    if print_flag:
        print(f"Generation complete. Total SMILES: {len(all_smiles)}")
        print(f"Total valid SMILES: {len(valid_smiles)}")
        print(f"Total invalid molecules: {len(invalid_molecules)}")
        print("Sample SMILES:", valid_smiles[:5])
        print("Sample invalid SMILES:", invalid_molecules[:5])
        print("Sample Tokens:", generated_ids[:5])
        print("Sample Invalid Tokens:", invalid_tokens[:5])
        print("Sample invalid Decoded:", invalid_decoded[:5])
    if return_frags:
        return frags_list

    return all_smiles, valid_smiles, invalid_molecules, valid_indices, valid_unique_indices


def evaluate_smiles(
    generated_ids, tokenizer, return_values=False, print_flag=True, exclude_salts=True, num_calc=1000, already_smiles=False, print_metrics=True, device="cuda:0", return_unique_indices=False, **kwargs
):
    num_samples = len(generated_ids)
    num_calc = min(num_calc, len(generated_ids))
    all_smiles, valid_smiles, invalid_molecules, valid_indices, valid_unique_indices = ids_to_smiles(generated_ids, print_flag=print_flag, exclude_salts=exclude_salts, tokenizer=tokenizer, already_smiles=already_smiles, )

    valid_count = len(valid_smiles)
    unique_valid_smiles = set(valid_smiles)

    # Get unique valid indices corresponding to unique valid SMILES
    unique_valid_indices = []
    seen_smiles = set()
    for i, smiles in enumerate(valid_smiles):
        if smiles not in seen_smiles:
            unique_valid_indices.append(valid_indices[i])
            seen_smiles.add(smiles)

    quality_percentage = 0.0
    validity = (valid_count / len(all_smiles)) * 100 if len(all_smiles) > 0 else 0.0
    uniqueness = (len(unique_valid_smiles) / valid_count) * 100 if valid_count > 0 else 0.0
    div_values = calculate_average_tanimoto(list(unique_valid_smiles)[:num_calc])
    diversity = np.nanmean(div_values) if div_values else 0.0

    if print_metrics:
        print(f"Validity: {validity:.2f}%")
        print(f"Uniqueness: {uniqueness:.2f}%")
        print(f"Diversity: {diversity:.4f}")
    try:
        average_sa, average_qed = 0.0, 0.0
        valid_smiles_filtered = [sm for sm in unique_valid_smiles if sm is not None]

        if valid_smiles_filtered:
            properties_list = compute_properties(valid_smiles_filtered[:num_calc])
            if properties_list and any(prop is not None for prop in properties_list):
                # Calculate SA scores: ensure that each property list is valid
                sa_scores = [prop[0] for prop in properties_list if prop is not None and len(prop) >= 1 and prop[0] is not None]
                qed_scores = [prop[1] for prop in properties_list if prop is not None and len(prop) >= 2 and prop[1] is not None]
                if qed_scores:
                    qed_scores = [score if score is not None else np.nan for score in qed_scores]
                    average_qed = np.nanmean(qed_scores)
                if sa_scores:
                    sa_scores = [score if score is not None else np.nan for score in sa_scores]
                    average_sa = np.nanmean(sa_scores)
                else:
                    print("No SA scores computed.")
                # Calculate quality metric
                quality_count = sum(1 for props in properties_list if props is not None and is_drug_like_and_synthesizable(props))
                quality_percentage = (quality_count / num_calc) * 100
                if print_metrics:
                    print(f"Average QED score: {average_qed:.4f}")
                    print(f"Average SA score: {average_sa:.4f}")

                    print(f"Quality (valid, unique, drug-like, synthesizable): {quality_percentage:.2f}%")

            else:
                print("Unable to compute properties: All properties are None or empty")
        else:
            if print_flag:
                print("No valid SMILES to compute properties")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error computing properties: {str(e)}")
        quality_percentage = 0.0  # Ensure a default value for synchronization
    # In a distributed setting, broadcast the metrics from rank 0 to all ranks
    metrics = {
        "validity": validity,
        "uniqueness": uniqueness,
        "diversity": diversity,
        "quality": quality_percentage,
        "sa": average_sa,
        "qed": average_qed,
    }


    if return_values:
        # When prompts are provided, we want to return the full list (with placeholders) to preserve block ordering.
        smiles_out = [s if s is not None else "INVALID" for s in all_smiles]
        # Ensure the list is exactly num_samples long (pad if necessary)
        if len(smiles_out) < num_samples:
            smiles_out += ["INVALID"] * (num_samples - len(smiles_out))
        if return_unique_indices:
            return valid_unique_indices, smiles_out, metrics
        return valid_indices, smiles_out, metrics
    else:
        return metrics
