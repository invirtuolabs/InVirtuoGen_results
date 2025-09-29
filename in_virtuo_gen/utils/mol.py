import os
import sys

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
from multiprocessing import Pool

import sascorer
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.QED import qed
from tqdm import tqdm
def canonicalize(smiles):
    """
    Canonicalize a single SMILES string.
    Returns None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def canonicalize_list(smiles_list):
    """
    Canonicalize a list of SMILES, dropping invalid ones.
    """
    return set(filter(None, (canonicalize(smi) for smi in smiles_list)))

def novelty_score(generated_smiles, training_smiles):
    """
    Compute novelty score of generated molecules.
    """
    canon_train = canonicalize_list([x[0] for x in training_smiles])
    canon_gen = canonicalize_list(generated_smiles)

    if not canon_gen:
        return 0.0

    num_novel = sum(1 for smi in canon_gen if smi not in canon_train)
    return num_novel / len(canon_gen)
def randomize_smiles(smiles):
    """
    Randomizes the atom order in a SMILES string while preserving its chemical structure.

    Args:
        smiles (str): A canonical SMILES string.

    Returns:
        str: A randomized SMILES string, or the input if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    return smiles


def compute_properties(smiles_list, num_processes=24):
    """
    Computes the QED and SA values for a list of SMILES strings in parallel.

    Args:
        smiles_list (list): A list of SMILES strings.
        num_processes (int): Number of processes to use in parallel.

    Returns:
        list of tuples: A list of tuples, each containing the QED and SA values for each SMILES string.
    """
    # with Pool(num_processes) as pool:
    properties = []
    for smiles in smiles_list:
        properties.append(compute_single_property(smiles))

    return properties


def compute_sa_score(smiles):
    """
    Computes the Synthetic Accessibility (SA) score for a single SMILES string.

    Args:
        smiles (str): A SMILES string.

    Returns:
        float: The SA score for the given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    sa_score = sascorer.calculateScore(mol)
    return sa_score


def compute_SA(smiles_list):
    """
    Computes the SA scores for a list of SMILES strings using parallel processing.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        list: A list of SA scores for each SMILES string.
    """
    num_processes = min(len(smiles_list), 24)

    with Pool(processes=num_processes) as pool:
        sa_scores = list(pool.imap(compute_sa_score, smiles_list), total=len(smiles_list))

    return sa_scores


def compute_QED(smiles_list):
    """
    Computes the QED values for a list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        list: A list of QED values for each SMILES string.
    """
    qed_values = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        qed_values.append(QED.qed(mol))

    return qed_values


def compute_molecular_weight(smiles_list):
    """
    Computes the molecular weights of a list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        list: Molecular weights of the molecules.
    """
    return [Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]


def compute_tpsa(smiles_list):
    """
    Computes the Topological Polar Surface Area (TPSA) for a list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        list: TPSA values for each molecule.
    """
    return [Descriptors.TPSA(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]


def compute_clogp(smiles_list):
    """
    Computes the cLogP (logarithm of partition coefficient) for a list of SMILES strings.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        list: cLogP values for each molecule.
    """
    return [Crippen.MolLogP(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]


def calculate_average_tanimoto(smiles_list, prompt=None):
    """
    Calculates the average Tanimoto distance for a list of SMILES strings based on ECFP4 fingerprints.

    Args:
        smiles_list (list): A list of SMILES strings.

    Returns:
        list: Average Tanimoto distances for each SMILES.
    """
    molecules = []
    fingerprints = []
    distances=[]
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append(mol)
            fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    # If a prompt is provided, calculate the distance from the prompt to each molecule
    if prompt is not None:
        prompt_mol = Chem.MolFromSmiles(prompt)
        if prompt_mol is None:
            raise ValueError("Invalid prompt SMILES string")
        prompt_fp = AllChem.GetMorganFingerprintAsBitVect(prompt_mol, 2, nBits=2048)
        for fp in fingerprints:
            distances.append(1 - DataStructs.FingerprintSimilarity( fp,prompt_fp))
        return np.nanmean(distances)

    else:
        # Calculate the average Tanimoto distance for each molecule (excluding self-comparison)
        average_distances = []
        for i, fp1 in enumerate(fingerprints):
            distances = [1 - DataStructs.FingerprintSimilarity(fp1, fp2) for j, fp2 in enumerate(fingerprints) if i != j]
            average_distances.append(np.mean(distances) if distances else 0)
        return [float("{:.3f}".format(dist)) for dist in average_distances]


def compute_single_property(smiles):
    """
    Computes multiple properties (SA score and QED) for a single SMILES string.

    Args:
        smiles (str): A SMILES string.

    Returns:
        tuple: A tuple containing the SA score and QED value.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    sa_score = sascorer.calculateScore(mol)
    qed_value = QED.qed(mol)

    return sa_score, qed_value


def is_drug_like_and_synthesizable(properties):
    """
    Checks if a molecule is drug-like and synthesizable based on its properties.

    Args:
        properties (tuple): A tuple containing SA score and QED value.

    Returns:
        bool: True if the molecule is drug-like and synthesizable, False otherwise.
    """
    if properties is None or len(properties) != 2:
        return False

    sa_score, qed = properties
    return qed >= 0.6 and sa_score <= 4.0


def calculate_metrics(all_smiles, valid_smiles):
    """
    Calculates validity, uniqueness, diversity, and quality metrics for a list of SMILES strings.

    Args:
        all_smiles (list): All generated SMILES strings.
        valid_smiles (list): Valid SMILES strings.

    Returns:
        tuple: Validity, uniqueness, diversity, and quality metrics as percentages.
    """
    if not all_smiles:
        return 0.0, 0.0, 0.0, 0.0

    validity = len(valid_smiles) / len(all_smiles) * 100
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles) * 100 if valid_smiles else 0.0

    diversity = np.mean(calculate_average_tanimoto(valid_smiles)) if len(valid_smiles) > 1 else 0.0

    properties_list = compute_properties(unique_smiles)
    quality_molecules = sum(1 for props in properties_list if props and is_drug_like_and_synthesizable(props))
    quality = (quality_molecules / len(all_smiles)) * 100 if all_smiles else 0.0
    return validity, uniqueness, diversity, quality


def is_valid_smiles(smiles, exclude_salts=True):
    """
    Validates a SMILES string by checking if it can be converted to an RDKit molecule.

    Args:
        smiles (str): A SMILES string.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        if smiles.find(".") != -1  and exclude_salts:#
            return None
        if len(smiles)==0:
            return None
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True)
    except:
        return None


def remove_salt(smiles):
    """
    Removes salts and keeps only the largest fragment from a SMILES string.

    Args:
        smiles (str): A SMILES string.

    Returns:
        str: SMILES string of the largest fragment.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fragments = Chem.GetMolFrags(mol, asMols=True)
    largest_fragment = max(fragments, key=lambda f: f.GetNumAtoms()) if fragments else mol

    return Chem.MolToSmiles(largest_fragment)


def get_canonical(smiles):
    """
    Converts a SMILES string to its canonical form.

    Args:
        smiles (str): A SMILES string.

    Returns:
        str: The canonical SMILES string, or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
