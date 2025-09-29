from rdkit import RDLogger, Chem

RDLogger.DisableLog("rdApp.*")
from .rBRICS_public import *
from collections import defaultdict
from collections import defaultdict
import re
import random
from collections import Counter
from contextlib import suppress

def remove_stereochemistry(smiles):
    """
    Removes the stereochemistry information from a given SMILES string.

    This function takes a SMILES string as input, parses it into an RDKit molecule,
    removes the stereochemistry from the molecule, and returns a new SMILES string
    of the modified molecule without stereochemistry.

    Parameters:
    - smiles (str): The SMILES string representing the chemical structure.

    Returns:
    - str: A new SMILES string of the molecule with stereochemistry removed.
    """

    # Parse the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Remove stereochemistry information from the molecule
    Chem.RemoveStereochemistry(mol)

    # Generate a new SMILES string from the modified molecule object
    new_smiles = Chem.MolToSmiles(mol)

    return new_smiles


def num_frags(smiles):
    """
    Calculates the number of possible fragments in a given SMILES string.

    This function identifies potential bonds for fragmentation in the molecule represented
    by the SMILES string, using the FindrBRICSBonds function. It returns the number fragments.

    Parameters:
    - smiles (str): The SMILES string representing the chemical structure.

    Returns:
    - int: The number of potential fragmentation points in the molecule plus one. Returns 101 if the SMILES string is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")

        # Use the FindrBRICSBonds function to identify bonds for fragmentation
        bonds = list(FindrBRICSBonds(mol))
        num = len(bonds)
    except:
        num = 0
    return num + 1


def frags2connections(fragment):
    """
    Converts a SMILES fragment string into a list of atoms and connections.

    This function parses a SMILES fragment string, identifies substitutions,
    and maps the connections between atoms in the fragment. It returns a list of atom symbols
    and a list of connections indicating how atoms in the molecule are connected.

    Parameters:
    - fragment (str): The SMILES string of the fragment

    Returns:
    - tuple(list, list): A tuple containing two lists; the first list contains the symbols of atoms in the fragment,
      and the second list contains the connections between these atoms. Returns empty lists if the fragment is invalid.
    """
    try:
        smiles = fragment
        substitutions = re.findall(r"\[\d+\*\]", smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid fragment SMILES string")
        id_frag = 0
        symbols = []

        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol == "*":
                if id_frag < len(substitutions):
                    atom_symbol = substitutions[id_frag]
                    id_frag += 1
            symbols.append(atom_symbol)

        connection_list = []
        for atom in mol.GetAtoms():
            bonded_atoms_info = [(bond.GetOtherAtom(atom).GetIdx(), bond.GetOtherAtom(atom).GetSymbol()) for bond in atom.GetBonds()]
            connection = [symbols[index] if letter == "*" else letter for index, letter in bonded_atoms_info]
            connection_list.append(connection)
        return symbols, connection_list
    except Exception as e:
        print(f"Error processing fragment {fragment}: {e}")
        return [], []


def frag2ais(input_string, connection_list, symbols):
    """
    Converts a SMILES fragment into atom-in-SMILES (AIS) notation.

    This function processes an input string representing a SMILES fragment, along with a list
    of connections and symbols for each fragment, to produce an atom-in-SMILES (AIS) notation string.
    The function identifies parts of the molecule marked with square brackets and replaces asterisks
    ('*') within these brackets with appropriate symbols from the 'symbols' list or connections
    from the 'connection_list'. This substitution is based on the position and count of asterisks.

    Parameters:
    - input_string (str): The input string representing the fragment in the AtomInSMILES notation
    - connection_list (list of lists): Each inner list contains connection information for one fragment
      of the molecule, specifically how each atom is connected to others.
    - symbols (list): A list of symbols that are used to replace the asterisks in the input string. Each
      symbol corresponds to a specific fragment.

    Returns:
    - str: The modified input string in atom-in-SMILES (AIS) notation, where asterisks in the original
      string are replaced with appropriate symbols from 'symbols' or connections from 'connection_list'.
    """

    # Splitting the input string to keep numbers and other brackets as well
    pattern = r"(\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\](?: \d | \( | \) )?)"
    elements = list(filter(None, re.split(pattern, input_string)))
    modified_elements = []
    pos_a = 0  # Position counter for square brackets in connection_list and symbols
    for element in elements:

        if "[" in element and "]" in element:  # Process only square bracket elements
            parts = element.split(";")
            if "*" in parts[0]:  # "*" is the first element
                replacement = symbols[pos_a]
                modified_element = element.replace("*", replacement, 1)
            elif "*" in parts[2]:  # "*" is the third element
                star_count = parts[2].count("*")  # Counting "*" in the third part
                connection_fragments = connection_list[pos_a]
                # Keep elements of the list connection_fragments if they have a "*"
                connection_fragments = [frag for frag in connection_fragments if "*" in frag]
                # Choose replacements based on star_count
                if star_count <= len(connection_fragments):
                    replacements = connection_fragments[:star_count]
                else:
                    # If there are more "*" than elements with "*", repeat the list elements
                    repeat_times = -(-star_count // len(connection_fragments))  # Ceiling division
                    full_list = (connection_fragments * repeat_times)[:star_count]
                    replacements = full_list
                # Joining all replacements for the current element
                replacement_str = "".join(replacements)
                modified_part = parts[2].replace("*", "{}", star_count).format(*replacements)
                modified_element = ";".join([parts[0], parts[1], modified_part])
            else:
                modified_element = element
            modified_elements.append(modified_element)
            pos_a += 1  # Increment pos_a for square bracket elements
        else:
            modified_elements.append(element)  # Add non-square bracket elements as is

    # Reconstructing the modified input string with all elements including numbers and other brackets
    ais = "".join(modified_elements)
    return ais



def BreakrBRICSBonds(mol, bonds=None, sanitize=True, silent=True):
    """
    Breaks specified r_BRICS bonds in a molecule and assigns unique bond IDs to dummy atoms.

    This function takes a molecule and  a list of the maximum number of bonds to break (useful for data augmentation).
    It breaks these bonds, replaces them with dummy atoms, and assigns unique bond IDs starting
    from [1*] for clarity and further processing. The function can optionally sanitize the resulting molecule.

    Parameters:
    - mol (rdkit.Chem.Mol): The RDKit molecule object to be fragmented.
    - bonds (int, optional): The maximum number of bonds to break.
      If not provided, the function uses Chem.FragmentOnBRICSBonds to automatically identify and break BRICS bonds.
    - sanitize (bool, optional): If True, sanitizes the resulting molecule. Defaults to True.
    - silent (bool, optional): If True, suppresses output messages. This parameter is present for signature compatibility
      but not used in the function logic.

    Returns:
    - rdkit.Chem.Mol: The fragmented molecule with dummy atoms representing broken bonds and unique bond IDs.
    """
    if not bonds:
        res = Chem.FragmentOnBRICSBonds(mol)
        if sanitize:
            Chem.SanitizeMol(res)
        return res
    eMol = Chem.EditableMol(mol)
    nAts = mol.GetNumAtoms()

    dummyPositions = []
    uniqueBondID = 1  # Start unique bond ID from 1
    for indices, dummyTypes in bonds:
        ia, ib = indices
        obond = mol.GetBondBetweenAtoms(ia, ib)
        bondType = obond.GetBondType()
        eMol.RemoveBond(ia, ib)

        # Assign unique bond ID to dummy atoms
        da, db = str(uniqueBondID), str(uniqueBondID)  # Use uniqueBondID for both atoms
        atoma = Chem.Atom(0)
        atoma.SetIsotope(int(da))
        atoma.SetNoImplicit(True)
        idxa = nAts
        nAts += 1
        eMol.AddAtom(atoma)
        eMol.AddBond(ia, idxa, bondType)

        atomb = Chem.Atom(0)
        atomb.SetIsotope(int(db))
        atomb.SetNoImplicit(True)
        idxb = nAts
        nAts += 1
        eMol.AddAtom(atomb)
        eMol.AddBond(ib, idxb, bondType)
        uniqueBondID += 1  # Increment the unique bond ID for the next broken bond

        if mol.GetNumConformers():
            dummyPositions.append((idxa, ib))
            dummyPositions.append((idxb, ia))
    res = eMol.GetMol()
    if sanitize:
        Chem.SanitizeMol(res)
    if mol.GetNumConformers():
        for conf in mol.GetConformers():
            resConf = res.GetConformer(conf.GetId())
            for ia, pa in dummyPositions:
                resConf.SetAtomPosition(ia, conf.GetAtomPosition(pa))
    return res


def smiles2frags(smiles, max_frags=5,canonical=True):
    """
    Breaks a molecule represented by a SMILES string into fragments by breaking random r_BRICS bonds.
    This function identifies BRICS bonds in a molecule represented by the given SMILES string,
    randomly selects and breaks up to a specified number of these bonds, ensuring that no more than
    N+1 fragments are generated. It utilizes BreakrBRICSBonds for the actual bond breaking and
    returns the resulting fragments as SMILES strings.

    Parameters:
    - smiles (str): The SMILES string representing the molecule to be fragmented.
    - max_frags (int, optional): The maximum number of fragments to generate, which limits the number
      of BRICS bonds to break. Defaults to 3, resulting in no more than 3 fragments.

    Returns:
    - list of str: A list containing the SMILES strings of the resulting fragments. Returns an empty list
      if the input SMILES string is invalid or an error occurs during processing.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")

        # Use the FindrBRICSBonds function to identify bonds for fragmentation
        bonds = list(FindrBRICSBonds(mol))

        # Randomly shuffle the bonds
        random.shuffle(bonds)

        # Limit the number of bonds to break to max_frags - 1, to ensure no more than max_frags fragments are generated
        bonds_to_break = bonds[: (max_frags - 1)]

        # Break the selected bonds and get the modified molecule
        modified_mol = BreakrBRICSBonds(mol, bonds=bonds_to_break, sanitize=True)

        # Extract fragments as separate molecules
        frags = Chem.GetMolFrags(modified_mol, asMols=True, sanitizeFrags=True)
        fragment_smiles = [Chem.MolToSmiles(frag, isomericSmiles=True, canonical=canonical) for frag in frags]

        return fragment_smiles
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return []

def order_fragments_by_attachment_points(fragments):
    """
    Order fragments by their attachment point numbers:
    - The first fragment should start with [1*]
    - The second fragment should start with [2*]
    - The third fragment should start with [3*]
    - And so on

    Args:
        fragments (list): List of fragment strings

    Returns:
        list: Reordered list of fragment strings
    """
    # If there's only one fragment or no fragments, return as is
    if len(fragments) <= 1:
        return fragments

    # Group fragments by their attachment point numbers
    attachment_point_groups = {}
    for fragment in fragments:
        # Extract the attachment point number
        match = re.match(r'\[(\d+)\*\]', fragment)
        if match:
            attachment_point = int(match.group(1))
            if attachment_point not in attachment_point_groups:
                attachment_point_groups[attachment_point] = []
            attachment_point_groups[attachment_point].append(fragment)

    # Order fragments by attachment point numbers
    ordered_fragments = []

    # Go through attachment points in numerical order (1, 2, 3, ...)
    for i in range(1, max(attachment_point_groups.keys()) + 1 if attachment_point_groups else 1):
        if i in attachment_point_groups and attachment_point_groups[i]:
            # Add the first fragment with this attachment point
            ordered_fragments.append(attachment_point_groups[i][0])

            # Remove the fragment we just added
            attachment_point_groups[i].pop(0)

    # Add any remaining fragments from all attachment point groups
    for ap in sorted(attachment_point_groups.keys()):
        ordered_fragments.extend(attachment_point_groups[ap])

    return ordered_fragments
def bridge_smiles_fragments(fragments, print_flag=False):
    """
    Reconnects fragmented molecules represented by a list of SMILES strings using isotopic labeling.

    This function takes a list of SMILES strings, each representing a fragment of a molecule where connections
    between fragments are indicated by isotopic labels. It combines these fragments into a single molecule,
    reconnects the fragments according to the isotopic labels, and returns a SMILES string of the reassembled molecule.

    The process involves:
    1. Concatenating the input fragments into a single SMILES string and creating a molecule from it.
    2. Identifying labeled atoms within this molecule.
    3. Inferring which atoms to connect based on these labels and their neighbors.
    4. Making connections between atoms to "bridge" the fragments back together.
    5. Removing the isotopic labels and returning the SMILES string of the reassembled molecule.

    Parameters:
    - fragments (list of str): A list of SMILES strings, where each string represents a molecule fragment.
      Isotopic labels in these fragments indicate where connections should be re-established.

    Returns:
    - str: A SMILES string representing the reassembled molecule after bridging the fragments.
      If the process fails or the input is invalid, it may return an unexpected result or an error.

    Important Warning:
    - This code has been designed for and tested on drug-like molecules, specifically using the REAL Diversity
      dataset. It generate 100% correct representation of the REAL diversity dataset. While it functions correctly for a
      wide range of drug-like compounds, it may produce errors or unexpected results when applied to more complex molecules.
      Users are advised to proceed with caution and validate the results, especially when working with molecules outside the scope
      of typical drug-like compounds.

    """
    # if num fragments is 1, return the fragment
    if len(fragments) == 1:
        return fragments[0]
    else:
        concatenated_smiles = ".".join(fragments)
        mol = Chem.MolFromSmiles(concatenated_smiles)
        if not mol and print_flag:
            print("Failed to create molecule from SMILES")
            return

        edmol = Chem.EditableMol(mol)

        # Mapping of isotopic labels to atom indices and their connected bond types, excluding structural isotopes
        labeled_atoms = {}
        for atom in mol.GetAtoms():
            label = atom.GetIsotope()
            if label > 0 and atom.GetSmarts().startswith("[") and "*" in atom.GetSmarts():
                labeled_atoms.setdefault(label, []).append(atom.GetIdx())

        connections_to_make = []
        for label, indices in labeled_atoms.items():
            if len(indices) != 2:
                continue
            atom1, atom2 = indices
            neighbors1 = [(n.GetIdx(), mol.GetBondBetweenAtoms(atom1, n.GetIdx()).GetBondType()) for n in mol.GetAtomWithIdx(atom1).GetNeighbors() if n.GetIsotope() == 0]
            neighbors2 = [(n.GetIdx(), mol.GetBondBetweenAtoms(atom2, n.GetIdx()).GetBondType()) for n in mol.GetAtomWithIdx(atom2).GetNeighbors() if n.GetIsotope() == 0]
            if neighbors1 and neighbors2:
                connections_to_make.append((neighbors1[0][0], neighbors2[0][0], neighbors1[0][1]))

        for atom1, atom2, bond_type in connections_to_make:
            edmol.AddBond(atom1, atom2, order=bond_type)

        to_remove = sorted([idx for indices in labeled_atoms.values() for idx in indices], reverse=True)
        for idx in to_remove:
            edmol.RemoveAtom(idx)

        new_mol = edmol.GetMol()
        final_smiles = Chem.MolToSmiles(new_mol, isomericSmiles=False, canonical=True)
        return final_smiles


def replace_identifiers(fragments):
    # Regular expression pattern to match [N*] identifiers
    pattern = r"(\[\d+\*\])"

    # Extract all identifiers from the fragments
    all_identifiers = [re.findall(pattern, fragment) for fragment in fragments]
    unique_identifiers = list(set(identifier for sublist in all_identifiers for identifier in sublist))

    # Shuffle the unique identifiers
    shuffled_identifiers = unique_identifiers.copy()
    random.shuffle(shuffled_identifiers)

    # Mapping from original to shuffled identifiers
    mapping = dict(zip(unique_identifiers, shuffled_identifiers))

    # Function to replace identifiers in a fragment using the mapping
    def replace_identifiers(fragment):
        return re.sub(pattern, lambda match: mapping[match.group()], fragment)

    # Replace the identifiers in each fragment
    replaced_fragments = [replace_identifiers(fragment) for fragment in fragments]

    return replaced_fragments


def reorder_fragments(fragments):
    """
    Reorders a list of chemical fragments based on specific label criteria.
    The first fragment should have at least one [1*], the second at least one [2*], and so on.
    Args:
        fragments (list of str): The list of chemical fragment strings with unique bond ID labels.
    Returns:
        list of str: The reordered list of fragments, considering the specified label criteria.
    """
    import re

    def has_label(fragment, label):
        return f"[{label}*]" in fragment

    def find_fragment_with_label(fragments, label):
        for fragment in fragments:
            if has_label(fragment, label):
                return fragment
        return None

    reordered = []
    remaining = fragments.copy()

    for i in range(1, len(fragments) + 1):
        fragment = find_fragment_with_label(remaining, i)
        if fragment:
            reordered.append(fragment)
            remaining.remove(fragment)
        else:
            break

    # Add any remaining fragments to the end
    reordered.extend(remaining)

    return reordered


def randomize_compatible_fragments(fragments):
    """
    Randomizes chemical fragments ensuring proper attachment point ordering.
    Each [1*] fragment must be immediately followed by a [2*] fragment.
    Returns original SMILES if no attachment points are present.
    Args:
        fragments (list of str): List of chemical fragments or SMILES
    Returns:
        list of str: Ordered and randomized fragments or original SMILES
    """
    # Handle input format
    if isinstance(fragments, str):
        fragments = fragments.split()

    # Quick check for attachment points
    has_attachments = False
    for fragment in fragments:
        if "[" in fragment and "*" in fragment:
            has_attachments = True
            break
    # If no attachment points, return original fragments
    if not has_attachments:
        return fragments
    # Initialize lists
    first_fragments = []  # Fragments starting with [1*]
    second_fragments = []  # Fragments starting with [2*]
    other_fragments = []  # Fragments with other attachment points or no attachment points
    for frag in fragments:
        # Find the first attachment point
        match = re.match(r"\[(\d+)\*\]", frag)
        if match:
            first_point = int(match.group(1))
            if first_point == 1:
                first_fragments.append(frag)
            elif first_point == 2:
                second_fragments.append(frag)
            else:
                other_fragments.append(frag)
        else:
            other_fragments.append(frag)
    # If no valid groups were found, return original
    if not first_fragments or not second_fragments:
        return fragments

    # Randomize each group
    random.shuffle(first_fragments)
    random.shuffle(second_fragments)
    # Create pairs of [1*] and [2*] fragments
    result = []
    for i in range(min(len(first_fragments), len(second_fragments))):
        result.append(first_fragments[i])
        result.append(second_fragments[i])

    # Add any remaining fragments
    result.extend(first_fragments[len(second_fragments) :])
    result.extend(second_fragments[len(first_fragments) :])
    result.extend(other_fragments)

    return result
from rdkit import Chem
from collections import Counter

def bridge_smiles_fragments_fix(fragments, print_flag=False):
    """
    Reconnect fragmented SMILES with [i*] attachment points,
    pairing both matching and odd-occurring labels to ensure full connectivity.
    """
    # 1) Quick return for single-fragment inputs
    if len(fragments) == 1:
        return fragments[0]

    # 2) Concatenate into a multi-fragment SMILES
    smiles = ".".join(fragments)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)  # parses & sanitizes  [oai_citation:0‡rdkit.org](https://www.rdkit.org/docs/GettingStartedInPython.html?utm_source=chatgpt.com)
    if mol is None:
        if print_flag:
            print("Failed to parse SMILES")
        return None

    # 3) Switch to an editable RWMol so we can both query and modify
    rw_mol = Chem.RWMol(mol)  # gives access to AddBond, GetBondBetweenAtoms, RemoveAtom…  [oai_citation:1‡rdkit.org](https://www.rdkit.org/docs/cppapi/classRDKit_1_1RWMol.html?utm_source=chatgpt.com)

    # 4) Map each dummy ([i*]) label to its atom indices
    labeled = {}
    for atom in rw_mol.GetAtoms():  # iterate all atoms  [oai_citation:2‡rdkit.org](https://www.rdkit.org/docs/source/rdkit.Chem.html?utm_source=chatgpt.com)
        iso = atom.GetIsotope()
        if iso > 0 and atom.GetSymbol() == '*':
            labeled.setdefault(iso, []).append(atom.GetIdx())  # group by isotope label  [oai_citation:3‡Python documentation](https://docs.python.org/3/library/collections.html?utm_source=chatgpt.com)

    # Helper: connect the two neighbors of two dummy atoms
    def connect_labels(idx1, idx2):
        neighs1 = [
            (nbr.GetIdx(), rw_mol.GetBondBetweenAtoms(idx1, nbr.GetIdx()).GetBondType())
            for nbr in rw_mol.GetAtomWithIdx(idx1).GetNeighbors()
            if nbr.GetSymbol() != '*'
        ]
        neighs2 = [
            (nbr.GetIdx(), rw_mol.GetBondBetweenAtoms(idx2, nbr.GetIdx()).GetBondType())
            for nbr in rw_mol.GetAtomWithIdx(idx2).GetNeighbors()
            if nbr.GetSymbol() != '*'
        ]
        if neighs1 and neighs2:
            n1, btype = neighs1[0]
            n2 = neighs2[0][0]
            # only add if bond doesn’t already exist
            if rw_mol.GetBondBetweenAtoms(n1, n2) is None:
                rw_mol.AddBond(n1, n2, order=btype)  # avoids duplicate-bond errors  [oai_citation:4‡rdkit.org](https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html?utm_source=chatgpt.com)

    # 5) First, connect any labels that appear exactly twice
    for iso, idxs in labeled.items():
        if len(idxs) == 2:
            connect_labels(idxs[0], idxs[1])

    # 6) Next, pair up any odd-occurring labels in sorted order
    odd = sorted([iso for iso, idxs in labeled.items() if len(idxs) == 1])
    for l1, l2 in zip(odd[0::2], odd[1::2]):  # zip-pairing of odd labels  [oai_citation:5‡W3Schools.com](https://www.w3schools.com/python/ref_func_zip.asp?utm_source=chatgpt.com)
        connect_labels(labeled[l1][0], labeled[l2][0])

    # 7) Remove all dummy atoms in reverse index order (to avoid reindexing bugs)
    to_remove = sorted([idx for idxs in labeled.values() for idx in idxs], reverse=True)  # safe deletion  [oai_citation:6‡rdkit.org](https://www.rdkit.org/docs/source/rdkit.Chem.html?utm_source=chatgpt.com)
    for idx in to_remove:
        rw_mol.RemoveAtom(idx)

    # 8) Return a fully connected, canonical SMILES
    return Chem.MolToSmiles(rw_mol.GetMol(), isomericSmiles=False, canonical=True)  # standardized output  [oai_citation:7‡GitHub](https://github.com/rdkit/rdkit/discussions/6335?utm_source=chatgpt.com)