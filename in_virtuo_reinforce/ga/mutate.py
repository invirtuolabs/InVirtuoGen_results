# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been taken from jensengroup/GB_GA.
#
# Source:
# https://github.com/jensengroup/GB_GA/blob/master/mutate.py
#
# The license for this can be found in license_thirdparty/LICENSE_GA.
# ---------------------------------------------------------------

import random
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from . import crossover as co
rdBase.DisableLog('rdApp.*')


def delete_atom():
    choices = ['[*:1]~[D1:2]>>[*:1]', '[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]',
               '[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]',
               '[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]',
               '[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]']
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom():
    choices = [['single', ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'], 7 * [1.0 / 7.0]],
               ['double', ['C', 'N', 'O'], 3 * [1.0 / 3.0]],
               ['triple', ['C', 'N'], 2 * [1.0 / 2.0]]]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == 'single':
        rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X', '-' + new_atom)
    if BO == 'double':
        rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X', '=' + new_atom)
    if BO == 'triple':
        rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X', '#' + new_atom)

    return rxn_smarts


def insert_atom():
    choices = [['single', ['C', 'N', 'O', 'S'], 4 * [1.0 / 4.0]],
               ['double', ['C', 'N'], 2 * [1.0 / 2.0]],
               ['triple', ['C'], [1.0]]]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == 'single':
        rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X', new_atom)
    if BO == 'double':
        rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X', new_atom)
    if BO == 'triple':
        rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X', new_atom)

    return rxn_smarts


def change_bond_order():
    choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]', '[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
               '[*:1]#[*:2]>>[*:1]=[*:2]', '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond():
    return '[*:1]@[*:2]>>([*:1].[*:2])'


def add_ring():
    choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
               '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1']
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)


def change_atom(mol):
    choices = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts('[' + X + ']')):
        X = np.random.choice(choices, p=p)
    Y = np.random.choice(choices, p=p)
    while Y == X:
        Y = np.random.choice(choices, p=p)

    return '[X:1]>>[Y:1]'.replace('X', X).replace('Y', Y)

def change_atom(mol):
    """
    Pick an atom type X that actually exists in `mol`, then pick
    a different atom type Y, and return the SMARTS reaction string
    '[X:1]>>[Y:1]'.
    """
    # full choice set (SMARTS uses atomic numbers like '#6' for C)
    choices = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']
    probs   = np.array([0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14])

    # 1) find which of those actually occur in mol
    present = []
    present_probs = []
    for c, p in zip(choices, probs):
        smarts = f"[{c}]"
        sub = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(sub):
            present.append(c)
            present_probs.append(p)

    if not present:
        # no eligible atom to change
        return None
        raise ValueError("Molecule contains none of the target atom types.")

    # normalize probabilities on just the present ones
    present_probs = np.array(present_probs, dtype=float)
    present_probs /= present_probs.sum()

    # 2) sample X from present atom types
    X = np.random.choice(present, p=present_probs.tolist())

    # 3) sample Y ≠ X from the full choices (or from present if you prefer)
    Y_choices = [c for c in choices if c != X]
    Y_probs   = [probs[choices.index(c)] for c in Y_choices]
    Y_probs   = np.array(Y_probs, dtype=float)
    Y_probs  /= Y_probs.sum()
    Y = np.random.choice(Y_choices, p=Y_probs.tolist())

    # 4) build the reaction SMARTS
    #   e.g. '[#6:1]>>[#7:1]'
    rxn = f"[{X}:1]>>[{Y}:1]"
    return rxn

def mutate(mol, mutation_rate):
    if random.random() > mutation_rate:
        return mol

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for i in range(10):
        rxn_smarts_list = 7 * ['']
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring()
        rxn_smarts_list[4] = delete_atom()
        rxn_smarts_list[5] = change_atom(mol)
        rxn_smarts_list[6] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=p)

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for m in new_mol_trial:
            m = m[0]
            if co.mol_ok(m) and co.ring_ok(m):
                new_mols.append(m)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None