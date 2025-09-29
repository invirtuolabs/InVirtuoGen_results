
import numpy as np
import rdkit.Chem as Chem
# TODO IMPLEMENT 2 ARMS PLUS 1 SCAFFOLD
from . import crossover as co
from . import mutate as mu


def choose_parents(population_mol, population_scores):
    population_scores = [s + 1e-4 for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    parents = np.random.choice(population_mol, p=population_probs, size=2)
    return parents


def reproduce(population, mutation_rate):
    population_mol = [Chem.MolFromSmiles(smiles) for smiles, _ in population]
    population_scores = [prop[0] for  smiles,prop in population]
    for tries in range(10):
        parent_a, parent_b = choose_parents(population_mol, population_scores)
        if parent_a is not None:
            parent_a = mu.mutate(parent_a, mutation_rate)
        if parent_a is not None:
            return Chem.MolToSmiles(parent_a), tries
    return None, tries