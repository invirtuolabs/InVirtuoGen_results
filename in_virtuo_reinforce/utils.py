
import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from in_virtuo_gen.utils.fragments import bridge_smiles_fragments, order_fragments_by_attachment_points, smiles2frags
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn
import os
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import wandb
from in_virtuo_gen.preprocess.preprocess_tokenize import custom_decode_sequence
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import re
import random
import torch
from collections import defaultdict


def count_attachment_points(fragment: str) -> int:
    """Count the number of attachment points [I*] in a fragment."""
    attachment_pattern = r"\[\d+\*\]"
    return len(re.findall(attachment_pattern, fragment))


class PromptTimeDataset(Dataset):
    """
    Wraps a base dataset of (ids, prompts) to emit, for each
    pair, n_steps different t values in [start_t, 1).
    """

    def __init__(self, base_dataset, start_t: float, dt: float):
        self.base = base_dataset
        self.start_t = start_t

        self.dt = dt
        # number of diffusion-time samples per example
        self.n_steps = int((1.0 - start_t) / dt)
        assert self.n_steps > 0, "dt too large or start_t >= 1"
        self.total_len = len(self.base) * self.n_steps

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # which base example
        sample_idx = idx // self.n_steps
        # which timestep for this example
        step_idx = idx % self.n_steps
        # compute t in [start_t, 1)
        t = self.start_t + step_idx * self.dt + random.uniform(0, self.dt)
        ids, scores = self.base[sample_idx]
        return ids, scores, torch.tensor(t, dtype=torch.float)


def make_dataloader(base_dataset, start_t, dt, batch_size, **dl_kwargs):
    ds = PromptTimeDataset(base_dataset, start_t, dt)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, **dl_kwargs)


def visualize_top_smiles(smiles_list, top_n=10, target="CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O", prompts=False, pairs=False, scores=None, oracle_name="", device=0, prefix=""):
    """
    Plot e top N generated SMILES by mean score.

    Parameters:
    - smiles_list: list of generated SMILES
    - top_n: number of top SMILES to display
    - target: target SMILES to display first
    """
    os.makedirs(f"plots/tdc/{oracle_name}", exist_ok=True)

    mols = []
    legends = []
    # Add target first
    target_mol = Chem.MolFromSmiles(target)
    if target_mol:
        mols.append(target_mol)
        legends.append("Target\n" + target)
    # Then each fragment
    for score, smi in zip(scores, smiles_list[:top_n]):  # type: ignore[attr-defined]
        m = Chem.MolFromSmiles(smi.replace(" ", ""))
        if m:
            mols.append(m)
            legends.append("Reward: %.2f" % score + "\n" + smi)  # "i=%d" % score[1] +
    from rdkit.Chem.Draw import rdMolDraw2D
    # 4) Draw grid (1 + top_n molecules)
    #    Adjust molsPerRow to fit nicely (e.g. 5 per row)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.legendFontSize = 25  # Set legend font size here
    opts.atomLabelFontSize = 14
    opts.bondLineWidth = 2
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=5, subImgSize=(300, 300), drawOptions=opts)
    try:
        img.save(f"plots/tdc/{oracle_name}/top_smiles_{device}.pdf" if not prefix else f"plots/tdc/{oracle_name}/{prefix}_{device}.pdf", format="pdf")
    except:
        pass
    plt.close()



def decompose_smiles(smi, max_frags=5, sort=False):
    try:
        frags = order_fragments_by_attachment_points(smiles2frags(smi, max_frags=max_frags)) if sort else smiles2frags(smi, max_frags=max_frags)
    except:
        frags = smiles2frags(smi, max_frags=max_frags)
        import traceback

        traceback.print_exc()
    return frags


def sort_frags(frags, tokenizer, ids=True, return_frags=False):
    if ids:
        frags = [decode(f, tokenizer) for f in frags]

    frags = order_fragments_by_attachment_points(frags)
    if return_frags:
        return [tokenizer.encode(f) for f in frags]
    frags = tokenizer.encode(" ".join(frags))

    return frags


def randomize_smiles(smiles: str, random_seed=None) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    atoms = list(mol.GetAtoms())  # type: ignore[attr-defined]
    idx = list(range(len(atoms)))
    random.seed(random_seed)
    random.shuffle(idx)
    mol = Chem.RenumberAtoms(mol, idx)
    return Chem.MolToSmiles(mol, canonical=False)


def augment_fragments(frags, num_augmentations=1):

    attempt_count = 0
    augmented_results = []
    while len(augmented_results) < num_augmentations and attempt_count < (100):
        try:
            if len(frags) > 1:
                if num_augmentations > 1:
                    smiles = bridge_smiles_fragments(frags)
                    smiles = randomize_smiles(smiles)  # type: ignore[attr-defined]
                frags = smiles2frags(smiles, max_frags=5, canonical=True)
                random.shuffle(frags)
                new_smiles = Chem.CanonSmiles(bridge_smiles_fragments(frags))

                augmented_results.append(" ".join(frags))
        except Exception as e:
            print(f"[augment_fragments] Fragment error for  {str(e)}")
        attempt_count += 1

    return augmented_results[:num_augmentations]


def decode(ids, tokenizer):
    seq = [tokenizer.decode(id) for id in ids]
    return "".join(seq)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Parameter(torch.ones(hidden_dim) * 1 / 50)  # context vector
        self.out = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, H, mask=None):
        """
        H: Tensor of shape (batch, seq_len, hidden_dim)
        mask: Optional BoolTensor of shape (batch, seq_len) where True=valid
        """

        z = self.norm(H) * mask.unsqueeze(-1)  # type: ignore[attr-defined]

        return self.out(torch.cat((z.sum(1) / mask.float().sum(1, keepdim=True), (mask.float().sum(1, keepdim=True) - 50) / 5), dim=-1))  # , alphas # type: ignore[attr-defined]


class ExperienceReplay:
    """Prioritized experience replay for highest scored sequences"""

    def __init__(self, max_size=1000, device="cuda"):
        self.memory = []
        self.max_size = max_size
        self.device = device

    def add_experience(self, experience):
        """Add new experiences to memory
        Args:
            experience: list of (sequence, score, prior_logprob) tuples
        """
        self.memory.extend(experience)

        # Remove duplicates based on sequence
        seen = set()
        unique_memory = []
        for exp in self.memory:
            seq_tuple = tuple(exp[0].tolist()) if torch.is_tensor(exp[0]) else tuple(exp[0])
            if seq_tuple not in seen:
                seen.add(seq_tuple)
                unique_memory.append(exp)
        self.memory = unique_memory

        # Keep only top scoring experiences
        if len(self.memory) > self.max_size:
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[: self.max_size]

    def sample(self, n):
        """Sample n experiences with probability proportional to score"""

        # Compute sampling probabilities based on scores
        ranks = np.arange(len(self.memory), dtype=np.float64)+1

        # 3) Compute weights ∝ 1 / (κ·N + rank)
        denom = 0.001 * len(self.memory) + ranks
        weights = 1.0 / denom
        probs = weights / weights.sum()
        # probs = scores / scores.sum()

        # Sample without replacement
        indices = np.random.choice(len(self.memory), size=min(n, len(self.memory)), replace=False, p=probs)
        sampled = [self.memory[i] for i in indices]

        sequences = [exp[0] for exp in sampled]
        scores = np.array([exp[1] for exp in sampled])
        # prior_logprobs = np.array([exp[2] for exp in sampled])
        unis = [exp[2] for exp in sampled]
        return sequences, scores, unis

    def __len__(self):
        return len(self.memory)


import numpy as np
from typing import Dict, List
import numpy as np

class PeakSeekerBandit:
    def __init__(self, prior_probs, lengths,
                 tau=0.5, floor=0.02,
                 q=0.9,           # target quantile
                 eta_q=0.1,       # quantile learning rate
                 w_best=0.6,      # weight on best-so-far
                 w_quant=0.4,     # weight on high-quantile
                 neigh_bw=3.0,    # neighborhood bandwidth
                 ucb_c=0.1,       # exploration bonus
                 **kwargs):

        self.prior = np.asarray(prior_probs, dtype=float)
        self.prior /= self.prior.sum()
        self.lengths = np.asarray(lengths, dtype=int)

        self.tau = float(tau)
        self.floor = float(floor)
        self.q = float(q)
        self.eta_q = float(eta_q)
        self.w_best = float(w_best)
        self.w_quant = float(w_quant)
        self.neigh_bw = float(neigh_bw)
        self.ucb_c = float(ucb_c)

        n = len(self.lengths)
        self.N = np.zeros(n, dtype=int)
        # Initialize to 0 instead of -inf to avoid NaN
        self.best = np.zeros(n, dtype=float)
        self.qhat = np.zeros(n, dtype=float)
        self.t = 1
        self.Lbest = None
        self._rbest = 0.0

    def _nearest_arm(self, L):
        return int(np.argmin(np.abs(self.lengths - int(L))))

    def _update_quantile(self, i, r):
        # Correct quantile SGD update (pinball loss gradient)
        if r >= self.qhat[i]:
            self.qhat[i] += self.eta_q * self.q
        else:
            self.qhat[i] -= self.eta_q * (1 - self.q)
        # Clip to valid range
        self.qhat[i] = np.clip(self.qhat[i], 0.0, 1.0)

    def _compute_scores(self):
        # For unvisited arms, use prior as fallback
        s = np.zeros(len(self.lengths))

        # Only use best/quantile scores for visited arms
        visited = self.N > 0

        if np.any(visited):
            # For visited arms: use best + quantile
            s[visited] = (self.w_best * self.best[visited] +
                         self.w_quant * self.qhat[visited])

            # For unvisited arms: use prior-based scores
            # Scale to match typical reward range [0, 1]
            s[~visited] = 0.5 * np.log(self.prior[~visited] + 1e-12)
        else:
            # At initialization (no visits yet): use log prior
            # This ensures we sample according to prior initially
            s = np.log(self.prior + 1e-12)
            # No need for UCB or neighborhood at t=1
            return s

        # UCB bonus for exploration (after initial phase)
        with np.errstate(divide='ignore', invalid='ignore'):
            ucb = self.ucb_c * np.sqrt(np.log(max(self.t, 2)) / np.maximum(self.N, 1))
            ucb[~visited] = self.ucb_c * 2.0  # Higher bonus for unvisited
        s = s + ucb

        # Neighborhood bump around global best
        if self.Lbest is not None:
            d = np.abs(self.lengths - self.Lbest)
            neigh = np.exp(-d**2 / (2 * max(self.neigh_bw**2, 0.01)))
            s = s + 0.2 * neigh  # Scaled neighborhood contribution

        return s

    def _probs(self):
        s = self._compute_scores()

        # Ensure no infinities
        s = np.clip(s, -100, 100)

        # Temperature-scaled softmax
        z = (s - np.max(s)) / max(self.tau, 1e-6)
        z = np.clip(z, -30, 30)

        p = np.exp(z)
        p = p / p.sum()

        # Floor probability
        p = (1.0 - self.floor) * p + self.floor / len(p)

        # Final normalization
        return p / p.sum()

    def select_length(self):
        p = self._probs()
        i = np.random.choice(len(p), p=p)
        return int(self.lengths[i])

    def update(self, sampled_length, reward, realized_length=None):
        if realized_length is None:
            realized_length = sampled_length

        i = self._nearest_arm(realized_length)
        r = float(np.clip(reward, 0.0, 1.0))

        self.N[i] += 1
        self.best[i] = max(self.best[i], r)
        self._update_quantile(i, r)

        # Track global best for neighborhood shaping
        if r > self._rbest:
            self._rbest = r
            self.Lbest = int(realized_length)

        self.t += 1

    def current_probs(self):
        return self._probs()

    def get_stats(self):
        """Return current statistics for debugging"""
        return {
            'best_rewards': self.best,
            'quantiles': self.qhat,
            'counts': self.N,
            'global_best_length': self.Lbest,
            'global_best_reward': self._rbest,
            'current_probs': self._probs()
        }
    def plot_distribution(self, save_path=None, log_wandb=False, global_step=0, smiles_list=None):
        import matplotlib.pyplot as plt
        try:
            import wandb
        except ImportError:
            wandb = None

        p_cur = self._probs()
        x = self.lengths
        w = 0.8

        pastel_colors = ["#AEC6CF", "#FFB347", "#98D8C8", "#F4B6C2"]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Panel 1: Actual generated lengths histogram
        if smiles_list:
            lengths = [len(s) for s in smiles_list]
            axs[0,0].hist(lengths, bins=range(min(lengths), max(lengths)+2),
                        color=pastel_colors[0], edgecolor="gray", alpha=0.7)
        axs[0,0].set_title("Generated Lengths")
        axs[0,0].set_xlabel("Sequence length")
        axs[0,0].set_ylabel("Count")

        # Panel 2: Initial prior distribution
        axs[0,1].bar(x, self.prior, color=pastel_colors[1], edgecolor="gray", width=w, alpha=0.7)
        axs[0,1].set_title("Initial Prior π₀")
        axs[0,1].set_xlabel("Sequence length")
        axs[0,1].set_ylabel("Probability")

        # Panel 3: Current sampling distribution
        axs[1,0].bar(x, p_cur, color=pastel_colors[2], edgecolor="gray", width=w, alpha=0.7)
        # Highlight global best length if exists
        if self.Lbest is not None:
            best_idx = self._nearest_arm(self.Lbest)
            axs[1,0].bar(x[best_idx], p_cur[best_idx], color='red', edgecolor="darkred",
                        width=w, alpha=0.9, label=f'Best L={self.Lbest}')
            axs[1,0].legend()
        axs[1,0].set_title("Current Sampling π")
        axs[1,0].set_xlabel("Sequence length")
        axs[1,0].set_ylabel("Probability")

        # Make y-axes consistent for the two probability plots
        max_prob = max(self.prior.max(), p_cur.max()) * 1.1
        axs[0,1].set_ylim(0, max_prob)
        axs[1,0].set_ylim(0, max_prob)

        # Panel 4: Peak metrics - best rewards and quantiles
        x_pos = np.arange(len(x))
        width = 0.35

        # Only plot for visited arms
        visited = self.N > 0
        if np.any(visited):
            axs[1,1].bar(x_pos[visited] - width/2, self.best[visited], width,
                        label='Best reward', color=pastel_colors[3], edgecolor="gray", alpha=0.7)
            axs[1,1].bar(x_pos[visited] + width/2, self.qhat[visited], width,
                        label=f'q={self.q:.1f} quantile', color=pastel_colors[2], edgecolor="gray", alpha=0.7)

            # Mark global best
            if self.Lbest is not None:
                best_idx = self._nearest_arm(self.Lbest)
                axs[1,1].scatter(x_pos[best_idx], self.best[best_idx],
                               color='red', s=100, zorder=5, marker='*',
                               label=f'Global best: {self._rbest:.3f}')

        axs[1,1].set_title("Peak Metrics (Best & Quantiles)")
        axs[1,1].set_xlabel("Sequence length")
        axs[1,1].set_ylabel("Reward")
        axs[1,1].set_xticks(x_pos[::max(1, len(x)//10)])  # Show subset of ticks if many arms
        axs[1,1].set_xticklabels(x[::max(1, len(x)//10)])
        axs[1,1].set_ylim(0, 1.05)
        axs[1,1].legend(loc='upper right', fontsize=8)

        # Light grid for readability
        for ax in axs.flatten():
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

        plt.suptitle(f"Peak-Seeking Length Distribution (Step {global_step})", fontsize=12)
        plt.tight_layout()

        if log_wandb and wandb and wandb.run:
            wandb.log({"bandit_distribution": wandb.Image(plt)}, step=global_step)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
class SoftmaxBandit:
    def __init__(
        self,
        prior_probs,  # initial prior, sums to 1
        lengths,  # list/array of actual lengths
        lr=0.1,  # Q-learning rate
        beta=0.95,  # how strongly to stick to old prior
        mean_reward=0.0,
    ):
        self.prior = np.array(prior_probs, dtype=float)
        assert np.all(self.prior >= 0) and abs(self.prior.sum() - 1.0) < 1e-6

        self.lengths = np.array(lengths)
        self.index_of = {l: i for i, l in enumerate(self.lengths)}
        self.max_length = self.lengths.max()
        self.lr = lr
        self.beta = beta
        self.Q = np.ones_like(self.prior)*mean_reward
        self.mean_reward = mean_reward

        self._history = []

    def _compute_probs(self):
        logits = np.log(self.prior + 1e-8) + (self.Q )
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        # Ensure minimum 1% probability per arm
        probs = 0.95 * probs + 0.05 / len(probs)
        return probs

    def select_length(self):
        p = self._compute_probs()
        arm = np.random.choice(len(p), p=p)
        self._history.append(p.copy())
        return int(self.lengths[arm])  # +random.randint(-2,2)

    def update(self, length, reward):
        # 1) Q‐update
        length = min(length, self.max_length)
        arm = self.index_of[length]
        self.Q[arm] += self.lr * (reward - self.Q[arm])

    #     # 2) compute new posterior p
    #     # p = self._compute_probs()

    #     # # 3) blend into prior
    #     # self.prior = self.beta * self.prior + (1 - self.beta) * p
    #     # self.prior /= self.prior.sum()  # renormalize

    def current_probs(self):
        return self._compute_probs()

    def plot_distribution(self, save_path=None, log_wandb=False, global_step=0, smiles_list=None):
        p_cur = self._compute_probs()

        x = self.lengths
        w = 0.8

        pastel_colors = ["#AEC6CF", "#FFB347", "#98D8C8"]

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        # Panel 1: Actual generated lengths histogram
        if smiles_list:
            lengths = [len(s) for s in smiles_list]
            axs[0,0].hist(lengths, bins=range(min(lengths), max(lengths)+2),
                        color=pastel_colors[0], edgecolor="gray", alpha=0.7)
        axs[0,0].set_title("Generated Lengths")
        axs[0,0].set_xlabel("Sequence length")
        axs[0,0].set_ylabel("Count")

        # Panel 2: Initial prior distribution
        axs[0,1].bar(x, self.prior, color=pastel_colors[1], edgecolor="gray", width=w, alpha=0.7)
        axs[0,1].set_title("Initial Prior π₀")
        axs[0,1].set_xlabel("Sequence length")
        axs[0,1].set_ylabel("Probability")

        # Panel 3: Current sampling distribution
        axs[1,0].bar(x, p_cur, color=pastel_colors[2], edgecolor="gray", width=w, alpha=0.7)
        axs[1,0].set_title("Current Sampling π")
        axs[1,0].set_xlabel("Sequence length")
        axs[1,0].set_ylabel("Probability")

        # Make y-axes consistent for the two probability plots
        max_prob = max(self.prior.max(), p_cur.max()) * 1.1
        axs[1,0].set_ylim(0, max_prob)

        # Panel 4: Q-values
        axs[1,1].bar(x, self.Q, color=pastel_colors[2], edgecolor="gray", width=w, alpha=0.7)
        axs[1,1].bar(x , np.log(self.prior + 1e-8), color=pastel_colors[1], edgecolor="gray", width=w, alpha=0.7)
        axs[1,1].set_title("Q-values")
        axs[1,1].set_xlabel("Sequence length")
        axs[1,1].set_ylabel("Q-value")

        # Light grid for readability
        for ax in axs.flatten():
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

        plt.suptitle(f"Length Distribution Evolution (Step {global_step})", fontsize=12)
        plt.tight_layout()

        if log_wandb and wandb.run:
            wandb.log({"bandit_distribution": wandb.Image(plt)}, step=global_step)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class GeneticPrompter:
    """
    Modular prompt builder: selects parent fragments, constructs crossover prompts,
    builds a vocab mask, and maintains/upates its own fragment vocabularies.
    """

    def __init__(
        self,
        tokenizer,
        bandit,
        offspring_size: int = 2,
        kappa: float = 0.001,
        always_ok: Any = None,
        max_frags: int = 5,
        K: int = 2,
        pad_id: int = 3,
        score_based=False,
        vocab_size=10,
        min_tanimoto_dist=0.7,
        start_rank=0,
    ):
        self.tokenizer = tokenizer
        self.bandit = bandit
        self.offspring_size = offspring_size
        self.kappa = kappa
        self.always_ok = list(always_ok) if always_ok is not None else []
        self.max_frags = max_frags
        self.pad_id = pad_id
        self.K = K
        # internal population for selection
        self.vocab = {}
        self.vocab_fps = {}
        self.population = []
        self.close = False
        self.start_rank = start_rank
        self.min_tanimoto_dist = min_tanimoto_dist
        self.score_based = score_based
        self.vocab_size = vocab_size
        # vocab of fragments (with attachment points)

    def update_with_score(self, smiles: str, score: float) -> None:
        """
        If just_update=False:
            (your normal logic, not shown here)
        If just_update=True:
            1) Find the single existing vocab entry whose fingerprint has the highest
               Tanimoto similarity to the new molecule (but only consider sims >= threshold).
            2) If none found → do nothing.
            3) If found and new score > old_score → delete that old entry and insert the new one.
               Otherwise → do nothing.
        """
        if smiles is None:
            print("similes is None")
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        fp_new = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # type: ignore[attr-defined]
        threshold = 1.0 - self.min_tanimoto_dist
        fragments = decompose_smiles(smiles, 5, sort=False)
        frag_key = tuple(self.tokenizer.encode(" ".join(fragments)))

        highest_sim = 0
        highest_sim_key = None
        highest_sim_score = 0

        for key, (fp_old, old_score) in self.vocab_fps.items():
            sim = DataStructs.TanimotoSimilarity(fp_new, fp_old)
            if sim > highest_sim:
                highest_sim = sim
                highest_sim_key = key
                highest_sim_score = old_score

        if highest_sim > threshold:
            if score > highest_sim_score:
                del self.vocab[highest_sim_key]
                del self.vocab_fps[highest_sim_key]
                self.vocab[frag_key] = score
                self.vocab_fps[frag_key] = (fp_new, score)
            return  # Exit early - just replace if similar molecule exists

        self.vocab[frag_key] = score
        self.vocab_fps[frag_key] = (fp_new, score)

        # prune vocab to top‐K, and keep fps in sync
        self._prune(self.vocab)
        keep = set(self.vocab)
        for k in list(self.vocab_fps):
            if k not in keep:
                del self.vocab_fps[k]

    def pad_seqs(self, seqs: List[torch.Tensor]) -> torch.Tensor:
        """Pad list of 1D tensors to same length using pad token."""
        if len(seqs) > 0:
            return torch.nn.utils.rnn.pad_sequence(
                seqs,
                batch_first=True,
                padding_value=self.pad_id,
            )
        else:
            return torch.tensor([])

    def _prune(self, vocab_scores: Dict[Tuple[int, ...], float]) -> None:
        """Keep only top-K items by score."""
        if len(vocab_scores) > self.vocab_size:
            # sort by descending score and keep top K keys
            topk = sorted(vocab_scores.items(), key=lambda x: x[1], reverse=True)[: self.vocab_size]
            # rebuild dict
            vocab_scores.clear()
            vocab_scores.update({k: v for k, v in topk})

    def build_prompts_and_masks(self, dev) -> Tuple[Any, List[List[int]]]:
        """
        Selects parents, builds crossover prompts, and returns:
          - prompts: LongTensor [B, Lmax]
          - raw_prompts: List of token-ID lists (for updating bandits/vocab)
        """
        pad_id = self.pad_id
        B = self.offspring_size
        V = len(self.tokenizer)

        # select parent pairs from internal population
        p1_list, p2_list, n_oracle = [], [], []
        for _ in range(self.offspring_size):


            i1, i2 = rank_based_sampling(self.vocab, 2, kappa=self.kappa, start_rank=self.start_rank    )
            i1 = torch.tensor(i1)
            i2 = torch.tensor(i2)
            p1_list.append(i1)
            p2_list.append(i2)
            n_oracle.append(self.bandit.select_length())

        P1 = self.pad_seqs(p1_list).to(dev)
        P2 = self.pad_seqs(p2_list).to(dev)

        prompt_tensors: List[torch.Tensor] = []
        for b in range(B):
            ids1 = P1[b][P1[b] != pad_id]
            ids2 = P2[b][P2[b] != pad_id]
            toks = self._fragment_prompter(ids1, ids2, n_oracle[b])
            prompt_tensors.append(torch.tensor(toks, device=dev))
        # pad and sort
        prompts = self.pad_seqs(prompt_tensors).long()

        return prompts, n_oracle

    def _fragment_prompter(self, p1_ids: torch.Tensor, p2_ids: torch.Tensor, n_oracle: int) -> List[int]:
        """
        Fragment-level crossover to flat token-ID list.
        """
        smi1 = bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, p1_ids).split())
        smi2 = bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, p2_ids).split())
        fr1 = decompose_smiles(smi1, self.max_frags, sort=True)
        fr2 = decompose_smiles(smi2, self.max_frags, sort=True)
        if random.random() < 0.5:
            random.shuffle(fr1)
            random.shuffle(fr2)
        frags = fr1[:-1] + fr2[-1:]
        kept = " ".join(frags)  # +" "
        return self.tokenizer.encode(kept)[:n_oracle]#


def rank_based_sampling(smiles_scores: Dict[str, float], n_select: int, kappa: float = 0.1, start_rank: int = 0) -> List[str]:
    """
    Rank‐based sampling without replacement over a smiles→score dict.

    Args:
      smiles_scores: mapping from SMILES string to its numeric score
      n_select:      how many SMILES to pick
      kappa:         small constant to control flatness of the distribution

    Returns:
      A list of selected SMILES (length = n_select or fewer if dict is smaller).
    """
    keys = list(smiles_scores.keys())
    N = len(keys)
    if n_select >= N:
        return keys[0],keys[0]  # return all if asking for too many

    # 1) collect scores in same order as keys
    scores = np.array([smiles_scores[s] for s in keys], dtype=np.float64)

    # 2) sort descending, get indices
    sorted_idx = np.argsort(-scores)  # best‐score first

    # 3) assign ranks (best rank=1, second=2, ..., worst=N)
    ranks = np.empty(N, dtype=np.int64)

    ranks[sorted_idx] = np.arange(start_rank, N+start_rank)

    # 4) compute weight ∝ 1 / (κ·N + rank)
    denom = kappa * N + ranks
    weights = 1.0 / denom
    weights /= weights.sum()

    # 5) draw without replacement
    chosen_idx = np.random.choice(N, size=2, replace=True, p=weights)
    # return the corresponding SMILES strings
    return [keys[i] for i in chosen_idx]


import numpy as np
from typing import Dict, List


def score_based_selection(smiles_scores: Dict[str, float], n_select: int, kappa: float = 0.001) -> List[str]:
    """
    Rank‐based sampling without replacement over a smiles→score dict.

    Args:
      smiles_scores: mapping from SMILES string to its numeric score
      n_select:      how many SMILES to pick
      kappa:         small constant to control flatness of the distribution

    Returns:
      A list of selected SMILES (length = n_select or fewer if dict is smaller).
    """
    keys = list(smiles_scores.keys())
    N = len(keys)
    if n_select >= N:
        return keys[:]  # return all if asking for too many

    # 1) collect scores in same order as keys
    scores = np.array([smiles_scores[s] for s in keys], dtype=np.float64)
    if scores.min() < 1e-4:
        scores += 1e-6
    scores = scores / scores.std() if len(scores) > 0 and scores.std() > 0 else scores
    # 5) draw without replacement
    chosen_idx = np.random.choice(N, size=n_select, replace=True, p=scores / scores.sum())
    # return the corresponding SMILES strings
    return [keys[i] for i in chosen_idx]