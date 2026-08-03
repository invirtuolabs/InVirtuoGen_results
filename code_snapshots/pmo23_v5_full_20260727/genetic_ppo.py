import heapq
import json
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightning
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import ConcatDataset

from lightning.pytorch import Trainer
import random
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import get_cosine_schedule_with_warmup
from in_virtuo_gen.utils.mol import compute_properties
import wandb
from in_virtuo_gen.models.invirtuofm import InVirtuoFM
from in_virtuo_gen.preprocess.preprocess_tokenize import custom_decode_sequence
from in_virtuo_gen.train_utils.metrics import evaluate_smiles
from in_virtuo_gen.utils.fragments import bridge_smiles_fragments, order_fragments_by_attachment_points, smiles2frags
from contextlib import nullcontext
from .pbo import BaseOptimizer
from .utils import SoftmaxBandit, decompose_smiles, visualize_top_smiles, ExperienceReplay, GeneticPrompter, PeakSeekerBandit
from .ga.ga import reproduce
from .train_utils import compute_kl_divergence, compute_seq_logp, sample_path, filter_valid_new, custom_collate
from dataclasses import asdict, dataclass
import sascorer
from rdkit import Chem
from rdkit.Chem import QED


@dataclass
class OptimizerConfig:
    """Configuration class for InVirtuoFMOptimizer"""

    ckpt: str
    oracle: str
    lr: float
    target: str
    offspring_size: int
    device: int
    start_t: float
    num_timesteps: int
    num_reinforce_steps: int

    big: bool
    alpha: float
    beta: float
    output_dir: str
    dt: float
    temperature: float
    use_prompter: bool
    clip_eps: float
    c_neg: float
    use_mutation: bool
    identifier: str
    reverse_kl: bool
    start_seed: int
    seed: int
    mutation_size: int
    task: str
    experience_replay_size: int
    train_mutation: bool
    aggressive_bandit: bool
    entropy_bonus: bool
    kl_coef: float
    # Additional configs that might be passed via *args, **kwargs
    max_oracle_calls: int
    no_sample_uni: bool
    use_prescreen: bool
    vocab_size: int
    first_pop_size: int
    rl_lr: float
    tot_offspring_size:int
    no_mask: bool
    no_bandit: bool
    classic_ga: bool
    eta: float
    start_rank: int = 0
    max_frags: int = 5
    pretrained_dir: Optional[str] = None
    save_pretrained_dir: Optional[str] = None
    batch_size: Optional[int] = None
    min_tanimoto_dist: float = 0.7
    single_prompt: bool = False
    parent_pool_size: int = 10000
    ga_elite_parents: int = 1
    # PeakSeekerBandit quantile step size (eta_q): how fast each arm's reward estimate
    # tracks what it observes. Was hardcoded at the class default of 0.1.
    bandit_lr: float = 0.3
    # Softmax temperature over arm scores: z = (s - max(s)) / tau, so SMALLER = sharper pi.
    # 0.3 concentrates ~0.81 of the mass within +-10 of the best arm against 0.65 at the old
    # 0.5; going lower locks on prematurely (0.15 converged on a wrong length in testing).
    bandit_tau: float = 0.3
    # Uniform probability mixed into every arm regardless of score. Was hardcoded at 0.01.
    bandit_floor: float = 0.005
    # PPO minibatch, and the chunk construct_dataset scores in. 0 keeps the old behaviour of
    # reusing tot_offspring_size, which is fine while that is ~100 but turns into thousands
    # of tiny forward passes once tot_offspring_size is small.
    ppo_batch_size: int = 0
    # Stage 1 opens with a one-off random init pool sized as a fraction of max_oracle_calls.
    # After that every round is offspring_size GA mutants on top of the model's own
    # offspring, of which the best elite_frac * offspring_size are trained on - so
    # init_random_frac is relative to the budget and elite_frac to offspring_size.
    # init_ga_frac adds a second, GA-generated init pool; off by default because it costs a
    # tenth of the budget without measurably improving on what the per-round GA already does.
    init_random_frac: float = 0.1
    init_ga_frac: float = 0.0
    # Fraction of the random init pool kept to seed the first reinforce step. The init pool
    # is thousands of untargeted samples, so training on all of it would drown the first
    # update; ordinary rounds are already selective by construction and are not cut.
    init_elite_frac: float = 0.1
    # 1.0 = no per-round selection: every molecule a round scores is trained on. Below 1 it
    # keeps the best elite_frac * offspring_size of the round instead.
    elite_frac: float = 1.0
    # eta for the random init pool only. h = min(eta*dt, 1-t), so a large eta collapses
    # generation to a single jump to the model's x_1 prediction - cheap, high-diversity
    # samples, which is all the init pool needs.
    init_eta: float = 999.0


def as_cpu_tensor(seq):
    """
    A 1-D CPU tensor for `seq`, whether it arrives as a list or as a tensor on the GPU.

    model.sample returns device tensors; torch.tensor() on one of those silently copies to
    the same device, so a naive conversion keeps every candidate of every round resident in
    VRAM. Candidates are only needed on the GPU inside the reinforce step, which moves them
    itself, so they are parked on the host in between.
    """
    if torch.is_tensor(seq):
        return seq.detach().to("cpu")
    return torch.as_tensor(seq, device="cpu")


def safe_qed_sa(smi):
    """
    QED/SA for a SMILES, as (qed, sa), or (nan, nan) when RDKit cannot handle the molecule.

    QED.qed re-sanitizes via Chem.RemoveHs and raises KekulizeException on molecules that
    MolFromSmiles accepted but cannot be kekulized. These values are logging-only, so a
    pathological mutant must not abort a run.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return float("nan"), float("nan")
    try:
        return QED.qed(mol), sascorer.calculateScore(mol)
    except Exception:
        return float("nan"), float("nan")


def sample_path(t, x_0, x_1, n=1):
    sigma_t = 1 - t**n
    source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t.unsqueeze(-1)
    return torch.where(condition=source_indices, input=x_0, other=x_1), source_indices


def filter_valid_new(valid, smiles, off_seqs, all_x_0, mol_buffer, n=None):

    valid_new = [i for i in valid if smiles[i] not in mol_buffer]
    valid_new_seqs = [off_seqs[i] for i in valid_new]
    valid_new_smiles = [smiles[i] for i in valid_new]
    valid_new_x_0 = [all_x_0[i] for i in valid_new]
    if n is not None:
        n = [n[i] for i in valid_new]
        return valid_new_smiles, valid_new_seqs, valid_new_x_0, n

    return valid_new_smiles, valid_new_seqs, valid_new_x_0


def compute_seq_logp(logits, ids, source_mask):
    # logits: [B, T, V], ids: [B, T]
    # 1) log-softmax over vocab
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    # 2) pick each token’s log-prob
    token_lp = log_probs.gather(-1, ids.unsqueeze(-1))  # [B, T, 1]
    token_lp = token_lp.squeeze(-1)  # [B, T]
    # 3) mask pads and sum over the sequence
    # [B, T]
    return (token_lp * source_mask).sum(dim=-1) / torch.max(torch.ones_like(source_mask[:, 0]), (source_mask.float().sum(dim=-1)))


class InVirtuoFMOptimizer(BaseOptimizer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        base_optimizer_kwargs = {"device": kwargs.get("device", "cpu"), "output_dir": kwargs.get("output_dir", "target_prop_opt"), "max_oracle_calls": kwargs.get("max_oracle_calls", 10000)}

        # Remove base optimizer args from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["device", "output_dir", "max_oracle_calls"] or k in OptimizerConfig.__dataclass_fields__}

        # Create config from provided arguments
        self.config = OptimizerConfig(**filtered_kwargs)
        print(f"self.config.max_oracle_calls: {self.config.max_oracle_calls}")

        # Initialize base optimizer
        super().__init__(**base_optimizer_kwargs)

        # Initialize wandb with config
        wandb.init(project="tdc-ppo-finetune", config=asdict(self.config))
        # Initialize instance variables from config
        self.device = torch.device(self.config.device)
        self.sa, self.qed, self.qed_ga, self.sa_ga, self.ga_scores, self.ga_tries = [], [], [], [], [], []
        self.global_step = 0
        self.stop_counter = 0
        self.ga_best_counter = 0
        # Rounds in which the elite parent ran out of novel mutants and the draw widened.
        self.ga_widen_counter = 0
        self.ga_start = 0
        # Model initialization
        ckpt = self.config.ckpt
        pretrained = None
        self.elite = None  # stage-1 top-K molecules by score, used to seed stage 2
        if self.config.pretrained_dir:
            pretrained_path = os.path.join(self.config.pretrained_dir, f"{self.config.oracle}.ckpt")
            pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if pretrained["oracle"] != self.config.oracle:
                raise ValueError(f"Pretrained checkpoint is for {pretrained['oracle']}, not {self.config.oracle}")
            self.config.use_prompter |= pretrained.get("prompter_vocab") is not None
            self.elite = pretrained.get("elite")

        self.model = InVirtuoFM.load_from_checkpoint(ckpt, map_location="cpu").to(self.device)
        if pretrained:
            self.model.load_state_dict(pretrained["model_state_dict"])

        self.prior = InVirtuoFM.load_from_checkpoint(ckpt, map_location="cpu").to(self.device)
        if pretrained:
            self.prior.load_state_dict(pretrained["model_state_dict"])
        self.prior.requires_grad_(False)
        self.prior.eval()
        self.tokenizer = self.model.tokenizer
        loaded = torch.load("configs/zinc_dist.pt")
        self.seq_lengths = loaded["unique_lengths"]
        self.seq_prior_probs = loaded["probs"]
        self.seq_len_dist = Categorical(self.seq_prior_probs.clone())
        # Initialize bandit
        if self.config.use_prescreen:
            mean = pd.read_csv("in_virtuo_reinforce/vocab/zinc250k.csv").sort_values(by=self.config.oracle, ascending=False)[: self.config.first_pop_size][self.config.oracle].mean()
        else:
            mean = 0.0
        self.bandit = SoftmaxBandit(prior_probs=self.seq_prior_probs.cpu().numpy(), lengths=self.seq_lengths, lr=self.config.lr, beta=self.config.beta, mean_reward=mean)
        if self.config.aggressive_bandit:
            self.bandit = PeakSeekerBandit(prior_probs=self.seq_prior_probs.cpu().numpy(), lengths=self.seq_lengths, floor=self.config.bandit_floor, w_best=0.6, w_quant=0.8, neigh_bw=3.0, ucb_c=0.1, eta_q=self.config.bandit_lr, tau=self.config.bandit_tau)
        if pretrained and pretrained.get("bandit_state"):
            self.bandit.__dict__.update(pretrained["bandit_state"])
        if self.config.use_prompter:
            self.prompter = GeneticPrompter(
                tokenizer=self.tokenizer,
                bandit=self.bandit,
                offspring_size=int(self.config.offspring_size * 2),
                vocab_size=self.config.vocab_size,
                kappa=0.001,
                always_ok=None,
                max_frags=self.config.max_frags,
                pad_id=3,
                score_based=False,
                K=1,  # if self.config.use_prescreen else 2,
                start_rank=self.config.start_rank,
                min_tanimoto_dist=self.config.min_tanimoto_dist,
            )
            if pretrained:
                self.prompter.vocab = pretrained["prompter_vocab"]
                self.prompter.vocab_fps = pretrained["prompter_vocab_fps"]
                top_vocab = dict(sorted(self.prompter.vocab.items(), key=lambda item: item[1], reverse=True)[: self.config.vocab_size])
                self.prompter.vocab = top_vocab
                self.prompter.vocab_fps = {
                    key: value for key, value in self.prompter.vocab_fps.items() if key in top_vocab
                }
            self.prompter.combine_prompts = not self.config.single_prompt

        self.model_opt = optim.Adam(self.model.parameters(), lr=self.config.rl_lr, betas=(0.9, 0.999))
        # One reinforce call issues num_reinforce_steps optimizer steps, and a round spends
        # model_offspring_per_round + ga_per_round oracle calls, after the init pool has
        # taken its share. Deriving the schedule length from that keeps the cosine decay
        # aligned however the round is split between the model and the GA.
        calls_per_round = max(1, self.model_offspring_per_round + self.ga_per_round)
        rounds = max(1, (self.config.max_oracle_calls - self.init_budget) // calls_per_round)
        num_steps = max(1, rounds * self.config.num_reinforce_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.model_opt, num_warmup_steps=min(10, max(1, num_steps // 10)), num_training_steps=num_steps
        )

        # Initialize trainer
        trainer = Trainer(max_steps=1, limit_train_batches=0.0, logger=False)
        self.model.trainer = trainer
        self.used_mutation = not self.config.use_mutation
        self.prev_validity = 1
        self.train_ids, self.train_prompts, self.train_scores, self.train_old_logprobs, self.train_x_0 = [], [], [], [], []
        # (seq, x_0, score) for everything the current round scored, model offspring and GA
        # mutants alike; drained by select_elite() at the end of the round.
        self.round_pool = []

        self.config.experience_replay_size = self.config.experience_replay_size
        self.experience = ExperienceReplay(max_size=self.config.experience_replay_size , device=self.device)
        self.c_neg = self.config.c_neg
        self.config.mutation_size = self.config.mutation_size if self.config.use_mutation else 0
        self.max_oracle_calls = self.config.max_oracle_calls

        self.prev_novelty = 1

    def set_c_neg(self, novelty):
        """
        Set the negative sample weight value based on the number of valid new sequences.
        Also controls the stop counter.
        """
        self.global_step += 1
        wandb.log({"train/c_neg": self.c_neg, "num_oracle_calls": len(self.mol_buffer.values())})
        if novelty >= 0.5 * (1 - self.config.start_t):
            self.c_neg = self.config.c_neg
            self.stop_counter -= 1
            self.stop_counter = max(0, self.stop_counter)
        # elif novelty > 0.1 and self.c_neg > 0:
        #     self.c_neg = 0
        #     self.config.c_neg *= 0.99
        elif novelty < 0.05:
            # self.c_neg = 0
            # self.config.c_neg *= 0.9
            self.stop_counter += 1
        self.global_step += 1
        wandb.log({"train/c_neg": self.c_neg, "train/stop_counter": self.stop_counter, "num_oracle_calls": len(self.mol_buffer.values())})

    @property
    def ga_per_round(self):
        """Mutants a single round asks the GA for. 0 when mutation is off (e.g. stage 2)."""
        return self.config.tot_offspring_size if self.config.use_mutation else 0

    @property
    def ppo_batch(self):
        """Rows per PPO minibatch / per construct_dataset scoring chunk."""
        return max(1, self.config.ppo_batch_size or self.config.tot_offspring_size or 1)

    @property
    def model_offspring_per_round(self):
        """
        Offspring the model itself proposes each round: the full offspring_size.

        Stage 2 keeps the old tot_offspring_size cap, because it trains on everything it
        scores (elite_frac is forced to 1.0 there), so anything beyond one reinforce batch
        would be oracle-scored for nothing. In stage 1 the surplus is the point - it is what
        elite selection picks from.
        """
        n = self.config.offspring_size
        if self.config.pretrained_dir and self.config.tot_offspring_size:
            n = min(n, self.config.tot_offspring_size)
        return n

    @property
    def elite_per_round(self):
        """
        How many of a round's molecules reach the training set.

        At elite_frac >= 1 that is the whole round; below 1 it is elite_frac * offspring_size,
        so the cut scales with what the model proposes rather than with the pool it lands in.
        """
        if self.config.elite_frac >= 1.0:
            return self.model_offspring_per_round + self.ga_per_round
        return max(1, int(self.config.elite_frac * self.config.offspring_size))

    @property
    def reinforce_trigger(self):
        """
        Training set size that fires a reinforce step.

        A round yields elite_per_round molecules, which is below tot_offspring_size once
        elite_frac starts biting; gating on tot_offspring_size alone would then stall the
        model for several rounds at a time instead of training after every round.
        """
        return max(1, min(self.config.tot_offspring_size, self.elite_per_round))

    @property
    def init_budget(self):
        """Oracle calls the stage-1 init pools consume. 0 in stage 2, which seeds from the elite."""
        if self.config.pretrained_dir:
            return 0
        n = max(1, int(self.config.init_random_frac * self.config.max_oracle_calls))
        if self.config.use_mutation:
            n += int(self.config.init_ga_frac * self.config.max_oracle_calls)
        return n

    def select_elite(self):
        """
        Promote the best elite_frac of this round's scored molecules into the training set.

        Everything a round produces - the model's own offspring and the GA mutants - lands in
        self.round_pool first, so selection sees the whole round and can take a true top-k
        rather than each source deciding on its own against a stale threshold.
        """
        if not self.round_pool:
            return
        k = min(self.elite_per_round, len(self.round_pool))
        elite = heapq.nlargest(k, self.round_pool, key=lambda item: item[2])
        for seq, x_0, score in elite:
            self.train_ids.append(seq)
            self.train_scores.append(score)
            self.train_x_0.append(x_0)
        self.round_pool = []

    def run_ga(self, max_tries=10):
        """
        Mutate the buffer's elite with the mol-opt GA and score the mutants.

        A round asks the GA for ga_per_round molecules, so the candidates are collected first
        and handed to the oracle in one call instead of one call per mutant. The scored
        mutants go into round_pool, where they compete with the model's own offspring for the
        round's elite slots rather than being admitted against a threshold of their own.

        Parents come from the top ga_elite_parents of the buffer - rank 1 by default, which
        is what actually produces new global bests. Mutating only the champion eventually
        exhausts its novel neighbourhood, though, and a round that can no longer find an
        unseen mutant would burn its whole attempt budget issuing no oracle calls at all, so
        a long run of rejections widens the draw to the top parent_pool_size for the rest of
        the round.
        """
        self.used_mutation = True
        # Never mutate past the budget: score_smi silently returns 0 once the buffer is full,
        # which would look like a round of worthless molecules.
        quota = min(self.ga_per_round, max(0, self.config.max_oracle_calls - len(self.mol_buffer)))
        if quota <= 0:
            return
        # One nlargest pass per round, rather than the three full-buffer sorts that used to
        # sit inside the retry loop (O(n log n) on up to 100k entries, once per mutation).
        elite_pool = heapq.nlargest(
            max(1, self.config.ga_elite_parents), self.mol_buffer.items(), key=lambda x: x[1][0]
        )
        if not elite_pool:
            return
        parent_pool, widened = elite_pool, False
        stall_limit = max_tries * 10

        ga_smiles, seen, tries, stall = [], set(), 0, 0
        max_attempts = quota * max_tries
        while len(ga_smiles) < quota and tries < max_attempts:
            tries += 1
            smi, tries_temp = reproduce([random.choice(parent_pool)], 1)
            tries += tries_temp
            mol = Chem.MolFromSmiles(smi) if smi else None
            smi = Chem.MolToSmiles(mol) if mol is not None else None
            if smi is None or smi in seen or smi in self.mol_buffer:
                stall += 1
                if not widened and stall >= stall_limit:
                    parent_pool = heapq.nlargest(
                        max(self.config.parent_pool_size, 10), self.mol_buffer.items(), key=lambda x: x[1][0]
                    )
                    widened = True
                continue
            stall = 0
            seen.add(smi)
            ga_smiles.append(smi)
        if widened:
            self.ga_widen_counter += 1
        if not ga_smiles:
            return

        ga_scores = self.oracle(ga_smiles)
        best_so_far = max(self.scores) if self.scores else 0
        for smi, score in zip(ga_smiles, ga_scores):
            if score is None:
                continue
            self.ga_scores.append(score)
            qed_ga_, sa_ga_ = safe_qed_sa(smi)
            self.qed_ga.append(qed_ga_)
            self.sa_ga.append(sa_ga_)
            if score > best_so_far:
                self.ga_best_counter += 1
            new_seq = torch.tensor(self.tokenizer.encode(" ".join(decompose_smiles(smi))))
            if self.config.train_mutation:
                self.round_pool.append((new_seq, torch.randint(4, 203, (len(new_seq),)), score))
            if self.config.use_prompter and score > 0:
                self.prompter.update_with_score(smi, score)
                self.prompter.bandit.update(len(new_seq[new_seq != self.model.pad_token_id]), score)

        if len(self.ga_scores[self.ga_start:]) > 0:
            self.global_step += 1
            wandb.log(
                {
                    "ga/num_generated": len(self.ga_scores) - self.ga_start,
                    "ga/best_ga_scores": np.max(self.ga_scores[self.ga_start:]),
                    "num_oracle_calls": len(self.mol_buffer.values()),
                    "ga/tries": tries / max_tries,
                    "ga/qed": np.nanmean(self.qed_ga[self.ga_start:]),
                    "ga/sa": np.nanmean(self.sa_ga[self.ga_start:]),
                    "ga/best_ga_counter": self.ga_best_counter,
                    "ga/widen_counter": self.ga_widen_counter,
                },
            )
            self.ga_start = len(self.ga_scores)


    def evaluate_offspring(self, valid_new_seqs, valid_new_x_0, valid_new_smiles, n_valid=None):
        """
        Evaluate the offspring sequences.
        """
        for i, (seq, x_0, smi) in enumerate(zip(valid_new_seqs, valid_new_x_0, valid_new_smiles)):
            if self.config.classic_ga:
                counter = 0
                while counter<10:
                    smi = reproduce([(smi,[1])], 1)[0]

                    if smi and smi not in self.mol_buffer:
                        seq = self.tokenizer.encode(" ".join(decompose_smiles(smi)))
                        break
                    counter += 1
                if counter == 10:
                    continue

            # Hold candidates on the CPU. These come straight off model.sample, so keeping
            # them as GPU tensors pins a whole round's worth of sequences in VRAM alongside
            # the model; construct_dataset moves them back for the reinforce step anyway.
            seq = as_cpu_tensor(seq)
            x_0 = as_cpu_tensor(x_0)
            qed, sa = safe_qed_sa(smi)
            self.qed.append(qed)
            self.sa.append(sa)
            score = self.oracle(smi)
            self.scores.append(score)


            if score > 0:
                if self.config.use_prompter:
                    self.prompter.update_with_score(smi, score)  # type: ignore[attr-defined]
                    if not self.config.no_bandit:
                        self.prompter.bandit.update(len(seq[seq != self.model.pad_token_id]), score)
                else:
                    self.bandit.update(len(seq[seq != self.model.pad_token_id]), score)

            self.round_pool.append((seq, x_0, score))

        return

    def add_experience(self):
        """
        Add the experience to the experience replay buffer.
        """
        new_experience = []
        for i in range(len(self.train_ids)):
            seq = self.train_ids[i]
            uni_ = self.train_x_0[i]
            valid_len = (seq != self.model.pad_token_id).sum()
            seq_trimmed, uni_trimmed = seq[:valid_len], uni_[:valid_len]
            new_experience.append((seq_trimmed, self.train_scores[i], uni_trimmed))
        self.experience.add_experience(new_experience)


    def pad_seqs(self, ids):
        """
        Pad the sequences to the same length.
        """
        return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.model.pad_token_id).to(self.device)

    def construct_dataset(self, ids, scores, uni, timesteps=10):
        ids_tensor = self.pad_seqs([i.clone().detach() for i in ids]).to(self.device)
        uni_tensor = self.pad_seqs([torch.tensor(u) if self.config.no_sample_uni else torch.randint(4, 203, (len(u),), device=self.device) for u in uni]).to(self.device)
        scores_tensor = torch.tensor(scores, device=self.device)
        old_logprobs, x_ts_list, source_masks_list, t_list = [], [], [], []
        t_roll = torch.rand(len(ids_tensor), device=self.device)
        # Score the training set in chunks of tot_offspring_size. compute_logprobs holds
        # [B, L, L] attention masks and a [B, L, V] softmax, so running it over the whole
        # set at once made peak memory scale with the training set - which elite selection
        # can now make an order of magnitude larger than the reinforce batch.
        chunk = self.ppo_batch
        for i in range(timesteps):
            current_t = (t_roll / 10 + i * (1.0 - self.config.start_t) / timesteps) % 1
            current_t = current_t.clamp(0, 0.99)
            lp_parts, x_t_parts, mask_parts = [], [], []
            for s in range(0, len(ids_tensor), chunk):
                e = min(s + chunk, len(ids_tensor))
                with torch.no_grad():
                    old_lp, _, x_t, source_mask, _ = self.compute_logprobs(
                        ids_tensor[s:e], uni_tensor[s:e], current_t[s:e]
                    )
                lp_parts.append(old_lp.detach())
                x_t_parts.append(x_t)
                mask_parts.append(source_mask)
            old_logprobs.append(torch.cat(lp_parts, dim=0).unsqueeze(1))
            x_ts_list.append(torch.cat(x_t_parts, dim=0).unsqueeze(1).clone())
            source_masks_list.append(torch.cat(mask_parts, dim=0).unsqueeze(1).clone().to(self.device))
            t_list.append(current_t)
        ids_tensor = ids_tensor.repeat(timesteps, 1)
        scores_tensor = scores_tensor.repeat(timesteps)
        uni_tensor = uni_tensor.repeat(timesteps, 1)
        old_logprobs = torch.cat(old_logprobs, dim=0).squeeze(1)
        x_ts_all = torch.cat(x_ts_list, dim=0).squeeze(1)
        source_masks_all = torch.cat(source_masks_list, dim=0).squeeze(1)
        t_list = torch.cat(t_list, dim=0)
        new_dataset = TensorDataset(ids_tensor, scores_tensor, uni_tensor, old_logprobs, x_ts_all, source_masks_all, t_list)
        return new_dataset

    def construct_loader(self, ids, scores, uni):

        if self.config.experience_replay_size > 0 and len(self.experience) > 0:
            exp_seqs, exp_scores, exp_uni = self.experience.sample(len(self.experience))
            ids = ids + exp_seqs
            scores = scores + exp_scores.tolist()
            uni = uni + exp_uni
            combined = list(zip(ids, scores, uni))
            random.shuffle(combined)
            ids, scores, uni = map(list, zip(*combined))
        combined_dataset = self.construct_dataset(ids, scores, uni, self.config.num_timesteps)
        loader = DataLoader(
            combined_dataset, batch_size=self.ppo_batch, shuffle=True, collate_fn=custom_collate, drop_last=False
        )
        return loader

    def compute_logprobs(self, ids, uni, t, prior=False, x_t=None, source_mask=None):
        """Compute log probabilities under the prior model"""
        B, L = ids.size()
        mask = ids != self.model.pad_token_id
        attn = (~mask).unsqueeze(1).expand(B, L, L).float()
        attn = attn.masked_fill(attn.bool(), float("-inf")).unsqueeze(1)
        if x_t is None or source_mask is None:
            x_t, source_mask = sample_path(t, uni, ids)
        else:
            x_t = x_t.clone()
        source_mask = source_mask & mask if not self.config.no_mask else mask
        x_t[~mask] = self.model.pad_token_id

        # Get logitssample
        with torch.no_grad() if prior else nullcontext():
            # self.prior.train()
            logits = self.prior.model(x=x_t, t=t, attn_mask=attn, return_hidden=False) if prior else self.model.model(x=x_t, t=t, attn_mask=attn, return_hidden=False)
            logprobs_ = compute_seq_logp(logits, ids, source_mask.float())

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        step_entropy = dist.entropy() * source_mask.float()
        mask_sum = source_mask.float().sum(dim=-1)
        seq_entropy = step_entropy.sum(dim=1) / torch.clamp(mask_sum, min=1.0)

        return logprobs_, seq_entropy, x_t, source_mask, logits

    def reinforce(self, ids, scores, uni):
        """PPO-style reinforce with (optional) experience replay."""
        self.prior.eval()
        self.model.train()

        loader = self.construct_loader(ids, scores, uni)

        # reinforce/reward below is a single minibatch's mean, sampled from a shuffled loader,
        # so it is dominated by sampling noise. Log the round's whole elite set too, which is
        # the quantity that actually says whether training data is improving.
        if scores:
            self.global_step += 1
            wandb.log(
                {
                    "reinforce/elite_reward_mean": float(np.mean(scores)),
                    "reinforce/elite_reward_min": float(np.min(scores)),
                    "reinforce/elite_reward_max": float(np.max(scores)),
                    "reinforce/num_batches": len(loader),
                    "num_oracle_calls": len(self.mol_buffer.values()),
                },
            )

        num_batches = len(loader)
        self.model_opt.zero_grad()
        avg_grad_norm = 0
        # Averaged over every minibatch that contributed gradients to the coming optimizer
        # step. Reporting a single minibatch instead made these dominated by shuffling noise:
        # one batch is 100 of num_batches*100 rows, so reinforce/reward swung ~0.06
        # peak-to-peak on a training set whose true mean was not moving at all.
        acc = dict.fromkeys(("loss", "pg_loss", "kl_div", "ratio", "reward", "adv", "entropy"), 0.0)
        acc_n = 0
        log_every = max(1, len(loader) // self.config.num_reinforce_steps)
        for i, (batch_ids, batch_scores, batch_uni, batch_t, batch_old_lp, batch_x_ts, batch_masks) in enumerate(loader):

            agent_lp, entropy, _, _, agent_logits = self.compute_logprobs(
                batch_ids, batch_uni, batch_t, x_t=batch_x_ts, source_mask=batch_masks
            )
            if self.config.kl_coef > 0:
                _, _, _, _, reference_logits = self.compute_logprobs(
                    batch_ids, batch_uni, batch_t, prior=True, x_t=batch_x_ts, source_mask=batch_masks
                )
                kl_div = compute_kl_divergence(
                    agent_logits, reference_logits, batch_masks, reverse=self.config.reverse_kl
                )
            else:
                kl_div = agent_logits.new_zeros(())
            ratio = torch.exp((agent_lp - batch_old_lp))
            score_std = batch_scores.std()
            if batch_scores.numel() > 1 and not score_std < 1e-8:
                adv = (batch_scores - batch_scores.mean()) / (score_std + 1e-8)
            else:
                continue
            adv = torch.where(adv > 0, adv, self.c_neg * adv)*1/(1-batch_t**2)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv
            pg_loss = -torch.min(surr1, surr2).mean()
              # - 0.01 * entropy.mean()
            if self.config.entropy_bonus:
                loss = (pg_loss + self.config.kl_coef * kl_div - 0.01 * entropy.mean()) / num_batches
            else:
                loss = (pg_loss + self.config.kl_coef * kl_div) / num_batches
            if torch.isnan(loss) or torch.isinf(loss):
                print("loss is nan or inf")
                continue

            loss.backward()
            grad_norm = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None]).norm().item()
            avg_grad_norm += grad_norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2)

            acc["loss"] += loss.item()
            acc["pg_loss"] += pg_loss.item()
            acc["kl_div"] += kl_div.item()
            acc["ratio"] += ratio.mean().item()
            acc["reward"] += batch_scores.mean().item()
            acc["adv"] += adv.mean().item()
            acc["entropy"] += entropy.mean().item()
            acc_n += 1

            if i % log_every == log_every - 1:
                if acc_n:
                    self.global_step += 1
                    wandb.log(
                        {
                            "reinforce/loss": acc["loss"] / acc_n,
                            "reinforce/pg_loss": acc["pg_loss"] / acc_n,
                            "reinforce/kl_div": acc["kl_div"] / acc_n,
                            "reinforce/ratio": acc["ratio"] / acc_n,
                            "reinforce/reward": acc["reward"] / acc_n,
                            "reinforce/adv": acc["adv"] / acc_n,
                            # Norm of the accumulated gradient at the step, not a batch mean.
                            "reinforce/grad_norm": grad_norm,
                            "reinforce/batches_in_step": acc_n,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "reinforce/entropy": acc["entropy"] / acc_n,
                            "num_oracle_calls": len(self.mol_buffer.values()),
                        },
                    )
                    acc = dict.fromkeys(acc, 0.0)
                    acc_n = 0
                self.lr_scheduler.step()
                self.model_opt.step()
                self.model_opt.zero_grad()


    def initialize_population(self):
        """
        Initialize population by sampling from the pretrained model.

        Returns:
            List of (sequence, score, smile) triples
        """
        if self.config.pretrained_dir:
            # stage 2: do not reinitialize by random sampling — seed the population from
            # stage 1's best. Prefer the saved elite (actual top-K by score); fall back to
            # the diversity-deduplicated prompter vocab for older checkpoints without one.
            k = self.config.first_pop_size
            if self.elite:
                smiles = [smi for smi, _ in self.elite[:k]]
                samples = [torch.tensor(self.tokenizer.encode(" ".join(decompose_smiles(smi, max_frags=self.config.max_frags)))) for smi in smiles]
            else:
                vocab_items = sorted(self.prompter.vocab.items(), key=lambda kv: kv[1], reverse=True)[: self.config.vocab_size]
                samples = [torch.tensor(list(kk)) for kk, _ in vocab_items]
                smiles = [bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, list(s)).split(" ")) for s in samples]
            init_ids = [torch.randint(4, 203, (len(s),)) for s in samples]
            scores = self.oracle(smiles)
        else:
            # stage 1: spend init_random_frac of the oracle budget on a random pool and
            # init_ga_frac on GA mutants of its best, update the prompter with ALL of them,
            # then keep only the best tot_offspring_size to seed the first reinforce step.
            n_random = max(1, int(self.config.init_random_frac * self.config.max_oracle_calls))
            n_ga_init = int(self.config.init_ga_frac * self.config.max_oracle_calls)
            gen_bs = self.config.batch_size or self.config.offspring_size * 2
            pool_samples, pool_init_ids, pool_smiles = [], [], []
            remaining = n_random
            while remaining > 0:
                b = min(gen_bs, remaining)
                n_oracle = [self.bandit.select_length() for _ in range(b)]
                gen_seqs, gen_x0 = self.model.sample(
                    num_samples=b,
                    temperature=1.0,
                    noise=1.0,
                    oracle=n_oracle,
                    # init_eta, not the run's eta: h = min(eta*dt, 1-t) saturates, so this is
                    # a single jump to the model's x_1 prediction instead of a 1/dt-step
                    # trajectory. The init pool wants breadth cheaply, not refined samples.
                    eta=self.config.init_eta,
                    return_uni=True,
                )
                valid, gen_smiles, _ = evaluate_smiles(
                    generated_ids=gen_seqs,
                    tokenizer=self.tokenizer,
                    return_values=True,
                    print_flag=False,
                    print_metrics=False,
                    exclude_salts=False,
                )
                # Park the pool on the host: at init_random_frac of a 100k budget this is
                # ~10k sequences, which have no reason to sit in VRAM next to the model.
                pool_samples += [as_cpu_tensor(gen_seqs[i]) for i in valid]
                pool_init_ids += [as_cpu_tensor(gen_x0[i]) for i in valid]
                pool_smiles += [gen_smiles[i] for i in valid]
                remaining -= b
                del gen_seqs, gen_x0
            if not pool_samples:
                raise RuntimeError("Could not generate a valid initial molecule in the init pool")
            pool_scores = self.oracle(pool_smiles)

            # also mutate: produce n_ga_init molecules via GA mutation of the best of the
            # generated pool, score them, and fold them into the init pool.
            if self.config.use_mutation and n_ga_init > 0:
                top_pool = sorted(zip(pool_smiles, pool_scores), key=lambda x: x[1], reverse=True)[:100]
                population = [(smi, [sc]) for smi, sc in top_pool]
                seen = set(pool_smiles)
                mut_smiles, attempts, target = [], 0, n_ga_init
                while len(mut_smiles) < target and attempts < target * 5:
                    attempts += 1
                    smi, _ = reproduce(population, 1)
                    if not smi:
                        continue
                    m = Chem.MolFromSmiles(smi)
                    if m is None:
                        continue
                    smi = Chem.MolToSmiles(m)
                    if smi in seen:
                        continue
                    seen.add(smi)
                    mut_smiles.append(smi)
                if mut_smiles:
                    mut_scores = self.oracle(mut_smiles)
                    mut_samples = [torch.tensor(self.tokenizer.encode(" ".join(decompose_smiles(s, max_frags=self.config.max_frags)))) for s in mut_smiles]
                    pool_samples += mut_samples
                    pool_init_ids += [torch.randint(4, 203, (len(s),)) for s in mut_samples]
                    pool_smiles += mut_smiles
                    pool_scores = list(pool_scores) + list(mut_scores)

            # update the prompter with ALL generated molecules
            if self.config.use_prompter:
                for smi_i, sc_i in zip(pool_smiles, pool_scores):
                    if sc_i > 0:
                        self.prompter.update_with_score(smi_i, sc_i)

            # keep only the best init_elite_frac of the pool to train on
            k = max(1, int(self.config.init_elite_frac * len(pool_scores)))
            order = sorted(range(len(pool_scores)), key=lambda i: pool_scores[i], reverse=True)[:k]
            samples = [pool_samples[i] for i in order]
            init_ids = [pool_init_ids[i] for i in order]
            smiles = [pool_smiles[i] for i in order]
            scores = [pool_scores[i] for i in order]

        if self.config.use_prescreen:
            # 1. Load ZINC SMILES
            assert self.config.offspring_size - self.config.first_pop_size > 0, "offspring_size must be greater than first_pop_size"
            df = pd.read_csv("in_virtuo_reinforce/vocab/zinc250k.csv").sort_values(by=self.config.oracle, ascending=False)[:self.config.offspring_size - self.config.first_pop_size]

            print(f"Loaded {len(smiles)} ZINC SMILES")
            samples = [torch.tensor(self.tokenizer.encode(" ".join(decompose_smiles(sm, max_frags=self.config.max_frags)))) for sm in df["smiles"].values.tolist()] + samples
            smiles = df["smiles"].values.tolist() + smiles
            scores = df[self.config.oracle].values.tolist() + scores
            assert len(samples) == len(smiles)
            init_ids = [torch.randint(4, 203, (len(s),)) for s in samples] + init_ids
        self.scores.extend(scores)
        qed_sa = [safe_qed_sa(smi) for smi in smiles]
        self.qed.extend([q for q, _ in qed_sa])
        self.sa.extend([s for _, s in qed_sa])

        for i in range(len(samples)):
            if scores[i] > 0:
                if self.config.use_prompter:
                    self.prompter.update_with_score(smiles[i], scores[i])
                    if not self.config.no_bandit:
                        self.prompter.bandit.update(len(samples[i][samples[i] != self.model.pad_token_id]), scores[i])
                else:
                    if not self.config.no_bandit:
                        self.bandit.update(len(samples[i][samples[i] != self.model.pad_token_id]), scores[i])
        self.train_ids  = [s[s != self.model.pad_token_id] for s in samples]
        self.train_x_0 = [i[: len(s)] for i, s in zip(init_ids, self.train_ids)]
        self.train_scores = scores
        if self.train_scores and max(self.train_scores) > 0 and self.config.num_timesteps > 0:
            self.reinforce([item for item in self.train_ids], [item for item in self.train_scores], [item for item in self.train_x_0])
            self.add_experience()
        self.train_ids, self.train_scores, self.train_x_0 = [], [], []

    def generate_offspring_batch(self):
        n_oracle = []
        num_samples = self.model_offspring_per_round
        if self.config.use_prompter and max(self.scores) > 0:
            self.prompter.offspring_size = num_samples
            prompts, n_oracle = self.prompter.build_prompts_and_masks(dev=self.device)
            assert len(prompts) == num_samples
        else:
            prompts = None
            for i in range(num_samples):
                n_oracle.append(self.bandit.select_length())
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            all_seqs, all_x_0 = self.model.sample(
                prompt=prompts,
                num_samples=num_samples,
                temperature=self.config.temperature,
                noise=0,
                oracle=n_oracle,
                start_t=self.config.start_t,
                fade_prompt=False,
                dt=self.config.dt,
                temperature_scaling=False,
                eta=self.config.eta,
                return_uni=True,
            )
        valid, all_smiles, metrics = evaluate_smiles(all_seqs, self.tokenizer, exclude_salts=True, return_values=True, print_flag=False, print_metrics=False, return_unique_indices=True)
        valid_new_smiles, valid_new_seqs, valid_new_x_0, n_valid = filter_valid_new(valid, all_smiles, all_seqs, all_x_0, self.mol_buffer, n_oracle)
        novelty = len(valid_new_seqs) / len(all_seqs)
        self.prev_novelty = novelty
        valid_new_smiles, valid_new_seqs, valid_new_x_0 = valid_new_smiles, valid_new_seqs, valid_new_x_0
        n_valid = n_valid
        self.prev_validity = metrics["validity"]

        self.evaluate_offspring(valid_new_seqs, valid_new_x_0, valid_new_smiles, n_valid)
        if not self.used_mutation and not self.config.classic_ga:
            self.run_ga()
        # Both sources have reported in; now take the round's top elite_frac.
        round_pool_size = len(self.round_pool)
        self.select_elite()

        # Logged after selection, not before: the previous round's reinforce step drains
        # train_ids, so reading it at the top of a round reports 0 every time.
        self.global_step += 1
        wandb.log(
            {
                "control/num_samples": num_samples,
                "control/lenght_train_ids": len(self.train_ids),
                "control/round_pool": round_pool_size,
                "control/length": len(all_seqs),
                "control/sum_valid_new": len(valid_new_seqs),
                "control/num_prompts": len(prompts) if prompts is not None else 0,
                "smiles/novelty": novelty,
                "smiles/validity": metrics["validity"],
                "smiles/uniqueness": metrics["uniqueness"],
                "smiles/diversity": metrics["diversity"],
                "smiles/quality": metrics["quality"],
                "smiles/qed": metrics["qed"],
                "smiles/sa": metrics["sa"],
                "num_oracle_calls": len(self.mol_buffer.values()),
            },
        )
        self.set_c_neg(novelty)

    def evolve_population(self):
        self.generate_offspring_batch()
        if len(self.train_ids) >= self.reinforce_trigger and self.config.num_timesteps > 0 and max(self.train_scores) > 0:
            self.used_mutation = not self.config.use_mutation
            self.reinforce([item for item in self.train_ids], [item for item in self.train_scores], [item for item in self.train_x_0])  # type: ignore[attr-defined]

            self.add_experience()
            self.train_ids, self.train_scores, self.train_x_0 = [], [], []

        elif len(self.train_ids) >= self.reinforce_trigger:
            print("resetting train_ids, train_scores, train_x_0")
            self.train_ids, self.train_scores, self.train_x_0 = [], [], []

        best_score = sorted(self.mol_buffer.values(), key=lambda x: x[0], reverse=True)[0][0]

        self.global_step += 1

        wandb.log({"best_score": best_score, "num_oracle_calls": len(self.mol_buffer.values())})

    def _optimize(self, oracle, config):
        """
        config is expected to be a dict with keys:
          - batch_size: how many candidates to propose per oracle call
        """
        self.oracle.assign_evaluator(oracle)
        self.model.gen_batch_size = self.config.batch_size or self.config.offspring_size * 2
        print("Initializing population...")
        self.initialize_population()
        stopping_top_n = 100 if self.config.pretrained_dir else 10
        # valsartan_smarts caps out at low scores, so avg-based early stopping cuts it
        # off prematurely; let it run to the full oracle budget instead.
        disable_avg_stopping = self.config.oracle == "valsartan_smarts"
        initial_top = sorted((score[0] for score in self.mol_buffer.values()), reverse=True)[:stopping_top_n]
        best_top_avg = sum(initial_top) / len(initial_top)
        top_stale_iterations = 0
        while not self.finish:
            self.evolve_population()
            top100 = sorted((score[0] for score in self.mol_buffer.values()), reverse=True)[:100]
            if not disable_avg_stopping and len(top100) >= 10 and sum(top100[:10]) / 10 >= 1.0:
                print("Early stopping: top-10 average reached 1.0")
                break
            stopping_top = top100[:stopping_top_n]
            stopping_avg = sum(stopping_top) / len(stopping_top)
            if max(stopping_top) <= 0:
                top_stale_iterations = 0
            elif stopping_avg > best_top_avg:
                best_top_avg = stopping_avg
                top_stale_iterations = 0
            else:
                top_stale_iterations += 1
            if not disable_avg_stopping and max(stopping_top) > 0 and top_stale_iterations >= 50:
                print(
                    f"Early stopping: top-{stopping_top_n} average did not increase "
                    f"for 50 iterations ({best_top_avg:.4f})"
                )
                break

            if len(self.mol_buffer) > 0:
                smis_, true_scores_ = [], []
                for smi, score in list(self.mol_buffer.items()):
                    smis_.append(smi)
                    true_scores_.append(score[0])
                self.global_step += 1
                visualize_top_smiles(smis_, scores=true_scores_, target=self.config.target, oracle_name=self.config.oracle, device=self.device)  # type: ignore[attr-defined]
                if self.config.use_prompter:
                    sorted_vocab = sorted(self.prompter.vocab.items(), key=lambda x: x[1], reverse=True)[:10]
                    smiles = [bridge_smiles_fragments(custom_decode_sequence(self.tokenizer, list(x[0])).split(" ")) for x in sorted_vocab]
                    scores = [x[1] for x in sorted_vocab]
                    visualize_top_smiles(smiles, scores=scores, target=self.config.target, oracle_name=self.config.oracle, device=self.device, prefix="vocab")
                if not self.config.use_prompter:
                    bandit = self.bandit
                else:
                    bandit = self.prompter.bandit
                bandit.plot_distribution(save_path=f"plots/tdc/{self.config.oracle}/length_bandit_{self.device}.png", log_wandb=True, global_step=self.global_step, smiles_list=smis_)

                try:
                    self.global_step += 1
                    wandb.log({"train/auc": wandb.Image("plots/tdc/" + self.config.oracle + "/tdc_avg_top10" + str(self.device) + ".png")})
                except:
                    traceback.print_exc()
                    print("wandb error")
        # final logging & save
        self.log_intermediate(finish=True)
        self.save_result_tot(self.config.oracle)
        self.save_qed_sa(self.config.oracle)
        if self.config.save_pretrained_dir:
            os.makedirs(self.config.save_pretrained_dir, exist_ok=True)
            pretrained_path = os.path.join(self.config.save_pretrained_dir, f"{self.config.oracle}.ckpt")
            bandit = self.prompter.bandit if self.config.use_prompter else self.bandit
            # elite = the actual top-K molecules by raw oracle score (no diversity filter),
            # so stage 2 can seed its population from stage 1's best instead of the
            # diversity-deduplicated prompter vocab.
            elite = sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True)[:100]
            elite = [(smi, float(v[0])) for smi, v in elite]
            torch.save(
                {
                    "oracle": self.config.oracle,
                    "model_state_dict": self.model.state_dict(),
                    "prompter_vocab": self.prompter.vocab if self.config.use_prompter else None,
                    "prompter_vocab_fps": self.prompter.vocab_fps if self.config.use_prompter else None,
                    "bandit_state": bandit.__dict__,
                    "elite": elite,
                },
                pretrained_path,
            )
            print(f"Saved pretrained checkpoint to {pretrained_path}")
        wandb.finish()


if __name__ == "__main__":

    oracles = [
        "albuterol_similarity",
        "amlodipine_mpo",
        "celecoxib_rediscovery",
        "deco_hop",
        "drd2",
        "fexofenadine_mpo",
        "gsk3b",
        "isomers_c7h8n2o2",
        "isomers_c9h10n2o2pf2cl",
        "jnk3",
        "median1",
        "median2",
        "mestranol_similarity",
        "osimertinib_mpo",
        "perindopril_mpo",
        "qed",
        "ranolazine_mpo",
        "scaffold_hop",
        "sitagliptin_mpo",
        "thiothixene_rediscovery",
        "troglitazone_rediscovery",
        "valsartan_smarts",
        "zaleplon_mpo",
    ]
    targets = {
        "albuterol_similarity": "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        "amlodipine_mpo": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
        "celecoxib_rediscovery": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "fexofenadine_mpo": "CC(C)(C)OC(=O)N1CCC(CC1)C2=CC=CC=C2C(C3=CC=CC=C3)OCC(=O)O",
        "osimertinib_mpo": "CN[C@H]1CN(C[C@H]1C2=NC=CC(=C2)C#N)C3=CC(=CC=C3)NC4=CC=CC=C4",
        "perindopril_mpo": "CC(C)C[C@H](C(=O)N1CCCC1C(=O)O)NC(CC2=CC=CC=C2)C(=O)O",
        "ranolazine_mpo": "CC1=CC=C(C=C1)C2=CC=CC=C2C(=O)NCCN(CC)CC",
        "sitagliptin_mpo": "CC1=CN(C2=CN=C(N=C2C1)N3CCN(CC3)C(=O)C4CC4)C",
        "zaleplon_mpo": "CC(=O)N1C=CN=C1C2=CC=CC=N2C",
        "thiothixene_rediscovery": "CN(C)CCCN1C2=CC=CC=C2SC3=CC=CC=C13",
        "troglitazone_rediscovery": "CC1=C(C=C(C=C1)O)C(CC2=CC(=C(C=C2)O)OCC(=O)C)C3=CC=CC=C3",
        "mestranol_similarity": "C#C[C@]1(CC[C@H]2[C@@H]3CCC4=CC(=O)CCC4=C3CC[C@H]12)OC",
        "valsartan_smarts": "CC(C)C[C@@H](C(=O)N1CCC[C@H]1C(=O)O)NC(Cc2ccccc2)C(=O)O",
    }
    # for oracle in oracles:
    # try:
    import argparse

    difficult_ones = ["amlodipine_mpo", "celecoxib_rediscovery", "valsartan_smarts", "thiothixene_rediscovery", "scaffold_hop", "troglitazone_rediscovery", "sitagliptin_mpo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/invirtuo_gen.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--oracle", type=str, default="albuterol_similarity")
    parser.add_argument("--start_t", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--rl_lr", type=float, default=1e-4)

    parser.add_argument("--offspring_size", type=int, default=64)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--num_reinforce_steps", type=int, default=10)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--big", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--use_prompter", action="store_true")
    parser.add_argument("--clip_eps", type=float, default=0.5)
    parser.add_argument("--c_neg", type=float, default=0.2)
    parser.add_argument("--use_mutation", action="store_true")
    parser.add_argument("--reverse_kl", action="store_true")
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--mutation_size", type=int, default=10)
    parser.add_argument("--experience_replay_size", type=int, default=24)
    parser.add_argument("--no_sample_uni", action="store_true")
    parser.add_argument("--train_mutation", action="store_true")
    parser.add_argument("--use_prescreen", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--identifier", type=str, default="")
    parser.add_argument("--end_task", type=int, default=len(oracles))
    parser.add_argument("--first_pop_size", type=int, default=50)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--max_frags", type=int, default=5)
    parser.add_argument("--difficult_ones", action="store_true")
    parser.add_argument("--easy_ones", action="store_true")
    parser.add_argument("--aggressive_bandit", action="store_true")
    parser.add_argument("--no_bandit", action="store_true")
    parser.add_argument("--tot_offspring_size", type=int, default=None)
    parser.add_argument("--start_rank", type=int, default=0)
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--classic_ga", action="store_true")
    parser.add_argument("--entropy_bonus", action="store_true")
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Load per-oracle pretrained checkpoints from this directory")
    parser.add_argument("--save_pretrained_dir", type=str, default=None, help="Save per-oracle pretrained checkpoints to this directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Generation batch size")
    parser.add_argument("--min_tanimoto_dist", type=float, default=0.7, help="Prompter vocab diversity: entries must differ by at least this Tanimoto distance; max allowed pairwise similarity = 1 - min_tanimoto_dist")
    parser.add_argument("--single_prompt", action="store_true", help="Stage 2: do not combine prompts via crossover; seed each offspring from a single best-performing prompt")
    parser.add_argument("--ga_elite_parents", type=int, default=1, help="Mutation: draw parents from the top-N of the buffer; 1 = elite GA on the single best molecule")
    parser.add_argument("--bandit_lr", type=float, default=0.3, help="PeakSeekerBandit quantile step size (eta_q): how fast an arm's reward estimate tracks its observations")
    parser.add_argument("--bandit_tau", type=float, default=0.3, help="PeakSeekerBandit softmax temperature over arm scores; SMALLER = sharper sampling distribution")
    parser.add_argument("--bandit_floor", type=float, default=0.005, help="PeakSeekerBandit uniform probability floor mixed into every arm regardless of score")
    parser.add_argument("--ppo_batch_size", type=int, default=0, help="PPO minibatch and construct_dataset chunk; 0 reuses tot_offspring_size")
    parser.add_argument("--parent_pool_size", type=int, default=10000, help="Mutation: widen the parent draw to the top-N once the elite parent stops yielding novel mutants for a round")
    parser.add_argument("--init_random_frac", type=float, default=0.1, help="Stage 1: fraction of max_oracle_calls spent on the random init pool (sampled at --init_eta)")
    parser.add_argument("--init_ga_frac", type=float, default=0.0, help="Stage 1: fraction of max_oracle_calls spent on a GA init pool seeded from the random pool's best; 0 disables it and leaves mutation to the per-round GA")
    parser.add_argument("--elite_frac", type=float, default=1.0, help="Per-round selection: 1.0 trains on everything a round scores; below 1 keeps the best elite_frac * offspring_size")
    parser.add_argument("--init_elite_frac", type=float, default=0.1, help="Stage 1: fraction of the random init pool kept to seed the first reinforce step")
    parser.add_argument("--init_eta", type=float, default=999.0, help="eta for the random init pool only; large values collapse generation to a single jump to the model's x_1 prediction")
    # Then in your config setup, add:
    args = parser.parse_args()
    device = f"cuda:{args.device}" if len(args.device) == 1 else args.device
    # Convert args to dict for easy passing
    config_dict = dict(vars(args))
    config_dict["device"] = device
    if args.pretrained_dir:
        config_dict["start_t"] = 0.95
        config_dict["use_mutation"] = False
        config_dict["train_mutation"] = False
        config_dict["kl_coef"] = 0.0
        config_dict["vocab_size"] = 10
        # The budget split is a stage-1 device. Stage 2 has no init pools and no GA, and
        # trains on every offspring it scores, so elite selection must be a no-op here.
        config_dict["elite_frac"] = 1.0
    config_dict.pop("start_task")
    config_dict.pop("end_task")
    num_seeds = config_dict.pop("num_seeds")
    # Now you can simply pass all arguments
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, f"target_property_optimization")
    import datetime

    identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if not args.identifier else args.identifier
    output_dir += "/" + identifier
    print("Output dir: ", output_dir)
    difficult_ones_flag = config_dict.pop("difficult_ones")
    easy_ones_flag = config_dict.pop("easy_ones")
    for seed in range(args.start_seed, num_seeds):
        for oracle in oracles[args.start_task : args.end_task]:
            try:
                if difficult_ones_flag and oracle not in difficult_ones:
                    continue
                if easy_ones_flag and oracle in difficult_ones:
                    continue
                lightning.seed_everything(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                config_dict["seed"] = seed
                config_dict["target"] = targets.get(oracle, "")
                config_dict["oracle"] = oracle
                config_dict["identifier"] = identifier
                config_dict["output_dir"] = output_dir
                config_dict["task"] = oracle
                # config_dict["tot_offspring_size"] = config_dict["offspring_size"] + config_dict["experience_replay_size"] if config_dict["tot_offspring_size"] is None else config_dict["tot_offspring_size"]
                print("Target: ", config_dict["target"])
                print("Oracle: ", oracle)
                lr = args.lr if oracle != "isomers_c7h8n2o2" else 0.2
                beta = 0.99 if oracle != "isomers_c7h8n2o2" else 0.95

                optimizer = InVirtuoFMOptimizer(
                    **config_dict,
                )

                optimizer.optimize(oracle=oracle, seed=seed)
                del optimizer
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                raise
