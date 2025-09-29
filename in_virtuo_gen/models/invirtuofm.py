import os
from abc import ABC
from itertools import product
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.rank_zero import rank_zero_only
from torch import Tensor, nn
from torch.distributions import Categorical, Gumbel
from tqdm import tqdm

from ..train_utils.sampling import create_mask, dynamic_temperature_sampling_cont, smooth, top_k_top_p_filtering
from .invirtuobase import InVirtuoBase
from .transformer.model_ddit import DDiT


def sample_dfm(p_1t, x_t, t, dt):
    step_probs = (dt / (1 - t[0]) * p_1t).clamp(max=1.0)  # (B, D, S)
    step_probs.scatter_(-1, x_t[:, :, None], 0.0)
    step_probs.scatter_(-1, x_t[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0))
    return Categorical(step_probs).sample()


def sample_purity(p_1t, x_t: Tensor, candidates: Tensor, t: Tensor, dt: float,noise: float =0.):
    x_1 = categorical(p_1t)
    candidates = (x_1 != x_t) & candidates
    if torch.isclose(t[0].float() + dt, torch.ones_like(t[0]).float()):
        x_t[candidates] = x_1[candidates]
        return x_t
    delta_1 = F.one_hot(x_1, num_classes=p_1t.shape[-1]).to(x_1.dtype)
    candidate_purity = p_1t.gather(dim=2, index=x_1.unsqueeze(-1)).squeeze(-1)
    candidate_purity = torch.where(candidates, candidate_purity, torch.full_like(candidate_purity, float("-inf")))
    gumbel_noise = torch.distributions.Gumbel(0, 1).sample(candidate_purity.shape).to(x_t.device) # type: ignore[attr-defined]
    candidate_purity = torch.where(candidates, candidate_purity + noise * (1 - t[0]) * gumbel_noise, torch.full_like(candidate_purity, float("-inf")))
    chosen = torch.argmax(candidate_purity, dim=1)  # shape: (B,)
    max_scores = candidate_purity.max(dim=1)[0]
    row_valid = max_scores != float("-inf")
    batch_indices = torch.arange(x_t.size(0), device=x_t.device)
    x_t[batch_indices[row_valid], chosen[row_valid]] = x_1[batch_indices[row_valid], chosen[row_valid]]
    return x_t


def sample_meta_dfm(p_1t, x_t, candidates, t, dt, div_free_t=0.):
    x_1 = categorical(p_1t)
    delta_1 = F.one_hot(x_1, p_1t.shape[-1]).float()
    delta_t = F.one_hot(x_t, p_1t.shape[-1]).float()
    candidates = candidates & (x_1 != x_t)
    if torch.isclose(t[0].float() + dt, torch.ones_like(t[0]).float()):
        x_t[candidates] = x_1[candidates]
        return x_t
    u = (delta_1) / (1.0 - t[0])
    if div_free_t > 0:
        u = u + div_free_t / (t[0] * (1 - t[0]) + 1e-5) * ((1 - t[0]) * 1 / p_1t.shape[-1] + t[0] * delta_1)
    u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
    u[~candidates] = 0
    intensity = u.sum(dim=-1)
    mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(-dt * intensity)

    if mask_jump.sum() > 0:
        x_t[mask_jump & candidates] = categorical(u[mask_jump & candidates])
    return x_t


def sample_mine(p_1t, x_t, candidates, t, dt, eta, div_free_t=0.):
    one_hot_x_t = F.one_hot(x_t, p_1t.shape[-1]).float()
    u = (p_1t - one_hot_x_t) / (1 - t[0])
    h = min(eta*dt, 1 - t[0])

    probs = one_hot_x_t + h * u
    x_t[candidates] = torch.distributions.Categorical(probs=probs.clamp(0)).sample()[candidates]
    return x_t


def get_confidence(p_1t: Tensor, x_t: Tensor, valid: Tensor, t: Tensor, unmasking_noise: float):
    B, L, V = p_1t.shape

    # 1) clamp away negatives and flatten
    flat_w = p_1t.view(B*L, V).clamp(min=0)

    # 2) compute a proper normalization so we know what dist.log_prob would have used
    denom = flat_w.sum(dim=1, keepdim=True).clamp(min=1e-10)
    flat_p = flat_w / denom              # shape [B*L, V], each row sums to 1

    # 3) sample with multinomial (no need to renormalize for sampling)
    idx = torch.multinomial(flat_w, 1, replacement=False).squeeze(1)  # [B*L]
    candidate_tokens = idx.view(B, L)

    # 4) gather the *normalized* probability, then log it
    logp = torch.log(flat_p.gather(1, idx.unsqueeze(1)) + 1e-10).squeeze(1)
    candidate_log_probs = logp.view(B, L)       # same as torch.log(probs.gather(...))

    gumbel_noise = Gumbel(0, 1).sample((B, L)).to(x_t.device) # type: ignore[attr-defined]
    # note: (~t).float() == 1 - t if t is 0/1 mask
    confidence = candidate_log_probs + unmasking_noise * gumbel_noise * (1-t).float().view(-1, 1)
    confidence = confidence.masked_fill(~valid, -float("inf"))
    return confidence, candidate_tokens


def sample_mdm(p_1t, x_t, valid, t, unmasking_noise):
    B, L = x_t.shape
    confidence, candidate_tokens = get_confidence(p_1t, x_t, valid, t, unmasking_noise)
    confidence = torch.where(valid, confidence, torch.full_like(confidence, -float("inf")))
    # sorted_confidence, sorted_indices = confidence.sort(dim=1, descending=True)
    max_inds = confidence.argmax(dim=1)
    mask = torch.zeros_like(valid)
    mask[torch.arange(B, device=mask.device), max_inds] = True
    mask &= valid
    x_t[mask] = candidate_tokens[mask]
    return x_t


def sample_path(t, x_0, x_1, n=1):
    sigma_t = 1 - t**n
    source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t.unsqueeze(-1)
    return torch.where(condition=source_indices, input=x_0, other=x_1)


def categorical(probs: Tensor) -> Tensor:
    r"""Categorical sampler according to weights in the last dimension of ``probs`` using :func:`torch.multinomial`.

    Args:
        probs (Tensor): probabilities.

    Returns:
        Tensor: Samples.
    """

    return torch.multinomial(probs.clamp(min=0).flatten(0, -2), 1, replacement=True).view(*probs.shape[:-1])


def sample_timesteps(k):
    """
    Sample k timesteps that cover [0,1] more evenly by using a single random number u0.

    Args:
        k (int): Number of timesteps to sample.

    Returns:
        torch.Tensor: A tensor of shape (k,) with timesteps sampled from U[0,1].
    """
    # Sample a single random number from U[0,1]
    u0 = torch.rand(1)

    # Create a tensor for indices: 1, 2, ..., k
    indices = torch.arange(1, k + 1, dtype=torch.float32)

    # Compute timesteps using the formula: t_i = (u0 + i/k) mod 1
    timesteps = (u0 + indices / k) % 1.0
    return timesteps


class ModelWrapper(ABC, nn.Module):
    """
    This class is used to wrap around another model, adding custom forward pass logic.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        attn_mask: Tensor,
        temperature: float,
        conds: Optional[Tensor] = None,
        noise=0,
        p=2,
        temperature_scaling=False,
        min_temp=0.5,
        vocab_mask=None,
        curve_factor=0.7,
        top_p=None,
        top_k=None,
        **extras
    ) -> Tuple[Tensor, Tensor]:

        logits = self.model(x=x, t=t, attn_mask=attn_mask, conds=conds).float()
        t = t.unsqueeze(-1).unsqueeze(-1)
        if noise > 0:
            logits += torch.distributions.Gumbel(0, 1).sample(logits.shape).to(x.device) * (1 - t) * noise # type: ignore[attr-defined]
        #         logits+=torch.distributions.Laplace(0, 1).sample(logits.shape).to(x.device)*smooth(t,p)*noise

        if temperature_scaling:
            logits = dynamic_temperature_sampling_cont(logits, base_temp=temperature, t=t.reshape(-1)[0], curve_factor=curve_factor, min_temp=min_temp)
        else:
            logits /= temperature
        if top_p is not None or top_k is not None:
            logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)
        if vocab_mask is not None:
            forbidden = ~vocab_mask
            logits.masked_fill(forbidden.unsqueeze(1), float('-inf'))

        probs = torch.softmax(logits, -1)
        return probs, logits


class InVirtuoFM(InVirtuoBase):
    """Fragment-based molecular generation model using BERT architecture.

    Implements a masked language modeling approach for generating molecular SMILES strings,
    with progressive denoising during generation.

    Args:
        model (str): Model architecture type, default is "bert"
        vocab_size (int): Size of the SMILES vocabulary
        block_size (int): Maximum sequence length
        n_cond (int): Number of molecular properties to condition on
        n_layer (int): Number of transformer layers
        n_head (int): Number of attention heads
        n_embd (int): Embedding dimension
        embd_pdrop (float): Embedding dropout rate
        resid_pdrop (float): Residual dropout rate
        attn_pdrop (float): Attention dropout rate
        num_steps (int): Number of denoising steps
        noise (float): Noise level for masking
        gen_batch_size (int): Batch size for generation
        num_strings (int): Number of strings to generate
        **kwargs: Additional arguments passed to InVirtuoBase
    """

    def __init__(
        self,
        model: str = "bert",
        vocab_size: int = 300,
        block_size: int = 150,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        n_conds: int = 0,
        dropout: float = 0.3,
        noise: float = 0.1,
        gen_batch_size: int = 180,
        num_samples: int = 180,
        dt: float = 1e-2,
        loss_fc: str = "cross_entropy",
        n: float = 1.0,
        cond_dim: int = 0,
        masked: bool = False,
         classification=False,
         num_classes=1,
         already_smiles=False,
         no_bos=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.vocab_size =self.vocab_size + int(masked)
        self.model = DDiT(
            n_layer=n_layer,
            vocab_size=self.vocab_size,
            dropout=dropout,
            n_heads=n_head,
            hidden_size=n_embd,
            n_conds=n_conds,
            classification=classification,
            num_classes=num_classes
        )
        self.gen_batch_size = gen_batch_size
        self.loss = lambda x, y: F.cross_entropy(x, y, ignore_index=self.pad_token_id, reduction="none")

        self.sample_source = lambda x: torch.randint_like(x, 4, self.vocab_size) if not self.hparams.masked else torch.full_like(x, self.mask_token_id) # type: ignore[attr-defined]

    def calculate_loss(self, batch) -> torch.Tensor:
        """Calculate masked language modeling loss.

        Applies noise to input sequences and computes loss
        between model predictions and noising process.

        Args:
            batch: Tuple containing:
                - x: Input token ids [batch_size, seq_len]
                - targets: Target token ids [batch_size, seq_len]
                - cond (optional): Property values for conditioning

        Returns:
            torch.Tensor: Computed loss value
        """
        if isinstance(batch, tuple) and len(batch) == 3:
            x_1, targets, conds = batch
        elif isinstance(batch, tuple) and len(batch) == 2:
            x_1, conds = batch["input_ids"].long(), batch["conds"] # type: ignore[attr-defined]
        else:
            if len(batch.shape) == 1: # type: ignore[attr-defined]
                batch = batch.unsqueeze(0) # type: ignore[attr-defined]
            x_1 = batch.long().clone() # type: ignore[attr-defined]
            conds = None
        t = sample_timesteps(len(x_1)).to(x_1.device) * (1.0 - 1e-3)
        x_1 = x_1[:, 1:] if not self.hparams.no_bos else x_1 # remove BOS and EOS token
        x_0 = torch.randint_like(x_1, low=4, high=self.vocab_size) if not self.hparams.masked else torch.ones_like(x_1) * self.mask_token_id # type: ignore[attr-defined]
        B, L = x_1.shape

        prompt = None
        token_mask = (x_1 != self.pad_token_id) & (x_1 != self.eos_token_id)
        x_0[~token_mask] = self.pad_token_id
        x_1[~token_mask] = self.pad_token_id
        attn_mask = ~token_mask.unsqueeze(1).expand(B, L, L)
        attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf")).unsqueeze(1)

        x_t = sample_path(t=t.to(self.device), x_0=x_0.to(self.device), x_1=x_1.to(self.device))  # Sample the conditional path
        logits = self.model(x_t, t, attn_mask=attn_mask.to(self.device), conds=conds)
        if self.hparams.masked: # type: ignore[attr-defined]
            x_1[x_t != self.mask_token_id] = self.pad_token_id
        loss = F.cross_entropy(logits.transpose(1, 2), x_1.to(self.device), reduction="none", ignore_index=self.pad_token_id)
        weights = 1 / ((1 - t.to(self.device)**self.hparams.n) + 1e-3)  # shape: (batch,) # type: ignore[attr-defined]
        weights = weights.unsqueeze(1)  # shape: (batch, 1)
        valid_tokens = (x_1 != self.pad_token_id) if not self.hparams.masked else (x_t == self.mask_token_id) & (x_1 != self.pad_token_id) # type: ignore[attr-defined]

        loss_mean = (loss * weights).sum() / valid_tokens.sum()

        if loss_mean.isnan():
            self.log("skipped_iter", 1, on_step=True, prog_bar=True)
            loss_mean = torch.tensor(1e-8, device=self.device, requires_grad=True)
        return loss_mean

    def sample(
        self,
        num_samples: int =1000,
        dt: float = 0.01,
        prompt: Optional[Tensor] = None,
        temperature: float =1.,
        temperature_scaling: bool =False,
        conds: Optional[Tensor] = None,
        T_min: float =0.25,
        p: float =1,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        eta: float =1,
        noise: float =0,
        unmasking_noise: float =0,
        meta: bool =False,
        div_free_t: float =0.0,
        dfm: bool =False,
        fade_prompt: bool =False,
        force_prompt: bool =False,
        oracle: Optional[List] = None,
        top_k: Optional[float] = None,
        purity: bool =False,
        start_t: float =0.,
        end_t: float =1.,
        vocab_mask: Optional[Tensor] = None,
        min_length: bool =False,
        return_trajectory: bool =False,
        return_uni: bool =False,
        **kwargs
    ) -> Union[List[str], Tuple[List[str], List[List[List[int]]]], Tuple[List[str], Tensor]]:
        """Generate SMILES strings using iterative denoising.

        Implements iterative demasking of sequences, sampling from model predictions
        at each step according to a temperature-scaled categorical distribution.

        Args:
            num_samples (int, optional): Number of sequences to generate. Defaults to 1000.
            num_sampling_steps (Optional[int], optional): Number of denoising steps.
                Defaults to self.hparams.num_steps.

        Returns:
            List[str]: Generated SMILES strings
        """
        assert not (dfm & meta) | (dfm & purity) | (meta & purity), "Mine,Purity and Meta cannot be true at the same time"
        # prompt=prompt[:5]\
        with torch.no_grad():
            if prompt is not None and prompt.shape[0]!=num_samples:
                prompt = prompt.unsqueeze(0).repeat(num_samples, 1)
            self.model.eval()
            model = ModelWrapper(self.model)
            samples = []
            trajectory = []
            sample_logps = []
            uni = []
            padded_batches = self.sample_seq_lengths(num_samples=num_samples, batch_size=self.gen_batch_size,
            oracle=oracle, min_length=0 if not min_length or prompt is None else int(prompt.shape[1])+5)

            for i, batch in enumerate(padded_batches):
                batch = batch.to(self.device) if prompt is None else batch[:prompt.shape[0]].to(self.device)
                B, S = batch.shape
                token_mask = (batch != self.pad_token_id)  # make sure to not redefine that!
                batch = self.sample_source(batch)
                batch[~token_mask] = self.pad_token_id
                candidates = token_mask
                if prompt is not None:
                    prompt = prompt.to(self.device)
                    Lp = min(prompt.size(1), S)
                    prompt_mask = torch.zeros_like(token_mask)
                    start = i * batch.shape[0]
                    end   = (i + 1) * batch.shape[0]
                    prompt_slice = prompt[start:end, : Lp]
                    prompt_mask = torch.zeros_like(token_mask)            # (B, S)
                    is_not_special = (prompt_slice != self.pad_token_id) & (prompt_slice != self.mask_token_id)
                    prompt_mask[:, :Lp] = is_not_special                  # only first Lp cols can be True
                    candidates = token_mask #& ~prompt_mask
                    if not isinstance(prompt, list) and not fade_prompt:
                        batch[prompt_mask] = prompt_slice[prompt_mask[:, : prompt.shape[1]]]
                attn_mask = ~token_mask.unsqueeze(1).expand(B, S, S)
                attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf")).unsqueeze(1)
                batch = batch.to(self.device)
                conds = self.prepare_conds(batch, conds)
                x_t = batch.clone()
                uni.extend(batch.clone().cpu().tolist())
                t = torch.zeros_like(batch[:, 0]).float() if not self.hparams.masked else (( batch!=self.mask_token_id)& (batch!=self.pad_token_id)).sum(1) / (token_mask).sum(1) # type: ignore[attr-defined]
                t+= start_t
                dt = 1 / (batch != self.pad_token_id).sum(1) if self.hparams.masked else dt # type: ignore[attr-defined]
                while (t <end_t - dt / 2).any() and candidates.any():
                    if prompt is not None:
                        if force_prompt:
                            x_t[prompt_mask] = prompt_slice[prompt_mask[:, : prompt.shape[1]]]
                        # elif t==:
                        #     x_t[prompt_mask]=prompt_slice[prompt_mask[:, : prompt.shape[1]]]#sample_path(t=t, x_0=batch[:,:Lp], x_1=x_t[:,:Lp])[is_not_special]


                    p_1t, logits = model(x=x_t, t=t, attn_mask=attn_mask, conds=conds, temperature=temperature, noise=noise, p=p, temperature_scaling=temperature_scaling, min_temp=T_min, top_p=top_p, vocab_mask=vocab_mask)
                    if self.hparams.masked: # type: ignore[attr-defined]
                        token_mask = candidates & (x_t == self.mask_token_id)
                        x_t = sample_mdm(p_1t, x_t, token_mask, t, unmasking_noise=unmasking_noise,)
                    elif meta:
                        x_t = sample_meta_dfm(p_1t, x_t, candidates, t, dt, div_free_t=div_free_t)
                    elif dfm:
                        x_t = sample_dfm(p_1t, x_t, candidates, t)  # (B, D)
                    elif purity:
                        x_t = sample_purity(p_1t, x_t, candidates, t, dt, noise=noise)
                    else:
                        x_t = sample_mine(p_1t, x_t, candidates, t, dt, eta=eta, div_free_t=div_free_t)
                    t = t + dt
                    if return_trajectory:
                        trajectory.append(x_t.cpu().tolist())
                if force_prompt:
                    x_t[prompt_mask] = prompt_slice[prompt_mask[:, : prompt.shape[1]]]
                samples.extend(x_t.cpu().tolist())

        if not return_trajectory and not return_uni:
            return samples
        elif return_uni:
            return samples, uni
        else:
            return samples, trajectory
