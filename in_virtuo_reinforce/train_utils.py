"""
Complete script for InVirtuoFM optimization with batch docking oracle support.
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



def compute_kl_divergence(logits_new, logits_old, source_mask, reverse=False):
    """Compute KL divergence for PPO"""
    if reverse:
        logits_new, logits_old = logits_old, logits_new

    p_new = F.softmax(logits_new, dim=-1)
    log_p_new = F.log_softmax(logits_new, dim=-1)
    log_p_old = F.log_softmax(logits_old, dim=-1)

    kl_per_token = (p_new * (log_p_new - log_p_old)).sum(dim=-1)
    kl = (kl_per_token * source_mask.float()).sum() / source_mask.float().sum().clamp(min=1)
    return kl


def pad_collate_with_masks(batch):
    """Collate function for batches."""
    ids, scores, uni, t, logprobs, x_ts, masks = zip(*batch)

    ids_padded = pad_sequence(list(ids), batch_first=True, padding_value=0)
    uni_padded = pad_sequence(list(uni), batch_first=True, padding_value=0)
    scores = torch.tensor(scores, dtype=torch.float, device=ids_padded.device)
    t = torch.tensor(t, dtype=torch.float, device=ids_padded.device)
    logprobs = torch.stack(logprobs, dim=0)

    batch_size = len(batch)
    num_timesteps = x_ts[0].shape[0]
    max_seq_len = ids_padded.shape[1]

    x_ts_padded = torch.zeros(batch_size, num_timesteps, max_seq_len, dtype=x_ts[0].dtype, device=ids_padded.device)
    masks_padded = torch.zeros(batch_size, num_timesteps, max_seq_len, dtype=masks[0].dtype, device=ids_padded.device)

    for i, (x_t, mask) in enumerate(zip(x_ts, masks)):
        seq_len = x_t.shape[-1]
        x_ts_padded[i, :, :seq_len] = x_t
        masks_padded[i, :, :seq_len] = mask

    return ids_padded, scores, uni_padded, t, logprobs, x_ts_padded, masks_padded


def sample_path(t, x_0, x_1, n=1):
    sigma_t = 1 - t**n
    source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t.unsqueeze(-1).to(x_1.device)
    return torch.where(condition=source_indices, input=x_0, other=x_1), source_indices

def filter_valid_new(valid, smiles, off_seqs, all_x_0, mol_buffer):

    valid_new = [i for i in valid if smiles[i] not in mol_buffer]
    valid_new_seqs = [off_seqs[i] for i in valid_new]
    valid_new_smiles = [smiles[i] for i in valid_new]
    valid_new_x_0 = [all_x_0[i] for i in valid_new]

    return valid_new_smiles, valid_new_seqs, valid_new_x_0

def compute_seq_logp(logits, ids, source_mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return (token_lp * source_mask).sum(dim=-1) / torch.max(torch.ones_like(source_mask[:, 0]), (source_mask.float().sum(dim=-1)))

def custom_collate(batch):
    """Collate function that handles variable length sequences."""
    ids, scores, uni,  old_lp, x_t, mask,t = zip(*batch)
    ids_padded = pad_sequence(list(ids), batch_first=True, padding_value=0)
    uni_padded = pad_sequence(list(uni), batch_first=True, padding_value=0)
    scores = torch.stack(scores)
    t = torch.stack(t)
    old_lp = torch.stack(old_lp)

    # Pad x_t and mask
    max_len = max(x.shape[0] for x in x_t)
    batch_size = len(batch)
    x_t_padded = torch.zeros(batch_size, max_len, dtype=x_t[0].dtype, device=x_t[0].device)
    mask_padded = torch.zeros(batch_size, max_len, dtype=mask[0].dtype, device=mask[0].device)

    for i, (x, m) in enumerate(zip(x_t, mask)):
        x_t_padded[i, :x.shape[0]] = x
        mask_padded[i, :x.shape[0]] = m

    return ids_padded, scores, uni_padded, t, old_lp, x_t_padded, mask_padded
