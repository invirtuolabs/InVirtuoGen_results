import torch
import torch.nn.functional as F
def create_mask(batch, device, pad_token_id):

    attn_mask = (batch == pad_token_id).to(device)
    B, S = batch.shape
    bool_mask = attn_mask.unsqueeze(1).expand(B, S, S)
    attn_mask = bool_mask.float().masked_fill(bool_mask, float('-inf')).unsqueeze(1)
    return bool_mask,attn_mask



def smooth(x,p):
    if not isinstance(x,torch.Tensor):
        x=torch.tensor(x)
    x=torch.min(x,torch.ones_like(x))
    return 1- torch.pow(x, p) / (torch.pow(x, p) + torch.pow(torch.ones_like(x) - x, p))


def top_k_top_p_filtering(logits, top_k=None, top_p=None, filter_value=-float('Inf')):
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.
    Assumes logits is a 2D tensor of shape (batch_size, vocab_size).

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, vocab_size)
        top_k (int): Keep only top k tokens with highest probability.
        top_p (float): Keep the top tokens with cumulative probability >= top_p.
        filter_value (float): The value to assign for filtered logits.

    Returns:
        torch.Tensor: Logits after top-k/top-p filtering.
    """
    # Top-k filtering: keep only the tokens with the top k highest logits
    if top_k is not None and top_k > 0:
        # Get the kth largest value for each row and create a threshold
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        # Set logits below the threshold to filter_value
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)

    # Top-p (nucleus) filtering: keep tokens with cumulative probability <= top_p
    elif top_p is not None and top_p > 0.0:
        # Sort logits in descending order along the vocab dimension
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # Compute softmax probabilities over sorted logits and then cumulative sum
        sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        # Shift the mask right to always keep at least one token.
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create a boolean mask for logits in original order
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        # Scatter the sorted mask back to the original ordering

        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def dynamic_temperature_sampling(logits, base_temp=1.0, token_position=0, max_position=200, curve_factor=0.7,min_temp=0.2):
    """
    Dynamically adjust temperature based on token position.
    Start with higher temperature and gradually decrease it.
    """
    position_factor = 1 - (token_position / max_position) * curve_factor
    dynamic_temp = max(min_temp,
                        base_temp * position_factor)  # Ensure temperature doesn't get too close to 0
    return logits / dynamic_temp

def dynamic_temperature_sampling_cont(logits, base_temp, t, curve_factor,min_temp):
    """
    Dynamically adjust temperature based on token position.
    Start with higher temperature and gradually decrease it.
    """

    position_factor =1-t * curve_factor
    dynamic_temp = max(min_temp,
                        base_temp * position_factor)  # Ensure temperature doesn't get too close to 0
    return logits / dynamic_temp



def selective_sampling(logits, top_k=30, min_tokens=5, entropy_scaling=0.8):
    """
    Apply more selective sampling for important structural tokens.

    Args:
        logits: The model logits
        top_k: Maximum number of tokens to consider
        min_tokens: Minimum number of tokens to consider
        entropy_scaling: Scaling factor for entropy-based k adjustment
    """
    try:
        # Get token probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Calculate entropy using torch operations
        prob_entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        vocab_size = probs.size(-1)

        # Calculate dynamic k using torch operations with scaling parameter
        entropy_factor = prob_entropy / torch.log(torch.tensor(vocab_size, dtype=torch.float))

        # Ensure k is within valid range (at least min_tokens, at most vocab_size)
        dynamic_k = max(min_tokens, min(vocab_size, int(top_k * entropy_factor.item() * entropy_scaling)))

        # Apply top-k filtering
        return top_k_top_p_filtering(logits, top_k=dynamic_k)
    except Exception as e:
        logging.error(f"Error in selective sampling: {str(e)}")
        # Fall back to a safe value for top_k
        return top_k_top_p_filtering(logits, top_k=min(50, logits.size(-1)))  # Safe fallback
