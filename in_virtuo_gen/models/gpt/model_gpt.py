"""
GPT Model Description:
----------------------

- The model employs token embeddings (for discrete tokens) and optional property embeddings
  (if `n_conds > 0`) to form the initial input.
- A series of Transformer blocks (each containing a self-attention mechanism and MLP) processes
  the sequence in a residual fashion.
- Rotary embeddings are used for positional information.
- A final linear head maps the outputs to `vocab_size` logits (e.g., for next-token prediction).
- An additional GPTForBinaryClassification subclass adds a binary classification head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#----------------------
#  Rotary Positional Embeddings
# ----------------------
class RotaryPositionalEmbedding(nn.Module):
    """
    Implements rotary positional embeddings.
    Adapted from GPT-NeoX / GPT-J style embeddings.
    """

    def __init__(self, dim, max_position_embeddings=150, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, seq_len):
        """
        Generate rotary embeddings for a sequence of length `seq_len`.
        `x` is just used for device/type context here.
        """
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


def rotate_half(x):
    """
    Helper to swap the halves for rotary transform.
    """
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs):
    """
    Applies rotary positional embeddings to q and k.
    """
    q_rot = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k_rot = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q_rot, k_rot
class DummyConfig:
    def __init__(self, model_type="gpt", tie_word_embeddings=False):
        self.model_type = model_type
        self.tie_word_embeddings = tie_word_embeddings

    def get(self, key, default=None):
        return getattr(self, key, default)

# ----------------------
#  Causal Self-Attention
# ----------------------
class CausalSelfAttention(nn.Module):
    """
    Standard multi-head self-attention using a causal mask.
    Rotary embeddings for position encoding.
    """

    def __init__(self, max_tokens, n_embd, n_head, resid_pdrop, attn_pdrop):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_tokens = max_tokens

        # Key, Query, Value projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Dropouts
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask
        mask = torch.tril(torch.ones(max_tokens, max_tokens))
        self.register_buffer("mask", mask.view(1, 1, max_tokens, max_tokens))

        # Rotary positional embeddings
        head_dim = n_embd // n_head
        self.rotary_emb = RotaryPositionalEmbedding(head_dim)

    def forward(self, x, attn_mask=None, is_causal=True, need_weights=False):
        B, T, C = x.size()

        # Compute key, query, value
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Rotary embeddings
        rotary_pos_emb = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, rotary_pos_emb)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Aggregate
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_drop(self.proj(y))
        return y



# ----------------------
#  RMSNorm
# ----------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Scales the input by the RMS of the last dimension
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # RMS-based normalization
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Float cast for numerical stability, then revert
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        """
        Args:
            max_len (int): Maximum sequence length.
            d_model (int): Dimensionality of embeddings.
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Token embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Token embeddings with added positional encodings.
        """
        batch_size, seq_len, d_model = x.size()
        # Create position indices (0, 1, ..., seq_len-1) for each sample in the batch.
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        # Get the positional embeddings.
        pos_enc = self.pos_embedding(positions)  # (batch_size, seq_len, d_model)
        # Add positional embeddings to token embeddings.
        return x + pos_enc
#
# ----------------------
#  Transformer Block
# ----------------------
class Block(nn.Module):
    """
    A single Transformer block with:
    - RMSNorm
    - Causal Self-Attention
    - MLP
    """

    def __init__(self, max_tokens, n_embd, n_head, resid_pdrop, attn_pdrop,rotary=False, **kwargs):
        super().__init__()
        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)
        self.pos_encoder = None
        self.rotary = None
        if not rotary or rotary == 'learnable':
            self.attn = torch.nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=attn_pdrop, batch_first=True)
        else:
            self.rotary = True
            self.attn = CausalSelfAttention(max_tokens, n_embd, n_head, attn_pdrop, attn_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, attn_mask=None):

        attn_mask_flag = attn_mask is None

        attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(torch.bool) if  attn_mask is None else attn_mask
        if self.rotary is None:
            x = x + self.attn(self.rms1(x), self.rms1(x), self.rms1(x), is_causal= attn_mask_flag , need_weights=False, attn_mask=attn_mask)[0]
        else:
            x = x + self.attn(self.rms1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.rms2(x))#
        return x


# ----------------------
#  GPT Model
# ----------------------
class GPT(nn.Module):
    """
    GPT-like Transformer model for next-token prediction.
    Removed optimizer config to keep it purely a model definition.
    """

    def __init__(
        self, max_tokens: int, vocab_size: int, n_embd: int, n_layer: int, n_head: int, embd_pdrop: float, resid_pdrop: float, attn_pdrop: float, n_conds: int = 0, binary_out: bool = False, rotary= False, multiclass=False, **kwargs
    ):

        super().__init__()
        self.block_args = {
            "max_tokens": max_tokens,
            "n_embd": n_embd,
            "n_head": n_head,
            "resid_pdrop": resid_pdrop,
            "attn_pdrop": attn_pdrop,
            "rotary": rotary
        }

        self.pos = LearnablePositionalEncoding(max_tokens, n_embd) if rotary == 'learnable' else None
        self.max_tokens = max_tokens
        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.config = DummyConfig('gpt')
        # Optional property embedding
        self.cond_emb = None
        if n_conds > 0:
            self.cond_emb = nn.Linear(n_conds, n_embd)

        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[Block(**self.block_args) for _ in range(n_layer)])
        self.rms_f = RMSNorm(n_embd)

        # Final projection to vocab logits
        self.head = nn.Linear(n_embd, vocab_size, bias=True)

        # Initialize weights
        self.apply(self._init_weights)
        if binary_out:
            self.cls_token = nn.Parameter(torch.randn(n_embd))
            self.activation_fn = nn.GELU()
            self.class_drop = nn.Dropout(embd_pdrop)
            self.dense = nn.Linear(n_embd, 4 if multiclass else 1)


            self.binary_out = nn.Linear(10, 1)

        else:
            self.binary_out = None

    def forward(self, input_ids=None, conds=None, attn_mask=None, **kwargs):
        # Allow compatibility: if input_ids is None, try to use 'input_ids' from kwargs.
        if input_ids is None and "idx" in kwargs:
            input_ids = kwargs["idx"]
        if input_ids is None:
            raise ValueError("No input_ids provided.")
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        b, t = input_ids.size()
        if t > self.get_max_tokens():
            raise ValueError(f"Cannot forward, block size is {self.get_max_tokens()}, but input seq length is {t}")
        # Token embeddings
        token_embeddings = self.tok_emb(input_ids)
        if self.pos is not None:
            token_embeddings = self.pos(token_embeddings)
        x = self.drop(token_embeddings)

        # Add property embeddings if specified
        if conds is not None:
            cond_emb = self.cond_emb(conds).unsqueeze(1).expand(-1, t, -1)
            x += cond_emb
        if self.binary_out is not None:
            x = torch.cat([x, self.cls_token.unsqueeze(0).expand(b, -1, -1)], dim=1)
            padding_mask = torch.cat(((input_ids==3).to(input_ids.device), torch.zeros(b, 1).to(input_ids.device)), dim=1).to(input_ids.device).bool()

        # Transformer blocks]
        for b in self.blocks:
            x = b(x,attn_mask=attn_mask, )

        # Final output logits
        logits = self.head( self.rms_f(x))
        if self.binary_out is not None:
            x[padding_mask] = 0
            assert not (x[:,-1]==0).all(), "Last token should not be masked"
            bce_logits = self.dense(x[:, -1, :])

            return logits, bce_logits
        return logits

    def get_max_tokens(self):
        return self.max_tokens

    def _init_weights(self, module):
        """
        Standard weight initialization for GPT-like models.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Simply return a dictionary with input_ids.
        # Adapt this if your generation requires more than just input_ids.
        return {"input_ids": input_ids}