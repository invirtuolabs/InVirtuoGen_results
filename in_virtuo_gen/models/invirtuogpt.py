# model/litGPT.py

import os
from itertools import product
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..train_utils.metrics import evaluate_smiles
from ..train_utils.sampling import dynamic_temperature_sampling, selective_sampling, smooth, top_k_top_p_filtering
from ..utils.mol import calculate_average_tanimoto

# Example placeholders for your actual code:
from .gpt.model_gpt import GPT
from .invirtuobase import InVirtuoBase


class InVirtuoGPT(InVirtuoBase):
    """
    A PyTorch Lightning Module for molecule generation tasks.

    This module integrates GPT and xLSTM models for molecule generation,
    handling training, validation, and inference phases. It also includes
    optimizer configuration with parameter grouping and learning rate scheduling.

    Attributes:
        model (str): The model type ('gpt' or 'xlstm') for molecule generation.
        tokenizer (Tokenizer): Tokenizer instance for processing input and output sequences.
        best_validity (float): Best recorded validity metric during validation.
        best_uniqueness (float): Best recorded uniqueness metric during validation.
        best_diversity (float): Best recorded diversity metric during validation.
    """

    def __init__(
        self,
        vocab_size: int = 300,
        block_size: int = 150,
        n_conds: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.3,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.2,
        generation_property_value: Optional[Any] = None,
        temperature: float = 1.2,
        max_tokens: int = 150,
        gen_batch_size: int = 180,
        num_samples: int = 5,
        rotary: Any = False,
        binary_out: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Initialize GPT model
        self.model = GPT(
            max_tokens=self.hparams.max_tokens,
            vocab_size=self.hparams.vocab_size,
            n_embd=self.hparams.n_embd,
            n_layer=self.hparams.n_layer,
            n_head=self.hparams.n_head,
            embd_pdrop=self.hparams.embd_pdrop,
            resid_pdrop=self.hparams.resid_pdrop,
            attn_pdrop=self.hparams.attn_pdrop,
            n_conds=self.hparams.n_conds,
            rotary=self.hparams.rotary,
            binary_out=self.hparams.binary_out,
        )

    def forward(self, idx: torch.Tensor, conds: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GPT model.

        Args:
            idx (torch.Tensor): Input indices tensor.
            cond (Optional[torch.Tensor], optional): Optional cond tensor. Defaults to None.
            targets (Optional[torch.Tensor], optional): Optional targets tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Model outputs including logits and loss.
        """
        return self.model(idx, conds=conds)

    def calculate_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute loss for the model."""
        # Unpack batch

        if len(batch) == 3 and not isinstance(batch, torch.Tensor):
            idx, targets, conds = batch
        elif len(batch) == 2 and not isinstance(batch, torch.Tensor):
            idx, conds = batch["input_ids"].long(), batch["conds"]
            targets = idx[:, 1:]
            idx = idx[:, :-1]
        else:
            if len(batch.shape) == 1:
                batch = batch.unsqueeze(0)
            batch = batch.long()
            idx = batch.clone()[:, :-1]
            targets = batch.clone()[:, 1:]
            conds = None
        # Forward pass
        outputs = self.model(idx, conds=conds)

        # Handle multiple outputs case
        if isinstance(outputs, tuple):
            logits, cls_logits = outputs
            targets, cls_targets = targets
            # Compute combined loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_token_id)
            if cls_logits is not None and cls_targets is not None:
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets.float())
                loss = loss + cls_loss
        else:
            # Single output case
            # [batch_size, seq_length, vocab_size] -> [batch_size * seq_length, vocab_size]
            outputs = outputs.reshape(-1, outputs.size(-1))  # Using reshape instead of view
            targets = targets.reshape(-1)  # Using reshape instead of view

            # Add shape check for debugging
            if outputs.size(0) != targets.size(0):
                raise ValueError(f"Shape mismatch: outputs: {outputs.shape}, targets: {targets.shape}")

            loss = F.cross_entropy(outputs, targets, ignore_index=self.pad_token_id)
        return loss

    def sample(
        self,
        num_samples: int = 1000,
        temperature: float = 1.0,
        batch_size=None,
        conds_orig=None,
        temperature_scaling: bool = False,
        prompt=None,
        top_p: float = None,
        p: float = 2,
        T_min: float = 1,
        curve_factor: float = 0.7,
        min_temp_factor: float = 0.2,
        entropy_scaling: float = 0.8,
        min_tokens: int = 5,
        top_k: int = 30,
        **kwargs,
    ) -> List[str]:
        """
        Sample molecules from the model.

        Args:
            num_samples (int): Number of molecules to sample.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 150.
            temperature_scaling (bool): Whether to use dynamic temperature scaling.
            top_p (float, optional): If provided, use nucleus (top-p) sampling.
            p (float): Exponent used for smoothing in the original implementation (if needed).
            T_min (float): Minimum temperature for scaling.

        Returns:
            List[str]: List of sampled SMILES strings.
        """
        self.model.eval()
        generated_ids = []
        batch_size = self.hparams.gen_batch_size if batch_size is None else batch_size
        if T_min is None:
            T_min = 0.01

        with torch.no_grad():
            for k, _ in enumerate(tqdm(range(0, num_samples, batch_size))):
                current_batch_size = min(batch_size, num_samples - len(generated_ids))
                # Initialize with BOS token
                input_ids = torch.full((current_batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
                prompt_len = 0
                if prompt is not None:
                    prompt_len = prompt.shape[1]
                    input_ids = torch.cat([input_ids, prompt[k * current_batch_size : (k + 1) * current_batch_size].clone().to(self.device)], dim=1)
                # Create flag tensor to track finished sequences (those that generated EOS)
                finished = torch.zeros(current_batch_size, dtype=torch.bool, device=self.device)
                if self.hparams.n_conds > 0 and conds_orig is None:
                    conds = torch.ones(current_batch_size, 2).to(self.device).float() * 3
                    conds[:, -1] *= -1
                elif conds_orig is not None:
                    conds = conds_orig[_ : _ + current_batch_size]
                else:
                    conds = None
                for i in range(self.hparams.max_tokens - prompt_len - 1):
                    logits = self(input_ids, conds=conds)  # shape: (batch_size, seq_len, vocab_size)
                    # Take the logits for the last generated token.
                    last_logits = logits[:, -1, :]
                    if temperature_scaling:
                        # Apply dynamic temperature sampling using the full set of parameters.
                        scaled_logits = dynamic_temperature_sampling(
                            last_logits, base_temp=temperature, token_position=i, max_position=self.hparams.max_tokens - prompt_len - 1, curve_factor=curve_factor, min_temp_factor=min_temp_factor
                        )
                        probs = torch.softmax(scaled_logits, dim=-1)
                    else:
                        # assert top_p>0 or top_k>0
                        # if top_p is not None and top_p > 0:
                        #     # Use nucleus (top-p) sampling filtering.
                        #     filtered_logits = top_k_top_p_filtering(last_logits, top_k=0, top_p=top_p)
                        # else:
                        # Use selective sampling with the extra parameters.
                        filtered_logits = selective_sampling(last_logits, top_k=top_k, entropy_scaling=entropy_scaling, min_tokens=min_tokens)
                        probs = torch.softmax(filtered_logits, dim=-1)

                    # Sample the next token.
                    next_token = torch.multinomial(probs, num_samples=1)  # shape: (batch_size, 1)
                    # For sequences already finished, override next_token with pad token.
                    mask_tensor = torch.full((current_batch_size, 1), self.pad_token_id, dtype=torch.long, device=self.device)
                    next_token = torch.where(finished.unsqueeze(1), mask_tensor, next_token)
                    # Append the new token.
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    # Update finished flags for sequences that generated EOS.
                    finished |= next_token.squeeze(1) == self.eos_token_id
                    if finished.all():
                        break

                generated_ids.extend(input_ids.tolist())
        self.model.train()
        return generated_ids

        def on_fit_end(self, *args, **kwargs):
            """Initialize scheduler and loss function."""
            self = self.to("cuda")
            pairs = product([0.8, 1, 1.3, 1.5, 1.75, 2], [0])
            scan_quality_vs_diversity(
                self,
                self.datamodule,
                pairs=pairs,
                outpath=self.checkpoint_dir,
                num_samples=30000,
                log_plot=True,
                temperature_scaling=False,
                p=2,
                model_name="InVirtuoGPT",
                reference_smiles=self.reference_smiles,
            )

            # scan_quality_vs_diversity(self, self.datamodule,temperatures=[0.6,0.8,1,1.2,1.4,1.6,1.8,2], etas=[1,2,3], outpath='checkpoints', dt=self.hparams.dt, num_samples=self.hparams.num_samples, log_plot=True )
            self = self.to("cpu")
