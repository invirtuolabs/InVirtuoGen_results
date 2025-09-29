import os
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.nn.utils.rnn import pad_sequence
from transformers import  PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from tokenizers import Tokenizer
from ..train_utils.param_groups import get_param_groups
from ..train_utils.metrics import evaluate_smiles

from ..utils.fragments import bridge_smiles_fragments
from ..preprocess.preprocess_tokenize import custom_decode_sequence
from ..utils.fragments import bridge_smiles_fragments
import torch.distributed as dist
from ..utils.mol import  is_valid_smiles

class InVirtuoBase(pl.LightningModule):
    """A PyTorch Lightning base class for molecular generation models.

    Provides common functionality for training and evaluating generative models that produce SMILES strings,
    including learning rate scheduling, optimization, logging, and evaluation metrics.

    Attributes:
        tokenizer (PreTrainedTokenizerFast): SMILES tokenizer for encoding/decoding molecules
        model (nn.Module): Underlying model architecture (to be defined by child classes)
        criterion (nn.Module): Loss function (to be defined by child classes)
        best_validity (float): Best molecular validity score achieved during training
        best_uniqueness (float): Best molecular uniqueness score achieved during training
        best_diversity (float): Best molecular diversity score achieved during training
        best_quality (float): Best molecular quality score achieved during training

    Args:
        ckpt_path (Optional[str]): Path to model checkpoint for loading weights
        model (str): Model architecture name ("gpt" by default)
        tokenizer_path (str): Path to tokenizer file
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay factor for regularization
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        max_epochs (int): Maximum number of training epochs
        train_batch_size (int): Batch size during training
        gen_batch_size (int): Batch size during evaluation
        accumulate_grad_batches (int): Number of batches for gradient accumulation
        betas (Tuple[float, float]): Adam optimizer betas
        lr_warmup_iters (int): Number of warmup iterations for learning rate
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model: str = "gpt",
        tokenizer_path: str = "tokenizer/smiles.json",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_warmup_steps: int = 1000,
        max_epochs: int = 100,
        train_batch_size: int = 32,
        gen_batch_size: int = 180,
        betas: Tuple[float, float] = (0.9, 0.999),
        lr_warmup_iters: int = 1000,
        temperature: float = 1.0,
        dist_path: str = "configs/zinc_dist.pt",
        already_smiles: bool = False,
        safe :bool=False,
        **kwargs
    ):
        """Initialize the base model.

        Args:
            ckpt_path: Path to checkpoint for loading weights
            model: Model architecture to use
            tokenizer_path: Path to tokenizer file
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay factor
            lr_warmup_steps: Number of warmup steps for learning rate
            max_epochs: Maximum number of training epochs
            train_batch_size: Batch size during training
            gen_batch_size: Batch size during evaluation
            accumulate_grad_batches: Number of batches for gradient accumulation
        """
        super().__init__()
        self.save_hyperparameters()
        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path) if not safe else Tokenizer.from_file('tokenizer/safe.json')

        # Initialize tokenizer
        if not safe:
            self.bos_token_id = self.tokenizer.encode("[BOS]")[0]
            self.eos_token_id = self.tokenizer.encode("[EOS]")[0]
            self.unk_token_id = self.tokenizer.encode("[UNK]")[0]
            self.pad_token_id = self.tokenizer.encode("[PAD]")[0]
            self.space_token_id = self.tokenizer.encode(" ")[0]
            self.mask_token_id = len(self.tokenizer)
            self.vocab_size = len(self.tokenizer)

        else:

            self.bos_token_id = self.tokenizer.encode("[BOS]").ids[0] # type: ignore[attr-defined]
            self.eos_token_id = self.tokenizer.encode("[EOS]").ids[0] # type: ignore[attr-defined]
            self.mask_token_id = self.tokenizer.encode("[MASK]").ids[0] # type: ignore[attr-defined]
            self.pad_token_id = self.tokenizer.encode("[PAD]").ids[0] # type: ignore[attr-defined]
            assert self.mask_token_id!=3
            self.vocab_size = len(self.tokenizer.get_vocab())
        # Placeholders for model-specific components
        self.model = None
        self.criterion = None
        # Initialize best metrics
        self.best_validity = 0.0
        self.best_uniqueness = 0.0
        self.best_diversity = 0.0
        self.best_quality = 0.0
        self.best_fcd = 10000.0
        self.gen_batch_size = gen_batch_size


    def setup_model(self):
        """Setup model architecture - to be implemented by child classes."""
        raise NotImplementedError

    def num_steps(self) -> int:
        """Calculate the total number of training steps.


        Estimates training steps based on dataset size, number of devices,
        maximum epochs, and gradient accumulation settings.

        Returns:
            int: Total number of training steps across all epochs.
        """
        if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps is not None and  self.trainer.max_steps>-1 :
            return self.trainer.max_steps
        dataloader = self.trainer.datamodule.train_dataloader() # type: ignore[attr-defined]
        dataset_size = len(dataloader)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * max(1,self.trainer.max_epochs) // (self.trainer.accumulate_grad_batches ) # type: ignore[attr-defined]
        num_steps/= self.hparams.batch_size
        print("total number of steps is", num_steps)
        return num_steps

    def load_datamodule(self, datamodule: pl.LightningDataModule):
        """Load datamodule for training and validation."""
        self.datamodule = datamodule

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers.

        Sets up AdamW optimizer with parameter groups for weight decay and
        cosine learning rate schedule with warmup.

        Returns:
            dict: Configuration dictionary containing:
                - optimizer: AdamW optimizer instance
                - lr_scheduler: Dict with scheduler settings including:
                    - scheduler: Cosine schedule with warmup
                    - interval: Update frequency ('step')
                    - frequency: How often to call scheduler
                    - reduce_on_plateau: Whether to reduce on plateau
                    - monitor: Metric to monitor ('val_loss')
        """
        warmup_iters = self.hparams.lr_warmup_iters # type: ignore[attr-defined]

        # Separate parameters into decay vs. no_decay sets
        optim_groups = get_param_groups(self, self.hparams.weight_decay) # type: ignore[attr-defined]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate, # type: ignore[attr-defined]
            betas=self.hparams.betas, # type: ignore[attr-defined]
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(warmup_iters, self.num_steps() // 10),  # warmup_iters,
            num_training_steps=self.num_steps(),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'epoch'
                "frequency": 1,  # How often to call the scheduler
                "reduce_on_plateau": False,
                "monitor": "val_loss",
            },
        }

    def on_fit_start(self):
        """Initialize training steps calculations.

        Calculates total steps and steps per epoch based on dataloader length,
        maximum epochs, and gradient accumulation settings.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.trainer.is_global_zero:
            self.logger.experiment.log({"total_params": total_params}) # type: ignore[attr-defined]
            self.logger.experiment.log({"trainable_params": trainable_params}) # type: ignore[attr-defined]
        wandb_experiment = getattr(self.logger, "experiment", None)
        if wandb_experiment is not None:
            wandb_experiment.log_code(".")
        if self.trainer is not None:
            self.total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs // self.trainer.accumulate_grad_batches # type: ignore[attr-defined]
            self.steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches # type: ignore[attr-defined]
            self.load_datamodule(self.trainer.datamodule) # type: ignore[attr-defined]
        if isinstance(wandb_experiment.name, str): # type: ignore[attr-defined]
            wandb_run_name = wandb_experiment.name # type: ignore[attr-defined]
        elif callable(wandb_experiment.name): # type: ignore[attr-defined]
            wandb_run_name = wandb_experiment.name() # type: ignore[attr-defined]
        self.checkpoint_dir = os.path.join('checkpoints',wandb_run_name) # type: ignore[attr-defined]

    def calculate_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute loss for the model.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of data.

        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """
        Training step executed during each training batch.

        Args:
            batch (Tuple[torch.Tensor, ...]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        # Handle optional property tensor
        loss = self.calculate_loss(batch)
        if loss is not None:
            self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_start(self) -> None:
        self.reference_smiles=[]
        print("reference smiles", len(self.reference_smiles))

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        if len(batch) == 3:
            x_1, targets, conds = batch
        elif len(batch) == 2:
            x_1, conds = batch["input_ids"].long(), batch["conds"] # type: ignore[attr-defined]

        else:
            x_1 = batch.long().clone() # type: ignore[attr-defined]
            conds = None
        self.reference_smiles.extend(x_1.tolist())
        val_loss = self.calculate_loss(batch)
        self.log(
            "validation/val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return val_loss

    def sample_seq_lengths(self, num_samples: int, batch_size: int, min_length=0, oracle=None) -> List[torch.Tensor]:
        # ------------------------------------------------------------------------------
        # Setup and Hyperparameters
        # ------------------------------------------------------------------------------

        loaded_distribution = torch.load(f=self.hparams.dist_path, weights_only=False) # type: ignore[attr-defined]

        unique_lengths = loaded_distribution["unique_lengths"]
        max_length = max(loaded_distribution["unique_lengths"])
        probs = loaded_distribution["probs"]
        # Create a Categorical distribution over the probabilities.
        length_dist = Categorical(probs)
        if max_length<int(min_length*1.05):
            samples = torch.tensor([int(min_length*1.05)]*num_samples)
        elif min_length > 0:
            indices = length_dist.sample((200*num_samples,))  # Sample indices (shape: (num_samples,))'
            samples = unique_lengths[indices]
            samples = samples[samples > int(min_length*1.05)][:num_samples]
            while len(samples) < num_samples:
                indices = length_dist.sample((200*num_samples,))  # Sample indices (shape: (num_samples,))'
                samples = torch.cat([samples, unique_lengths[indices]], dim=0)  # Concatenate unique_lengths[indices]
                samples = samples[samples >= min_length*1.05][:num_samples]
        else:
            indices = length_dist.sample((num_samples,))  # Sample indices (shape: (num_samples,))'
            samples = unique_lengths[indices]
        if oracle is not None:
            samples = torch.tensor(oracle).to(self.device)
        B = self.gen_batch_size  # Batch size
        if oracle is None:
            samples = sorted(samples) # type: ignore[assignment]
            # For each sampled length S, create a 1D tensor of shape (S,) filled with mask_index.
        samples = [torch.full((int(length.item()),), self.mask_token_id if self.hparams.masked else 0, dtype=torch.long) for length in samples] # type: ignore[attr-defined]
        # Sorting groups samples of similar lengths together; the overall distribution remains unchanged.


        if oracle is None:
            samples = sorted(samples, key=lambda x: x.shape[0])
        # Simply take consecutive samples (after sorting) into batches of size B.
        batches = [samples[i : i + B] for i in range(0, len(samples), B)]
        padded_batches = []
        for batch in batches:
            batch_tensor = pad_sequence(batch, batch_first=True, padding_value=self.pad_token_id  )
            padded_batches.append(batch_tensor)
        return padded_batches

    def prepare_conds(self, batch,conds):
        if self.hparams.n_conds > 0 and conds is None: # type: ignore[attr-defined]
            conds = torch.zeros_like(batch[:, :n_conds]).float().to(self.device) # type: ignore[attr-defined]
        elif conds is not None:
            conds = conds.to(self.device)
        return conds
    def on_validation_epoch_end(self) -> None:
        if self.global_step > 0:
            # Only perform sampling and evaluation on rank 0
            if self.trainer.is_global_zero:
                generated_ids = self.sample(num_samples=self.hparams.num_samples) # type: ignore[attr-defined]
                if generated_ids:
                    # Here, we set return_values=True so that we get a dict of metrics
                    metrics = evaluate_smiles(
                        generated_ids,
                        tokenizer=self.tokenizer,
                        return_values=False,
                        reference_smiles=self.reference_smiles,
                        already_smiles=self.hparams.already_smiles, # type: ignore[attr-defined]
                        device=self.device # type: ignore[attr-defined]
                    )
                else:
                    # Fallback default metrics if sampling fails
                    metrics = {"validity": 0.0, "uniqueness": 0.0, "diversity": 0.0, "quality": 0.0, "fcd": 0.0}
            else:
                # For non-global-zero ranks, initialize default values
                metrics = {"validity": 0.0, "uniqueness": 0.0, "diversity": 0.0, "quality": 0.0, "fcd": 0.0}

            # Pack the metrics into a tensor for broadcast.
            # Order must be consistent across ranks.
            metrics_tensor = torch.tensor(
                [
                    metrics["validity"],
                    metrics["uniqueness"],
                    metrics["diversity"],
                    metrics["quality"],
                    metrics["fcd"],
                ],
                device=self.device
            )

            if "novelty" in metrics.keys():
                metrics_tensor = torch.cat((metrics_tensor, torch.tensor([metrics["novelty"]], device=self.device)))

            # If in distributed mode, broadcast the metrics from rank 0 to all other ranks.
            if dist.is_initialized():
                dist.broadcast(metrics_tensor, src=0)

            # Reconstruct the metrics dictionary from the broadcasted tensor.
            metrics = {
                "validity": metrics_tensor[0].item(),
                "uniqueness": metrics_tensor[1].item(),
                "diversity": metrics_tensor[2].item(),
                "quality": metrics_tensor[3].item(),
                "fcd": metrics_tensor[4].item(),
            }
            if "novelty" in metrics.keys():
                metrics["novelty"] = metrics_tensor[5].item()

            # Update best metrics on rank 0 and log collectively (all ranks must call self.log)
            if self.trainer.is_global_zero:
                if metrics["validity"] > self.best_validity:
                    self.best_validity = metrics["validity"]
                    self.log("best/best_validity", self.best_validity, rank_zero_only=True)
                if metrics["uniqueness"] > self.best_uniqueness:
                    self.best_uniqueness = metrics["uniqueness"]
                    self.log("best/best_uniqueness", self.best_uniqueness, rank_zero_only=True)
                if metrics["diversity"] > self.best_diversity:
                    self.best_diversity = metrics["diversity"]
                    self.log("best/best_diversity", self.best_diversity, rank_zero_only=True)
                if metrics["quality"] > self.best_quality:
                    self.best_quality = metrics["quality"]
                    self.log("best/best_quality", self.best_quality, rank_zero_only=True)
                if metrics["fcd"] < self.best_fcd and metrics["fcd"] > 0:
                    self.best_fcd = metrics["fcd"]
                    self.log("best/best_fcd", self.best_fcd, rank_zero_only=True)
                if "novelty" in metrics.keys():
                    if metrics["novelty"] > self.best_novelty:
                        self.best_novelty = metrics["novelty"]
                        self.log("best/best_novelty", self.best_novelty, rank_zero_only=True)
            if dist.is_initialized():
                dist.barrier()

            # Log current metrics (all processes call self.log with same keys)
            self.log("validation/validity", metrics["validity"], prog_bar=False, rank_zero_only=True)
            self.log("validation/uniqueness", metrics["uniqueness"], prog_bar=False, rank_zero_only=True)
            self.log("validation/diversity", metrics["diversity"], prog_bar=False, rank_zero_only=True)
            self.log("validation/quality", metrics["quality"], prog_bar=False, rank_zero_only=True)
            self.log("validation/fcd", metrics["fcd"], prog_bar=False, rank_zero_only=True)
            if "novelty" in metrics.keys():
                self.log("validation/novelty", metrics["novelty"], prog_bar=False, rank_zero_only=True)
    def sample(self, num_samples: int = 1000) -> List[str]:
        """
        Sample SMILES strings from the model.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            List[str]: List of generated SMILES strings.
        """
        raise NotImplementedError

    def convert_to_smiles(self, generated_ids: torch.Tensor) -> List[str]:
        # Ensure generated_string is defined outside the try/except block
        generated_string = ""
        smiles = []
        for ids in generated_ids:
            try:
                generated_string = custom_decode_sequence(self.tokenizer, ids)
                fragments_list = [frag for frag in generated_string.split() if frag]
                # If you need to do more processing, consider renaming the variable for clarity.
                full_string = bridge_smiles_fragments(fragments_list, print_flag=False)


                if is_valid_smiles(full_string):
                    smiles.append(full_string)
                else:
                    # Use a placeholder error message if needed
                   continue
            except Exception as e:
                print(e)
                continue


        return smiles

    def generate_smiles(
        self,
        gen_kwargs: Dict[str, Any],
    ) -> List[str]:
        """Execute prediction step for inference - not that this is a hack, it
        does not actually predict anything (i.e. supervised), but rather generates molecules and computes metrics.

        Generates molecules and computes their evaluation metrics.

        Args:
            batch (Tuple[Any, ...]): Batch of input data (unused).
            batch_idx (int): Index of current batch.
            gen_kwargs (Dict[str, Any]): Generation parameters passed to sample().

        Returns:
            Tuple[List[str], float, float, float, float]: Tuple containing:
                - List of generated SMILES strings
                - Validity score
                - Uniqueness score
                - Diversity score
                - Quality score
        """
        num_samples = int(1.1*gen_kwargs["num_samples"])

        generated_ids = self.sample(**gen_kwargs)
        smiles = self.convert_to_smiles(generated_ids) # type: ignore[attr-defined]
        if len(generated_ids) == 0:
            raise ValueError("No valid SMILES generated.")
        if len(smiles) <num_samples:
            num_samples = int((num_samples-len(smiles))*2)
            generated_ids = self.sample(num_samples=num_samples)
            smiles.extend(self.convert_to_smiles(generated_ids)[num_samples-len(smiles):]) # type: ignore[attr-defined]
        return smiles[:gen_kwargs["num_samples"]]


