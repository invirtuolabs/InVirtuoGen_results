# utils/callbacks.py
import math
import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
import yaml
from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.cli import SaveConfigCallback
from torch.utils.data import DataLoader, IterableDataset

# (Assume here that you have your BucketSampler, DistributedBucketSampler,
# and BucketDataset classes defined exactly as you provided.)


class BucketDatasetStateCallback(Callback):
    """
    A callback that saves and restores the state of your BucketDataset
    so that you can resume training mid-epoch without fast-forwarding the data.
    """

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # We assume the training DataLoader’s dataset is a BucketDataset.
        if hasattr(pl_module, "train_dataloader"):
            dataloader = pl_module.train_dataloader()
            if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "state_dict"):
                checkpoint["bucket_dataset_state"] = dataloader.dataset.state_dict()
                print("Saved bucket dataset state.")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Restore the state if it exists in the checkpoint.
        if "bucket_dataset_state" in checkpoint:
            if hasattr(pl_module, "train_dataloader"):
                dataloader = pl_module.train_dataloader()
                if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "load_state_dict"):
                    dataloader.dataset.load_state_dict(checkpoint["bucket_dataset_state"])
                    print("Loaded bucket dataset state.")




import os
import yaml
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

class DynamicCheckpointCallback(Callback):
    def __init__(self, base_dir: str = "checkpoints"):
        super().__init__()
        self.base_dir = base_dir

    def on_train_start(self, trainer, pl_module):
        # Only proceed on the main process
        if trainer.is_global_zero:
            # Check if the logger is WandbLogger
            if isinstance(trainer.logger, pl.loggers.WandbLogger):
                wandb_exp = trainer.logger.experiment
                wandb_run_name = None

                # Try to get the run name properly.
                if hasattr(wandb_exp, "name"):
                    if isinstance(wandb_exp.name, str):
                        wandb_run_name = wandb_exp.name
                    elif callable(wandb_exp.name):
                        wandb_run_name = wandb_exp.name()

                # Fallbacks if no run name is available.
                if not wandb_run_name:
                    if hasattr(wandb_exp, "id") and wandb_exp.id:
                        wandb_run_name = str(wandb_exp.id)
                    else:
                        wandb_run_name = f"default_run_{int(time.time())}"

                checkpoint_dir = os.path.join(self.base_dir, wandb_run_name)
                self.checkpoint_dir = checkpoint_dir
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Find existing ModelCheckpoint callback and update its settings
                found = False
                for cb in trainer.callbacks:
                    if isinstance(cb, ModelCheckpoint):
                        cb.dirpath = checkpoint_dir
                        cb.filename = "checkpoint"
                        cb.save_top_k = 2
                        print(f"Updated ModelCheckpoint dirpath to: {checkpoint_dir}")
                        found = True

                if not found:
                    # Create and add a new ModelCheckpoint callback.
                    new_ckpt = ModelCheckpoint(
                        monitor="val_loss",
                        mode="min",  # Lower val_loss is better.
                        save_top_k=2,
                        dirpath=checkpoint_dir,
                        filename="checkpoint",
                    )
                    trainer.callbacks.append(new_ckpt)
                    print(f"Added new ModelCheckpoint with dirpath: {checkpoint_dir}")

                # Save hyperparameters to config.yaml in the checkpoint directory.

                config_file = os.path.join(checkpoint_dir, "config.yaml")
                try:
                    with open(config_file, "w") as outfile:
                        yaml.dump(pl_module.hparams, outfile, default_flow_style=False)
                except:
                    simple_hparams = {
                        k: v
                        for k, v in vars(pl_module.hparams).items()
                        if isinstance(v, (str, bool, int, float, list, dict))
                    }
                    with open(config_file, "w") as outfile:

                        yaml.dump(simple_hparams, outfile, default_flow_style=False)
                print(f"Saved hyperparameters to {config_file}")

                self.checkpoint_dir_set = True
            else:
                print("Logger is not WandbLogger. Checkpoints will use default settings.")

class DebugCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Debug mode enabled: Training has started.")

    def on_train_end(self, trainer, pl_module):
        print("Debug mode enabled: Training has ended.")
