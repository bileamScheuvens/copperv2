import os

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.constants import ROOT
from src.data.pubmeddata import PubMedDatamodule
from src.models.copper import CoPPeR
from src.utils.configs import ExperimentConfig, as_dict, get_wandb_names

import time
import torch
from lightning.pytorch.callbacks import Callback


class StepTimer(Callback):
    """Prints dataloader wait time and model (train step) time for every batch."""

    def __init__(self):
        self._last_end = None

    def _now(self, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        return time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._last_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start = time.perf_counter()
        if self._last_end is not None:
            print(f"[{batch_idx}] data wait: {self._start - self._last_end:.4f}s")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._last_end = self._now(pl_module)
        print(f"[{batch_idx}] model:     {self._last_end - self._start:.4f}s")


def train_copper(cfg: ExperimentConfig):
    # prepare dirs
    EXPERIMENT_DIR = ROOT / "experiments" / cfg.experiment_name
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # start trainer
    experiment_name, group_name = get_wandb_names(cfg)
    run = wandb.init(
        name=experiment_name,
        group=group_name,
        config=as_dict(cfg),
        project=cfg.logger.project,
        mode=cfg.logger.mode,  # ty:ignore[invalid-argument-type]
        resume="allow",
    )

    model = CoPPeR(cfg.model, load_encoder=False)
    datamodule = PubMedDatamodule(cfg.data)
    trainer = L.Trainer(
        **as_dict(cfg.trainer),
        logger=WandbLogger(save_dir=EXPERIMENT_DIR, experiment=run),
        default_root_dir=EXPERIMENT_DIR,
        callbacks=[
            ModelCheckpoint(
                dirpath=EXPERIMENT_DIR / "checkpoints",
                every_n_train_steps=50,
                save_on_exception=True,
            ),
            # EarlyStopping("val/loss", patience=patience),
            TQDMProgressBar(),
            StepTimer(),
        ],
    )

    # train
    trainer.fit(model, datamodule, ckpt_path=cfg.model.ckpt_path, weights_only=False)
    # cleanup (necessary for multirun)
    wandb.finish()
    del model
    del trainer
    del datamodule
