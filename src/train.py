import os

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.constants import ROOT
from src.data.scidocdata import ExtractionsDatamodule
from src.models.copper import CoPPeR
from src.utils.configs import ExperimentConfig, as_dict, get_wandb_names


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

    model = CoPPeR(cfg.model)
    datamodule = ExtractionsDatamodule(cfg.data, tokenizer=model.tokenizer)
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
        ],
    )

    # train
    trainer.fit(model, datamodule, ckpt_path=cfg.model.ckpt_path, weights_only=False)
    # cleanup (necessary for multirun)
    wandb.finish()
    del model
    del trainer
    del datamodule
