# -*- coding: utf-8 -*-
# @Time    : 6/17/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : train_net.py

import logging
from pathlib import Path

import pytorch_lightning as pl
from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader

from cocap.modeling.lm_cocap import cocap_lm_cfg

logger = logging.getLogger(__name__)


def train(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        trainer: pl.Trainer
):
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    store(
        train,
        model=cocap_lm_cfg,
        train_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            populate_full_signature=True,
        ),
        val_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            populate_full_signature=True,
        ),
        trainer=builds(pl.Trainer, populate_full_signature=True),
        populate_full_signature=True,
        name="train",
    )
    store.add_to_hydra_store()

    zen(train).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        version_base=None,
    )
