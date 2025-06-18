# -*- coding: utf-8 -*-
# @Time    : 7/19/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : loss.py

__all__ = [
    "LossBase",
    "LabelSmoothingLoss",
    "label_smoothing_loss_cfg"
]

import logging
from abc import abstractmethod

import torch
import torch.nn.functional as F
from hydra_zen import builds
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class LossBase(nn.Module):
    @abstractmethod
    def forward(self, inputs, outputs) -> Tensor:
        """Compute loss."""


class LabelSmoothingLoss(LossBase):
    def __init__(self, label_smoothing=0.1, target_vocab_size=49408, ignore_index=0):
        assert 0.0 < label_smoothing <= 1.0

        super().__init__()

        self.tgt_vocab_size = target_vocab_size
        self.ignore_index = ignore_index

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (self.tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((self.tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, target, output):
        output = output["prediction_scores"]
        output = output.view(-1, self.tgt_vocab_size)
        target = target['input_labels'].reshape(-1).long()
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1).to(target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


# Build configs for organizing modules with hydra
label_smoothing_loss_cfg = builds(LabelSmoothingLoss, populate_full_signature=True)
