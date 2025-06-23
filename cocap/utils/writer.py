# -*- coding: utf-8 -*-
# @Time    : 2022/11/13 00:41
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : writer.py

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


class DummySummaryWriter:
    """
    Issue: https://github.com/pytorch/pytorch/issues/24236
    """

    def __init__(*args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


def get_writer(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        return SummaryWriter(*args, **kwargs)
    else:
        return DummySummaryWriter()
