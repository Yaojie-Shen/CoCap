# -*- coding: utf-8 -*-
# @Time    : 8/6/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : compressed_video_captioner.py

__all__ = [
    "CaptionHead",
    "CompressedVideoCaptioner",
    "caption_head_cfg",
    "caption_head_pretrained_cfg",
    "compressed_video_captioner_cfg",
    "compressed_video_captioner_pretrained_cfg",
]

import logging
from typing import *

import numpy as np
import torch
from easydict import EasyDict as edict
from hydra_zen import builds
from torch import Tensor
from torch import nn

from cocap.modules.bert import BertSelfEncoder, BertLMPredictionHead
from cocap.modules.clip.clip import get_model_path
from cocap.modules.clip.model import CLIP
from cocap.modules.compressed_video.compressed_video_transformer import CompressedVideoTransformer, \
    compressed_video_transformer_pretrained_cfg, compressed_video_transformer_cfg

logger = logging.getLogger(__name__)


class CaptionHead(nn.Module):

    def __init__(
            self,
            word_embedding_size: int, visual_feature_size: int,
            max_v_len: int, max_t_len: int, hidden_size: int,
            vocab_size: int, verbose: Optional[Union[int, bool]] = False
    ):
        super(CaptionHead, self).__init__()
        self.model_network = "Self"
        self.cap_config = edict(
            word_vec_size=word_embedding_size,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=hidden_size,
            video_feature_size=visual_feature_size,
            layer_norm_eps=1e-12,  # bert layernorm
            hidden_dropout_prob=0.1,  # applies everywhere except attention
            num_hidden_layers=2,  # number of transformer modules
            num_attention_heads=8,
            share_wd_cls_weight=False,
            vocab_size=vocab_size,
            BOS_id=vocab_size - 2,
            EOS_id=vocab_size - 1,
            PAD_id=0
        )
        logger.debug("Caption Head Configuration: %s", self.cap_config)
        self.cap_sa_decoder = BertSelfEncoder(self.cap_config)
        self.prediction_head = BertLMPredictionHead(self.cap_config, self.cap_sa_decoder.word_embeddings.weight)
        # debug output cfgs
        if verbose:
            if isinstance(verbose, bool):
                self.log_interval = 1
            else:
                self.log_interval = int(verbose)
        else:
            self.log_interval = float("inf")
        self.step_counter = 1

    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        predict_ids = predict_scores.max(-1)[1]
        return CaptionHead.ids2text(predict_ids)

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids: Union[np.ndarray, Tensor]):
        from cocap.trainer.cocap_trainer import convert_ids_to_sentence
        if isinstance(gt_ids, np.ndarray) or isinstance(gt_ids, Tensor):
            assert 0 < len(gt_ids.shape) <= 2, f"gt_ids should be a 1 dim or 2 dim array/tensor, got {gt_ids.shape}"
        else:
            raise ValueError("gt_ids should be np.ndarray or Tensor")
        if isinstance(gt_ids, Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()
        if len(gt_ids.shape) == 1:
            return convert_ids_to_sentence(gt_ids.tolist())
        else:
            return [convert_ids_to_sentence(_gt_ids) for _gt_ids in gt_ids.tolist()]

    def forward(self, visual_output, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"

        input_types = torch.concat(
            [
                torch.full((visual_output["feature_context"].size(0), visual_output["feature_context"].size(1)),
                           fill_value=1, dtype=torch.long, device=visual_output["feature_context"].device),
                torch.full((visual_output["feature_action"].size(0), visual_output["feature_action"].size(1)),
                           fill_value=0, dtype=torch.long, device=visual_output["feature_action"].device),
                torch.full((input_ids.size(0), input_ids.size(1)),
                           fill_value=2, dtype=torch.long, device=input_ids.device)
            ], dim=1
        )
        visual_output = torch.cat([visual_output["feature_context"], visual_output["feature_action"]], dim=1)
        input_mask = torch.concat(
            [
                torch.ones(size=(visual_output.size(0), visual_output.size(1)),
                           dtype=torch.long, device=visual_output.device),
                input_mask
            ], dim=1
        )
        hidden = self.cap_sa_decoder.forward(visual_output, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        if self.step_counter % self.log_interval == 0:
            logger.debug("GT  : %s", self.ids2text(input_ids))
            logger.debug("Pred: %s", self.probability2text(prediction_scores))
        self.step_counter += 1
        return prediction_scores

    @classmethod
    def from_pretrained(
            cls,
            pretrained_clip_name_or_path: str = "ViT-B/16", max_v_len: int = 8 * 2, max_t_len: int = 77,
            verbose: Optional[Union[int, bool]] = False
    ):
        model_path = get_model_path(pretrained_clip_name_or_path, download_root="model_zoo/clip_model")
        pretrained_model: CLIP = torch.jit.load(model_path, map_location="cpu")
        state_dict = pretrained_model.state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]

        head = cls(
            word_embedding_size=transformer_width,
            visual_feature_size=embed_dim,
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=embed_dim,
            vocab_size=vocab_size,
            verbose=verbose
        )
        logger.debug(
            "Pretrained embedding parameters: %s",
            [k for k, v in state_dict.items() if k.startswith("token_embedding")]
        )
        pretrained_embedding = {k.lstrip("token_embedding."): v for k, v in state_dict.items()
                                if k.startswith("token_embedding")}
        head.cap_sa_decoder.word_embeddings.load_state_dict(pretrained_embedding, strict=True)
        head.prediction_head.decoder.load_state_dict(pretrained_embedding, strict=True)
        assert torch.equal(head.cap_sa_decoder.word_embeddings.weight, head.prediction_head.decoder.weight)
        return head


class CompressedVideoCaptioner(nn.Module):

    def __init__(
            self,
            compressed_video_transformer: CompressedVideoTransformer,
            caption_head: CaptionHead,
            motion_dropout_prob: float = 0.2,
            residual_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.compressed_video_transformer = compressed_video_transformer
        self.caption_head = caption_head

        self.dropout_motion = nn.Dropout(motion_dropout_prob)
        self.dropout_residual = nn.Dropout(residual_dropout_prob)

    def forward(self, inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]]):
        """

        :param inputs:
            video:
                iframe:         batch_size n_gop c h w
                motion_vector:  batch_size n_gop n_mv c=4|9 h/4 w/4
                residual:       batch_size n_gop n_res c h w
                input_mask_gop: batch_size n_gop
                input_mask_mv:  batch_size n_gop n_mv
        :return:
        """
        if "visual_output" not in inputs:
            iframe = inputs["video"]["iframe"]
            motion = inputs["video"]["motion_vector"]
            residual = inputs["video"]["residual"] / 128 - 1  # for saving memory
            bp_type_ids = inputs["video"]["type_ids_mv"]

            motion = self.dropout_motion(motion)
            residual = self.dropout_residual(residual)
            compressed_visual_features = self.compressed_video_transformer(
                iframe=iframe,
                motion=motion,
                residual=residual,
                bp_type_ids=bp_type_ids
            )
        else:
            # reuse pre-extracted visual features
            compressed_visual_features = inputs["visual_output"]

        prediction_scores = self.caption_head(
            compressed_visual_features,
            inputs["input_ids"],
            inputs["input_mask"],
        )
        return {"prediction_scores": prediction_scores, "visual_output": compressed_visual_features}


# Build configs for organizing modules with hydra
caption_head_cfg = builds(CaptionHead, populate_full_signature=True)
caption_head_pretrained_cfg = builds(CaptionHead.from_pretrained, populate_full_signature=True)

compressed_video_captioner_cfg = builds(
    CompressedVideoCaptioner,
    compressed_video_transformer=compressed_video_transformer_cfg,
    caption_head=caption_head_cfg,
    populate_full_signature=True
)
compressed_video_captioner_pretrained_cfg = builds(
    CompressedVideoCaptioner,
    compressed_video_transformer=compressed_video_transformer_pretrained_cfg,
    caption_head=caption_head_pretrained_cfg,
    populate_full_signature=True
)
