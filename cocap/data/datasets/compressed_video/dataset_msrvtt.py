# -*- coding: utf-8 -*-
# @Time    : 2022/11/17 16:54
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : dataset_msrvtt.py

import os
import random
from collections import defaultdict
from typing import Literal

import torch
from torch.utils import data
from torchvision import transforms

from cocap.modules.clip import clip
from cocap.utils.json import load_json
from .transforms import (DictNormalize, DictCenterCrop, DictRandomHorizontalFlip)
from .video_readers import VIDEO_READER_REGISTRY
from .video_text_base import get_video, CVConfig


class MSRVTTCaptioningDataset(data.Dataset):

    def __init__(
            self,
            video_root: str,
            max_words: int,
            max_frames: int,
            unfold_sentences: False,
            video_size: tuple[int, int],
            metadata: str,
            video_reader: str,
            cv_config: CVConfig,
            split: Literal["train", "test"],
    ):
        self.split = split
        self.video_root = video_root
        self.max_words = max_words
        self.max_frames = max_frames
        self.unfold_sentences = unfold_sentences  # only affect the train split
        self.height, self.width = video_size
        self.sentences = []  # (vid, [sentence, ...])
        self.h265_cfg = cv_config
        metadata = load_json(metadata)
        video_ids = [metadata['videos'][idx]['video_id'] for idx in range(len(metadata['videos']))]
        all_split_video_ids = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497],
                               "test": video_ids[6513 + 497:]}

        split_video_ids = all_split_video_ids[split].copy()
        if self.unfold_sentences:
            for item in metadata["sentences"]:
                if item["video_id"] in split_video_ids:
                    self.sentences.append([item["video_id"], [item["caption"]]])
                    if split == "test":
                        split_video_ids.remove(item["video_id"])
        else:
            vid2sentence = defaultdict(list)
            for item in metadata["sentences"]:
                if item["video_id"] in split_video_ids:
                    vid2sentence[item["video_id"]].append(item["caption"])
            self.sentences = list(vid2sentence.items())

        # self.sentences = self.sentences[:50000]
        self.video_reader = VIDEO_READER_REGISTRY.get(video_reader)
        # transforms
        normalize = DictNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if split == "train":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                DictRandomHorizontalFlip(),
                normalize
            ])
        elif split == "test":
            self.transform = transforms.Compose([
                DictCenterCrop((self.height, self.width)),
                normalize
            ])
        else:
            raise NotImplementedError

        if split == "test":
            json_ref = {k: [] for k in all_split_video_ids[split]}
            for sentence in metadata["sentences"]:
                if sentence["video_id"] in json_ref:
                    json_ref[sentence["video_id"]].append(sentence["caption"])
            # verify
            assert all(len(v) == 20 for _, v in json_ref.items())
            self.json_ref = {k[len("video"):]: v for k, v in json_ref.items()}

    def __len__(self):
        return len(self.sentences)

    def _get_video(self, video_id):
        video, video_mask = get_video(video_reader=self.video_reader,
                                      video_path=os.path.join(self.video_root, f"{video_id}.mp4"),
                                      max_frames=self.max_frames,
                                      sample="rand" if self.split == "train" else "uniform",
                                      hevc_config=self.h265_cfg)
        if self.transform is not None:
            video = self.transform(video)
        return video, video_mask

    def __getitem__(self, idx):
        video_id, sentence_list = self.sentences[idx]
        sentence = random.choice(sentence_list)

        input_ids = clip.tokenize(sentence, context_length=self.max_words, truncate=True)[0]
        input_mask = torch.zeros(self.max_words, dtype=torch.long)
        input_mask[:len(clip._tokenizer.encode(sentence)) + 2] = 1

        video, video_mask = self._get_video(video_id)
        input_labels = torch.cat((input_ids[1:], torch.IntTensor([0])))
        return {
            # video
            "video": video,
            "video_mask": video_mask,
            # text
            "input_ids": input_ids,
            "input_labels": input_labels,
            "input_mask": input_mask,
            # metadata
            "metadata": (video_id, sentence)
        }
