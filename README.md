# Accurate and Fast Compressed Video Captioning

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/accurate-and-fast-compressed-video-captioning/video-captioning-on-msr-vtt-1)](https://paperswithcode.com/sota/video-captioning-on-msr-vtt-1?p=accurate-and-fast-compressed-video-captioning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/accurate-and-fast-compressed-video-captioning/video-captioning-on-msvd-1)](https://paperswithcode.com/sota/video-captioning-on-msvd-1?p=accurate-and-fast-compressed-video-captioning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/accurate-and-fast-compressed-video-captioning/video-captioning-on-vatex-1)](https://paperswithcode.com/sota/video-captioning-on-vatex-1?p=accurate-and-fast-compressed-video-captioning)


✨This is the official implementation of ICCV 2023 paper *[Accurate and Fast Compressed Video Captioning](https://arxiv.org/abs/2309.12867)*.

🚀 This code is a revised version of the original release, incorporating Hydra and PyTorch Lightning. For the original implementation, please refer to [this commit](https://github.com/Yaojie-Shen/CoCap/tree/initial_release).

## Introduction

In this work, we propose an end-to-end video captioning method based on compressed domain information from the encoded H.264 videos. Our approach aims to accurately generate captions for compressed videos in a fast and efficient manner.

![Framework](./assets/framework.svg)

By releasing this code, we hope to facilitate further research and development in the field of compressed video processing. If you find this work useful in your own research, please consider citing our paper as a reference.

## Preparation

### 1. Install the Requirements

To run the code, please install the dependency libraries by using the following command:

```shell
sudo apt update && sudo apt install default-jre -y  # required by pycocoevalcap
pip3 install -e . # See `requirements.txt` for exact versions used in development
```

Additionally, you will need to install the compressed video reader as described in the README.md of [Compressed-Video-Reader](https://github.com/yaojie-shen/Compressed-Video-Reader).


### 2. Prepare the Pretrained Models

Our model is based on the pretrained CLIP. You can run the following script to download the weights before training to avoid any network issues:

```bash
sudo apt update && sudo apt install aria2 -y  # install aria2
bash model_zoo/download_model.sh
```

This will download the CLIP model to `model_zoo/clip_model`. Note that this directory is hard-coded in our code.

### 3. Prepare the Data

We have conducted experiments on three video caption datasets: MSRVTT, MSVD, and VATEX. The datasets are stored in the `dataset` folder under the project root. For detailed instructions on downloading and preparing the training data, please refer to [dataset/README.md](./dataset/README.md).

## Training & Evaluation

The training is configured using YAML, and all the configurations are listed in [`configs/compressed_video`](./configs/compressed_video). You can use the following commands to run the experiments:

```shell
# msrvtt
python3 tools/train_net.py --config-name=exp/train/msrvtt_captioning
# msvd
python3 tools/train_net.py --config-name=exp/train/msvd_captioning
# vatex
python3 tools/train_net.py --config-name=exp/train/vatex_captioning
```

By default, the logs and results will be saved to `./logs/<experiment_name>/`. The loss and metrics are visualized using tensorboard.

## Citation

```text
@inproceedings{cocap,
      title={Accurate and Fast Compressed Video Captioning}, 
      author={Yaojie Shen and Xin Gu and Kai Xu and Heng Fan and Longyin Wen and Libo Zhang},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2023}
}
```
