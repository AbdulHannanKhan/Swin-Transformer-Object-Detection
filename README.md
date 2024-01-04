# Efficient Traffic Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Updates

***04/01/2024*** Code Released

## Results and Models

### LSFM on NuImages

| Backbone | Pretrain | Lr Schd | mAP | RTOP | FPS | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HRNet | ImageNet-1K | 24x | 48.1 | 33.5 | 14.3 | [config](configs/tju/lsfm_nuim_4x4.py) | |
| ConvMLP Pin | ImageNet-1K | 24x | 46.1 | 46.1 | 30.3 | [config](configs/tju/lsfm_tiny_nuim_4x4.py) | |
