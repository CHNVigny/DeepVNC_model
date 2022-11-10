
## Overview
This is the model that implemented on [DeepVNC](https://github.com/CHNVigny/DeepVNC).
This model is improved from [A Unified End-to-End Framework for Efficient Deep Image Compression](https://arxiv.org/abs/2002.03370).



## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)

## Prerequisites

You should install the libraries of this repo.

```
pip install -r requirements.txt
```

## Data Preparation

We need to first prepare the training and validation data.
The trainging data is from flicker.com.
You can obtain the training data according to description of [CompressionData](https://drive.google.com/file/d/1EK04NO6o3zbcFv5G-vtPDEkPB1dWblEF/view).

The validation data is the popular kodak dataset.
```
bash data/download_kodak.sh
```

## Training 

For high bitrate (1024, 2048, 4096), the out_channel_N is 192 and the out_channel_M is 320 in 'config_high.json'.
For low bitrate (128, 256, 512), the out_channel_N is 128 and the out_channel_M is 192 in 'config_low.json'.

For high bitrate of 4096, we first train from scratch as follows.

```
CUDA_VISIBLE_DEVICES=0 python train_improve_cbam_ssim.py --config examples/example/config_high.json -n baseline_8192 --train flicker_path --val kodak_path
```
For other high bitrate (1024, 2048), we use the converged model of 8192 as pretrain model and set the learning rate as 1e-5.
The training iterations are set as 500000.

The low bitrate (128, 256, 512) training process follows the same strategy.


## Author
Wang Hejun@ICA/CAEP
