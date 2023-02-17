# ShadowFormer (AAAI'23)
This is the official implementation of the AAAI 2023 paper [ShadowFormer: Global Context Helps Image Shadow Removal](https://arxiv.org/pdf/2302.01650.pdf).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowformer-global-context-helps-image/shadow-removal-on-istd)](https://paperswithcode.com/sota/shadow-removal-on-istd?p=shadowformer-global-context-helps-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowformer-global-context-helps-image/shadow-removal-on-adjusted-istd)](https://paperswithcode.com/sota/shadow-removal-on-adjusted-istd?p=shadowformer-global-context-helps-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shadowformer-global-context-helps-image/shadow-removal-on-srd)](https://paperswithcode.com/sota/shadow-removal-on-srd?p=shadowformer-global-context-helps-image)

## Introduction
To trackle image shadow removal problem, we propose a novel transformer-based method, dubbed ShadowFormer, for exploiting non-shadow
regions to help shadow region restoration. A multi-scale channel attention framework is employed to hierarchically
capture the global information. Based on that, we propose a Shadow-Interaction Module (SIM) with Shadow-Interaction Attention (SIA) in the bottleneck stage to ef-
fectively model the context correlation between shadow and non-shadow regions. 
For more details, please refer to our [orginal paper](https://arxiv.org/pdf/2302.01650.pdf)

<p align=center><img width="80%" src="doc/pipeline.jpg"/></p>

<p align=center><img width="80%" src="doc/details.jpg"/></p>

## Requirement
* Python 3.7
* Pytorch 1.7
* CUDA 11.1
```bash
pip install -r requirements.txt
```

## Datasets
* ISTD [[link]](https://github.com/DeepInsight-PCALab/ST-CGAN)  
* ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID)
* SRD (please email the [authors](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf) to get assess)

## Pretrained models
[ISTD]() | [ISTD+]() | [SRD]()

Please download the corresponding pretrained model and modify the `weights` in `test.py`.

## Test
You can directly test the performance of the pre-trained model as follows

## Train

## Evaluation

## Results

## References
Our implementation is based on [Uformer](https://github.com/ZhendongWang6/Uformer). We would like to thank them.

Citation
-----
Preprint available [here](https://arxiv.org/pdf/2302.01650.pdf). 

In case of use, please cite our publication:

L. Guo, S. Huang, D. Liu, H. Cheng and B. Wen, "ShadowFormer: Global Context Helps Image Shadow Removal," AAAI 2023.

Bibtex:
```
@article{guo2023shadowformer,
  title={ShadowFormer: Global Context Helps Image Shadow Removal},
  author={Guo, Lanqing and Huang, Siyu and Liu, Ding and Cheng, Hao and Wen, Bihan},
  journal={arXiv preprint arXiv:2302.01650},
  year={2023}
}
```

## Contact
If you have any questions, please contact lanqing001@e.ntu.edu.sg
