# Quaternion Wavelet Transform for Neural Models in Medical Imaging

This repository is under construction.

[Luigi Sigillo](), [Eleonora Grassucci](), [Aurelio Uncini](), and [Danilo Comminiello]()

## Generalizing Medical Image Representations via Quaternion Wavelet Networks

Official implementation of [Generalizing Medical Image Representations via Quaternion Wavelet Networks](https://arxiv.org/abs/2310.10224), ArXiv preprint:2310.10224, 2023.

#### Abstract

Neural network generalizability is becoming a broad research field due to the increasing availability of datasets from different sources and for various tasks. This issue is even wider when processing medical data, where a lack of methodological standards causes large variations being provided by different imaging centers or acquired with various devices and cofactors. To overcome these limitations, we introduce a novel, generalizable, data- and task-agnostic framework able to extract salient features from medical images. The proposed quaternion wavelet network (QUAVE) can be easily integrated with any pre-existing medical image analysis or synthesis task, and it can be involved with real, quaternion, or hypercomplex-valued models, generalizing their adoption to single-channel data. QUAVE first extracts different sub-bands through the quaternion wavelet transform, resulting in both low-frequency/approximation bands and high-frequency/fine-grained features. Then, it weighs the most representative set of sub-bands to be involved as input to any other neural model for image processing, replacing standard data samples. We conduct an extensive experimental evaluation comprising different datasets, diverse image analysis, and synthesis tasks including reconstruction, segmentation, and modality translation. We also evaluate QUAVE in combination with both real and quaternion-valued models. Results demonstrate the effectiveness and the generalizability of the proposed framework that improves network performance while being flexible to be adopted in manifold scenarios.

#### Cite

```
@Article{GrassucciEntropy2021,
author = {Sigillo, L. and Grassucci, E. and Uncini, A. and Comminiello, D.},
title = {Generalizing Medical Image Representations via Quaternion Wavelet Networks},
journal = {ArXiv preprint: arXiv:2310.10224},
year = {2023},
}
```

## GROUSE: A Task and Model Agnostic Wavelet-Driven Framework for Medical Imaging

Official implementation of [GROUSE: A Task and Model Agnostic Wavelet-Driven Framework for Medical Imaging](https://ieeexplore.ieee.org/abstract/document/10268972), IEEE Signal Processing Letters, 2023.

#### Abstract

In recent years, deep learning has permeated the field of medical image analysis gaining increasing attention from clinicians. However, medical images always require specific preprocessing that often includes downscaling due to computational constraints. This may cause a crucial loss of information magnified by the fact that the region of interest is usually a tiny portion of the image. To overcome these limitations, we propose GROUSE, a novel and generalizable framework that produces salient features from medical images by grouping and selecting frequency sub-bands that provide approximations and fine-grained details useful for building a more complete input representation. The framework provides the most enlightening set of bands by learning their statistical dependency to avoid redundancy and by scoring their informativeness to provide meaningful data. This set of representative features can be fed as input to any neural model, replacing the conventional image input. Our method is task- and model-agnostic, thus it can be generalized to any medical image benchmark, as we extensively demonstrate with different tasks, datasets, and model domains. We show that the proposed framework enhances model performance in every test we conduct without requiring ad-hoc preprocessing or network adjustments.

#### Cite

```
@Article{GrassucciEntropy2021,
author = {Grassucci, E. and Sigillo, L. and Uncini, A. and Comminiello, D.},
title = {GROUSE: A Task and Model Agnostic Wavelet-Driven Framework for Medical Imaging},
journal = {IEEE Signal Processing Letters},
volume = {30},
year = {2023},
pages = {1397 - 1401},
}
```
