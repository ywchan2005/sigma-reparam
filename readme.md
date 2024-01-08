# An unofficial implementation of $\sigma$-Reparam

## Overview

This repository contains an implementation of $\sigma$-Reparam, which is proposed in [Stabilizing Transformer Training by Preventing Attention Entropy Collapse](https://proceedings.mlr.press/v202/zhai23a/zhai23a.pdf) (Zhai et al. 2023) at ICML 2023.

Compared to spectral norm, $\sigma$-Reparam introduces a dimensionless learnable variable $\gamma$ to force the updates of spectral norm to be dimensionality independent.

$$
\hat{W} = \frac{\gamma}{\sigma(W)}W
$$

Feedbacks and discussions are welcome on how we could make use of $\sigma$-Reparam to enhance our models.

## Compatibility

The implementation is based on `torch.nn.utils.parametrizations.spectral_norm` in [PyTorch v2.1.0](https://github.com/pytorch/pytorch/releases/tag/v2.1.0). Incompability may arise in newer versions.

## Reference

Please refer to the [original repository](https://github.com/apple/ml-sigma-reparam) for the official implementation.
