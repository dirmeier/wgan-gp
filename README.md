# WGAN-GP

[![ci](https://github.com/dirmeier/wgan-gp/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/wgan/actions/workflows/ci.yaml)

## About

This repository implements the [Wasserstein GAN with gradient penalty](https://arxiv.org/abs/1704.00028) loss for testing.
The implementations are in JAX and Flax/NNX.

## Example usage

An experiment where we train a WGAN-GP on MNIST can be found in [`experiments/mnist/`](experiments/mnist/).
To run the example, first download the latest release and install all dependencies via:

```bash
wget -qO- https://github.com/dirmeier/wgan-gp/archive/refs/tags/<TAG>.tar.gz | tar zxvf -
uv sync --all-groups
```

To train a model and make visualizations, call:

```bash
cd experiments/eight_gaussians_two_moons
python main.py
```

Below are the results from training the GN using the hyperparameters defined in [`experiments/mnist/config.py`](experiments/mnist/config.py).
A sample after training 20k steps (i.e., gradient steps) is shown below.

<div align="center">
  <img src="experiments/mnist/figures/samples.png" width="700">
</div>

## Installation

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/dirmeier/wgan@<TAG>
```

## Author

Simon Dirmeier <a href="mailto:simd23@pm.me">simd23 @ pm dot me</a>
