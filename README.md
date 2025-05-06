# DeltaNet PyTorch Minimal implementation

A single-file implementation of the DeltaNet with pure PyTorch.

The model was proposed by [Parallelizing Linear Transformers with the Delta Rule
over Sequence Length](https://arxiv.org/abs/2406.06484). The official implementation requires [flash-linear-attention](https://www.github.com/fla-org/flash-linear-attention) to run, which only supports Linux. This code can be run with any platform that supports PyTorch (e.g., MacOS and Windows).

This also supports both forward and backward pass, since it's purely written with PyTorch. But the speed is worse.

## Installation

You pretty much only need PyTorch, einops, and safetensors (for loading the weights). This code was tested with:
- PyTorch 2.6
- Python 3.12

## How to Run?

Let's say you want to run the offical 1.3B checkpoint released by the authors at <https://huggingface.co/fla-hub/delta_net-1.3B-100B>.

1. Download the `model.safetensors` checkpoint file from: <https://huggingface.co/fla-hub/delta_net-1.3B-100B/tree/main>
2. See `generate_with_deltanet.ipynb` for an inference code example.


## Efficiency

Since this runs with pure PyTorch, it will be much slower than the kernels of flash-linear-attention.

## Acknowledgements

Some of the code was taken from:

- [flash-linear-attention](https://www.github.com/fla-org/flash-linear-attention)
- [mamba2-minimal](https://github.com/johnma2006/mamba-minimal)
