# DeltaNet PyTorch Minimal implementation

This repo contains a minimal implementation of the DeltaNet proposed by [Parallelizing Linear Transformers with the Delta Rule
over Sequence Length](https://arxiv.org/abs/2406.06484) with **just PyTorch**.

The official implementation requires [flash-linear-attention](https://www.github.com/fla-org/flash-linear-attention) to run, which only supports Linux. This code can be run with any platform that supports PyTorch (e.g., MacOS and Windows).

## How to Run?

See `generate_with_deltanet.ipynb` for an example.
