# Machine-Learning Learning Lab

This directory contains hands-on deep-learning exercises completed while
studying the fundamentals of deep learning with PyTorch.

These experiments are separate from the production FastAPI application.

## Current Lab: MNIST Data Inspection

The MNIST dataset contains grayscale images of handwritten digits from 0 to 9.

For this classification task:

- **Features:** the pixel values of each handwritten-digit image
- **Label:** the correct digit represented by the image

## Tensors

A tensor is a multidimensional numerical structure used by PyTorch.

Examples include:

- Scalar: one number
- Vector: one-dimensional sequence
- Matrix: rows and columns
- Image tensor: channels, height and width
- Image batch: multiple image tensors grouped together

MNIST batches use this shape:

```text
[batch_size, channels, height, width]