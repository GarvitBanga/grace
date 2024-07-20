# GRACE - GRAdient ComprEssion for distributed deep learning

## Updated GRACE for new version of Horovod in Pytorch
Updated Patch files



## Context
Powerful computer clusters are used nowadays to train complex deep neural networks (DNN) on large datasets. Distributed training workloads increasingly become communication bound. For this reason, many lossy compression techniques have been proposed to reduce the volume of transferred data, and consequently, accelerate training.

## What is GRACE?
GRACE is a GRAdient ComprEssion framework for distributed deep learning.
It provides a general framework and an API for implementing gradient compression methods.
GRACE includes the implementation of popular compression algorithms surveyed in [GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://sands.kaust.edu.sa/papers/grace.icdcs21.pdf).
GRACE supports both TensorFlow and Pytorch, and offers easy integration with training scripts. This simplifies
the process of implemeting and comparing across compression methods.
