# Pose Estimation Model 

### Description
Application for experimenting in training pose estimation model using SimCC architecture and DARK refinement.

## Prerequisites
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/releases/) (both C++ and Python)
- [CMake](https://cmake.org/download/)
- Visual Studio C++ Compiler

## Model Architecture
- Pose Model uses resnet50 as backbone, a few transpose convolution (deconvolution) layers to upscale the feature maps and finally go through SimCC head to produce x (horizontal) and y (vertical) labels.

- Then the predicted labels go through a decoding process which utilizes DARK refinement to produce the final joints coordinates.
