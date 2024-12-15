# Image Denoiser and Image Segmentation
Optimising neural networks for low-resource devices

## Project Overview

This project focuses on optimizing neural networks for low-resource devices by developing lightweight image denoising and segmentation models. The research aims to reduce computational demands while maintaining high performance, making deep learning models more suitable for real-world deployment.


## Key Objectives

- Design a lightweight U-Net for image segmentation
- Reduce model size and computational complexity
- Develop an efficient AutoEncoder-based denoising approach
- Maintain high accuracy with optimized models

## Major Achievements

- **Model Size Reduction:** 12x smaller U-Net model
- **Computational Efficiency:** 3.2x reduction in multiply-accumulate operations (MACs)
- **Performance Improvements:**
  - 17% faster inference time
  - 52% faster training time
  - Maintained segmentation and denoising accuracy

## Technical Approach

### Lightweight U-Net Design
- Depth-wise separable convolutions
- Channel pruning
- Quantization-aware training (QAT)
- Lightweight attention mechanisms

### AutoEncoder-based Denoising
- Compressed encoder-decoder architecture
- Skip connections for feature preservation
- Efficient latent space representation

## Performance Metrics

### U-Net Optimization

| Metric              | Original U-Net | Optimized U-Net |
|---------------------|----------------|-----------------|
| Model Size (MB)     | 240            | 20              |
| MACs (Giga)         | 60             | 18.75           |
| Segmentation mIoU   | 84.2%          | 83.6%           |
| Training Time (hrs) | 24             | 11.5            |
| Inference Time (ms) | 120            | 100             |

### AutoEncoder Optimization

| Metric               | Baseline | Optimized |
|----------------------|----------|-----------|
| Training Time (hrs)  | 20       | 9.6       |
| Inference Time (ms)  | 50       | 41.5      |
| Denoising PSNR (dB)  | 32.1     | 32.0      |



## Datasets

- Denoising: BSDS500
- Segmentation: Pascal VOC

## Future Work

1. Extend lightweight U-Net to 3D segmentation tasks
2. Explore advanced model compression techniques
3. Develop on-device learning capabilities for mobile platforms

## References

1. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."
2. Zhang, K., et al. (2017). "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising."
