# MatConvNet Feature Extraction Demo
This project demonstrates manual implementation of CNN forward and reverse operations using MatConvNet in MATLAB(MATLAB CNN toolbox).

Instead of relying on high-level deep learning toolboxes, this project implemented convolution, pooling, unpooling, and deconvolution operations layer by layer based on the VGG16 architecture.

## What I did
- Loaded pretrained VGG16 model (using MatConvNet)
- Performed layer-wise convolution and pooling operations
- Rescaled and projected intermediate feature representations
- Reconstructed images by manually reversing pooling and convolution operations
- Implemented filter-only operations without standard end-to-end feedforward inference

## Technical Highlights
- Manual layer-by-layer CNN execution
- Explicit control of intermediate feature maps
- Custom feature rescaling and projection
- Reverse mapping from feature space to image space
- MATLAB-based CNN implementation using MatConvNet

This project emphasizes understanding of:
- Convolutional filter behavior
- Feature transformation across layers
- Pooling and unpooling mechanics
- Deconvolution-based reconstruction

*** The key point of these operations are basically hand crafted using "MatConvNet" rather than referencing the preexisting vgg model in the deep-learning toolbox ***

## Files
- from_NSD_creatANDgen_deconv.m – main script

## Purpose
Exploration of generating deep feature-controlled images using Matlab 

## Pipeline
```
┌─────────────────┐
│  Colored Images │
└────────┬────────┘
         ▼
┌─────────────────────────────────────┐
│  filter-only convolution operation  │
└─────────────────┬───────────────────┘
                  ▼
        ┌────────────────────┐
        │  pooling operation │
        └─────────┬──────────┘
                  .
                  .
                  ▼
┌─────────────────────────────────────┐
│   Rescaling, combining, projecting  │
└─────────────────┬───────────────────┘
                  .
                  .
                  ▼
        ┌────────────────────┐
        │  unpooling operation │
        └─────────┬──────────┘
                  ▼
┌─────────────────────────────────────┐
│ filter-only deconvolution operation │
└─────────────────┬───────────────────┘
                  ▼
          ┌───────────────┐
          │  output images │
          └───────────────┘
```

## Why This Matters (CV Engineering Perspective)

Many applications use high-level deep learning frameworks where internal operations are abstracted away.

This project demonstrates:
- Structural understanding of CNN architecture
- Ability to operate directly on feature maps
- Capability to modify and control intermediate representations
- Experience working with deep learning toolkits (MatConvNet)
---
