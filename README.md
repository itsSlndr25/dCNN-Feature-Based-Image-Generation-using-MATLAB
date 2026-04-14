# MatConvNet Feature Extraction Demo

This small project demonstrates feature extraction, CNN layer-wise operation & reverse operation to generate images using MatConvNet (MATLAB CNN toolbox).

## What I did
- Loaded pretrained vgg16 model
- Operated input images using convolution operation and pooling operation following the structure of vgg16
- Rescaling, combining, and projecting feature vectors  
- Operated the rescaled feature by reversing the convolution and pooling operation back to the input stage

*** The key point of these operations are basically hand crafted using "MatConvNet" rather than referencing the preexisting vgg model in the deep-learning toolbox ***

*** 整個卷積、pooling和反卷積、unpooling的過程是利用開源的MatConvNet一個階段一個階段的操作，目的是可以對input進行filter-only的操作而不是feed-forward到目標層 ***


## Files
- from_NSD_creatANDgen_deconv.m – main script

## Purpose
Exploration of generating deep feature-controlled images using MATLAB 

## Pipeline

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
        │ unpooling operation │
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

---
