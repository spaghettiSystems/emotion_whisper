# Towards Generalizable SER: Soft Labeling and Data Augmentation for Modeling Temporal Emotion Shifts in Large-Scale Multilingual Speech

This repository contains a model weights release and an example notebook.


## Overview

The paper presents a novel approach to SER that addresses the issue of bias in cross-corpus emotion detection. Key aspects include:

- Amalgamation of 16 diverse datasets resulting in 375 hours of multilingual speech data.
- Introduction of a soft labeling system to capture gradational emotional intensities.
- Use of the Whisper encoder and a unique data augmentation method inspired by contrastive learning.
- Validation on four multilingual datasets demonstrating significant zero-shot generalization.

## How to use

- Download the weights from the releases
- Clone the repository
- Check example.ipynb for examples on how to load and use