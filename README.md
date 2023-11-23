# Towards Generalizable SER: Soft Labeling and Data Augmentation for Modeling Temporal Emotion Shifts in Large-Scale Multilingual Speech
Paper on [arxiv](https://arxiv.org/abs/2311.08607).

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

If your clip is shorter than 30s:

After you downsample your audio to 16KHz get its length (# of samples) and downsample by the same factor as Whisper as follows:
```
effective_length = clip.shape[-1]//160//2
```
And once you perform inference take the mean of the logits 0:effective_length similarly to this
```
torch.mean(extracted_features[:, :length, :], dim=1)
```

Do note that you can downsample the predictions in a different way so as to get predictions every t frames.