# LLM-GAODE
This repository contains an implementation of the Large-Language-Model Augmented Neural Ordinary Differential Equations Network for Video Nystagmography Classification.

For details about the augmented neural ode network and Lyapunov loss, please refer to:
https://github.com/ivandariojr/LyapunovLearning/tree/master

https://github.com/EmilienDupont/augmented-neural-odes/tree/master

Regarding the construction of the dataset in this article, please refer to:
https://github.com/XiheQiu/Gram-AODE

## Basic Usage
```python
python train.py --network odenet --gpu 0 --use_vllm_fallback True --vllm_confidence_threshold 0.6
```
