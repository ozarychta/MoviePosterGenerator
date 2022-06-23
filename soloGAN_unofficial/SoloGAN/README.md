# Implementation of SoloGAN

This repository is an unofficial implementation of SoloGAN in *Multimodal Image-to-Image Translation via a Single Generative Adversarial Network*
https://arxiv.org/pdf/2008.01681.pdf

## Dependency:
PyTorch, Numpy, torchvision, PIL, tensorboardX


## Dataset:
Data will be placed under route_path/datasets/animal, each folder contains images from one domain and is named 'trainX' (X=A,B,C...)


## Training:
### start training 
```
python train.py
```
 
### resume training:
```
python train.py --resume [.pth file route](e.g.: ./results/trial/00069.pth)
```

### tensorboard loss monitoring:
```
tensorboard --logdir=logs/trial
```
Visualise the loss and image at: http://localhost:6006/