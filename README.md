# SEF
This is the PyTorch implementation of our IEEE Signal Processing Letters paper **["Learning Semantically Enhanced Feature for Fine-Grained Image Classification"](https://arxiv.org/pdf/2006.13457v3.pdf)**. We experimented on 4 fine-grained benchmark datasets --- CUB-200-2011, Stanford Cars, Stanford Dogs, and VGG-Aircraft. You should first download these datasets from their project homepages before runing SEF.

**SEF can achieve comparable performance to the state-of-the-art methods with considerably less computation cost and training epochs. It has about 1.7% more parameters than its ResNet-50 backbone on the CUB-Birds dataset and can be easily integrated into other DCNNs.**

## Appoach

![alt text](https://github.com/cswluo/SEF/blob/master/figs/sef.png)

## Requirements

- PyTorch (>1.5), 
- CUDA 10.2 
- Python 3.6
- Tensorboard (>2.2.2)

## Learning and Evaluation
A "x-imdb.py" is provided for each dataset to generate Python pickle files, which are then used to prepare train/val/trainval/test data. Run "x-imdb.py" in the folder of your dataset to generate corresponding pickle file (imdb.pkl) should be the very first step.

- main.py trains your own SEF model.

- eval.py outputs classification accuracy by employing pretrained SEF models.   

Due to the random generation of train/val/test data on some datasets, the classification accuracy may have a bit fluctuation but it should be in a reasonable range.

The pretrained SEF models can be download from [HERE](https://pan.baidu.com/s/1r-mP0mQop20bSGnau6SUGg) with code **`i5wk`**. 

## Results

### Accuracy
SEF-18 and SEF-50 are results of models with different backbones (ResNet-18 and ResNet-50)
|              |        SEF-18    | SEF-50 |
|:-------------|:---------------:|:----------------:|
|CUB-200-2011  |84.8%            |87.3%             |
|Stanford Cars |91.8%            |94.0%             |
|Stanford Dogs |83.1%            |88.8%             |
|VGG-Aircraft  |89.3%            |92.1%             |

### Visualization
![correlation matrices](https://github.com/cswluo/SEF/blob/master/figs/corr.png)

![visualization](https://github.com/cswluo/SEF/blob/master/figs/visualization.png)

## Citation
```
@inproceedings{sef@luowei,
author = {Wei Luo and Hengmin Zhang and Jun Li and Xiu-Shen Wei},
title = {Learning Semantically Enhanced Feature for Fine-Grained Image Classification},
booktitle = {arXiv preprint arXiv:2006.13457},
year = {2020},
}
```
