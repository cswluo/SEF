# SEF
This is PyTorch implementation of our paper ["Learning Semantically Enhanced Feature for Fine-Grained Image Classification"](about:blank). We experimented on 4 fine-grained benchmark datasets --- CUB-200-2011, Stanford Cars, Stanford Dogs, and VGG-Aircraft. You should first download these datasets from their project homepages before runing SEF.


## Appoach

![alt text](https://github.com/cswluo/SEF/blob/master/sef.pdf)

## Requirements

- PyTorch (>1.5), 
- CUDA 10.2 
- Python 3.6
- Tensorboard (>2.2.2)

## Learning and Evaluation
A "x-imdb.py" is provided for each dataset to generate Python pickle files, which are then used to prepare train/val/trainval/test data. Run "x-imdb.py" in the folder of your dataset to generate corresponding pickle file (imdb.pkl) should be the very first step.

- demo.py is used to train your own SEF model from scratch.

- prediction.py outputs classification accuracy by employing pretrained SEF models.   

Due to the random generation of train/val/test data on some datasets, the classification accuracy may have a bit fluctuation but it should be in a reasonable range.

The pretrained SEF models can be download from [HERE](https://pan.baidu.com/s/1r-mP0mQop20bSGnau6SUGg) with code `<i5wk>`. 

## Results

SEF-18 and SEF-50 are results from different backbones (ResNet-18 and ResNet-50)
|              |        SEF-18    | SEF-50 |
|:-------------|:---------------:|:----------------:|
|CUB-200-2011  |84.8%            |87.3%             |
|Stanford Cars |83.1%            |94.0%             |
|Stanford Dogs |91.8%            |88.8%             |
|VGG-Aircraft  |89.3%            |92.1%             |

## Citation
```
@inproceedings{sef@luowei,
author = {Wei Luo and Hengmin Zhang and Jun Li and Xiu-Shen Wei},
title = {Learning Semantically Enhanced Feature for Fine-Grained Image Classification},
booktitle = {arXiv},
year = {2020},
}
```
