# Diffusion in the Dark

Official repository for Diffusion in the Dark: A Diffuion Model for Low-Light Text Recognition\
Cindy M. Nguyen, Eric R. Chan, Alexander W. Bergman, Gordon Wetzstein\
WACV 2024

[Project Page](https://ccnguyen.github.io/diffusion-in-the-dark/) | [Paper](https://arxiv.org/pdf/2303.04291.pdf)

[![arXiv](https://img.shields.io/badge/arXiv-2303.04291-b31b1b.svg)](https://arxiv.org/abs/2303.04291)

## Updates
- 11/06/23: 📣 Training code, inference code, and pretrained models are released.
- 10/23/23: ✨ Diffusion in the Dark was accepted into WACV!


## Setup
You can use the following to set up a Conda environment.\
Environment and code was tested on Linux, NVIDIA Driver Version 545.23.06, CUDA Version 12.3. 
Code was run on a single NVIDIA RTX 3090.

```
conda env create -f environment.yml
conda activate did
pip install -r requirements.txt
pip install -e .
```
Download our pretrained model [HERE](https://drive.google.com/drive/folders/1EoDN4Yt6JbyA1DZbzkZx21JGbERXaNZc?usp=sharing). Put the pretrained model in a `runs` folder.

We put the datasets into a folder outside of the repo to create the following file structure
```python
data
    LOL
    lowLightDataset
diffusion-in-the-dark
```

### LOL (LOw-Light) dataset from [here](https://daooshee.github.io/BMVC2018website/)
File structure should be as follows:
```python
LOL
    our485
        high
        low
    eval15
        high
        low
```
### Low Light Dataset (based off of Seeing in the Dark) 
From Xu, Ke, et al. "Learning to restore low-light images via decomposition-and-enhancement."
This can be downloaded from [here](https://xinyangdut.github.io/enhancement/index.html).\
File structure should be as follows:
```python
lowLightDataset
    gt
        test
        train
    input
        test
        train
```


## Usage
To train on LowLightDataset or LOL dataset:
```
python train.py --outdir=runs --data=../data/lowLightDataset --batch 1 --curve linear --dataset lowlight --add_noise True --scale_norm True --use_lpips True

python train.py --outdir=runs --data=../data/LOL --batch 1 --curve linear --dataset lol --add_noise True --scale_norm True --use_lpips True
```

If you want to train on a custom dataset use `data/raw_img_lol.ipynb` as a base to sample 30 random images to find
the mean and std of your data to perform data normalization with. Add the mean and std to `training/constants.py`


To test:
```
python inference.py --dataset lol
```



## Reference
If you find this work useful, please consider citing us!
```python
@inproceedings{nguyen2024diffusion,
  author    = {Nguyen, Cindy M and Chan, Eric R and Bergman, Alexander W and Wetzstein, Gordon},
  title     = {Diffusion in the Dark: A Diffusion Model for Low-Light Text Recognition},
  journal   = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2024},
}
```


## Acknowledgements
We thank the authors of [EDM](https://github.com/NVlabs/edm) from which our repo is based off of.

