### Improving Depth Completion via Depth Feature Upsampling (CVPR 2024)

[Project Page](https://npucvr.github.io/DFU/), [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Improving_Depth_Completion_via_Depth_Feature_Upsampling_CVPR_2024_paper.pdf)

### Environment
```
CUDA 12.0
CUDNN 8.5.0
torch 1.7.1
torchvision 0.8.0
pip install -r LRRU/requirements.txt
pip3 install opencv-python
pip3 install opencv-python-headless
```

#### Wandb

We used <a href="https://wandb.ai/" target="_blank">WANDB</a> to visualize and track our experiments.

#### NVIDIA Apex

We used NVIDIA Apex for multi-GPU training as <a href="https://github.com/zzangjinsun/NLSPN_ECCV20" target="_blank">NLSPN</a>.

Apex can be installed as follows:

```bash
$ cd PATH_TO_INSTALL
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
```


### Dataset

#### KITTI Depth Completion (KITTI DC)

KITTI DC dataset is available at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) and the data structure is:

```
.
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```

### Usage

#### Training

```bash

$ sh train.sh

```

#### Testing

```bash
# download the pretrained model and add it to corresponding path.

$ sh val.sh

```


### Pretrained models

## Models on the KITTI validate dataset.

[LRRU_with_Three_DFU](https://drive.google.com/file/d/1IYoobWIImsD1GwFJfnv4RkVL1cAFZ7sX/view?usp=drive_link)


### Acknowledgments

Thanks the ACs and the reviewers for their insightful comments, which are very helpful to improve our paper!



### Citation
```
@inproceedings{wang2024improving,
  title={Improving Depth Completion via Depth Feature Upsampling},
  author={Wang, Yufei and Zhang, Ge and Wang, Shaoqian and Li, Bo and Liu, Qi and Hui, Le and Dai, Yuchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21104--21113},
  year={2024}
}
```
