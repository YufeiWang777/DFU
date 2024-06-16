### Improving Depth Completion via Depth Feature Upsampling (CVPR 2024)

[Project Page](https://npucvr.github.io/DFU/), [openaccess](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Improving_Depth_Completion_via_Depth_Feature_Upsampling_CVPR_2024_paper.pdf)

### Environment
```
CUDA 12.0
CUDNN 8.5.0
torch 1.7.1
torchvision 0.8.0
pip install -r DFU/requirements.txt
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

# train LRRU_Mini model
# python LRRU/train_apex.py -c train_lrru_mini_kitti.yml

# train LRRU_Tiny model
# python LRRU/train_apex.py -c train_lrru_tiny_kitti.yml

# train LRRU_Small model
# python LRRU/train_apex.py -c train_lrru_small_kitti.yml

# train LRRU_Base model
# python LRRU/train_apex.py -c train_lrru_base_kitti.yml
```

#### Testing

```bash
# download the pretrained model and add it to corresponding path.

$ sh val.sh

# val LRRU_Mini model
# python LRRU/val.py -c val_lrru_mini_kitti.yml

# val LRRU_Tiny model
# python LRRU/val.py -c val_lrru_tiny_kitti.yml

# val LRRU_Small model
# python LRRU/val.py -c val_lrru_small_kitti.yml

# val LRRU_Base model
# python LRRU/val.py -c val_lrru_base_kitti.yml
```


### Pretrained models

#### Models on the KITTI validate dataset.
|   Methods  | Pretrained Model  |   Loss  | RMSE[mm] | MAE[mm] | iRMSE[1/km] | iMAE[1/km] |
|:----------:|-------------------|:-------:|:--------:|:-------:|:-----------:|:----------:|
|  LRRU-Mini | [download link](https://drive.google.com/file/d/18je8eR_EqgtS8IM5dKvr0uy9jBoiMZe6/view?usp=sharing) | L1 + L2 |   806.3  |  210.0  |     2.3     |     0.9    |
|  LRRU-Tiny | [download link](https://drive.google.com/file/d/1nEoC1eUkvB_eZF-t6V_ykogwo0YXoA2l/view?usp=sharing) | L1 + L2 |   763.8  |  198.9  |     2.1     |     0.8    |
| LRRU-Small | [download link](https://drive.google.com/file/d/1YtldwyFsTUwmii4H2_fk8z9OiRLdZniI/view?usp=sharing) | L1 + L2 |   745.3  |  195.7  |     2.0     |     0.8    |
|  LRRU-Base | [download link](https://drive.google.com/file/d/10WTVS7a_5Hjo4f5iNgY0v_KsYuftoDZk/view?usp=sharing) | L1 + L2 |   729.5  |  188.8  |     1.9     |     0.8    |

### Acknowledgments

Thanks the ACs and the reviewers for their insightful comments, which are very helpful to improve our paper!


### Citation
```
@InProceedings{DFU_CVPR_2024,
  author    = {Wang, Yufei and Zhang, Ge and Wang, Shaoqian and Li, Bo and Liu, Qi and Hui, Le and Dai, Yuchao},
  title     = {Improving Depth Completion via Depth Feature Upsampling},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21104--21113},
  year      = {2024}
}
```

