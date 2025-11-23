# Voxel4D Core

The official code and data for the benchmark with baselines for my paper: [Voxel4D Core: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications](https://arxiv.org/abs/2311.17663)

**Author:** Rohan Tennety

<img src="./benchmark.png" width="49%"/> <img src="./VoxelNet.png" width="49%"/>



## Citation
If you use Voxel4D Core in an academic work, please cite my paper:

	@inproceedings{tennety2024cvpr,
		author = {Rohan Tennety},
		title = {{Voxel4D Core: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications}},
		year = 2025
	}
 
## Installation

* Create a conda virtual environment and activate it
```bash
conda create -n voxel4d python=3.7 -y
conda activate voxel4d
```
* Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3)
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
* Install gcc>=5 in conda env
```bash
conda install -c omgarcia gcc-6
```
* Install mmcv, mmdet, and mmseg
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```
* Install mmdet3d from the source code
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
* Install other dependencies
```bash
pip install timm
pip install open3d-python
pip install PyMCubes
pip install spconv-cu113
pip install fvcore
pip install setuptools==59.5.0

pip install lyft_dataset_sdk # for lyft dataset
```
* Install occupancy pooling
```
git clone <your-voxel4d-core-repo>
cd Voxel4D Core
export PYTHONPATH="."
python setup.py develop
```

## Data Structure

### nuScenes dataset
* Please link your [nuScenes V1.0 full dataset](https://www.nuscenes.org/nuscenes#download) to the data folder. 
* [nuScenes-Occupancy](https://drive.google.com/file/d/1vLL6bdqSC7WxtvQ6ODTw0pixAZCbPkpq/view?usp=sharing), [nuscenes_occ_infos_train.pkl](https://drive.google.com/file/d/1i6ktiV2951r5k9ABCLi2w2lIteRpoZ7c/view?usp=sharing), and [nuscenes_occ_infos_val.pkl](https://drive.google.com/file/d/1hs2P1tipydKRgq-VeuS8NtjjVCzAXVy8/view?usp=sharing) are also provided by the previous work. If you only want to reproduce the forecasting results with "inflated" form, nuScenes dataset and Voxel4D Core are all you need.

### Lyft dataset
* Please link your [Lyft dataset](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data) to the data folder.
* The required folders are listed below.

Note that the folders under `Core` will be generated automatically once you first run my training or evaluation scripts.

```bash
Voxel4D Core
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   ├── nuscenes_occ_infos_val.pkl
│   ├── nuScenes-Occupancy/
│   ├── lyft/
│   │   ├── maps/
│   │   ├── train_data/
│   │   ├── images/   # from train images, containing xxx.jpeg
│   ├── Core
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── GMO_lyft/
│   │   │   ├── ...
│   │   ├── MMO_lyft/
│   │   │   ├── ...
```
Alternatively, you could manually modify the path parameters in the [config files](./projects/configs/baselines) instead of using the default data structure, which are also listed here:
```
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
voxel4d_core_dataset_path = "./data/Core/"
nusc_root = './data/nuscenes/'
```

## Training and Evaluation

I directly integrate the Voxel4D Core dataset generation pipeline into the dataloader, so you can directly run training or evaluate scripts and just wait :smirk:

Optionally, you can set `only_generate_dataset=True` in the [config files](./projects/configs/baselines) to only generate the Voxel4D Core data without model training and inference.

### Train VoxelNetV1.1 with 8 GPUs

VoxelNetV1.1 can forecast inflated GMO and others. In this case, _vehicle_ and _human_ are considered as one unified category.

For the nuScenes dataset, please run

```bash
bash run.sh ./projects/configs/baselines/VoxelNet_in_Voxel4D_Core_V1.1.py 8
```

For the Lyft dataset, please run

```bash
bash run.sh ./projects/configs/baselines/VoxelNet_in_Voxel4D_Core_V1.1_lyft.py 8
```
### Train VoxelNetV1.2 with 8 GPUs

VoxelNetV1.2 can forecast inflated GMO including _bicycle_, _bus_, _car_, _construction_, _motorcycle_, _trailer_, _truck_, _pedestrian_, and others. In this case, _vehicle_ and _human_ are divided into multiple categories for clearer evaluation on forecasting performance.

For the nuScenes dataset, please run

```bash
bash run.sh ./projects/configs/baselines/VoxelNet_in_Voxel4D_Core_V1.2.py 8
```

For the Lyft dataset, please run

```bash
bash run.sh ./projects/configs/baselines/VoxelNet_in_Voxel4D_Core_V1.2_lyft.py 8
```

* The training/test process will be accelerated several times after you generate datasets by the first epoch.

### Test VoxelNet for different tasks

If you only want to test the performance of occupancy prediction for the present frame (current observation), please set `test_present=True` in the [config files](./projects/configs/baselines). Otherwise, forecasting performance on the future interval is evaluated.

```bash
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
# e.g. bash run_eval.sh ./projects/configs/baselines/VoxelNet_in_Voxel4D_Core_V1.1.py ./work_dirs/VoxelNet_in_Voxel4D_Core_V1.1/epoch_20.pth  8
```
Please set `save_pred` and `save_path` in the config files once saving prediction results is needed.

`VPQ` evaluation of 3D instance prediction will be refined in the future.

### Visualization

Please install the dependencies as follows:

```bash
sudo apt-get install Xvfb
pip install xvfbwrapper
pip install mayavi
```
where `Xvfb` may be needed for visualization in your server.

**Visualize ground-truth occupancy labels**. Set `show_time_change = True` if you want to show the changing state of occupancy in time intervals. 

```bash
cd viz
python viz_gt.py
```
<img src="./viz_occupancy.png" width="100%"/>

**Visualize occupancy forecasting results**. Set `show_time_change = True` if you want to show the changing state of occupancy in time intervals. 

```bash
cd viz
python viz_pred.py
```
<img src="./viz_pred.png" width="100%"/>

There is still room for improvement. Camera-only 4D occupancy forecasting remains challenging, especially for predicting over longer time intervals with many moving objects. I envision this benchmark as a valuable evaluation tool, and my VoxelNet can serve as a foundational codebase for future research on 4D occupancy forecasting.



## Pretrained Models


~~Please download my pretrained models (for epoch=20) to resume training or reproduce results.~~


| V1.2 | [link](https://drive.google.com/file/d/18IFs8LOu0dZe22rtZ78jbrbbQrg9E7LC/view?usp=sharing)  [VoxelNet_in_Voxel4D_Core_V1.2.py](./projects/configs/baselines/VoxelNet_in_Voxel4D_Core_V1.2.py) |


## Other Baselines

I also provide the evaluation on the forecasting performance of [other baselines](./other_baselines) in Voxel4D Core.

## TODO
New Pretrained models coming soon.



## Contact

**Author:** Rohan Tennety  
**Email:** rtennety@gmail.com
