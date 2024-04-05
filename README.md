<div align="center">
<h2>🏡Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning</h2>

[**Rui Li**](https://ruili3.github.io/)<sup>1</sup> · [**Tobias Fischer**](https://tobiasfshr.github.io/)<sup>1</sup> · [**Mattia Segu**](https://mattiasegu.github.io/)<sup>1</sup> · [**Marc Pollefeys**](https://people.inf.ethz.ch/pomarc/)<sup>1</sup> <br>
[**Luc Van Gool**](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html)<sup>1</sup> · [**Federico Tombari**](https://federicotombari.github.io/)<sup>2,3</sup>

<sup>1</sup>ETH Zürich  · <sup>2</sup>Google  · <sup>3</sup>Technical University of Munich

**CVPR 2024**

<a href="https://arxiv.org/abs/2404.03658"><img src='https://img.shields.io/badge/arXiv-KYN-red' alt='Paper PDF'></a>
<a href='https://ruili3.github.io/kyn/'><img src='https://img.shields.io/badge/Project_Page-KYN-green' alt='Project Page'></a>
<a href='https://huggingface.co/'><img src='https://img.shields.io/badge/Hugging_Face-KYN (coming soon)-yellow' alt='Hugging Face'></a>
</div>

This work presents _Know-Your-Neighbors_ (KYN), a single-view 3D reconstruction method that disambiguates occluded scene geometry by utilizing Vision-Language semantics and spatial reasoning.

![teaser](/media/assets/teaser.png)


### 🔗 Environment Setup
```bash
# python virtual environment
python -m venv kyn
source kyn/bin/activate
pip install -r requirements.txt
```

### 🚀 Quick Start
Download our [pre-trianed model](https://drive.google.com/file/d/1wul-WjsH1iaccfMOGwqIJ55vJnfMUywp/view?usp=drive_link) and the [LSeg model](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing), put them into `./checkpoints`. Then run the demo:
```bash
python scripts/demo.py --img media/example/0000.png --model_path checkpoints/kyn.pt --save_path /your/save/path
```
Herein `--img` specifies the input image path, `--model_path` is the model checkpoint path, and `--save_path` stores the resulting depth map, BEV map, as well as 3D voxel grids.

### 📁 Dataset Setup
We use the KITTI-360 dataset and process it as follows:
1. Register at [https://www.cvlibs.net/datasets/kitti-360/index.php](https://www.cvlibs.net/datasets/kitti-360/index.php) and download perspective images, fisheye images, raw Velodyne scans, calibrations, and vehicle poses. The required KITTI-360 official scripts & data are:
    ```
    download_2d_fisheye.zip
    download_2d_perspective.zip
    download_3d_velodyne.zip
    calibration.zip
    data_poses.zip
    ```
2. Preprocess with the Python script below. It rectifies the fisheye views, resizes all images, and stores them in separate folders:
    ```
    python datasets/kitti_360/preprocess_kitti_360.py --data_path ./KITTI-360 --save_path ./KITTI-360
    ```
3. The final folder structure should look like:
    ```
    KITTI-360
       ├── calibration
       ├── data_poses
       ├── data_2d_raw
       │   ├── 2013_05_28_drive_0003_sync
       │   │   ├── image_00
       │   │   │    ├── data_192x640
       │   │   │    └── data_rect
       │   │   ├── image_01
       │   │   ├── image_02
       │   │   │    ├── data_192x640_0x-15
       │   │   │    └── data_rgb
       │   │   └── image_03
       │   └── ...
       └── data_3d_raw
               ├── 2013_05_28_drive_0003_sync
               └── ...
    ```

### 📊 Evaluation
#### Quantitative Evaluation
1. The data directory is set to `./KITTI-360` by default.
2. Download and unzip the pre-computed [GT occupancy maps](https://drive.google.com/file/d/17FvEShQdCRBSH91iQSMhcoocb8j9x3at/view?usp=drive_link) into `./KITTI-360`. You can also compute and store your customized GT occupancy maps by setting `read_gt_occ_path: ''` and specifying `save_gt_occ_map_path` in `configs/eval_kyn.yaml`. 
3. Download and unzip the [object labels](https://drive.google.com/file/d/1ELY2Hxy5hRP52J7ewzLYFWMk-Qu5QViQ/view?usp=drive_link) to `./KITTI-360`.
4. Download our [pre-trianed model](https://drive.google.com/file/d/1wul-WjsH1iaccfMOGwqIJ55vJnfMUywp/view?usp=drive_link) and the [LSeg model](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing), put them into `./checkpoints`.
4. Run the following command for evaluation:
    ```bash
    python eval.py -cn eval_kyn
    ```

#### Voxel Visualization
Run the following command to generate 3D voxel models on the KITTI-360 test set:
```bash
python scripts/gen_kitti360_voxel.py -cn gen_voxel
```

### 💻 Training
Download the [LSeg model](https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing) and put it into `./checkpoints`. Then run:
```bash
torchrun --nproc_per_node=<num_of_gpus> train.py -cn train_kyn
```
where `<num_of_gpus>` denotes the number of available GPUs. Models will be saved in `./result` by defualt. 


### 📰 Citation
Please cite our paper if you use the code in this repository:
```
@inproceedings{li2024know,
      title={Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning}, 
      author={Li, Rui and Fischer, Tobias and Segu, Mattia and Pollefeys, Marc and Van Gool, Luc and Tombari, Federico},
      booktitle={CVPR},
      year={2024}
}
``` 

<!-- ### 🌟 Star History
<div style="text-align: center;">
<a href="https://star-history.com/#ruili3/Know-Your-Neighbors&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ruili3/Know-Your-Neighbors&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ruili3/Know-Your-Neighbors&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ruili3/Know-Your-Neighbors&type=Date" width="600"/>
  </picture>
</a>
</div> -->