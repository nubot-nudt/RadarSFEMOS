# [RAL2025] Self-Supervised Diffusion-Based Scene Flow Estimation and Motion Segmentation with 4D Radar

This is the official repository of the **RadarSFEMOS**, a self-supervised method for 4D radar SFE. 
For technical details, please refer to our paper on [RAL 2025](https://ieeexplore.ieee.org/document/10974572)

## Citation
If you find our work useful, please consider citing:

```shell
@article{liu2025ral,
  author={Liu, Yufei and Chen, Xieyuanli and Wang, Neng and Andreev, Stepan and Dvorkovich, Alexander and Fan, Rui and Lu, Huimin},
  journal={IEEE Robotics and Automation Letters}, 
  title={Self-Supervised Diffusion-Based Scene Flow Estimation and Motion Segmentation With 4D Radar}, 
  year={2025},
  volume={10},
  number={6},
  pages={5895-5902},
}
```

## Abstract 
Scene flow estimation (SFE) and motion segmentation (MOS) using 4D radar are emerging yet challenging tasks in robotics and autonomous driving applications. Existing LiDAR- or RGB-D-based point cloud processing methods often deliver suboptimal performance on radar data due to radar signals' highly sparse, noisy, and artifact-prone nature. Furthermore, for radar-based SFE and MOS, the lack of annotated datasets further aggravates these challenges. To address these issues, we propose a novel self-supervised framework that exploits denoising diffusion models to effectively handle radar noise inputs and predict point-wise scene flow and motion status simultaneously. To extract key features from the raw input, we design a transformer-based feature encoder tailored to address the sparsity of 4D radar data. Additionally, we generate self-supervised segmentation signals by exploiting the discrepancy between robust rigid ego-motion estimates and scene flow predictions, thereby eliminating the need for manual annotations. Experimental evaluations on the View-of-Delft (VoD) dataset and TJ4DRadSet demonstrate that our method achieves state-of-the-art performance for both radar-based SFE and MOS.
