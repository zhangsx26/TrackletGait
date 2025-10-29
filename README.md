# TrackletGait: A Robust Framework for Gait Recognition in the Wild

Paper [[arXiv](https://arxiv.org/pdf/2508.02143)] [[IEEE](https://ieeexplore.ieee.org/abstract/document/11154018)]

This project is based on [OpenGait](https://github.com/ShiqiYu/OpenGait),  
with additional dependencies from [torch-dwt](https://github.com/KeKsBoTer/torch-dwt).

Currently, only the modified parts of the code are provided.  
The full project will be released later after further organization.


## Usage
Replace the files in the original OpenGait project with the ones provided here (overwrite files with the same name).


## Result and Checkpoint

**Dataset: Gait3D**

[checkpoint](https://drive.google.com/file/d/18KKxCVshTKX6ewwOn2rwf815uM32kIET/view?usp=sharing)
```
{'scalar/test_accuracy/Rank-1': 77.7999997138977, 'scalar/test_accuracy/Rank-5': 88.99999856948853, 'scalar/test_accuracy/Rank-10': 92.10000038146973, 'scalar/test_accuracy/mAP': 70.20399214273307, 'scalar/test_accuracy/mINP': 51.6445367958063}
```
---

**Dataset: CASIA-B**
[checkpoint](https://drive.google.com/file/d/1wMoltPohD52_Zeogmjx84VaJE5FTafSP/view?usp=sharing)

(Model: small ver.)
```
===Rank-1 (Exclude identical-view cases)===
NM@R1: [98.20 100.00 100.00 99.90 98.10 98.00 98.40 99.50 100.00 99.50 96.00]
BG@R1: [95.10 98.00 98.10 97.37 94.50 91.80 94.50 97.70 98.20 97.47 93.80]
CL@R1: [80.50 93.10 93.80 92.20 86.80 83.60 85.30 89.30 90.10 88.50 77.80]
NM@R1: 98.87%     BG@R1: 96.05%   CL@R1: 87.36%
```
---

**Dataset: CCPG**
[checkpoint](https://drive.google.com/file/d/1LTeBi_x18bo4LUEyT3lgRPhQdk3iO5RV/view?usp=sharing)

(Model: small ver.)
(data_in_use: [True, False, False, False])
```
===Rank-1 (Exclude identical-view cases for Person Re-Identification)===
CL: 92.486,       UP: 96.707,     DN: 96.000,     BG: 97.611
===mAP (Exclude identical-view cases for Person Re-Identification)===
CL: 66.840,       UP: 85.867,     DN: 84.064,     BG: 89.956
===mINP (Exclude identical-view cases for Person Re-Identification)===
CL: 21.678,       UP: 61.313,     DN: 60.162,     BG: 68.736
===Rank-1 (Include identical-view cases)===
CL: 84.589,       UP: 89.717,     DN: 88.463,     BG: 93.396
===Rank-1 (Exclude identical-view cases)===
CL: 83.191,       UP: 88.738,     DN: 87.389,     BG: 92.912
===Rank-1 of each angle (Exclude identical-view cases)===
CL: [75.57 77.56 84.11 86.50 84.00 83.94 81.38 86.00 85.20 87.65]
UP: [85.15 84.67 88.22 90.22 90.89 91.22 85.99 90.78 86.68 93.56]
DN: [79.89 86.38 85.70 89.99 87.78 89.00 86.28 89.22 88.43 91.22]
BG: [88.69 89.54 95.11 96.78 94.33 95.67 92.04 91.44 90.92 94.61]
```
---

## Citation

```
@article{zhang2025trackletgait,
  title={TrackletGait: A Robust Framework for Gait Recognition in the Wild},
  author={Zhang, Shaoxiong and Zheng, Jinkai and Zhu, Shangdong and Yan, Chenggang},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

```
Zhang S, Zheng J, Zhu S, et al. TrackletGait: A Robust Framework for Gait Recognition in the Wild[J]. IEEE Transactions on Multimedia, 2025.
```
