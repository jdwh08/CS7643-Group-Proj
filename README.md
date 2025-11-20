# Team Water is Not Wet: Flood Segmentation using Sen1Floods11 dataset

## Development Environment Setup

### 1 - Install conda

### 2 - Create the project environment from environment.yml

From your repo folder:
`conda env create -f environment.yml`
This will create an environment called cs7643-flood

### 3 - Activate the environment

`conda activate cs7643-flood`

### 4 - Updating the environment

if the environment.yml is updated, run:
`conda env update -f environment.yml --prune`

## Potential Repo Structure

cs7643-group-proj/
│
├── configs/  
│ ├── unet_s1weak.yaml
│ ├── unet_s1hand.yaml
│ ├── vit_s1weak.yaml
│ ├── vit_s1hand.yaml
│ ├── prithvi_s1weak_finetune.yaml
│ └── prithvi_s1hand_finetune.yaml
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_unet_results.ipynb
│ ├── 03_vit_results.ipynb
│ └── 04_prithvi_results.ipynb
│
├── src/
│ ├── dataloaders.py  
│ ├── utils.py # iou, dice, seeding, logging helpers
│ │
│ ├── models/
│ │ ├── unet.py  
│ │ ├── vit_seg.py  
│ │ └── prithvi/
│ │ ├── **init**.py
│ │ └── prithvi_seg_head.py
│ │
│ ├── train_unet.py  
│ ├── train_vit.py  
│ └── train_prithvi.py  
│
├── scripts/
│ ├── train_unet_s1weak.slurm  
│ ├── train_unet_s1hand.slurm
│ ├── train_vit_s1weak.slurm
│ ├── train_vit_s1hand.slurm
│ ├── train_prithvi_s1weak.slurm
│ └── train_prithvi_s1hand.slurm
│
├── results/
│ ├── unet_s1weak/
│ ├── unet_s1hand/
│ ├── vit_s1weak/
│ ├── vit_s1hand/
│ ├── prithvi_s1hand/
│ └── metrics_global.csv
│
├── environment.yml  
├── README.md
└── .gitignore
