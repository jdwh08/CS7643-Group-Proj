## Team Water is Not Wet: Flood Segmentation using Sen1Floods11 dataset

### Development Environment Setup

#### 1 - Install conda

#### 2 - Create the project environment from environment.yml

From your repo folder:
`conda env create -f environment.yml`
This will create an environment called cs7643-flood

#### 3 - Activate the environment

`conda activate cs7643-flood`

#### 4 - Updating the environment

if the environment.yml is updated, run:
`conda env update -f environment.yml --prune`

### Data Setup

#### 1 - Install gcloud sdk

Follow instructions:  
https://docs.cloud.google.com/sdk/docs/install-sdk

After installation:

```
gcloud init
gsutil --version     # should show gsutil version: 5.x
```

You may select any project during gcloud init; we only access a public bucket.

#### 2 - Syncing the data in local

From project root:  
`bash scripts/sync_local.sh`

This will download:

- Full hand-labeled dataset
  - HandLabeled/S1Hand/
  - HandLabeled/LabelHand/
- Hand-labeled splits
  - splits/flood_handlabeled/
- Partial weakly-labeled dataset (50 random chips)
  - WeaklyLabeled/S1Weak/
  - WeaklyLabeled/S1OtsuLabelWeak/

All files are stored under:  
`./data/`
unless you override DATA_ROOT.

### 3 - Optional

Explore the dataset with:

- src/data/dataloaders.py
- notebooks/data_exploration.ipynb
