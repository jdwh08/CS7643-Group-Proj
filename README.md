## Team Water is Not Wet: Flood Segmentation using Sen1Floods11 dataset

### PACE Setup

#### 1.  SSH through terminal or your IDE
Please refer to the Ed PACE guide for this part. 

Once you're connected, **make sure you are inside ~/scratch**.

#### 2. Clone the repo inside scratch

```
git clone https://github.com/jdwh08/CS7643-Group-Proj.git
cd CS7643-Group-Proj
```

#### 3. Install uv inside scratch

```
curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$HOME/scratch/.local sh
source $HOME/scratch/.local/env
uv --version 
```

#### 4. Create the virtual environment
Inside the repo root:
```
uv venv
source .venv/bin/activate 
```

#### 5. Load CUDA before syncing dependencies

```
module purge
module load cuda/12.1.1
nvcc --version #should be 12.1.1
```

#### 6. Install project dependencies
```
uv sync
```

### 7. Sync the dataset
If you just want to run the data exploration notebook and train unet:
```
bash scripts/sync_partial.sh
```
If you need to do ViT or transfer learning using Prithivi:
```
bash scripts/sync_full.sh #the whole thing shoud take about 10 mins
```

### Running the data exploration notebook (optional)
#### 1. Go to [PACE OnDemand](https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard) in the browser.

#### 2. Activate the venv in terminal
Go to Files/Home Directory, open terminal:
```
cd scratch/CS7643-Group-Proj
source .venv/bin/activate
```
#### 3. Install an IPython kernel that points to the venv

Run inside venv:
```
python -m ipykernel install --user --name cs7643-env --display-name "CS7643 Env"
```
#### 4. Launch Jupyter from OnDemand using ANY base environment
Pick Anaconda 2023.03 or PyTorch 2.1.0, doesn’t matter.

#### 5. Inside Jupyter → Open the notebook → choose your kernel
Kernel → Change Kernel → CS7643 Env

### Running the training scripts
For Unet, at the project root:
```
sbatch --array=0-1 scripts/slurm_unet.sh
```
This will run full training on S1hand and S1weak with the correct config files.