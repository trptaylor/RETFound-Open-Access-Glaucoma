## RETFound OAG - RETFound Open Access Glaucoma ##

This is a adapted version of RETFound (including weights) for the purpose of glaucoma detection from colour fundus photos (CFP)

It was fine-tuned using open access glaucoma images available online. Full details of the fine-tuning recipe will be made available in a forthcoming paper.

Please see the official RETFound GitHub for more information:
- https://github.com/rmaphoh/RETFound
- With thanks to Yukun Zhou

```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

You are recommended to first process your raw images through AutoMorph https://github.com/rmaphoh/AutoMorph using the processing module only.
Once completed, retrieve your processed images from AutoMorph/Results/M0/images and put them in an image folder inside /RETFound-Open-Access-Glaucoma

## Pipeline

### Prerequisites

- System with a CUDA-enabled GPU
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- All source fundus images stored in a single directory (`.jpg`, `.png`, or `.tif` formats)
- A reference dataframe mapping image filenames to `study ID`, `eye laterality`, and `disease labels`

---

### 🔧 Install environment

### Step 1: Create environment with conda:

```
conda create -n retfoundOAG python=3.11.0 -y
conda activate retfoundOAG
```

### Step 2: Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/trptaylor/RETFound-Open-Access-Glaucoma/
cd RETFound-Open-Access-Glaucoma
pip install -r requirements.txt
```


### Step 3: Crop Gradable Images Around the Optic Disc

### 1. Environment Setup
Create and activate a dedicated environment for the cropping tool:

```bash
conda deactivate
conda create --name fundus_image_toolbox python=3.9.19 pip -y
conda activate fundus_image_toolbox
pip install fundus_image_toolbox
```

### 2. Configure Paths
Navigate to the `RETFound-Open-Access-Glaucoma` root directory and open `crop_script.py`. Under `# 3. Setup Paths`, update the following variables:

```python
input_dir = Path("/path/to/your/images")
out_dir = Path("/path/to/your/cropped_images")
```

### 3. Run Cropping
Execute the script and deactivate the environment when finished:

```bash
python crop_script.py
conda deactivate
```

---

### Step 4: Generate Disease Predictions

### 1. Environment Setup
Reactivate the RETFound environment:

```bash
conda activate retfoundOAG
```

### 2. Download Weights
Save the weights into your RETFound-Open-Access-Glaucoma` directory

```bash
wget https://github.com/trptaylor/RETFound-Open-Access-Glaucoma/releases/download/v1.0.0/RETFound.OAG.pth
```

### 3. Configure Execution Script
Open `RETFound_OAG.sh` in the `RETFound-Open-Access-Glaucoma` directory and update the target paths:

```bash
DATA_PATH="/path/to/your/cropped_images"
CKPT="/path/to/your/checkpoint/RETFound OAG.pth"
```

### 4. Run Inference
From the `RETFound-Open-Access-Glaucoma` folder, run:

```bash
sh RETFound_OAG.sh
```

### 5. Outputs
Predictions will be saved to:

```text
RETFound-Open-Access-Glaucoma/output_dir/RETFound OAG/predictions.csv
```

Take `predictions.csv` forward for your subsequent results analysis alongside your reference metadata dataframe.
