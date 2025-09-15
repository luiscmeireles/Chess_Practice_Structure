# Reproducibility Repository

This repository contains the code and datasets accompanying the paper  
**"_<Practice Structure Predicts Skill Growth in Online Chess: A Behavioral Modeling Approach.>_"**.

It includes:
- 2 Jupyter notebooks (`.ipynb`)
- 2 Python scripts (`.py`)
- 3 datasets (CSV or similar)
- a setup script to install the Stockfish chess engine
- a setup script to download the large dataset from Zenodo

---

## ⚙️ Setup Instructions

Follow these steps to reproduce the experiments.

---

### 1. Install Python dependencies

bash -
python -m pip install \
  numpy pandas matplotlib tqdm scipy \
  scikit-learn imbalanced-learn seaborn statsmodels \
  python-chess stockfish jupyter

### 2. Place stockfish in the appropriate folder

The code expects the Stockfish engine binary at:
'stockfish/stockfish-macos-m1-apple-silicon'

Run this script to place it there automatically:
bash - scripts/get_stockfish.sh

### 3. Download games (large 3.5Gb) dataset

bash scripts/get_data.sh

After setup, you can run the notebooks or scripts as usual.

REPOSITORY STRUCTURE

├── notebooks/
│   ├── 01_experiment.ipynb
│   └── 02_analysis.ipynb
├── src/
│   ├── feature.py
│   ├── run_match.py
│   └── eval_dataset.py
├── data/
│   └── (df_final.feather will download here)
├── scripts/
│   ├── get_stockfish.sh
│   └── get_data.sh
└── README.md





