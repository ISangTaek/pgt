# PGT: A Multitask Molecular Graph Learning Framework for AI-Enabled Drug Discovery

This repository contains the official implementation of **PGT** (Pre-trained Graph Transformer), a deep learning framework designed for molecular property prediction. It leverages Graph Neural Networks (GNNs) with spatial positioning and supports multi-task learning.

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation & Compilation](#-installation--compilation)
- [Data Preprocessing](#-data-preprocessing)
- [Usage](#-usage)
  - [Training](#1-training)
  - [Single Molecule Inference](#2-single-molecule-inference)
  - [Batch Inference](#3-batch-inference)
- [License](#-license)

## 📂 Project Structure

```text
PGT-Project/
├── algos/                  # Cython modules for graph algorithms
├── architecture/           # Model definitions (PGT, Graphormer, etc.)
├── data/                   # Raw CSV data storage
├── processed_data/         # Preprocessed PyG graph data (generated)
├── ckpt/                   # Model checkpoints
├── main.py                 # Main entry point for training and inference
├── preprocess_data.py      # Data preprocessing script
├── setup.py                # Build script for Cython extensions
└── requirements.txt        # Python dependencies
```

## 📦 Requirements
- Python 3.8+
- PyTorch >= 1.10
- PyTorch Geometric
- RDKit
- Cython
- Pandas, Numpy, Tqdm 

You can install the dependencies Using:
```bash
pip install -r requirements.txt
```
## ⚙️ Installation && Compilation
**Crucial Step:** This project utilizes **Cython** to accelerate complex graph algorithms (e.g., Floyd-Warshall for
spatial positioning). You **must** compile the C extensions locally before running any code.
Run the following command in the root directory:
```bash
python setup.py build_ext --inplace
```
*Note:* If successful, you will see `.so` (Linux/Mac) or `.pyd` (Windows) files generated in the `algos/` directory. And then you can run the code right now.

## 🧪Data Preprocessing
- Please unzip the dataset in the `data/` directory before running the preprocessing script.
- Before training or inference, raw molecular data (SMILES format in CSV) must be converted into graph objects.

1. Place your raw data (e.g., `toxacute.csv`) in the `data/` folder.

2. Run the preprocessing script:
```bash
python preprocess_data.py \
  --raw_csv_path ./data/toxacute.csv \
  --task_list "man_oral_TDLo,women_oral_TDLo,human_oral_TDLo" \
  --output_dir ./processed_data
```
- `--raw_csv_path`: Path to your raw CSV file containing SMILES and labels.

- `--task_list`: Comma-separated list of target columns to process (or use `all` to process all tasks).

- `--output_dir`: Directory where processed graph files (`.pt`) will be saved.

## 🚀 Usage
The `main.py` script handles training, evaluation, and inference modes.
1. **Training**To train the model using **Uncertainty Weighting (UW)** for multitask learning, you can also choose other strategy:
```bash
 python main.py \
  --mode train \
  --dataset toxacute \
  --preprocessed_data_dir ./processed_data \
  --arch PGT \
  --weighting UW \
  --epochs 50 \
  --bs 64 \
  --gpu_id 0 \
  --save_path ./ckpt
```
**Key Arguments:**

- `--arch`: Model architecture (default: `PGT`).

- `--weighting`: Multitask weighting strategy (`UW`, `EW`, `DWA`).

- `--dataset`: Dataset name (used for task configuration).
2. **Single Molecule Inference**
To predict properties for a single SMILES string using a trained model:
```bash
 python main.py \
  --mode single_inference \
  --load_path ./ckpt \
  --ckpt_name PGT_ckpt \
  --arch PGT \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```
3. **Batch Inference**
To perform inference on a specific task folder within your preprocessed data:
```bash
 python main.py \
  --mode batch_inference \
  --preprocessed_data_dir ./processed_data \
  --inference_task human_oral_TDLo \
  --load_path ./ckpt \
  --ckpt_name PGT_ckpt \
  --inference_output_path ./predictions.csv
```

## 🪪License
This project is released under the MIT License. See the LICENSE file for details.