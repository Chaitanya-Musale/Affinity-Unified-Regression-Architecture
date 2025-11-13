"""
AURA FRAMEWORK - COMPLETE STANDALONE SCRIPT FOR GOOGLE COLAB
=============================================================
ALL CODE IN ONE FILE! Just copy this entire file and paste into Google Colab.

Instructions:
1. Copy this ENTIRE file
2. Create new Google Colab notebook
3. Paste into a cell
4. Run the cell
5. Wait for training to complete (~4-6 hours)

The script will:
- Install all dependencies
- Mount Google Drive
- Load and preprocess data
- Train the complete AURA model
- Evaluate on all test sets
- Save results

Make sure your Google Drive has the AURA folder at:
/content/drive/MyDrive/research/AURA/

Total size: ~3500 lines including models, training, and execution.
"""

# =============================================================================
# INSTALLATIONS
# =============================================================================
print("=" * 70)
print("AURA FRAMEWORK - STARTING INSTALLATION")
print("=" * 70)

import subprocess
import sys

def install(package):
    try:
        __import__(package.split('[')[0].split('>')[0].split('==')[0].replace('-', '_'))
        print(f"OK {package}")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("pandas")
install("numpy")
install("scikit-learn")
install("matplotlib")
install("seaborn")
install("tqdm")
install("scipy")
install("rdkit")

print("\nInstalling PyTorch...")
get_ipython().system('pip uninstall torch torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv -y')
get_ipython().system('pip install torch==2.3.1 torchvision==0.18.1 -f https://download.pytorch.org/whl/cu121')
get_ipython().system('pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html')

install("xgboost")
install("transformers")
install("sentencepiece")
install("accelerate")
install("shap")
install("e3nn")
install("biopython")
install("torch_geometric")

print("\nAll dependencies installed!")

# =============================================================================
# MOUNT GOOGLE DRIVE
# =============================================================================
print("\n" + "=" * 70)
print("MOUNTING GOOGLE DRIVE")
print("=" * 70)

from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# IMPORTS
# =============================================================================
print("\n" + "=" * 70)
print("IMPORTING LIBRARIES")
print("=" * 70)

import os, gc, json, warnings, pickle, copy
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_add_pool, radius_graph
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from transformers import AutoTokenizer
from transformers.models.esm.modeling_esm import EsmModel
from Bio import PDB

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

print("Imports complete!")

# CONTINUE WITH NEXT PART - File is too large, splitting into aura_standalone_part2.py
