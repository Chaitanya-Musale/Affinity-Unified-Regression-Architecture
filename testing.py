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
import shap

print("Imports complete!")

# =============================================================================
# CONFIGURATION MODULE
# =============================================================================
print("\n" + "=" * 70)
print("SETTING UP CONFIGURATION")
print("=" * 70)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# Training hyperparameters
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8
MAX_EPOCHS_STAGE_B = 50
MAX_EPOCHS_STAGE_C = 50
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
GRADIENT_CLIP_NORM = 1.0
WARMUP_STEPS = 500
L2_REG_WEIGHT = 1e-5

# Model architecture parameters
GNN_2D_NODE_DIM = 6
GNN_2D_EDGE_DIM = 3
HIDDEN_DIM = 128
PLM_HIDDEN_DIM = 320
OUTPUT_DIM = 1
PLM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Data processing parameters
N_CONFORMERS = 5
ECFP_N_BITS = 2048
MAX_PROTEIN_LENGTH = 1024
PLM_PAD_ID = 1

# Normalization configuration
NORMALIZATION_METHOD = 'zscore'

# Paths configuration (Google Colab defaults)
BASE_DIR = '/content/drive/MyDrive/research/AURA'
SPLITS_DIR = f'{BASE_DIR}/Data_splits'

TRAIN_CSV = f'{SPLITS_DIR}/train_split.csv'
VAL_CSV = f'{SPLITS_DIR}/validation_split.csv'
TEST_GENERAL_CSV = f'{SPLITS_DIR}/test_split.csv'
TEST_REFINED_CSV = f'{SPLITS_DIR}/test_refined_2020.csv'
TEST_CASF_CSV = f'{SPLITS_DIR}/test_casf_2016.csv'

STRUCTURE_PATHS = [
    f'{BASE_DIR}/Data/PDBbind_v2020_other_PL/v2020-other-PL',
    f'{BASE_DIR}/Data/PDBbind_v2020_refined/refined-set',
    f'{BASE_DIR}/Data/CASF-2016/CASF-2016/coreset'
]

OUTPUT_DIR = f'{BASE_DIR}/Normalized_Model/v1'

CONFORMER_CACHE = 'all_conformers_multipath_v2.pkl'
ECFP_CACHE = 'all_ecfp_precomputed.pkl'
TOKEN_CACHE = 'protein_tokens_multipath_v2.pkl'
NORMALIZER_CACHE = 'affinity_normalizer.pkl'

# Dataloader configuration
NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# XGBoost configuration
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_RANDOM_STATE = 42
XGB_EARLY_STOPPING_ROUNDS = 20

# Visualization configuration
VISUALIZATION_FIGURE_SIZE = (16, 10)
VISUALIZATION_DPI = 100

# Logging configuration
PRINT_TRAINING_INFO = True
SAVE_TRAINING_HISTORY = True

def update_pad_id(pad_id):
    global PLM_PAD_ID
    PLM_PAD_ID = pad_id

def print_config():
    print("\n" + "="*60)
    print("AURA FRAMEWORK CONFIGURATION")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Mixed Precision Training: {USE_AMP}")
    print(f"Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs (Stage B/C): {MAX_EPOCHS_STAGE_B}/{MAX_EPOCHS_STAGE_C}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Normalization Method: {NORMALIZATION_METHOD}")
    print(f"Number of Conformers: {N_CONFORMERS}")
    print("="*60 + "\n")

print("Configuration set!")

# =============================================================================
# DATA NORMALIZATION MODULE
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING AFFINITY NORMALIZER")
print("=" * 70)

class AffinityNormalizer:
    """
    Handles normalization and denormalization of affinity values.
    Supports two normalization methods:
    - 'zscore': Z-score normalization (mean=0, std=1)
    - 'minmax': Min-max scaling to [0, 1]
    """

    def __init__(self, method='zscore'):
        if method not in ['zscore', 'minmax']:
            raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'.")
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False

    def fit(self, values):
        values = np.array(values).flatten()
        if len(values) == 0:
            raise ValueError("Cannot fit normalizer on empty values array")

        if self.method == 'zscore':
            self.mean = values.mean()
            self.std = values.std()
            if self.std == 0:
                print("Warning: Standard deviation is 0, setting to 1.0 to prevent division by zero")
                self.std = 1.0
        elif self.method == 'minmax':
            self.min = values.min()
            self.max = values.max()
            if self.max == self.min:
                print("Warning: Max equals min, setting max to min+1 to prevent division by zero")
                self.max = self.min + 1.0

        self.fitted = True
        return self

    def transform(self, values):
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transforming")

        values = np.array(values)
        original_shape = values.shape
        values = values.flatten()

        if self.method == 'zscore':
            normalized = (values - self.mean) / self.std
        elif self.method == 'minmax':
            normalized = (values - self.min) / (self.max - self.min)

        return normalized.reshape(original_shape)

    def inverse_transform(self, values):
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before inverse transforming")

        values = np.array(values)
        original_shape = values.shape
        values = values.flatten()

        if self.method == 'zscore':
            denormalized = values * self.std + self.mean
        elif self.method == 'minmax':
            denormalized = values * (self.max - self.min) + self.min

        return denormalized.reshape(original_shape)

    def save(self, path):
        params = {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'fitted': self.fitted
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(params, f)
        except Exception as e:
            raise IOError(f"Failed to save normalizer to {path}: {str(e)}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalizer file not found: {path}")

        try:
            with open(path, 'rb') as f:
                params = pickle.load(f)
            self.method = params['method']
            self.mean = params['mean']
            self.std = params['std']
            self.min = params['min']
            self.max = params['max']
            self.fitted = params['fitted']
            return self
        except Exception as e:
            raise IOError(f"Failed to load normalizer from {path}: {str(e)}")

    def get_params(self):
        return {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'fitted': self.fitted
        }

    def __repr__(self):
        if not self.fitted:
            return f"AffinityNormalizer(method='{self.method}', fitted=False)"
        if self.method == 'zscore':
            return f"AffinityNormalizer(method='zscore', mean={self.mean:.4f}, std={self.std:.4f})"
        else:
            return f"AffinityNormalizer(method='minmax', min={self.min:.4f}, max={self.max:.4f})"

print("AffinityNormalizer defined!")

# =============================================================================
# DATA PREPROCESSING MODULE
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING PREPROCESSING FUNCTIONS")
print("=" * 70)

def extract_protein_sequence(pdb_file_path):
    """Extract protein sequence from PDB file - handles multiple chains correctly."""
    parser = PDB.PDBParser(QUIET=True)

    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file not found: {pdb_file_path}")
        return None

    try:
        structure = parser.get_structure('protein', pdb_file_path)

        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':
                        res_name = residue.resname
                        if res_name in three_to_one:
                            sequence.append(three_to_one[res_name])

        if not sequence:
            print(f"Warning: No valid residues found in {pdb_file_path}")
            return None

        return ''.join(sequence)

    except Exception as e:
        print(f"Error extracting sequence from {pdb_file_path}: {e}")
        return None


def generate_conformer_ensemble_with_crystal(smiles_string, crystal_mol=None, n_conformers=5):
    """Generates conformer ensemble, optionally using crystal structure as first conformer."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            print(f"Warning: Failed to parse SMILES: {smiles_string}")
            return None

        mol = Chem.AddHs(mol)

        use_crystal = False
        if crystal_mol is not None:
            try:
                crystal_mol_h = Chem.AddHs(crystal_mol, addCoords=True)

                if mol.GetNumAtoms() == crystal_mol_h.GetNumAtoms():
                    match = mol.GetSubstructMatch(crystal_mol_h)
                    if match and len(match) == mol.GetNumAtoms():
                        conf = Chem.Conformer(mol.GetNumAtoms())
                        for i, j in enumerate(match):
                            pos = crystal_mol_h.GetConformer().GetAtomPosition(j)
                            conf.SetAtomPosition(i, pos)
                        mol.AddConformer(conf)
                        use_crystal = True
            except Exception as e:
                pass

        if use_crystal:
            if n_conformers > 1:
                try:
                    additional_conf_ids = AllChem.EmbedMultipleConfs(
                        mol,
                        numConfs=n_conformers-1,
                        pruneRmsThresh=0.5,
                        randomSeed=42,
                        clearConfs=False
                    )
                except Exception as e:
                    additional_conf_ids = []
            conf_ids = list(range(mol.GetNumConformers()))
        else:
            try:
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers,
                    pruneRmsThresh=0.5,
                    randomSeed=42,
                    useRandomCoords=True
                )
            except Exception as e:
                conf_ids = []

            if len(conf_ids) == 0:
                try:
                    res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    if res == -1:
                        print(f"Warning: Failed to embed molecule: {smiles_string}")
                        return None
                    conf_ids = [0]
                except Exception as e:
                    print(f"Error embedding molecule {smiles_string}: {e}")
                    return None

        for i, conf_id in enumerate(conf_ids):
            if use_crystal and i == 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            except Exception as e:
                pass

        valid_conf_ids = []
        for conf_id in conf_ids:
            if conf_id < mol.GetNumConformers():
                valid_conf_ids.append(conf_id)

        if not valid_conf_ids:
            print(f"Warning: No valid conformers generated for {smiles_string}")
            return None

        return mol, valid_conf_ids

    except Exception as e:
        print(f"Error generating conformers for {smiles_string}: {e}")
        return None


def generate_conformer_ensemble(smiles_string, n_conformers=5):
    """Simple conformer generation without crystal structure."""
    return generate_conformer_ensemble_with_crystal(smiles_string, None, n_conformers)


def mol_to_3d_graph(mol, conf_id=0):
    """Converts a molecule with 3D coordinates to a PyG Data object with validation."""
    if mol is None:
        return None

    if conf_id >= mol.GetNumConformers():
        print(f"Warning: Invalid conformer ID {conf_id} (molecule has {mol.GetNumConformers()} conformers)")
        return None

    try:
        conf = mol.GetConformer(conf_id)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

        atom_features = []
        atomic_numbers = []
        for atom in mol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
            atom_features.append([
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs()
            ])

        z = torch.tensor(atomic_numbers, dtype=torch.long)
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_indices, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = [
                int(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_type, bond_type])

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        print(f"Error converting molecule to graph (conf_id={conf_id}): {e}")
        return None


def smiles_to_ecfp(smiles_string, n_bits=2048):
    """Converts a SMILES string to an ECFP (Morgan) fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            print(f"Warning: Failed to parse SMILES for ECFP: {smiles_string}")
            return np.zeros(n_bits, dtype=np.float32)

        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    except Exception as e:
        print(f"Error generating ECFP for {smiles_string}: {e}")
        return np.zeros(n_bits, dtype=np.float32)


def precompute_all_conformers(all_data_df, structure_paths, n_conformers=5, cache_path='conformers_precomputed.pkl'):
    """Pre-compute all conformers before training starts - supports multiple structure paths."""
    if os.path.exists(cache_path):
        try:
            print(f"Loading pre-computed conformers from {cache_path}")
            with open(cache_path, 'rb') as f:
                conformer_dict = pickle.load(f)
            print(f"Loaded {len(conformer_dict)} pre-computed conformer sets")
            return conformer_dict
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            print("Will regenerate conformers...")

    print("Pre-computing all conformers (this only happens once)...")
    conformer_dict = {}
    failed_entries = []

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Pre-computing conformers"):
        pdb_id = row['pdb_id']
        smiles = row['canonical_smiles']

        crystal_mol = None
        ligand_found = False

        for structure_path in structure_paths:
            ligand_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_ligand.sdf")

            if not os.path.exists(ligand_path):
                ligand_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_ligand.mol2")

            if os.path.exists(ligand_path):
                try:
                    if ligand_path.endswith('.sdf'):
                        ligand_supplier = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)
                        if ligand_supplier and len(ligand_supplier) > 0:
                            crystal_mol = ligand_supplier[0]
                    elif ligand_path.endswith('.mol2'):
                        crystal_mol = Chem.MolFromMol2File(ligand_path, removeHs=False, sanitize=True)
                    ligand_found = True
                    break
                except Exception as e:
                    pass

        mol_conf_data = generate_conformer_ensemble_with_crystal(smiles, crystal_mol, n_conformers)

        if mol_conf_data:
            mol, conf_ids = mol_conf_data
            graphs = []
            for conf_id in conf_ids[:n_conformers]:
                graph = mol_to_3d_graph(mol, conf_id)
                if graph:
                    graphs.append(graph)

            if graphs:
                key = f"{pdb_id}_{smiles}"
                conformer_dict[key] = graphs
            else:
                failed_entries.append(pdb_id)
        else:
            failed_entries.append(pdb_id)

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(conformer_dict, f)
        print(f"Saved to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save conformer cache: {e}")

    print(f"Pre-computed {len(conformer_dict)} conformer sets")
    if failed_entries:
        print(f"Failed to compute conformers for {len(failed_entries)} entries")

    return conformer_dict


def precompute_ecfp_fingerprints(all_data_df, cache_path='ecfp_precomputed.pkl'):
    """Pre-compute all ECFP fingerprints."""
    if os.path.exists(cache_path):
        try:
            print(f"Loading pre-computed ECFP fingerprints from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load ECFP cache from {cache_path}: {e}")
            print("Will regenerate ECFP fingerprints...")

    print("Pre-computing ECFP fingerprints...")
    ecfp_dict = {}

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Computing ECFP"):
        smiles = row['canonical_smiles']
        if smiles not in ecfp_dict:
            ecfp_dict[smiles] = smiles_to_ecfp(smiles)

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(ecfp_dict, f)
    except Exception as e:
        print(f"Warning: Failed to save ECFP cache: {e}")

    print(f"Pre-computed {len(ecfp_dict)} ECFP fingerprints")
    return ecfp_dict


def preprocess_and_cache_tokens(all_data_df, structure_paths, plm_tokenizer, cache_path):
    """Pre-tokenize all protein sequences for efficiency - supports multiple structure paths."""
    if os.path.exists(cache_path):
        try:
            print(f"Loading cached protein tokens from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load token cache from {cache_path}: {e}")
            print("Will regenerate tokens...")

    print("Pre-tokenizing protein sequences (this will only happen once)...")
    token_cache = {}

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Tokenizing proteins"):
        pdb_id = row['pdb_id']

        sequence = None
        for structure_path in structure_paths:
            protein_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_protein.pdb")

            if not os.path.exists(protein_path):
                protein_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_pocket.pdb")

            if os.path.exists(protein_path):
                sequence = extract_protein_sequence(protein_path)
                if sequence:
                    break

        if sequence:
            try:
                tokens = plm_tokenizer(
                    sequence, return_tensors='pt', padding='longest',
                    truncation=True, max_length=1024
                )
                token_cache[pdb_id] = {
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'sequence': sequence
                }
            except Exception as e:
                print(f"Error tokenizing protein {pdb_id}: {e}")

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(token_cache, f)
        print(f"Saved token cache with {len(token_cache)} proteins to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save token cache: {e}")

    return token_cache

print("Preprocessing functions defined!")

# =============================================================================
# DATASET MODULE
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING DATASET CLASSES")
print("=" * 70)

class OptimizedPDBbindDataset(Dataset):
    """Ultra-fast dataset using only pre-computed data with normalization support."""

    def __init__(self, dataframe, conformer_dict, token_cache, ecfp_dict, normalizer=None):
        self.conformer_dict = conformer_dict
        self.token_cache = token_cache
        self.ecfp_dict = ecfp_dict
        self.normalizer = normalizer

        valid_entries = []
        skipped_no_conformer = 0
        skipped_no_token = 0
        skipped_no_ecfp = 0

        for _, row in dataframe.iterrows():
            pdb_id = row['pdb_id']
            smiles = row['canonical_smiles']
            key = f"{pdb_id}_{smiles}"

            has_conformer = key in self.conformer_dict
            has_token = pdb_id in self.token_cache
            has_ecfp = smiles in self.ecfp_dict

            if has_conformer and has_token and has_ecfp:
                entry = row.to_dict()
                if self.normalizer and self.normalizer.fitted:
                    entry['normalized_affinity'] = self.normalizer.transform(
                        [entry['affinity']]
                    )[0]
                else:
                    entry['normalized_affinity'] = entry['affinity']
                valid_entries.append(entry)
            else:
                if not has_conformer:
                    skipped_no_conformer += 1
                if not has_token:
                    skipped_no_token += 1
                if not has_ecfp:
                    skipped_no_ecfp += 1

        self.data = valid_entries

        total_requested = len(dataframe)
        total_valid = len(self.data)
        print(f"Dataset initialized: {total_valid}/{total_requested} valid entries")
        if total_valid < total_requested:
            print(f"  Skipped - Missing conformers: {skipped_no_conformer}")
            print(f"  Skipped - Missing tokens: {skipped_no_token}")
            print(f"  Skipped - Missing ECFP: {skipped_no_ecfp}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        pdb_id = row['pdb_id']
        smiles = row['canonical_smiles']
        key = f"{pdb_id}_{smiles}"

        return {
            'graphs_3d': self.conformer_dict[key],
            'ecfp': torch.from_numpy(self.ecfp_dict[smiles]),
            'protein_tokens': self.token_cache[pdb_id],
            'label': torch.tensor(row['normalized_affinity'], dtype=torch.float32),
            'original_label': torch.tensor(row['affinity'], dtype=torch.float32),
            'pdb_id': pdb_id,
            'smiles': smiles
        }


def custom_collate(batch):
    """Collate function handling conformer ensembles and variable-length sequences."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    all_graphs = []
    conformer_batch_idx = []
    for i, item in enumerate(batch):
        graphs = item.get('graphs_3d', [])
        if not graphs:
            continue
        for graph in graphs:
            all_graphs.append(graph)
            conformer_batch_idx.append(i)

    if not all_graphs:
        print("Warning: No valid graphs in batch")
        return None

    try:
        max_seq_len = max(item['protein_tokens']['input_ids'].size(0) for item in batch)
    except Exception as e:
        print(f"Error getting max sequence length: {e}")
        return None

    padded_input_ids = []
    padded_attention_masks = []

    for item in batch:
        try:
            input_ids = item['protein_tokens']['input_ids']
            attention_mask = item['protein_tokens']['attention_mask']

            pad_len = max_seq_len - input_ids.size(0)

            if pad_len > 0:
                input_ids_padded = F.pad(input_ids, (0, pad_len), value=PLM_PAD_ID)
                attention_mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
            else:
                input_ids_padded = input_ids
                attention_mask_padded = attention_mask

            padded_input_ids.append(input_ids_padded)
            padded_attention_masks.append(attention_mask_padded)
        except Exception as e:
            print(f"Error padding sequence: {e}")
            return None

    try:
        collated_batch = {
            'graphs_3d': Batch.from_data_list(all_graphs),
            'conformer_batch_idx': torch.tensor(conformer_batch_idx, dtype=torch.long),
            'ecfp': torch.stack([item['ecfp'] for item in batch]),
            'protein_tokens': {
                'input_ids': torch.stack(padded_input_ids),
                'attention_mask': torch.stack(padded_attention_masks)
            }
        }

        if 'label' in batch[0]:
            collated_batch['label'] = torch.stack([item['label'] for item in batch])
            collated_batch['original_label'] = torch.stack([item['original_label'] for item in batch])
        if 'pdb_id' in batch[0]:
            collated_batch['pdb_id'] = [item['pdb_id'] for item in batch]
        if 'smiles' in batch[0]:
            collated_batch['smiles'] = [item['smiles'] for item in batch]

        return collated_batch

    except Exception as e:
        print(f"Error creating collated batch: {e}")
        return None

print("Dataset classes defined!")

# =============================================================================
# ENCODER MODELS
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING ENCODER MODELS")
print("=" * 70)

class PLM_Encoder(nn.Module):
    """Protein Language Model Encoder using ESM-2."""

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.model = EsmModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, tokens):
        out = self.model(**tokens)
        reps = out.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        summed = (reps * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = summed / denom
        return reps, pooled


class GNN_2D_Encoder(nn.Module):
    """2D Topology GNN Encoder using Graph Attention Networks."""

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_emb(x)

        if edge_attr.size(0) > 0:
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = torch.zeros((0, x.size(1)), device=x.device, dtype=x.dtype)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))

        return x


class GNN_3D_Encoder(nn.Module):
    """3D-aware GNN using distance-based message passing."""

    def __init__(self, hidden_channels=128, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_channels)
        self.distance_expansion = nn.Linear(20, hidden_channels)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                GATv2Conv(hidden_channels, hidden_channels, edge_dim=hidden_channels, heads=4, concat=False)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels))

        self.final_norm = nn.LayerNorm(hidden_channels)

    def gaussian_expansion(self, dist, start=0.0, stop=10.0, num_gaussians=20):
        mu = torch.linspace(start, stop, num_gaussians, device=dist.device)
        mu = mu.view(1, -1)
        sigma = (stop - start) / num_gaussians
        return torch.exp(-((dist.view(-1, 1) - mu) ** 2) / (2 * sigma ** 2 + 1e-8))

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        h = self.embedding(z.clamp(min=0, max=99))
        edge_index = radius_graph(pos, r=10.0, batch=batch)

        if edge_index.size(1) == 0:
            return h

        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1)
        edge_attr = self.distance_expansion(self.gaussian_expansion(dist))

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h) + h_in

        return self.final_norm(h)


class PhysicsInformedGNN(nn.Module):
    """Physics-informed GNN that processes protein pocket dynamics."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.pocket_encoder = nn.Sequential(
            nn.Linear(320, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.correlation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

    def forward(self, protein_residue_embs):
        h_pocket_dyn = self.pocket_encoder(protein_residue_embs)
        corr_features = self.correlation_head(h_pocket_dyn.mean(dim=1))
        return h_pocket_dyn, corr_features

print("Encoder models defined!")

# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING ATTENTION MECHANISMS")
print("=" * 70)

# Note: Simple CrossAttention class removed - using HierarchicalCrossAttention instead
# If you need the simpler version, it's available in git history


class HierarchicalCrossAttention(nn.Module):
    """Hierarchical Cross-Attention module with two levels."""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.pocket_attention = nn.MultiheadAttention(
            embed_dim, num_heads//2, batch_first=True
        )
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(self, ligand_emb, protein_residues, protein_mask):
        ligand_global = ligand_emb.mean(dim=1, keepdim=True)
        pocket_scores, _ = self.pocket_attention(
            ligand_global,
            protein_residues,
            protein_residues,
            key_padding_mask=~protein_mask.bool()
        )

        interaction_output, interaction_weights = self.interaction_attention(
            ligand_emb,
            protein_residues,
            protein_residues,
            key_padding_mask=~protein_mask.bool()
        )

        return interaction_output, interaction_weights, pocket_scores


class ConformerGate(nn.Module):
    """Learned attention mechanism for weighting conformers based on protein alignment."""

    def __init__(self, hidden_dim, protein_dim):
        super().__init__()
        self.conformer_scorer = nn.Sequential(
            nn.Linear(hidden_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, conformer_features, protein_global, n_conformers):
        conformer_global = conformer_features.mean(dim=1)
        protein_expanded = protein_global.unsqueeze(0).expand(n_conformers, -1)
        scores_input = torch.cat([conformer_global, protein_expanded], dim=-1)
        scores = self.conformer_scorer(scores_input)
        weights = F.softmax(scores, dim=0)
        weighted_features = (conformer_features * weights.unsqueeze(1)).sum(dim=0)
        return weighted_features, weights

print("Attention mechanisms defined!")

# =============================================================================
# KAN (KOLMOGOROV-ARNOLD NETWORK) MODULE
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING KAN LAYER")
print("=" * 70)

class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network (KAN) Linear Layer."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (1.0 / grid_size)
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h
        self.register_buffer("grid", grid.float())

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        with torch.no_grad():
            self.spline_weight.uniform_(-0.1, 0.1)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            denominator1 = self.grid[k:-1] - self.grid[:-(k + 1)]
            denominator2 = self.grid[k + 1:] - self.grid[1:(-k)]

            denominator1 = torch.where(
                denominator1 == 0,
                torch.ones_like(denominator1),
                denominator1
            )
            denominator2 = torch.where(
                denominator2 == 0,
                torch.ones_like(denominator2),
                denominator2
            )

            term1 = ((x - self.grid[:-(k + 1)]) / denominator1) * bases[:, :, :-1]
            term2 = ((self.grid[k + 1:] - x) / denominator2) * bases[:, :, 1:]

            bases = term1 + term2

        return bases

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        spline_basis = self.b_splines(x)
        spline_basis_flat = spline_basis.view(x.size(0), -1)
        spline_weight_flat = self.spline_weight.view(self.out_features, -1)
        spline_output = F.linear(spline_basis_flat, spline_weight_flat)
        return base_output + spline_output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'grid_size={self.grid_size}, spline_order={self.spline_order}'

print("KAN layer defined!")

# =============================================================================
# MAIN AURA MODEL
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING MAIN AURA MODEL")
print("=" * 70)

class AuraDeepLearningModel(nn.Module):
    """AURA (Affinity Unified Regression Architecture) Deep Learning Model."""

    def __init__(self, gnn_2d_node_dim, gnn_2d_edge_dim, hidden_dim, plm_hidden_dim, out_dim):
        super().__init__()

        self.plm_encoder = PLM_Encoder()
        self.gnn_2d_encoder = GNN_2D_Encoder(gnn_2d_node_dim, gnn_2d_edge_dim, hidden_dim)
        self.gnn_3d_encoder = GNN_3D_Encoder(hidden_channels=hidden_dim)
        self.physics_gnn = PhysicsInformedGNN(hidden_dim)

        self.proj_2d = nn.Linear(hidden_dim, plm_hidden_dim)
        self.proj_3d = nn.Linear(hidden_dim, plm_hidden_dim)
        self.proj_physics = nn.Linear(hidden_dim // 4, plm_hidden_dim)

        self.conformer_gate = ConformerGate(plm_hidden_dim, plm_hidden_dim)
        self.hierarchical_attention = HierarchicalCrossAttention(plm_hidden_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(plm_hidden_dim * 4, plm_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(plm_hidden_dim * 2, plm_hidden_dim)
        )

        self.kan_head = nn.Sequential(
            KANLinear(plm_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            KANLinear(64, out_dim)
        )

    def forward(self, batch):
        graphs_3d = batch['graphs_3d'].to(DEVICE)
        protein_tokens = {k: v.to(DEVICE) for k, v in batch['protein_tokens'].items()}
        conformer_batch_idx = batch['conformer_batch_idx'].to(DEVICE)

        protein_residue_embs, protein_global_emb = self.plm_encoder(protein_tokens)
        ligand_2d_nodes = self.gnn_2d_encoder(graphs_3d)
        ligand_3d_nodes = self.gnn_3d_encoder(graphs_3d)
        pocket_dyn, corr_features = self.physics_gnn(protein_residue_embs)

        ligand_2d_proj = self.proj_2d(ligand_2d_nodes)
        ligand_3d_proj = self.proj_3d(ligand_3d_nodes)
        ligand_fused_nodes = ligand_2d_proj + ligand_3d_proj

        batch_size = protein_tokens['input_ids'].size(0)
        ligand_nodes_list = []
        conformer_weights_list = []

        for i in range(batch_size):
            conformer_indices = (conformer_batch_idx == i).nonzero(as_tuple=True)[0]

            if len(conformer_indices) > 0:
                node_indices = []
                for conf_idx in conformer_indices:
                    node_mask = (graphs_3d.batch == conf_idx)
                    node_indices.extend(node_mask.nonzero(as_tuple=True)[0].tolist())

                if node_indices:
                    node_indices = torch.tensor(node_indices, device=DEVICE)
                    molecule_nodes = ligand_fused_nodes[node_indices]

                    n_conformers = len(conformer_indices)
                    n_atoms = len(node_indices) // n_conformers

                    conformer_features = molecule_nodes.reshape(n_conformers, n_atoms, -1)
                    weighted_nodes, conformer_weights = self.conformer_gate(
                        conformer_features,
                        protein_global_emb[i],
                        n_conformers
                    )
                    ligand_nodes_list.append(weighted_nodes)
                    conformer_weights_list.append(conformer_weights.squeeze(-1).detach().cpu())

        if not ligand_nodes_list:
            empty_info = {
                'interaction_weights': None,
                'pocket_scores': None,
                'conformer_weights': None
            }
            return torch.zeros(batch_size, 1, device=DEVICE), empty_info

        ligand_nodes_padded = nn.utils.rnn.pad_sequence(ligand_nodes_list, batch_first=True)

        interaction_output, interaction_weights, pocket_scores = self.hierarchical_attention(
            ligand_nodes_padded, protein_residue_embs, protein_tokens['attention_mask']
        )

        context_ligand_global = interaction_output.mean(dim=1)
        ligand_global_pool = torch.stack([nodes.mean(dim=0) for nodes in ligand_nodes_list])
        physics_proj = self.proj_physics(corr_features)

        final_fused = self.fusion_mlp(torch.cat([
            context_ligand_global,
            protein_global_emb,
            ligand_global_pool,
            physics_proj
        ], dim=1))

        prediction = self.kan_head(final_fused)

        interpretability_info = {
            'interaction_weights': interaction_weights,
            'pocket_scores': pocket_scores,
            'conformer_weights': conformer_weights_list if conformer_weights_list else None
        }
        return prediction, interpretability_info

print("Main AURA model defined!")

# =============================================================================
# ENSEMBLE MODEL
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING ENSEMBLE MODEL")
print("=" * 70)

class AuraEnsemble(nn.Module):
    """AURA Ensemble Model combining deep learning and XGBoost predictions."""

    def __init__(self, dl_model, normalizer=None):
        super().__init__()
        self.dl_model = dl_model
        self.xgb_model = None
        self.ensemble_weight = nn.Parameter(torch.tensor([0.5]))
        self.normalizer = normalizer
        self.initial_weights = None

    def set_xgb_model(self, xgb_model):
        self.xgb_model = xgb_model

    def store_initial_weights(self):
        self.initial_weights = {}
        for name, param in self.dl_model.named_parameters():
            self.initial_weights[name] = param.data.clone()

    def get_l2_regularization_loss(self):
        if self.initial_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        l2_loss = 0.0
        for name, param in self.dl_model.named_parameters():
            if name in self.initial_weights:
                l2_loss += torch.sum((param - self.initial_weights[name]) ** 2)

        return L2_REG_WEIGHT * l2_loss

    def forward(self, batch, xgb_features=None):
        dl_pred, interpretability_info = self.dl_model(batch)

        if self.xgb_model is not None and xgb_features is not None:
            try:
                xgb_pred = torch.tensor(
                    self.xgb_model.predict(xgb_features),
                    device=dl_pred.device,
                    dtype=dl_pred.dtype
                ).unsqueeze(1)

                w = torch.sigmoid(self.ensemble_weight)
                ensemble_pred = w * dl_pred + (1 - w) * xgb_pred

                return ensemble_pred, interpretability_info
            except Exception as e:
                print(f"Warning: XGBoost prediction failed: {e}")
                print("Falling back to deep learning predictions only")
                return dl_pred, interpretability_info

        return dl_pred, interpretability_info

    def get_ensemble_weight(self):
        return torch.sigmoid(self.ensemble_weight).item()

    def freeze_dl_model(self):
        for param in self.dl_model.parameters():
            param.requires_grad = False

    def unfreeze_dl_model(self):
        for param in self.dl_model.parameters():
            param.requires_grad = True

    def freeze_except_ensemble_weight(self):
        for name, param in self.named_parameters():
            param.requires_grad = (name == 'ensemble_weight')

print("Ensemble model defined!")

# =============================================================================
# METRICS MODULE
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING METRICS")
print("=" * 70)

def concordance_index(y_true, y_pred):
    """Calculate the concordance index (C-index)."""
    n = len(y_true)
    if n < 2:
        return 0.5

    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] != y_true[j]:
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    tied += 1
                else:
                    discordant += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5

    return (concordant + 0.5 * tied) / total


def regression_metrics(y_true, y_pred, denormalize=False, normalizer=None):
    """Calculate comprehensive regression evaluation metrics."""
    if len(y_true) == 0:
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'CI': 0.0,
            'Pearson_R': 0.0,
            'Spearman_R': 0.0
        }

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if denormalize and normalizer and normalizer.fitted:
        y_true = normalizer.inverse_transform(y_true)
        y_pred = normalizer.inverse_transform(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    ci = concordance_index(y_true, y_pred)

    if len(y_true) > 1:
        try:
            pearson_r, _ = pearsonr(y_true, y_pred)
            if np.isnan(pearson_r):
                pearson_r = 0.0
        except Exception:
            pearson_r = 0.0

        try:
            spearman_r, _ = spearmanr(y_true, y_pred)
            if np.isnan(spearman_r):
                spearman_r = 0.0
        except Exception:
            spearman_r = 0.0
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MSE': float(mse),
        'CI': float(ci),
        'Pearson_R': float(pearson_r),
        'Spearman_R': float(spearman_r)
    }


def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way."""
    if prefix:
        print(f"\n{prefix}:")
    print(f"  RMSE:       {metrics['RMSE']:.4f}")
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  MSE:        {metrics['MSE']:.4f}")
    print(f"  CI:         {metrics['CI']:.4f}")
    print(f"  Pearson R:  {metrics['Pearson_R']:.4f}")
    print(f"  Spearman R: {metrics['Spearman_R']:.4f}")

print("Metrics defined!")

# =============================================================================
# TRAINING UTILITIES
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING TRAINING UTILITIES")
print("=" * 70)

def train_one_epoch_with_accumulation(model, dataloader, optimizer, criterion,
                                     accumulation_steps=4, use_amp=False, scaler=None):
    """Training with gradient accumulation and gradient clipping."""
    model.train()
    total_loss = 0
    n_batches = 0

    if use_amp and scaler is None and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        if batch is None:
            continue

        try:
            if use_amp and torch.cuda.is_available():
                from torch.cuda.amp import autocast
                with autocast():
                    predictions, _ = model(batch)
                    predictions = predictions.view(-1)
                    labels = batch['label'].to(DEVICE).view(-1)
                    loss = criterion(predictions, labels)

                    if hasattr(model, 'get_l2_regularization_loss'):
                        loss = loss + model.get_l2_regularization_loss()

                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                predictions, _ = model(batch)
                predictions = predictions.view(-1)
                labels = batch['label'].to(DEVICE).view(-1)
                loss = criterion(predictions, labels)

                if hasattr(model, 'get_l2_regularization_loss'):
                    loss = loss + model.get_l2_regularization_loss()

                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            n_batches += 1

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

    if len(dataloader) % accumulation_steps != 0:
        if use_amp and torch.cuda.is_available():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
        optimizer.zero_grad()

    average_loss = total_loss / max(n_batches, 1)
    return average_loss, scaler


def evaluate_model(model, dataloader, normalizer=None):
    """Evaluate regression model - returns both normalized and original predictions."""
    model.eval()
    all_preds, all_labels, all_original_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue

            try:
                predictions, _ = model(batch)
                pred_cpu = predictions.squeeze(-1).cpu()
                if pred_cpu.dim() == 0:
                    pred_cpu = pred_cpu.unsqueeze(0)
                all_preds.append(pred_cpu)

                label_cpu = batch['label'].cpu()
                if label_cpu.dim() == 0:
                    label_cpu = label_cpu.unsqueeze(0)
                all_labels.append(label_cpu)

                if 'original_label' in batch:
                    orig_label_cpu = batch['original_label'].cpu()
                    if orig_label_cpu.dim() == 0:
                        orig_label_cpu = orig_label_cpu.unsqueeze(0)
                    all_original_labels.append(orig_label_cpu)

            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

    if not all_preds:
        return np.array([]), np.array([]), np.array([])

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if all_original_labels:
        all_original_labels = torch.cat(all_original_labels).numpy()
    else:
        if normalizer and normalizer.fitted:
            all_original_labels = normalizer.inverse_transform(all_labels)
        else:
            all_original_labels = all_labels

    return all_preds, all_labels, all_original_labels


def extract_features_for_xgboost(model, dataloader):
    """Extract features for XGBoost regression."""
    model.eval()
    features, labels, original_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting XGBoost Features"):
            if batch is None:
                continue

            try:
                dl_preds, _ = model(batch)
                dl_preds = dl_preds.squeeze(-1).cpu().numpy()
                if dl_preds.ndim == 0:
                    dl_preds = np.array([dl_preds])

                ecfp = batch['ecfp'].cpu().numpy()

                batch_features = np.hstack([
                    ecfp.astype(np.float32),
                    dl_preds.reshape(-1, 1).astype(np.float32)
                ])
                features.append(batch_features)

                label_cpu = batch['label'].cpu().numpy()
                if label_cpu.ndim == 0:
                    label_cpu = np.array([label_cpu])
                labels.append(label_cpu)

                if 'original_label' in batch:
                    orig_label_cpu = batch['original_label'].cpu().numpy()
                    if orig_label_cpu.ndim == 0:
                        orig_label_cpu = np.array([orig_label_cpu])
                    original_labels.append(orig_label_cpu)

            except Exception as e:
                print(f"Error extracting features: {e}")
                continue

    if not features:
        return np.array([]), np.array([]), np.array([])

    features_array = np.vstack(features)
    labels_array = np.concatenate(labels)
    original_labels_array = np.concatenate(original_labels) if original_labels else None

    return features_array, labels_array, original_labels_array

print("Training utilities defined!")

# =============================================================================
# TRAINER MODULE
# =============================================================================
print("\n" + "=" * 70)
print("DEFINING STAGED TRAINER")
print("=" * 70)

class StagedTrainer:
    """Implements staged training strategy with normalization support."""

    def __init__(self, ensemble_model, train_loader, val_loader, test_loaders_dict,
                 normalizer=None, device='cuda', use_amp=False, save_dir='./saved_models'):
        self.ensemble_model = ensemble_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loaders_dict = test_loaders_dict
        self.normalizer = normalizer
        self.device = device
        self.use_amp = use_amp
        self.save_dir = save_dir

        if use_amp and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        os.makedirs(save_dir, exist_ok=True)

        self.training_history = {
            'stage_b': {'loss': [], 'val_metrics': []},
            'stage_c': {'loss': [], 'val_metrics': []},
            'test_results': {},
            'normalization_params': normalizer.get_params() if normalizer else None
        }

    def freeze_backbone(self):
        for name, param in self.ensemble_model.dl_model.named_parameters():
            if 'kan_head' not in name and 'ensemble_weight' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.ensemble_model.parameters():
            param.requires_grad = True

    def stage_b_head_training(self, epochs=15):
        print("\n" + "="*70)
        print("STAGE B: Head Training (Backbone Frozen)")
        print("="*70)
        print("Training regression heads on normalized values...")

        if self.normalizer:
            print(f"Normalization: {self.normalizer.method}")
            if self.normalizer.method == 'zscore':
                print(f"  Mean: {self.normalizer.mean:.4f}, Std: {self.normalizer.std:.4f}")
            else:
                print(f"  Min: {self.normalizer.min:.4f}, Max: {self.normalizer.max:.4f}")

        self.freeze_backbone()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.ensemble_model.parameters()),
            lr=5e-4
        )
        criterion = nn.MSELoss()

        best_val_pearson = -1
        patience_counter = 0
        best_path = None

        for epoch in range(epochs):
            train_loss, self.scaler = train_one_epoch_with_accumulation(
                self.ensemble_model, self.train_loader, optimizer, criterion,
                accumulation_steps=ACCUMULATION_STEPS,
                use_amp=self.use_amp,
                scaler=self.scaler
            )

            val_preds, val_labels, val_original_labels = evaluate_model(
                self.ensemble_model, self.val_loader, self.normalizer
            )

            if len(val_labels) > 0:
                if self.normalizer and self.normalizer.fitted:
                    val_preds_denorm = self.normalizer.inverse_transform(val_preds)
                else:
                    val_preds_denorm = val_preds

                val_metrics = regression_metrics(val_original_labels, val_preds_denorm)
            else:
                val_metrics = {
                    'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0,
                    'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0
                }

            self.training_history['stage_b']['loss'].append(train_loss)
            self.training_history['stage_b']['val_metrics'].append(val_metrics)

            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f} (normalized scale)")
            print(f"    Val RMSE={val_metrics['RMSE']:.4f}, Pearson R={val_metrics['Pearson_R']:.4f} (original scale)")

            if val_metrics['Pearson_R'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson_R']
                best_path = os.path.join(self.save_dir, 'stage_b_best.pt')
                torch.save(self.ensemble_model.state_dict(), best_path)
                print(f"    New best model saved! (Pearson R: {val_metrics['Pearson_R']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

        if best_path and os.path.exists(best_path):
            self.ensemble_model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best Stage B model from {best_path}")

        self.ensemble_model.store_initial_weights()

    def stage_c_fine_tuning(self, epochs=25):
        print("\n" + "="*70)
        print("STAGE C: Fine-tuning (All Layers)")
        print("="*70)
        print("Fine-tuning with normalized values and proper warmup scheduling...")

        self.unfreeze_all()

        optimizer = torch.optim.AdamW(
            self.ensemble_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        criterion = nn.MSELoss()

        best_val_pearson = -1
        patience_counter = 0
        best_path = None

        global_step = 0

        for epoch in range(epochs):
            self.ensemble_model.train()
            total_loss = 0
            n_batches = 0

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Stage C Epoch {epoch+1}/{epochs}")):
                if batch is None:
                    continue

                try:
                    optimizer.zero_grad()

                    predictions, _ = self.ensemble_model(batch)
                    predictions = predictions.view(-1)
                    labels = batch['label'].to(self.device).view(-1)
                    loss = criterion(predictions, labels)

                    if hasattr(self.ensemble_model, 'get_l2_regularization_loss'):
                        loss = loss + self.ensemble_model.get_l2_regularization_loss()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.ensemble_model.parameters(),
                        GRADIENT_CLIP_NORM
                    )
                    optimizer.step()

                    if global_step < WARMUP_STEPS:
                        warmup_progress = global_step / WARMUP_STEPS
                        lr = LEARNING_RATE * (0.01 + 0.99 * warmup_progress)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    global_step += 1
                    total_loss += loss.item()
                    n_batches += 1

                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue

            train_loss = total_loss / max(n_batches, 1)
            val_preds, val_labels, val_original_labels = evaluate_model(
                self.ensemble_model, self.val_loader, self.normalizer
            )

            if len(val_labels) > 0:
                if self.normalizer and self.normalizer.fitted:
                    val_preds_denorm = self.normalizer.inverse_transform(val_preds)
                else:
                    val_preds_denorm = val_preds

                val_metrics = regression_metrics(val_original_labels, val_preds_denorm)
            else:
                val_metrics = {
                    'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0,
                    'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0
                }

            self.training_history['stage_c']['loss'].append(train_loss)
            self.training_history['stage_c']['val_metrics'].append(val_metrics)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f} (normalized), LR={current_lr:.6f}")
            print(f"    Val RMSE={val_metrics['RMSE']:.4f}, Pearson R={val_metrics['Pearson_R']:.4f} (original scale)")

            if val_metrics['Pearson_R'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson_R']
                best_path = os.path.join(self.save_dir, 'aura_final_best.pt')
                torch.save(self.ensemble_model.state_dict(), best_path)
                print(f"    New best model saved! (Pearson R: {val_metrics['Pearson_R']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

        if best_path and os.path.exists(best_path):
            self.ensemble_model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best Stage C model from {best_path}")

    def train_xgboost_component(self):
        print("\n" + "="*70)
        print("Training XGBoost Component (on normalized values)")
        print("="*70)

        X_train, y_train, _ = extract_features_for_xgboost(
            self.ensemble_model.dl_model, self.train_loader
        )
        X_val, y_val, _ = extract_features_for_xgboost(
            self.ensemble_model.dl_model, self.val_loader
        )

        if len(X_train) > 0 and len(X_val) > 0:
            xgb_model = xgb.XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                random_state=XGB_RANDOM_STATE,
                eval_metric='rmse',
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS
            )

            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            self.ensemble_model.set_xgb_model(xgb_model)

            xgb_path = os.path.join(self.save_dir, 'xgboost_model.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            print(f"Saved XGBoost model to {xgb_path}")

            self.fine_tune_ensemble_weight()
        else:
            print("Not enough data to train XGBoost component")

    def fine_tune_ensemble_weight(self, epochs=3):
        print("\n" + "="*70)
        print("Fine-tuning Ensemble Weight")
        print("="*70)

        for name, param in self.ensemble_model.named_parameters():
            param.requires_grad = (name == 'ensemble_weight')

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.ensemble_model.parameters()),
            lr=1e-3
        )
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.ensemble_model.train()

            for batch in tqdm(self.train_loader, desc=f"Weight Tuning {epoch+1}/{epochs}"):
                if batch is None:
                    continue

                try:
                    with torch.no_grad():
                        dl_pred, _ = self.ensemble_model.dl_model(batch)
                        dl_pred_np = dl_pred.cpu().numpy()
                        ecfp = batch['ecfp'].cpu().numpy()
                        xgb_features = np.hstack([ecfp, dl_pred_np.reshape(-1, 1)])

                    optimizer.zero_grad()
                    ensemble_pred, _ = self.ensemble_model(batch, xgb_features)
                    ensemble_pred = ensemble_pred.view(-1)
                    labels = batch['label'].to(self.device).view(-1)
                    loss = criterion(ensemble_pred, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.ensemble_model.parameters()),
                        GRADIENT_CLIP_NORM
                    )

                    optimizer.step()
                except Exception as e:
                    print(f"Error in weight tuning: {e}")
                    continue

            w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
            print(f"  Ensemble weight (DL portion): {w:.3f}")

        final_path = os.path.join(self.save_dir, 'aura_ensemble_final.pt')
        torch.save(self.ensemble_model.state_dict(), final_path)
        print(f"Saved final ensemble model to {final_path}")

    def evaluate_on_all_test_sets(self):
        print("\n" + "="*70)
        print("Final Model Evaluation on All Test Sets (Original Scale)")
        print("="*70)

        all_results = {}

        for test_name, test_loader in self.test_loaders_dict.items():
            print(f"\nEvaluating on {test_name}:")
            print("="*50)

            test_preds_dl, test_labels, test_original_labels = evaluate_model(
                self.ensemble_model.dl_model, test_loader, self.normalizer
            )

            if len(test_labels) == 0:
                print(f"No test samples available for {test_name}")
                continue

            if self.normalizer and self.normalizer.fitted:
                test_preds_dl_denorm = self.normalizer.inverse_transform(test_preds_dl)
            else:
                test_preds_dl_denorm = test_preds_dl

            metrics_results = {}
            metrics_results['Deep Learning'] = regression_metrics(
                test_original_labels, test_preds_dl_denorm
            )

            if self.ensemble_model.xgb_model is not None:
                X_test, _, _ = extract_features_for_xgboost(
                    self.ensemble_model.dl_model, test_loader
                )

                if len(X_test) > 0:
                    test_preds_xgb = self.ensemble_model.xgb_model.predict(X_test)

                    if self.normalizer and self.normalizer.fitted:
                        test_preds_xgb_denorm = self.normalizer.inverse_transform(test_preds_xgb)
                    else:
                        test_preds_xgb_denorm = test_preds_xgb

                    w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
                    test_preds_ensemble = w * test_preds_dl_denorm + (1 - w) * test_preds_xgb_denorm

                    metrics_results['XGBoost'] = regression_metrics(
                        test_original_labels, test_preds_xgb_denorm
                    )
                    metrics_results['Ensemble'] = regression_metrics(
                        test_original_labels, test_preds_ensemble
                    )

            for model_name in metrics_results:
                print_metrics(metrics_results[model_name], prefix=model_name)

            all_results[test_name] = metrics_results

        self.training_history['test_results'] = all_results

        results_path = os.path.join(self.save_dir, 'all_test_results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved all test results to {results_path}")
        except Exception as e:
            print(f"Warning: Failed to save results to JSON: {e}")

        return all_results

print("Staged trainer defined!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
print("\n" + "=" * 70)
print("STARTING MAIN EXECUTION")
print("=" * 70)

print_config()

print("\n--- Setting up paths ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

conformer_cache_path = os.path.join(OUTPUT_DIR, CONFORMER_CACHE)
ecfp_cache_path = os.path.join(OUTPUT_DIR, ECFP_CACHE)
token_cache_path = os.path.join(OUTPUT_DIR, TOKEN_CACHE)
normalizer_cache_path = os.path.join(OUTPUT_DIR, NORMALIZER_CACHE)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Structure paths:")
for i, path in enumerate(STRUCTURE_PATHS, 1):
    exists = os.path.exists(path)
    status = 'Found' if exists else 'Missing'
    print(f"  {i}. {status}: {path}")

# Load data splits
print("\n--- Loading data splits ---")
try:
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_general_df = pd.read_csv(TEST_GENERAL_CSV)
    test_refined_df = pd.read_csv(TEST_REFINED_CSV)
    test_casf_df = pd.read_csv(TEST_CASF_CSV)

    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test General: {len(test_general_df)} samples")
    print(f"Test Refined: {len(test_refined_df)} samples")
    print(f"Test CASF: {len(test_casf_df)} samples")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure all CSV files exist and paths are correct.")
    raise

# Initialize and fit normalizer
print("\n--- Setting up affinity normalizer ---")
affinity_normalizer = AffinityNormalizer(method=NORMALIZATION_METHOD)

if os.path.exists(normalizer_cache_path):
    try:
        affinity_normalizer.load(normalizer_cache_path)
        print(f"Loaded normalizer from cache")
    except Exception as e:
        print(f"Failed to load normalizer: {e}")
        print("Fitting new normalizer...")
        affinity_normalizer.fit(train_df['affinity'].values)
        affinity_normalizer.save(normalizer_cache_path)
else:
    affinity_normalizer.fit(train_df['affinity'].values)
    affinity_normalizer.save(normalizer_cache_path)
    print(f"Fitted and saved normalizer")

print(f"Normalizer: {affinity_normalizer}")

# Initialize tokenizer
print("\n--- Initializing protein language model tokenizer ---")
try:
    plm_tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    plm_pad_id = plm_tokenizer.pad_token_id or 1
    update_pad_id(plm_pad_id)
    print(f"Tokenizer initialized: {PLM_MODEL_NAME}")
    print(f"Pad token ID: {plm_pad_id}")
except Exception as e:
    print(f"Error initializing tokenizer: {e}")
    raise

# Pre-compute all data
print("\n" + "="*70)
print("PRE-COMPUTING ALL DATA (ONE-TIME COST)")
print("="*70)

all_data_df = pd.concat([
    train_df, val_df, test_general_df, test_refined_df, test_casf_df
], ignore_index=True).drop_duplicates(subset=['pdb_id'])

print(f"Total unique PDB entries: {len(all_data_df)}")

print("\n1 Pre-computing conformers...")
all_conformers = precompute_all_conformers(
    all_data_df, STRUCTURE_PATHS,
    N_CONFORMERS, conformer_cache_path
)

print("\n2 Pre-computing ECFP fingerprints...")
all_ecfp = precompute_ecfp_fingerprints(all_data_df, ecfp_cache_path)

print("\n3 Pre-computing protein tokens...")
token_cache = preprocess_and_cache_tokens(
    all_data_df, STRUCTURE_PATHS, plm_tokenizer, token_cache_path
)

print("\nAll pre-computation complete!")

# Create datasets
print("\n--- Creating optimized datasets ---")
train_dataset = OptimizedPDBbindDataset(
    train_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
)
val_dataset = OptimizedPDBbindDataset(
    val_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
)
test_general_dataset = OptimizedPDBbindDataset(
    test_general_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
)
test_refined_dataset = OptimizedPDBbindDataset(
    test_refined_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
)
test_casf_dataset = OptimizedPDBbindDataset(
    test_casf_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
)

# Create dataloaders
print("\n--- Creating dataloaders ---")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS
)

test_loaders_dict = {}
test_datasets_info = {
    'General Set 2020': test_general_dataset,
    'Refined Set 2020': test_refined_dataset,
    'CASF 2016': test_casf_dataset
}

for name, dataset in test_datasets_info.items():
    if len(dataset) > 0:
        test_loaders_dict[name] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        print(f"  {name}: {len(dataset)} samples")
    else:
        print(f"  {name}: No valid samples")

# Initialize model
print("\n--- Initializing AURA model ---")
dl_model = AuraDeepLearningModel(
    gnn_2d_node_dim=GNN_2D_NODE_DIM,
    gnn_2d_edge_dim=GNN_2D_EDGE_DIM,
    hidden_dim=HIDDEN_DIM,
    plm_hidden_dim=PLM_HIDDEN_DIM,
    out_dim=OUTPUT_DIM
).to(DEVICE)

ensemble_model = AuraEnsemble(dl_model, normalizer=affinity_normalizer).to(DEVICE)

total_params = sum(p.numel() for p in ensemble_model.parameters())
trainable_params = sum(p.numel() for p in ensemble_model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Initialize trainer
print("\n--- Initializing trainer ---")
trainer = StagedTrainer(
    ensemble_model,
    train_loader,
    val_loader,
    test_loaders_dict,
    normalizer=affinity_normalizer,
    device=DEVICE,
    use_amp=USE_AMP,
    save_dir=OUTPUT_DIR
)

# Run training
print("\n" + "="*70)
print("STARTING STAGED TRAINING")
print("="*70)
print("Training on normalized scale, evaluating on original scale")
print("Expected benefits:")
print("  • Better gradient flow")
print("  • Faster convergence")
print("  • Improved stability")

if MAX_EPOCHS_STAGE_B > 0:
    trainer.stage_b_head_training(MAX_EPOCHS_STAGE_B)
else:
    print("\nSkipping Stage B (MAX_EPOCHS_STAGE_B = 0)")

if MAX_EPOCHS_STAGE_C > 0:
    trainer.stage_c_fine_tuning(MAX_EPOCHS_STAGE_C)
else:
    print("\nSkipping Stage C (MAX_EPOCHS_STAGE_C = 0)")

trainer.train_xgboost_component()

# Final evaluation
if test_loaders_dict:
    all_test_results = trainer.evaluate_on_all_test_sets()
else:
    print("\nNo test sets available for evaluation")
    all_test_results = {}

# Save training history
print("\n--- Saving training history ---")
history_path = os.path.join(OUTPUT_DIR, 'training_history.pkl')
try:
    with open(history_path, 'wb') as f:
        pickle.dump(trainer.training_history, f)
    print(f"Training history saved to {history_path}")
except Exception as e:
    print(f"Warning: Failed to save training history: {e}")

# =============================================================================
# INTERPRETABILITY VISUALIZATIONS
# =============================================================================
print("\n" + "="*70)
print("GENERATING INTERPRETABILITY VISUALIZATIONS")
print("="*70)

# Need to define CompleteAuraInterpreter class for testing.py
class CompleteAuraInterpreter:
    """Complete interpretability suite for AURA model."""

    def __init__(self, ensemble_model, plm_tokenizer, normalizer=None):
        self.ensemble_model = ensemble_model.to(DEVICE)
        self.plm_tokenizer = plm_tokenizer
        self.normalizer = normalizer
        self.ensemble_model.eval()

        if ensemble_model.xgb_model is not None:
            try:
                self.shap_explainer = shap.TreeExplainer(ensemble_model.xgb_model)
            except Exception as e:
                print(f"Warning: Failed to initialize SHAP explainer: {e}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None

    def explain(self, smiles, protein_sequence, pdb_id):
        """Generate comprehensive explanations."""
        print(f"\n=== Generating Explanation for {pdb_id} ===")

        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Generate conformers
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFORMERS, randomSeed=42)
            if len(conf_ids) == 0:
                result = AllChem.EmbedMolecule(mol)
                if result == -1:
                    print(f"Failed to embed molecule for {pdb_id}")
                    return None
                conf_ids = [0]
        except Exception as e:
            print(f"Error in EmbedMultipleConfs for {pdb_id}: {e}")
            try:
                result = AllChem.EmbedMolecule(mol)
                if result == -1:
                    print(f"Failed to embed single conformer for {pdb_id}")
                    return None
                conf_ids = [0]
            except Exception as e2:
                print(f"Complete failure to embed conformers for {pdb_id}: {e2}")
                return None

        # Create graphs
        graphs_3d = []
        for conf_id in conf_ids[:N_CONFORMERS]:
            try:
                conf = mol.GetConformer(conf_id)
            except Exception as e:
                print(f"Warning: Failed to get conformer {conf_id} for {pdb_id}: {e}")
                continue
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

            atom_features = []
            atomic_numbers = []
            for atom in mol.GetAtoms():
                atomic_numbers.append(atom.GetAtomicNum())
                atom_features.append([
                    atom.GetDegree(), atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(), int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()), atom.GetTotalNumHs()
                ])

            z = torch.tensor(atomic_numbers, dtype=torch.long)
            x = torch.tensor(atom_features, dtype=torch.float)

            edge_indices, edge_features = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = [int(bond.GetBondType()), int(bond.GetIsConjugated()), int(bond.IsInRing())]
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_type, bond_type])

            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 3), dtype=torch.float)

            graphs_3d.append(Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr))

        # Check if we have at least one valid graph
        if not graphs_3d:
            print(f"No valid 3D graphs generated for {pdb_id}")
            return None

        # Generate ECFP
        from rdkit.Chem import rdFingerprintGenerator
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_N_BITS)
        fp = fp_gen.GetFingerprint(mol)
        ecfp = np.zeros((ECFP_N_BITS,), dtype=np.float32)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, ecfp)

        # Tokenize protein
        protein_tokens = self.plm_tokenizer(
            protein_sequence, return_tensors='pt',
            max_length=MAX_PROTEIN_LENGTH, truncation=True
        )

        # Create batch (need both label and original_label for custom_collate)
        batch = custom_collate([{
            'graphs_3d': graphs_3d,
            'ecfp': torch.from_numpy(ecfp),
            'protein_tokens': {k: v.squeeze(0) for k, v in protein_tokens.items()},
            'label': torch.tensor([0.0]),
            'original_label': torch.tensor([0.0]),
            'pdb_id': pdb_id,
            'smiles': smiles
        }])

        if batch is None:
            print("Failed to create batch")
            return None

        # Get predictions
        with torch.no_grad():
            dl_pred, interpretability_info = self.ensemble_model.dl_model(batch)
            dl_affinity_norm = dl_pred.squeeze().item()

            if self.normalizer and self.normalizer.fitted:
                dl_affinity = self.normalizer.inverse_transform(np.array([dl_affinity_norm]))[0]
            else:
                dl_affinity = dl_affinity_norm

            # Extract interpretability information
            attn_weights = interpretability_info.get('interaction_weights')
            pocket_scores = interpretability_info.get('pocket_scores')
            conformer_weights = interpretability_info.get('conformer_weights')

            # Extract interaction matrix
            # attn_weights shape: [batch, num_atoms, seq_len]
            if attn_weights is not None:
                interaction_matrix = attn_weights.squeeze(0).cpu().numpy()  # [num_atoms, seq_len]
            else:
                interaction_matrix = np.zeros((10, 10))

            # Extract pocket importance scores
            if pocket_scores is not None:
                pocket_importance = pocket_scores.squeeze().cpu().numpy()
            else:
                pocket_importance = None

            feature_importance = None
            if self.ensemble_model.xgb_model is not None:
                xgb_features = np.hstack([ecfp.reshape(1, -1), [[dl_affinity_norm]]])
                xgb_affinity_norm = self.ensemble_model.xgb_model.predict(xgb_features)[0]

                if self.normalizer and self.normalizer.fitted:
                    xgb_affinity = self.normalizer.inverse_transform(np.array([xgb_affinity_norm]))[0]
                else:
                    xgb_affinity = xgb_affinity_norm

                if self.shap_explainer is not None:
                    try:
                        shap_values = self.shap_explainer(xgb_features)
                        shap_array = shap_values.values[0]
                        ecfp_shap = shap_array[:-1]

                        top_positive = np.argsort(ecfp_shap)[-10:]
                        top_negative = np.argsort(ecfp_shap)[:10]

                        feature_importance = {
                            'positive_bits': [(int(i), float(ecfp_shap[i])) for i in top_positive if ecfp[i] == 1],
                            'negative_bits': [(int(i), float(ecfp_shap[i])) for i in top_negative if ecfp[i] == 1]
                        }
                    except Exception as e:
                        print(f"Warning: SHAP analysis failed: {e}")

                w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
                final_affinity = w * dl_affinity + (1 - w) * xgb_affinity
            else:
                xgb_affinity = None
                final_affinity = dl_affinity

            if attn_weights is not None:
                # attn_weights shape: [batch, num_atoms, seq_len]
                # Sum over protein dimension to get per-atom importance
                atom_scores = attn_weights.sum(dim=2)  # [batch, num_atoms]
                atom_attributions = atom_scores[0].cpu().numpy()  # Take first batch item

                # Normalize to [0, 1]
                atom_range = atom_attributions.max() - atom_attributions.min()
                if atom_range > 0:
                    atom_attributions = (atom_attributions - atom_attributions.min()) / atom_range
                else:
                    atom_attributions = np.zeros_like(atom_attributions)

                atom_attributions = atom_attributions[:graphs_3d[0].num_nodes]
            else:
                atom_attributions = np.zeros(graphs_3d[0].num_nodes)

        return {
            'predictions': {
                'deep_learning': float(dl_affinity),
                'xgboost': float(xgb_affinity) if xgb_affinity is not None else None,
                'ensemble': float(final_affinity),
                'ensemble_weight_dl': torch.sigmoid(self.ensemble_model.ensemble_weight).item()
            },
            'level2_interactions': interaction_matrix,
            'level3_features': feature_importance,
            'level4_atoms': atom_attributions,
            'pocket_importance': pocket_importance,
            'conformer_weights': [w.numpy() if w is not None else None for w in conformer_weights] if conformer_weights else None,
            'conformer_count': len(graphs_3d),
            'pdb_id': pdb_id,
            'smiles': smiles
        }

    def visualize_explanation(self, explanation_dict, save_path=None):
        """Create comprehensive visualization dashboard."""
        fig = plt.figure(figsize=(18, 12))  # Larger for 3x3 grid

        # 1. Prediction comparison
        ax1 = plt.subplot(3, 3, 1)
        preds = explanation_dict['predictions']
        names = ['DL', 'XGB', 'Ensemble']
        values = [
            preds['deep_learning'],
            preds.get('xgboost', 0) if preds.get('xgboost') is not None else 0,
            preds['ensemble']
        ]
        ax1.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Model Predictions', fontweight='bold')
        ax1.set_ylabel('Predicted Affinity')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Feature importance
        ax2 = plt.subplot(3, 3, 2)
        if explanation_dict['level3_features'] is not None:
            feat_imp = explanation_dict['level3_features']
            if feat_imp['positive_bits'] or feat_imp['negative_bits']:
                pos_indices = [x[0] for x in feat_imp['positive_bits'][:5]]
                pos_values = [x[1] for x in feat_imp['positive_bits'][:5]]
                neg_indices = [x[0] for x in feat_imp['negative_bits'][:5]]
                neg_values = [x[1] for x in feat_imp['negative_bits'][:5]]

                all_indices = pos_indices + neg_indices
                all_values = pos_values + neg_values
                colors = ['g'] * len(pos_indices) + ['r'] * len(neg_indices)

                ax2.barh(range(len(all_values)), all_values, color=colors)
                ax2.set_yticks(range(len(all_values)))
                ax2.set_yticklabels([f'Bit {i}' for i in all_indices], fontsize=8)
                ax2.set_xlabel('SHAP Value')
        ax2.set_title('ECFP Feature Importance', fontweight='bold')

        # 3. Ligand-Protein interactions
        ax3 = plt.subplot(3, 3, 3)
        interaction_data = explanation_dict['level2_interactions']
        if interaction_data.size > 0:
            im = ax3.imshow(interaction_data[:20, :50], cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax3)
            ax3.set_title('Ligand-Protein Interactions', fontweight='bold')

        # 4. Ensemble weight
        ax4 = plt.subplot(3, 3, 4)
        if explanation_dict['predictions']['xgboost'] is not None:
            w_dl = explanation_dict['predictions']['ensemble_weight_dl']
            w_xgb = 1 - w_dl
            ax4.pie([w_dl, w_xgb], labels=['DL', 'XGB'], autopct='%1.1f%%',
                   colors=['#1f77b4', '#ff7f0e'], startangle=90)
        ax4.set_title('Ensemble Weights', fontweight='bold')

        # 5. Atom attributions
        ax5 = plt.subplot(3, 3, 5)
        atom_scores = explanation_dict['level4_atoms']
        if len(atom_scores) > 0:
            ax5.plot(atom_scores, 'o-', color='#2ca02c')
            ax5.set_title('Atom-level Attributions', fontweight='bold')
            ax5.set_xlabel('Atom Index')
            ax5.set_ylabel('Importance')
            ax5.grid(True, alpha=0.3)

        # 6. Pocket importance scores (NEW!)
        ax6 = plt.subplot(3, 3, 6)
        pocket_scores_data = explanation_dict.get('pocket_importance')
        if pocket_scores_data is not None and len(pocket_scores_data) > 0:
            top_residues = min(20, len(pocket_scores_data))
            ax6.barh(range(top_residues), pocket_scores_data[:top_residues], color='#9467bd')
            ax6.set_xlabel('Importance Score')
            ax6.set_ylabel('Residue Index')
            ax6.set_title('Pocket Residue Importance', fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
            ax6.invert_yaxis()
        else:
            ax6.text(0.5, 0.5, 'No pocket data', ha='center', va='center')
            ax6.axis('off')

        # 7. Conformer weights (NEW!)
        ax7 = plt.subplot(3, 3, 7)
        conformer_weights_data = explanation_dict.get('conformer_weights')
        if conformer_weights_data is not None and len(conformer_weights_data) > 0:
            weights = conformer_weights_data[0]  # First batch sample
            if weights is not None:
                n_conf = len(weights)
                ax7.bar(range(n_conf), weights, color='#8c564b')
                ax7.set_xlabel('Conformer Index')
                ax7.set_ylabel('Attention Weight')
                ax7.set_title('Conformer Selection Weights', fontweight='bold')
                ax7.grid(axis='y', alpha=0.3)
                ax7.set_xticks(range(n_conf))
            else:
                ax7.text(0.5, 0.5, 'No conformer weights', ha='center', va='center')
                ax7.axis('off')
        else:
            ax7.text(0.5, 0.5, 'No conformer data', ha='center', va='center')
            ax7.axis('off')

        # 8. Summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        info_text = f"""
PDB: {explanation_dict['pdb_id']}
SMILES: {explanation_dict['smiles'][:40]}...
Conformers: {explanation_dict['conformer_count']}

Predicted: {explanation_dict['predictions']['ensemble']:.2f}

Components:
  2D GNN (Topology)
  3D GNN (Geometry)
  PLM (Protein)
  Physics-informed
        """
        ax8.text(0.1, 0.5, info_text, fontsize=9, family='monospace', va='center')
        ax8.set_title('Summary', fontweight='bold')

        # 9. Reserved for future visualization
        ax9 = plt.subplot(3, 3, 9)
        ax9.text(0.5, 0.5, 'Reserved\\nfor future\\nvisualization',
                ha='center', va='center', fontsize=10, alpha=0.5)
        ax9.axis('off')

        plt.suptitle('AURA Explanation Dashboard', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        return fig

# Initialize interpreter
print("\n--- Initializing interpreter ---")
interpreter = CompleteAuraInterpreter(ensemble_model, plm_tokenizer, affinity_normalizer)
print("Interpreter ready!")

# Generate example visualizations
try:
    num_examples = min(5, len(test_general_df))
    example_indices = np.random.choice(len(test_general_df), num_examples, replace=False)

    os.makedirs(os.path.join(OUTPUT_DIR, 'interpretability'), exist_ok=True)

    for idx in example_indices:
        row = test_general_df.iloc[idx]
        pdb_id = row['pdb_id']
        smiles = row['canonical_smiles']

        print(f"\nGenerating explanation for {pdb_id}...")

        if pdb_id in token_cache:
            protein_sequence = token_cache[pdb_id]['sequence']

            explanation = interpreter.explain(smiles, protein_sequence, pdb_id)

            if explanation is not None:
                save_path = os.path.join(OUTPUT_DIR, 'interpretability', f'{pdb_id}_explanation.png')
                interpreter.visualize_explanation(explanation, save_path)
                print(f"  Saved to {save_path}")

    print(f"\nGenerated {num_examples} interpretability visualizations")
    print(f"Location: {os.path.join(OUTPUT_DIR, 'interpretability')}")

except Exception as e:
    print(f"Warning: Failed to generate visualizations: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("AURA FRAMEWORK TRAINING COMPLETE")
print("="*70)
print(f"All models and results saved to: {OUTPUT_DIR}")
print("\nKey improvements from normalization:")
print("  • Improved gradient flow during backpropagation")
print("  • Faster convergence and better stability")
print("  • More consistent loss landscapes")
print("\nNext steps:")
print("  • Check test results in all_test_results.json")
print("  • Review training history in training_history.pkl")
print("="*70)

print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
