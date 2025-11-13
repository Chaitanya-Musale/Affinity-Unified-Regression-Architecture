# =============================================================================
# AURA FRAMEWORK - ENHANCED VERSION WITH AFFINITY NORMALIZATION
# =============================================================================

# =============================================================================
# SECTION 0: SETUP & INSTALLATIONS
# =============================================================================
import subprocess
import sys
import os

print("############################################################")
print("### ⚙️  SECTION 0: INSTALLING DEPENDENCIES ###")
print("############################################################")

def install(package):
    """Installs a package if it's not already installed."""
    try:
        __import__(package.split('[')[0].split('>')[0].split('==')[0].replace('-', '_'))
        print(f"✅ {package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✅ {package} installed.")

# Core ML and data handling
install("pandas")
install("numpy")
install("scikit-learn")
install("matplotlib")
install("seaborn")
install("tqdm")
install("scipy")
install("rdkit")
# PyTorch and related libraries
# Cell 1: Uninstall
!pip uninstall torch torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv -y

# Cell 2: Install a specific, stable PyTorch version (e.g., 2.3.1 with CUDA 12.1)
!pip install torch==2.3.1 torchvision==0.18.1 -f https://download.pytorch.org/whl/cu121
# Cell 3: Install the matching PyG dependencies
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
# Advanced models and tools
install("xgboost")
install("transformers")
install("sentencepiece")
install("accelerate")
install("shap")
install("nglview")
install("e3nn")
install("biopython")
install("torch_geometric")
print("\n✅ All dependencies are ready.")

# =============================================================================
# SECTION 1: IMPORTS & GLOBAL CONFIGURATION
# =============================================================================
print("\n############################################################")
print("### 🌎 SECTION 1: IMPORTS & GLOBAL CONFIGURATION ###")
print("############################################################")

# --- Standard Libraries ---
import gc
import json
import warnings
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
import copy  # Added for L2 regularization

# --- Data Science & ML ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_add_pool, radius_graph
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap

# --- Cheminformatics & Biology ---
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from transformers import AutoTokenizer
from transformers.models.esm.modeling_esm import EsmModel
from Bio import PDB
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
import nglview as nv
from IPython.display import display
from tqdm import tqdm

# --- Scheduler imports for warmup ---
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

# --- Global Configuration ---
BATCH_SIZE = 8  # Reduced for memory efficiency
ACCUMULATION_STEPS = 8  # Effective batch size = 64
MAX_EPOCHS_STAGE_B = 50  # Reduced for faster experimentation
MAX_EPOCHS_STAGE_C = 50  # Reduced for faster experimentation
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-4
N_CONFORMERS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
PLM_PAD_ID = 1  # Will be updated after tokenizer initialization
GRADIENT_CLIP_NORM = 1.0  # Added for gradient clipping
WARMUP_STEPS = 500  # Added for learning rate warmup
L2_REG_WEIGHT = 1e-5  # Added for L2 regularization

# NORMALIZATION CONFIGURATION
NORMALIZATION_METHOD = 'zscore'  # Options: 'zscore', 'minmax'
AFFINITY_MEAN = None  # Will be computed from training data
AFFINITY_STD = None   # Will be computed from training data
AFFINITY_MIN = None   # Will be computed from training data
AFFINITY_MAX = None   # Will be computed from training data

print(f"Using device: {DEVICE}")
print(f"Mixed Precision Training: {USE_AMP}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"Gradient Clipping Norm: {GRADIENT_CLIP_NORM}")
print(f"Normalization Method: {NORMALIZATION_METHOD}")

# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

class AffinityNormalizer:
    """Handles normalization and denormalization of affinity values."""

    def __init__(self, method='zscore'):
        """
        Initialize normalizer.

        Args:
            method: 'zscore' for z-score normalization, 'minmax' for min-max scaling
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False

    def fit(self, values):
        """Fit the normalizer on training data."""
        values = np.array(values).flatten()

        if self.method == 'zscore':
            self.mean = values.mean()
            self.std = values.std()
            if self.std == 0:
                self.std = 1.0  # Prevent division by zero
        elif self.method == 'minmax':
            self.min = values.min()
            self.max = values.max()
            if self.max == self.min:
                self.max = self.min + 1.0  # Prevent division by zero
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self.fitted = True
        return self

    def transform(self, values):
        """Normalize values."""
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
        """Denormalize values."""
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
        """Save normalizer parameters."""
        params = {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'fitted': self.fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load(self, path):
        """Load normalizer parameters."""
        with open(path, 'rb') as f:
            params = pickle.load(f)

        self.method = params['method']
        self.mean = params['mean']
        self.std = params['std']
        self.min = params['min']
        self.max = params['max']
        self.fitted = params['fitted']
        return self

    def get_params(self):
        """Get normalization parameters as dict."""
        return {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max
        }

# Global normalizer instance
affinity_normalizer = AffinityNormalizer(method=NORMALIZATION_METHOD)

# =============================================================================
# SECTION 2: HELPER FUNCTIONS & UTILITIES (WITH PRE-COMPUTATION)
# =============================================================================
print("\n############################################################")
print("### 🧬 SECTION 2: HELPER FUNCTIONS (WITH PRE-COMPUTATION) ###")
print("############################################################")

def extract_protein_sequence(pdb_file_path):
    """Extract protein sequence from PDB file - handles multiple chains correctly."""
    parser = PDB.PDBParser(QUIET=True)
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
                    if residue.id[0] == ' ':  # Standard amino acid
                        res_name = residue.resname
                        if res_name in three_to_one:
                            sequence.append(three_to_one[res_name])

        return ''.join(sequence)
    except Exception as e:
        print(f"Error extracting sequence from {pdb_file_path}: {e}")
        return None

def generate_conformer_ensemble_with_crystal(smiles_string, crystal_mol=None, n_conformers=5):
    """FIXED: Generates conformer ensemble, handling atom mismatch properly."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # If we have a crystal structure, try to use it as the first conformer
    use_crystal = False
    if crystal_mol is not None:
        try:
            # Check if the molecules match structurally
            crystal_mol_h = Chem.AddHs(crystal_mol, addCoords=True)

            # Check if number of atoms match
            if mol.GetNumAtoms() == crystal_mol_h.GetNumAtoms():
                # Try to match atom ordering
                match = mol.GetSubstructMatch(crystal_mol_h)
                if match and len(match) == mol.GetNumAtoms():
                    # Create conformer with crystal coordinates
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for i, j in enumerate(match):
                        pos = crystal_mol_h.GetConformer().GetAtomPosition(j)
                        conf.SetAtomPosition(i, pos)
                    mol.AddConformer(conf)
                    use_crystal = True
        except Exception as e:
            # Crystal structure couldn't be used, will generate all conformers
            pass

    # Generate remaining conformers
    if use_crystal:
        # We already have the crystal as conformer 0, generate n_conformers-1 more
        if n_conformers > 1:
            try:
                additional_conf_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers-1,
                    pruneRmsThresh=0.5,
                    randomSeed=42,
                    clearConfs=False  # Don't clear existing conformers
                )
            except:
                additional_conf_ids = []
        conf_ids = list(range(mol.GetNumConformers()))
    else:
        # Generate all conformers from scratch
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_conformers,
            pruneRmsThresh=0.5,
            randomSeed=42,
            useRandomCoords=True
        )

        if len(conf_ids) == 0:
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if res == -1:
                return None
            conf_ids = [0]

    # Optimize non-crystal conformers only
    for i, conf_id in enumerate(conf_ids):
        # Don't optimize the crystal pose (first conformer if use_crystal)
        if use_crystal and i == 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        except:
            pass

    # Validate conformer IDs
    valid_conf_ids = []
    for conf_id in conf_ids:
        if conf_id < mol.GetNumConformers():
            valid_conf_ids.append(conf_id)

    if not valid_conf_ids:
        return None

    return mol, valid_conf_ids

def generate_conformer_ensemble(smiles_string, n_conformers=5):
    """Simple conformer generation without crystal structure."""
    return generate_conformer_ensemble_with_crystal(smiles_string, None, n_conformers)

def mol_to_3d_graph(mol, conf_id=0):
    """FIXED: Converts a molecule with 3D coordinates to a PyG Data object with validation."""
    if mol is None:
        return None

    # Validate conformer ID
    if conf_id >= mol.GetNumConformers():
        return None

    try:
        conf = mol.GetConformer(conf_id)
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

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        return None

def smiles_to_ecfp(smiles_string, n_bits=2048):
    """Converts a SMILES string to an ECFP fingerprint."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = fp_gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr

# =============================================================================
# NEW: PRE-COMPUTATION FUNCTIONS WITH NORMALIZATION
# =============================================================================

def precompute_all_conformers(all_data_df, structure_paths, n_conformers=5, cache_path='conformers_precomputed.pkl'):
    """Pre-compute all conformers before training starts - MULTI-PATH VERSION"""
    if os.path.exists(cache_path):
        print(f"Loading pre-computed conformers from {cache_path}")
        with open(cache_path, 'rb') as f:
            conformer_dict = pickle.load(f)
            print(f"Loaded {len(conformer_dict)} pre-computed conformer sets")
            return conformer_dict

    print("Pre-computing all conformers (this only happens once)...")
    conformer_dict = {}
    failed_entries = []

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Pre-computing conformers"):
        pdb_id = row['pdb_id']
        smiles = row['canonical_smiles']

        # Try to find the structure in any of the provided paths
        crystal_mol = None
        ligand_found = False

        for structure_path in structure_paths:
            ligand_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_ligand.sdf")

            # Some datasets use different file naming conventions
            if not os.path.exists(ligand_path):
                # Try alternative naming for CASF
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
                except:
                    pass

        # Generate conformers
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

    # Save the pre-computed conformers
    with open(cache_path, 'wb') as f:
        pickle.dump(conformer_dict, f)

    print(f"✅ Pre-computed {len(conformer_dict)} conformer sets")
    print(f"⚠️  Failed to compute conformers for {len(failed_entries)} entries")
    print(f"💾 Saved to {cache_path}")

    return conformer_dict

def precompute_ecfp_fingerprints(all_data_df, cache_path='ecfp_precomputed.pkl'):
    """Pre-compute all ECFP fingerprints"""
    if os.path.exists(cache_path):
        print(f"Loading pre-computed ECFP fingerprints from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Pre-computing ECFP fingerprints...")
    ecfp_dict = {}

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Computing ECFP"):
        smiles = row['canonical_smiles']
        if smiles not in ecfp_dict:
            ecfp_dict[smiles] = smiles_to_ecfp(smiles)

    with open(cache_path, 'wb') as f:
        pickle.dump(ecfp_dict, f)

    print(f"✅ Pre-computed {len(ecfp_dict)} ECFP fingerprints")
    return ecfp_dict

def preprocess_and_cache_tokens(all_data_df, structure_paths, plm_tokenizer, cache_path):
    """Pre-tokenize all protein sequences for efficiency - MULTI-PATH VERSION."""
    if os.path.exists(cache_path):
        print(f"Loading cached protein tokens from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print("Pre-tokenizing protein sequences (this will only happen once)...")
    token_cache = {}

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Tokenizing proteins"):
        pdb_id = row['pdb_id']

        # Try to find the protein in any of the provided paths
        sequence = None
        for structure_path in structure_paths:
            protein_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_protein.pdb")

            # Some datasets use different naming conventions
            if not os.path.exists(protein_path):
                # Try alternative naming for CASF (pocket instead of protein)
                protein_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_pocket.pdb")

            if os.path.exists(protein_path):
                sequence = extract_protein_sequence(protein_path)
                if sequence:
                    break

        if sequence:
            tokens = plm_tokenizer(
                sequence, return_tensors='pt', padding='longest',
                truncation=True, max_length=1024
            )
            token_cache[pdb_id] = {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'sequence': sequence
            }

    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(token_cache, f)
    print(f"Saved token cache with {len(token_cache)} proteins to {cache_path}")

    return token_cache

# =============================================================================
# OPTIMIZED DATASET WITH NORMALIZATION
# =============================================================================

class OptimizedPDBbindDataset(Dataset):
    """Ultra-fast dataset using only pre-computed data with normalization support"""
    def __init__(self, dataframe, conformer_dict, token_cache, ecfp_dict, normalizer=None):
        self.conformer_dict = conformer_dict
        self.token_cache = token_cache
        self.ecfp_dict = ecfp_dict
        self.normalizer = normalizer

        # Filter to valid entries only
        valid_entries = []
        for _, row in dataframe.iterrows():
            pdb_id = row['pdb_id']
            smiles = row['canonical_smiles']
            key = f"{pdb_id}_{smiles}"

            if (key in self.conformer_dict and
                pdb_id in self.token_cache and
                smiles in self.ecfp_dict):
                entry = row.to_dict()
                # Normalize affinity if normalizer is provided
                if self.normalizer and self.normalizer.fitted:
                    entry['normalized_affinity'] = self.normalizer.transform(
                        np.array([entry['affinity']])
                    )[0]
                else:
                    entry['normalized_affinity'] = entry['affinity']
                valid_entries.append(entry)

        self.data = valid_entries
        print(f"Dataset has {len(self.data)} valid entries (from {len(dataframe)} total)")

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
        for graph in item['graphs_3d']:
            all_graphs.append(graph)
            conformer_batch_idx.append(i)

    if not all_graphs:
        return None

    # Handle variable-length protein sequences by padding
    max_seq_len = max(item['protein_tokens']['input_ids'].size(0) for item in batch)

    padded_input_ids = []
    padded_attention_masks = []

    for item in batch:
        input_ids = item['protein_tokens']['input_ids']
        attention_mask = item['protein_tokens']['attention_mask']

        # Calculate padding needed
        pad_len = max_seq_len - input_ids.size(0)

        if pad_len > 0:
            # Use PLM_PAD_ID from global
            input_ids_padded = F.pad(input_ids, (0, pad_len), value=PLM_PAD_ID)
            # Pad attention_mask with 0s
            attention_mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
        else:
            input_ids_padded = input_ids
            attention_mask_padded = attention_mask

        padded_input_ids.append(input_ids_padded)
        padded_attention_masks.append(attention_mask_padded)

    collated_batch = {
        'graphs_3d': Batch.from_data_list(all_graphs),
        'conformer_batch_idx': torch.tensor(conformer_batch_idx, dtype=torch.long),
        'ecfp': torch.stack([item['ecfp'] for item in batch]),
        'protein_tokens': {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks)
        }
    }

    # Only add optional fields if they exist
    if 'label' in batch[0]:
        collated_batch['label'] = torch.stack([item['label'] for item in batch])
        collated_batch['original_label'] = torch.stack([item['original_label'] for item in batch])
    if 'pdb_id' in batch[0]:
        collated_batch['pdb_id'] = [item['pdb_id'] for item in batch]
    if 'smiles' in batch[0]:
        collated_batch['smiles'] = [item['smiles'] for item in batch]

    return collated_batch

# =============================================================================
# SECTION 3: COMPLETE AURA MODEL ARCHITECTURE (NO CHANGES NEEDED)
# =============================================================================
print("\n############################################################")
print("### 🤖 SECTION 3: ENHANCED AURA MODEL ARCHITECTURE ###")
print("############################################################")

# [Keep all model architecture code unchanged - PLM_Encoder, CrossAttention, GNN_2D_Encoder, etc.]
# PLM Encoder
class PLM_Encoder(nn.Module):
    """Protein Language Model Encoder using ESM-2."""
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.model = EsmModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, tokens):
        out = self.model(**tokens)  # out.last_hidden_state: [B, L, H]
        reps = out.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)  # [B,L,1]
        summed = (reps * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = summed / denom
        return reps, pooled  # (residue_embs, global_emb)

# Cross Attention
class CrossAttention(nn.Module):
    """Cross-Attention module for ligand-protein interaction."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, ligand_nodes, protein_nodes, protein_mask):
        attn_output, attn_weights = self.multihead_attn(
            query=ligand_nodes,
            key=protein_nodes,
            value=protein_nodes,
            key_padding_mask=~protein_mask.bool()
        )
        return attn_output, attn_weights

# Stream A: 2D Topology GNN
class GNN_2D_Encoder(nn.Module):
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
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        return x

# Stream B: 3D-Aware GNN (WITH NUMERICAL STABILITY FIX)
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
        """Expand distances using Gaussian basis functions - WITH NUMERICAL STABILITY."""
        mu = torch.linspace(start, stop, num_gaussians, device=dist.device)
        mu = mu.view(1, -1)
        sigma = (stop - start) / num_gaussians
        # ENHANCED: Added epsilon for numerical stability
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

# Stream C: Physics-Informed GNN
class PhysicsInformedGNN(nn.Module):
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

# Hierarchical Attention
class HierarchicalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.pocket_attention = nn.MultiheadAttention(embed_dim, num_heads//2, batch_first=True)
        self.interaction_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, ligand_emb, protein_residues, protein_mask):
        ligand_global = ligand_emb.mean(dim=1, keepdim=True)
        pocket_scores, _ = self.pocket_attention(
            ligand_global, protein_residues, protein_residues,
            key_padding_mask=~protein_mask.bool()
        )
        interaction_output, interaction_weights = self.interaction_attention(
            ligand_emb, protein_residues, protein_residues,
            key_padding_mask=~protein_mask.bool()
        )
        return interaction_output, interaction_weights, pocket_scores

# KAN Layer
class KANLinear(nn.Module):
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
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        with torch.no_grad():
            self.spline_weight.uniform_(-0.1, 0.1)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - self.grid[:-(k + 1)]) / (self.grid[k:-1] - self.grid[:-(k + 1)]) * bases[:, :, :-1]) + \
                    ((self.grid[k + 1:] - x) / (self.grid[k + 1:] - self.grid[1:(-k)]) * bases[:, :, 1:])
        return bases

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1)
        )
        return base_output + spline_output

# NEW: Conformer Attention Module
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
        """
        Args:
            conformer_features: [n_conformers, n_atoms, hidden_dim]
            protein_global: [hidden_dim]
            n_conformers: int
        Returns:
            weighted_features: [n_atoms, hidden_dim]
        """
        # Get conformer-level representations
        conformer_global = conformer_features.mean(dim=1)  # [n_conformers, hidden_dim]

        # Expand protein representation
        protein_expanded = protein_global.unsqueeze(0).expand(n_conformers, -1)

        # Compute conformer scores
        scores_input = torch.cat([conformer_global, protein_expanded], dim=-1)
        scores = self.conformer_scorer(scores_input)  # [n_conformers, 1]
        weights = F.softmax(scores, dim=0)  # [n_conformers, 1]

        # Weight and aggregate conformers
        weighted_features = (conformer_features * weights.unsqueeze(1)).sum(dim=0)

        return weighted_features, weights

# Main AURA Model (ENHANCED VERSION)
class AuraDeepLearningModel(nn.Module):
    def __init__(self, gnn_2d_node_dim, gnn_2d_edge_dim, hidden_dim, plm_hidden_dim, out_dim):
        super().__init__()

        self.plm_encoder = PLM_Encoder()
        self.gnn_2d_encoder = GNN_2D_Encoder(gnn_2d_node_dim, gnn_2d_edge_dim, hidden_dim)
        self.gnn_3d_encoder = GNN_3D_Encoder(hidden_channels=hidden_dim)
        self.physics_gnn = PhysicsInformedGNN(hidden_dim)

        self.proj_2d = nn.Linear(hidden_dim, plm_hidden_dim)
        self.proj_3d = nn.Linear(hidden_dim, plm_hidden_dim)
        self.proj_physics = nn.Linear(hidden_dim // 4, plm_hidden_dim)

        # Conformer gating module
        self.conformer_gate = ConformerGate(plm_hidden_dim, plm_hidden_dim)

        self.hierarchical_attention = HierarchicalCrossAttention(plm_hidden_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(plm_hidden_dim * 4, plm_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(plm_hidden_dim * 2, plm_hidden_dim)
        )

        # Regression head
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

        # FIX 3: Properly map conformer indices to graph batch indices
        for i in range(batch_size):
            # Find which conformer indices belong to this batch sample
            conformer_indices = (conformer_batch_idx == i).nonzero(as_tuple=True)[0]

            if len(conformer_indices) > 0:
                # Collect all nodes belonging to these conformers
                node_indices = []
                for conf_idx in conformer_indices:
                    # graphs_3d.batch tells us which nodes belong to which graph
                    # conf_idx is the actual index in the batched graph
                    node_mask = (graphs_3d.batch == conf_idx)
                    node_indices.extend(node_mask.nonzero(as_tuple=True)[0].tolist())

                if node_indices:
                    node_indices = torch.tensor(node_indices, device=DEVICE)
                    molecule_nodes = ligand_fused_nodes[node_indices]

                    n_conformers = len(conformer_indices)
                    n_atoms = len(node_indices) // n_conformers

                    # Use learned attention over conformers
                    conformer_features = molecule_nodes.reshape(n_conformers, n_atoms, -1)
                    weighted_nodes, conformer_weights = self.conformer_gate(
                        conformer_features,
                        protein_global_emb[i],
                        n_conformers
                    )
                    ligand_nodes_list.append(weighted_nodes)

        if not ligand_nodes_list:
            return torch.zeros(batch_size, 1, device=DEVICE), None

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

        return prediction, interaction_weights

# ENSEMBLE MODEL WITH NORMALIZATION SUPPORT
class AuraEnsemble(nn.Module):
    def __init__(self, dl_model, normalizer=None):
        super().__init__()
        self.dl_model = dl_model
        self.xgb_model = None
        self.ensemble_weight = nn.Parameter(torch.tensor([0.5]))
        self.normalizer = normalizer

        # NEW: Store initial weights for L2 regularization
        self.initial_weights = None

    def set_xgb_model(self, xgb_model):
        self.xgb_model = xgb_model

    def store_initial_weights(self):
        """Store initial weights for L2 regularization during fine-tuning."""
        self.initial_weights = {}
        for name, param in self.dl_model.named_parameters():
            self.initial_weights[name] = param.data.clone()

    def get_l2_regularization_loss(self):
        """Calculate L2 regularization loss towards initial weights."""
        if self.initial_weights is None:
            return 0.0

        l2_loss = 0.0
        for name, param in self.dl_model.named_parameters():
            if name in self.initial_weights:
                l2_loss += torch.sum((param - self.initial_weights[name]) ** 2)

        return L2_REG_WEIGHT * l2_loss

    def forward(self, batch, xgb_features=None):
        dl_pred, attn = self.dl_model(batch)

        if self.xgb_model is not None and xgb_features is not None:
            # XGBoost expects normalized values if trained on normalized data
            xgb_pred = torch.tensor(
                self.xgb_model.predict(xgb_features),
                device=dl_pred.device, dtype=dl_pred.dtype
            ).unsqueeze(1)

            w = torch.sigmoid(self.ensemble_weight)
            ensemble_pred = w * dl_pred + (1 - w) * xgb_pred
            return ensemble_pred, attn

        return dl_pred, attn

# =============================================================================
# SECTION 4: TRAINING UTILITIES WITH NORMALIZATION
# =============================================================================
print("\n############################################################")
print("### 🚂 SECTION 4: ENHANCED TRAINING UTILITIES WITH NORMALIZATION ###")
print("############################################################")

def concordance_index(y_true, y_pred):
    """Calculate the concordance index (works with original or normalized values)."""
    n = len(y_true)
    if n < 2:
        return 0.0

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

    if concordant + discordant + tied == 0:
        return 0.5

    return (concordant + 0.5 * tied) / (concordant + discordant + tied)

def regression_metrics(y_true, y_pred, denormalize=False, normalizer=None):
    """
    Calculate comprehensive regression evaluation metrics.

    Args:
        y_true: True values (can be normalized or original)
        y_pred: Predicted values (should match y_true scale)
        denormalize: If True and normalizer provided, denormalize before computing metrics
        normalizer: AffinityNormalizer instance for denormalization
    """
    if len(y_true) == 0:
        return {
            'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0,
            'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0
        }

    # Ensure numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Denormalize if requested
    if denormalize and normalizer and normalizer.fitted:
        y_true = normalizer.inverse_transform(y_true)
        y_pred = normalizer.inverse_transform(y_pred)

    # MSE
    mse = mean_squared_error(y_true, y_pred)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # Concordance Index (order-based, so normalization doesn't matter)
    ci = concordance_index(y_true, y_pred)

    # Pearson correlation (scale-invariant)
    if len(y_true) > 1:
        pearson_r, _ = pearsonr(y_true, y_pred)
        spearman_r, _ = spearmanr(y_true, y_pred)
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'CI': ci,
        'Pearson_R': pearson_r,
        'Spearman_R': spearman_r
    }

def train_one_epoch_with_accumulation(model, dataloader, optimizer, criterion,
                                     accumulation_steps=4, use_amp=False, scaler=None):
    """Training with gradient accumulation and gradient clipping - works with normalized values."""
    model.train()
    total_loss = 0
    n_batches = 0

    if use_amp and scaler is None:
        from torch.cuda.amp import autocast, GradScaler
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
                    labels = batch['label'].to(DEVICE).view(-1)  # These are normalized
                    loss = criterion(predictions, labels)

                    # Add L2 regularization if available
                    if hasattr(model, 'get_l2_regularization_loss'):
                        loss = loss + model.get_l2_regularization_loss()
            else:
                predictions, _ = model(batch)
                predictions = predictions.view(-1)
                labels = batch['label'].to(DEVICE).view(-1)  # These are normalized
                loss = criterion(predictions, labels)

                # Add L2 regularization if available
                if hasattr(model, 'get_l2_regularization_loss'):
                    loss = loss + model.get_l2_regularization_loss()

            loss = loss / accumulation_steps

            if use_amp and torch.cuda.is_available():
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
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

    # Apply remaining gradients if batches not divisible by accumulation_steps
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

    return total_loss / max(n_batches, 1), scaler

def evaluate_model(model, dataloader, normalizer=None):
    """Evaluate regression model - returns both normalized and denormalized predictions."""
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

                label_cpu = batch['label'].cpu()  # Normalized labels
                if label_cpu.dim() == 0:
                    label_cpu = label_cpu.unsqueeze(0)
                all_labels.append(label_cpu)

                # Also collect original labels if available
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
        # If original labels not available, denormalize the normalized ones
        if normalizer and normalizer.fitted:
            all_original_labels = normalizer.inverse_transform(all_labels)
        else:
            all_original_labels = all_labels

    return all_preds, all_labels, all_original_labels

def extract_features_for_xgboost(model, dataloader):
    """Extract features for XGBoost regression - works with normalized values."""
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
                # Intentionally including DL predictions (normalized) for ensemble
                batch_features = np.hstack([
                    ecfp.astype(np.float32),
                    dl_preds.reshape(-1, 1).astype(np.float32)
                ])
                features.append(batch_features)

                label_cpu = batch['label'].cpu().numpy()  # Normalized labels
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

    return np.vstack(features), np.concatenate(labels), np.concatenate(original_labels) if original_labels else None

# =============================================================================
# SECTION 5: ENHANCED STAGED TRAINER WITH NORMALIZATION
# =============================================================================
print("\n############################################################")
print("### 🎯 SECTION 5: ENHANCED STAGED TRAINING WITH NORMALIZATION ###")
print("############################################################")

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

        if use_amp:
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
        """Freeze all layers except heads."""
        for name, param in self.ensemble_model.dl_model.named_parameters():
            if 'kan_head' not in name and 'ensemble_weight' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.ensemble_model.parameters():
            param.requires_grad = True

    def stage_b_head_training(self, epochs=15):
        """Stage B: Train only heads with frozen backbone - uses normalized values."""
        print("\n=== STAGE B: Head Training (Backbone Frozen) ===")
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
                accumulation_steps=ACCUMULATION_STEPS, use_amp=self.use_amp, scaler=self.scaler
            )

            val_preds, val_labels, val_original_labels = evaluate_model(
                self.ensemble_model, self.val_loader, self.normalizer
            )

            if len(val_labels) > 0:
                # Compute metrics on original scale for interpretability
                if self.normalizer and self.normalizer.fitted:
                    val_preds_denorm = self.normalizer.inverse_transform(val_preds)
                else:
                    val_preds_denorm = val_preds
                val_metrics = regression_metrics(val_original_labels, val_preds_denorm)
            else:
                val_metrics = {'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0, 'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0}

            self.training_history['stage_b']['loss'].append(train_loss)
            self.training_history['stage_b']['val_metrics'].append(val_metrics)

            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f} (normalized scale)")
            print(f"    Val RMSE={val_metrics['RMSE']:.4f}, Pearson R={val_metrics['Pearson_R']:.4f} (original scale)")

            # Early stopping check
            if val_metrics['Pearson_R'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson_R']
                best_path = os.path.join(self.save_dir, 'stage_b_best.pt')
                torch.save(self.ensemble_model.state_dict(), best_path)
                print(f"    ✨ New best model saved! (Pearson R: {val_metrics['Pearson_R']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  ⏹️  Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_path and os.path.exists(best_path):
            self.ensemble_model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best Stage B model from {best_path}")

        # Store initial weights for L2 regularization
        self.ensemble_model.store_initial_weights()

    def stage_c_fine_tuning(self, epochs=25):
        """Stage C: Fine-tune with normalized values."""
        print("\n=== STAGE C: Fine-tuning (All Layers) ===")
        print("Fine-tuning with normalized values and proper warmup scheduling...")

        self.unfreeze_all()

        optimizer = torch.optim.AdamW(self.ensemble_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        batches_per_epoch = len(self.train_loader)
        warmup_epochs = max(1, WARMUP_STEPS // batches_per_epoch)

        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

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
                    labels = batch['label'].to(self.device).view(-1)  # Normalized
                    loss = criterion(predictions, labels)

                    # Add L2 regularization
                    if hasattr(self.ensemble_model, 'get_l2_regularization_loss'):
                        loss = loss + self.ensemble_model.get_l2_regularization_loss()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ensemble_model.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()

                    # Step warmup scheduler per batch during warmup phase
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

            # Step main scheduler per epoch after warmup
            if epoch >= warmup_epochs:
                main_scheduler.step()

            train_loss = total_loss / max(n_batches, 1)
            val_preds, val_labels, val_original_labels = evaluate_model(
                self.ensemble_model, self.val_loader, self.normalizer
            )

            if len(val_labels) > 0:
                # Compute metrics on original scale
                if self.normalizer and self.normalizer.fitted:
                    val_preds_denorm = self.normalizer.inverse_transform(val_preds)
                else:
                    val_preds_denorm = val_preds
                val_metrics = regression_metrics(val_original_labels, val_preds_denorm)
            else:
                val_metrics = {'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0, 'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0}

            self.training_history['stage_c']['loss'].append(train_loss)
            self.training_history['stage_c']['val_metrics'].append(val_metrics)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f} (normalized), LR={current_lr:.6f}")
            print(f"    Val RMSE={val_metrics['RMSE']:.4f}, Pearson R={val_metrics['Pearson_R']:.4f} (original scale)")

            # Early stopping check
            if val_metrics['Pearson_R'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson_R']
                best_path = os.path.join(self.save_dir, 'aura_final_best.pt')
                torch.save(self.ensemble_model.state_dict(), best_path)
                print(f"    ✨ New best model saved! (Pearson R: {val_metrics['Pearson_R']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  ⏹️  Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_path and os.path.exists(best_path):
            self.ensemble_model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best Stage C model from {best_path}")

    def train_xgboost_component(self):
        """Train XGBoost for regression on normalized values."""
        print("\n=== Training XGBoost Component (on normalized values) ===")

        X_train, y_train, _ = extract_features_for_xgboost(
            self.ensemble_model.dl_model, self.train_loader
        )
        X_val, y_val, _ = extract_features_for_xgboost(
            self.ensemble_model.dl_model, self.val_loader
        )

        if len(X_train) > 0 and len(X_val) > 0:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='rmse',
                early_stopping_rounds=20
            )

            # Train on normalized values
            xgb_model.fit(
                X_train, y_train,  # Both normalized
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            self.ensemble_model.set_xgb_model(xgb_model)

            # Save XGBoost model
            xgb_path = os.path.join(self.save_dir, 'xgboost_model.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            print(f"Saved XGBoost model to {xgb_path}")

            self.fine_tune_ensemble_weight()

    def fine_tune_ensemble_weight(self, epochs=3):
        """Fine-tune only the ensemble weight for regression."""
        print("\n=== Fine-tuning Ensemble Weight (with normalized values) ===")

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
                        dl_pred_np = dl_pred.cpu().numpy()  # Normalized predictions
                        ecfp = batch['ecfp'].cpu().numpy()
                        xgb_features = np.hstack([ecfp, dl_pred_np.reshape(-1, 1)])

                    optimizer.zero_grad()
                    ensemble_pred, _ = self.ensemble_model(batch, xgb_features)
                    ensemble_pred = ensemble_pred.view(-1)
                    labels = batch['label'].to(self.device).view(-1)  # Normalized
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

        # Save final ensemble model
        final_path = os.path.join(self.save_dir, 'aura_ensemble_final.pt')
        torch.save(self.ensemble_model.state_dict(), final_path)
        print(f"Saved final ensemble model to {final_path}")

    def evaluate_on_all_test_sets(self):
        """Comprehensive evaluation on all test sets - reports on original scale."""
        print("\n=== Final Model Evaluation on All Test Sets (Original Scale) ===")

        all_results = {}

        for test_name, test_loader in self.test_loaders_dict.items():
            print(f"\n📊 Evaluating on {test_name}:")
            print("=" * 50)

            test_preds_dl, test_labels, test_original_labels = evaluate_model(
                self.ensemble_model.dl_model, test_loader, self.normalizer
            )

            if len(test_labels) == 0:
                print(f"No test samples available for {test_name}")
                continue

            # Denormalize predictions for metric computation
            if self.normalizer and self.normalizer.fitted:
                test_preds_dl_denorm = self.normalizer.inverse_transform(test_preds_dl)
            else:
                test_preds_dl_denorm = test_preds_dl

            metrics_results = {}
            metrics_results['Deep Learning'] = regression_metrics(test_original_labels, test_preds_dl_denorm)

            if self.ensemble_model.xgb_model is not None:
                X_test, _, _ = extract_features_for_xgboost(
                    self.ensemble_model.dl_model, test_loader
                )

                if len(X_test) > 0:
                    test_preds_xgb = self.ensemble_model.xgb_model.predict(X_test)

                    # Denormalize XGBoost predictions
                    if self.normalizer and self.normalizer.fitted:
                        test_preds_xgb_denorm = self.normalizer.inverse_transform(test_preds_xgb)
                    else:
                        test_preds_xgb_denorm = test_preds_xgb

                    w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
                    test_preds_ensemble = w * test_preds_dl_denorm + (1 - w) * test_preds_xgb_denorm

                    metrics_results['XGBoost'] = regression_metrics(test_original_labels, test_preds_xgb_denorm)
                    metrics_results['Ensemble'] = regression_metrics(test_original_labels, test_preds_ensemble)

            # Print results for this test set
            for model_name, metrics in metrics_results.items():
                print(f"\n{model_name}:")
                print(f"  RMSE:       {metrics['RMSE']:.4f}")
                print(f"  MAE:        {metrics['MAE']:.4f}")
                print(f"  MSE:        {metrics['MSE']:.4f}")
                print(f"  CI:         {metrics['CI']:.4f}")
                print(f"  Pearson R:  {metrics['Pearson_R']:.4f}")
                print(f"  Spearman R: {metrics['Spearman_R']:.4f}")

            all_results[test_name] = metrics_results

        # Save all results
        self.training_history['test_results'] = all_results

        # Save to JSON
        results_path = os.path.join(self.save_dir, 'all_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Saved all test results to {results_path}")

        return all_results

# =============================================================================
# SECTION 6: COMPLETE INTERPRETER WITH NORMALIZATION
# =============================================================================
print("\n############################################################")
print("### 📊 SECTION 6: COMPLETE AURA INTERPRETER WITH NORMALIZATION ###")
print("############################################################")

class CompleteAuraInterpreter:
    """Complete interpretability suite for AURA - handles normalized values."""

    def __init__(self, ensemble_model, plm_tokenizer, normalizer=None):
        self.ensemble_model = ensemble_model.to(DEVICE)
        self.plm_tokenizer = plm_tokenizer
        self.normalizer = normalizer
        self.ensemble_model.eval()

        if ensemble_model.xgb_model is not None:
            self.shap_explainer = shap.TreeExplainer(ensemble_model.xgb_model)
        else:
            self.shap_explainer = None

    def explain(self, smiles, protein_sequence, pdb_id):
        """Generate comprehensive multi-level explanations - returns original scale values."""
        print(f"\n=== AURA Explanation for {pdb_id} ===")

        mol_conf_data = generate_conformer_ensemble(smiles, N_CONFORMERS)
        if mol_conf_data is None:
            print("Failed to generate conformers")
            return None

        mol, conf_ids = mol_conf_data
        graphs_3d = [mol_to_3d_graph(mol, cid) for cid in conf_ids[:N_CONFORMERS]]
        graphs_3d = [g for g in graphs_3d if g is not None]

        if not graphs_3d:
            return None

        ecfp = smiles_to_ecfp(smiles)
        protein_tokens = self.plm_tokenizer(
            protein_sequence, return_tensors='pt',
            max_length=1024, truncation=True
        )

        batch = custom_collate([{
            'graphs_3d': graphs_3d,
            'ecfp': torch.from_numpy(ecfp),
            'protein_tokens': {k: v.squeeze(0) for k, v in protein_tokens.items()},
            'label': torch.tensor([0.0]),
            'pdb_id': pdb_id,
            'smiles': smiles
        }])

        if batch is None:
            return None

        with torch.no_grad():
            dl_pred, attn_weights = self.ensemble_model.dl_model(batch)
            dl_affinity_norm = dl_pred.squeeze().item()

            # Denormalize for display
            if self.normalizer and self.normalizer.fitted:
                dl_affinity = self.normalizer.inverse_transform(np.array([dl_affinity_norm]))[0]
            else:
                dl_affinity = dl_affinity_norm

            if attn_weights is not None:
                interaction_matrix = attn_weights.squeeze(0).mean(dim=0).cpu().numpy()
            else:
                interaction_matrix = np.zeros((10, 10))

            feature_importance = None
            if self.ensemble_model.xgb_model is not None:
                xgb_features = np.hstack([ecfp.reshape(1, -1), [[dl_affinity_norm]]])
                xgb_affinity_norm = self.ensemble_model.xgb_model.predict(xgb_features)[0]

                # Denormalize
                if self.normalizer and self.normalizer.fitted:
                    xgb_affinity = self.normalizer.inverse_transform(np.array([xgb_affinity_norm]))[0]
                else:
                    xgb_affinity = xgb_affinity_norm

                if self.shap_explainer is not None:
                    shap_values = self.shap_explainer(xgb_features)
                    feature_importance = self._extract_top_features(shap_values, ecfp)

                w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
                final_affinity = w * dl_affinity + (1 - w) * xgb_affinity
            else:
                xgb_affinity = None
                final_affinity = dl_affinity

            if attn_weights is not None:
                atom_attributions = self._compute_atom_attributions(
                    graphs_3d[0], attn_weights
                )
            else:
                atom_attributions = np.zeros(graphs_3d[0].num_nodes)

        return {
            'predictions': {
                'deep_learning': dl_affinity,
                'xgboost': xgb_affinity,
                'ensemble': final_affinity,
                'ensemble_weight_dl': torch.sigmoid(self.ensemble_model.ensemble_weight).item()
            },
            'level2_interactions': interaction_matrix,
            'level3_features': feature_importance,
            'level4_atoms': atom_attributions,
            'conformer_count': len(graphs_3d),
            'pdb_id': pdb_id,
            'smiles': smiles
        }

    def _extract_top_features(self, shap_values, ecfp, top_k=10):
        """Extract top chemical features from SHAP."""
        shap_array = shap_values.values[0]
        ecfp_shap = shap_array[:-1]

        top_positive = np.argsort(ecfp_shap)[-top_k:]
        top_negative = np.argsort(ecfp_shap)[:top_k]

        return {
            'positive_bits': [(i, ecfp_shap[i]) for i in top_positive if ecfp[i] == 1],
            'negative_bits': [(i, ecfp_shap[i]) for i in top_negative if ecfp[i] == 1]
        }

    def _compute_atom_attributions(self, graph, attn_weights):
        """Compute per-atom importance scores."""
        w = attn_weights.mean(dim=1)
        q_scores = w.sum(dim=2)
        atom = q_scores[0].cpu().numpy()
        atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-8)
        return atom[:graph.num_nodes]

    def visualize_explanation(self, explanation_dict):
        """Create visualization dashboard - shows original scale values."""
        fig = plt.figure(figsize=(16, 10))

        # Prediction comparison
        ax1 = plt.subplot(2, 3, 1)
        preds = explanation_dict['predictions']
        names = ['DL', 'XGB', 'Ensemble']
        values = [preds['deep_learning'],
                 preds.get('xgboost', 0) if preds.get('xgboost') is not None else 0,
                 preds['ensemble']]
        bars = ax1.bar(names, values)
        bars[2].set_color('gold')
        ax1.set_title('Model Predictions (Original Scale)')
        ax1.set_ylabel('Predicted Affinity (-logKd/Ki)')

        # Rest of visualization code remains the same...
        ax3 = plt.subplot(2, 3, 3)
        interaction_data = explanation_dict['level2_interactions']
        if interaction_data.size > 0:
            im = ax3.imshow(interaction_data[:min(20, interaction_data.shape[0]),
                                           :min(50, interaction_data.shape[1])],
                           cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax3)
        ax3.set_title('Ligand-Protein Interactions')
        ax3.set_xlabel('Protein Residues')
        ax3.set_ylabel('Ligand Atoms')

        ax5 = plt.subplot(2, 3, 5)
        atom_scores = explanation_dict['level4_atoms']
        if len(atom_scores) > 0:
            ax5.plot(atom_scores, 'o-')
        ax5.set_title('Atom-level Attributions')
        ax5.set_xlabel('Atom Index')
        ax5.set_ylabel('Importance Score')
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        info_text = f"""
PDB ID: {explanation_dict['pdb_id']}
SMILES: {explanation_dict['smiles'][:40]}...
Conformers: {explanation_dict['conformer_count']}
DL Weight: {explanation_dict['predictions']['ensemble_weight_dl']:.3f}

Predicted Affinity: {explanation_dict['predictions']['ensemble']:.2f}
        """
        ax6.text(0.1, 0.5, info_text, fontsize=10, family='monospace')
        ax6.set_title('Summary')

        plt.suptitle('AURA Regression Explanation Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

# =============================================================================
# SECTION 7: MAIN EXECUTION WITH NORMALIZATION
# =============================================================================

if __name__ == "__main__":
    print("\n############################################################")
    print("### 🚀 SECTION 7: MAIN EXECUTION WITH NORMALIZATION ###")
    print("############################################################")

    # =========================================================================
    # Paths Configuration
    # =========================================================================
    BASE_DIR = '/content/drive/MyDrive/research/AURA'
    SPLITS_DIR = os.path.join(BASE_DIR, 'Data_splits')

    TRAIN_CSV = os.path.join(SPLITS_DIR, 'train_split.csv')
    VAL_CSV = os.path.join(SPLITS_DIR, 'validation_split.csv')
    TEST_GENERAL_CSV = os.path.join(SPLITS_DIR, 'test_split.csv')
    TEST_REFINED_CSV = os.path.join(SPLITS_DIR, 'test_refined_2020.csv')
    TEST_CASF_CSV = os.path.join(SPLITS_DIR, 'test_casf_2016.csv')

    # Multiple structure paths for different datasets
    STRUCTURE_PATHS = [
        os.path.join(BASE_DIR, 'Data/PDBbind_v2020_other_PL/v2020-other-PL'),
        os.path.join(BASE_DIR, 'Data/PDBbind_v2020_refined/refined-set'),
        os.path.join(BASE_DIR, 'Data/CASF-2016/CASF-2016/coreset')
    ]

    OUTPUT_DIR = os.path.join(BASE_DIR, 'Normalized_Model/v1')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cache paths
    CONFORMER_CACHE_PATH = os.path.join(OUTPUT_DIR, 'all_conformers_multipath_v2.pkl')
    ECFP_CACHE_PATH = os.path.join(OUTPUT_DIR, 'all_ecfp_precomputed.pkl')
    TOKEN_CACHE_PATH = os.path.join(OUTPUT_DIR, 'protein_tokens_multipath_v2.pkl')
    NORMALIZER_CACHE_PATH = os.path.join(OUTPUT_DIR, 'affinity_normalizer.pkl')

    SAVE_DIR = OUTPUT_DIR

    print(f"\nConfiguration:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Normalization method: {NORMALIZATION_METHOD}")
    print(f"  Structure paths:")
    for i, path in enumerate(STRUCTURE_PATHS, 1):
        exists = os.path.exists(path)
        print(f"    {i}. {path}: {'✓ Found' if exists else '✗ Missing'}")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n--- Loading pre-split data ---")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_general_df = pd.read_csv(TEST_GENERAL_CSV)
    test_refined_df = pd.read_csv(TEST_REFINED_CSV)
    test_casf_df = pd.read_csv(TEST_CASF_CSV)

    print(f"\nData Split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test General: {len(test_general_df)} samples")
    print(f"  Test Refined: {len(test_refined_df)} samples")
    print(f"  Test CASF: {len(test_casf_df)} samples")

    # =========================================================================
    # FIT NORMALIZER ON TRAINING DATA
    # =========================================================================
    print("\n--- Fitting affinity normalizer on training data ---")
    if os.path.exists(NORMALIZER_CACHE_PATH):
        print(f"Loading normalizer from {NORMALIZER_CACHE_PATH}")
        affinity_normalizer.load(NORMALIZER_CACHE_PATH)
    else:
        train_affinities = train_df['affinity'].values
        affinity_normalizer.fit(train_affinities)
        affinity_normalizer.save(NORMALIZER_CACHE_PATH)
        print(f"Saved normalizer to {NORMALIZER_CACHE_PATH}")

    print(f"Normalization parameters:")
    if affinity_normalizer.method == 'zscore':
        print(f"  Method: Z-score normalization")
        print(f"  Mean: {affinity_normalizer.mean:.4f}")
        print(f"  Std: {affinity_normalizer.std:.4f}")
    else:
        print(f"  Method: Min-Max normalization")
        print(f"  Min: {affinity_normalizer.min:.4f}")
        print(f"  Max: {affinity_normalizer.max:.4f}")

    # =========================================================================
    # Initialize Tokenizer
    # =========================================================================
    print("\n--- Initializing tokenizer ---")
    plm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    PLM_PAD_ID = plm_tokenizer.pad_token_id or 1
    print(f"Using padding token ID: {PLM_PAD_ID}")

    # =========================================================================
    # PRE-COMPUTE EVERYTHING (ONE TIME COST)
    # =========================================================================
    print("\n############################################################")
    print("### 🏗️  PRE-COMPUTING ALL DATA (ONE-TIME COST) ###")
    print("############################################################")

    # Combine all unique data
    all_data_df = pd.concat([
        train_df, val_df, test_general_df, test_refined_df, test_casf_df
    ], ignore_index=True).drop_duplicates(subset=['pdb_id'])

    print(f"Total unique PDB entries: {len(all_data_df)}")

    # Pre-compute all data
    print("\n1️⃣ Pre-computing conformers...")
    all_conformers = precompute_all_conformers(
        all_data_df, STRUCTURE_PATHS, N_CONFORMERS, CONFORMER_CACHE_PATH
    )

    print("\n2️⃣ Pre-computing ECFP fingerprints...")
    all_ecfp = precompute_ecfp_fingerprints(all_data_df, ECFP_CACHE_PATH)

    print("\n3️⃣ Pre-computing protein tokens...")
    token_cache = preprocess_and_cache_tokens(
        all_data_df, STRUCTURE_PATHS, plm_tokenizer, TOKEN_CACHE_PATH
    )

    print("\n✅ All pre-computation complete!")

    # =========================================================================
    # Create Datasets with Normalization
    # =========================================================================
    print("\n--- Creating optimized datasets with normalization ---")
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

    # Check if we have valid test datasets
    test_datasets_info = {
        'General Set 2020': test_general_dataset,
        'Refined Set 2020': test_refined_dataset,
        'CASF 2016': test_casf_dataset
    }

    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    print("\n--- Creating optimized dataloaders ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loaders_dict = {}
    for name, dataset in test_datasets_info.items():
        if len(dataset) > 0:
            test_loaders_dict[name] = DataLoader(
                dataset, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=custom_collate, num_workers=2, pin_memory=True
            )
            print(f"  Created loader for {name}: {len(dataset)} samples")
        else:
            print(f"  ⚠️  Skipping {name}: No valid samples")

    # =========================================================================
    # Initialize Model & Trainer with Normalizer
    # =========================================================================
    print("\n--- Initializing AURA Model with Normalization Support ---")

    dl_model = AuraDeepLearningModel(
        gnn_2d_node_dim=6,
        gnn_2d_edge_dim=3,
        hidden_dim=128,
        plm_hidden_dim=320,
        out_dim=1
    ).to(DEVICE)

    ensemble_model = AuraEnsemble(dl_model, normalizer=affinity_normalizer).to(DEVICE)

    trainer = StagedTrainer(
        ensemble_model,
        train_loader,
        val_loader,
        test_loaders_dict,
        normalizer=affinity_normalizer,
        device=DEVICE,
        use_amp=USE_AMP,
        save_dir=SAVE_DIR
    )

    # =========================================================================
    # Run Training with Normalized Values
    # =========================================================================
    print("\n🏋️ Starting staged training with NORMALIZED affinities...")
    print(f"Training on normalized scale, evaluating on original scale")
    print(f"Expected benefits: Better gradient flow, faster convergence, improved stability")

    # Stage B: Head training
    if MAX_EPOCHS_STAGE_B > 0:
        trainer.stage_b_head_training(MAX_EPOCHS_STAGE_B)

    # Stage C: Fine-tuning
    if MAX_EPOCHS_STAGE_C > 0:
        trainer.stage_c_fine_tuning(MAX_EPOCHS_STAGE_C)

    # Train XGBoost and fine-tune ensemble
    trainer.train_xgboost_component()

    # Final evaluation
    if test_loaders_dict:
        all_test_results = trainer.evaluate_on_all_test_sets()
    else:
        print("\n⚠️  No test sets available for evaluation")
        all_test_results = {}

    # Save training history
    history_path = os.path.join(SAVE_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(trainer.training_history, f)

    print("\n✅ AURA Training with Normalization Complete!")
    print(f"All models and results saved to: {SAVE_DIR}")

    # =========================================================================
    # Initialize Interpreter with Normalizer
    # =========================================================================
    print("\n--- Initializing AURA Interpreter with normalization support ---")
    interpreter = CompleteAuraInterpreter(ensemble_model, plm_tokenizer, affinity_normalizer)
    print("✅ Interpreter ready for generating explanations (will show original scale values)")

    print("\n" + "="*60)
    print("AURA FRAMEWORK WITH NORMALIZATION COMPLETE")
    print("="*60)
    print("\n📊 Key improvements from normalization:")
    print("  • Improved gradient flow during backpropagation")
    print("  • Faster convergence and better stability")
    print("  • More consistent loss landscapes")
    print("  • Better handling of varying affinity ranges")
    print("  • All metrics reported on original scale for interpretability")