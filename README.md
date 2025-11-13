# AURA Framework
## Affinity Unified Regression Architecture for Protein-Ligand Binding Prediction

---

## Executive Summary

AURA is a novel deep learning framework designed to predict protein-ligand binding affinity using a multi-stream architecture that integrates molecular topology, 3D geometry, protein sequence information, and physics-informed features. The framework achieves robust predictions through an ensemble approach combining deep learning with gradient boosting, while providing comprehensive interpretability through attention mechanisms and SHAP analysis.

**Key Innovation**: Multi-modal fusion architecture with similarity-aware data splitting to prevent data leakage and ensure generalization to novel protein-ligand complexes.

---

## Table of Contents

- [Research Objective](#research-objective)
- [Problem Statement](#problem-statement)
- [Methodology Overview](#methodology-overview)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Training Strategy](#training-strategy)
- [Evaluation Strategy](#evaluation-strategy)
- [Model Components](#model-components)
- [Interpretability](#interpretability)
- [Results and Outputs](#results-and-outputs)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Research Objective

**Primary Goal**: Develop an accurate and interpretable deep learning model for predicting protein-ligand binding affinity that generalizes well to unseen protein-ligand pairs.

**Secondary Goals**:
- Address data leakage issues common in binding affinity prediction benchmarks
- Integrate multiple molecular representations for comprehensive feature extraction
- Provide multi-level interpretability for understanding model predictions
- Leverage conformer ensembles to capture molecular flexibility

---

## Problem Statement

Accurate prediction of protein-ligand binding affinity is crucial for drug discovery, but presents several challenges:

1. **Data Leakage**: Traditional random splits can lead to similar protein-ligand pairs appearing in both training and test sets, artificially inflating performance metrics
2. **Multi-scale Information**: Binding affinity depends on 2D topology, 3D geometry, protein sequence, and physical interactions
3. **Conformational Flexibility**: Ligands adopt multiple conformations when binding to proteins
4. **Interpretability**: Black-box models lack transparency needed for scientific validation

**Our Solution**: AURA addresses these challenges through similarity-aware data splitting, multi-stream architecture, conformer attention, and comprehensive interpretability modules.

---

## Methodology Overview

AURA follows a systematic approach to binding affinity prediction:

1. **Data Preparation**: Similarity-aware splitting to prevent leakage
2. **Feature Extraction**: Multi-modal molecular and protein representations
3. **Multi-stream Processing**: Parallel encoding of different information types
4. **Attention-based Fusion**: Learned integration of molecular features and protein context
5. **Ensemble Prediction**: Combination of deep learning and XGBoost
6. **Interpretability Analysis**: Multi-level explanations of predictions

---

## Architecture

### High-Level Overview

AURA employs a **five-stream architecture** that processes different aspects of the protein-ligand complex:

```
Stream A: 2D Molecular Topology (GNN)
Stream B: 3D Molecular Geometry (GNN)
Stream C: Protein Sequence (ESM-2 PLM)
Stream D: Physics-informed Features
Stream E: ECFP Fingerprints
```

These streams are integrated through:
- **Conformer Gating**: Attention-based selection of relevant conformers
- **Hierarchical Cross-Attention**: Modeling ligand-protein interactions
- **Fusion MLP**: Final integration of all streams
- **KAN Regression Head**: Non-linear prediction layer

### Architecture Diagram

The AURA architecture processes protein-ligand complexes through five parallel streams that are integrated via attention mechanisms:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          INPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Ligand SMILES                    Protein PDB File
               │                                │
               └────────┬───────────────────────┘
                        │
                        ▼

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    FEATURE EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    LIGAND SIDE                        PROTEIN SIDE
    ───────────                        ────────────

    5 Conformers ──┐                  ESM-2 PLM (frozen)
         │         │                         │
         ▼         ▼                         ▼
    ┌────────────────┐              ┌─────────────────┐
    │  2D-GNN  3D-GNN│              │ Residue Embeddings│
    │  (topology +   │              │  (sequence info)  │
    │   geometry)    │              └─────────────────┘
    └────────────────┘                       │
         │                                   ▼
         │                          ┌─────────────────┐
         │                          │  Physics-GNN    │
         │                          │ (binding pocket)│
         │                          └─────────────────┘
         │                                   │
         ▼                                   │
    ┌────────────────┐                      │
    │  Conformer     │◄─────────────────────┘
    │  Attention     │  (protein context)
    └────────────────┘
         │
         │
    ECFP Fingerprint
    (2048-bit Morgan)
         │

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    INTERACTION MODELING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                 ┌──────────────────────┐
                 │ Hierarchical Cross-  │
                 │    Attention         │
                 │ (Ligand ↔ Protein)   │
                 └──────────────────────┘
                          │
                          ▼

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    MULTI-STREAM FUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     Stream 1: Interaction features (from attention)
     Stream 2: Protein global embedding
     Stream 3: Ligand global pooling
     Stream 4: Physics features
     Stream 5: ECFP embedding
                          │
                          ▼
                 ┌──────────────────────┐
                 │   Fusion MLP         │
                 │   (concatenate all)  │
                 └──────────────────────┘
                          │
                          ▼
                 ┌──────────────────────┐
                 │    KAN Regression    │
                 │       Head           │
                 └──────────────────────┘
                          │
                          ▼

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ENSEMBLE PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

              DL Prediction      XGBoost
                  (w)           (1-w)
                   │               │
                   └───────┬───────┘
                           ▼
                   Final Affinity
                    (-logKd/Ki)
```

---

## Data Pipeline

### Datasets

AURA is trained and evaluated on the PDBbind dataset:

1. **PDBbind v2020 (Other Set)**: General protein-ligand complexes
2. **PDBbind v2020 (Refined Set)**: High-quality crystal structures
3. **CASF-2016 Benchmark**: Standard benchmark for affinity prediction

### Similarity-Aware Data Splitting

**Critical Innovation**: Traditional random splitting leads to data leakage because similar protein-ligand pairs appear in both train and test sets.

**Solution**: We employ **similarity-aware evaluation splits** that ensure:
- Training and test sets contain structurally dissimilar proteins
- Ligands in test set are chemically distinct from training ligands
- Model is evaluated on its ability to generalize to novel complexes

This approach provides realistic evaluation of model performance on truly unseen data, addressing a major limitation in prior binding affinity prediction work.

### Data Preprocessing

**Molecular Processing**:
1. **Conformer Generation**: Generate 5 conformers per ligand using RDKit
   - Uses crystal structure as first conformer when available
   - Generates additional conformers with MMFF optimization
   - Captures conformational flexibility

2. **2D Graph Representation**:
   - Node features: atom type, degree, charge, hybridization, aromaticity, hydrogens
   - Edge features: bond type, conjugation, ring membership

3. **3D Graph Representation**:
   - Node features: atomic numbers, 3D coordinates
   - Edges constructed via radius-based connectivity

4. **ECFP Fingerprints**: 2048-bit Morgan fingerprints (radius=2) for XGBoost

**Protein Processing**:
1. **Sequence Extraction**: Extract amino acid sequences from PDB files
2. **Tokenization**: ESM-2 tokenizer with max length 1024
3. **Embedding**: Pre-trained ESM-2 (8M parameters) frozen encoder

**Efficiency**: All preprocessing is cached to disk for rapid subsequent runs.

---

## Training Strategy

AURA employs a **staged training approach** with three phases:

1. **Stage B - Head Training**: Regression heads are trained first with frozen backbone encoders to initialize prediction layers
2. **Stage C - End-to-End Fine-tuning**: Full model is fine-tuned with learning rate warmup and L2 regularization to prevent overfitting
3. **Ensemble Training**: XGBoost model is trained on ECFP features and deep learning embeddings, then ensemble weights are optimized

**Key Technical Decisions**:
- Training on z-score normalized affinity values for improved gradient flow and stability
- All evaluation metrics reported on original scale for interpretability
- Early stopping based on validation set Pearson R correlation
- Mixed precision training for computational efficiency

---

## Evaluation Strategy

### Metrics

Performance is evaluated using standard regression metrics on original scale:

- **RMSE** (Root Mean Squared Error): Overall prediction error
- **MAE** (Mean Absolute Error): Average absolute deviation
- **Pearson R**: Linear correlation between predictions and true values
- **Spearman R**: Rank correlation (important for ranking compounds)
- **Concordance Index (CI)**: Pairwise ranking accuracy

### Test Sets

Model is comprehensively evaluated on three test sets:

1. **General Set 2020**: Diverse protein-ligand complexes
2. **Refined Set 2020**: High-quality crystal structures
3. **CASF-2016**: Standard benchmark for comparison with literature

All test sets are constructed using **similarity-aware splitting** to ensure no data leakage.

---

## Model Components

### 1. Protein Language Model Encoder (PLM_Encoder)

**Architecture**: ESM-2 (8M parameters)
- Pre-trained on 50M protein sequences
- Frozen during training (feature extractor)
- Outputs: Per-residue embeddings (320-dim) + global protein embedding

**Role**: Captures protein sequence context and evolutionary information

### 2. 2D GNN Encoder (GNN_2D_Encoder)

**Architecture**: Graph Attention Networks (GATv2)
- 3 layers, 4 attention heads per layer
- Node features: 6-dimensional (atom properties)
- Edge features: 3-dimensional (bond properties)
- Hidden dimension: 128

**Role**: Encodes molecular topology and chemical connectivity

### 3. 3D GNN Encoder (GNN_3D_Encoder)

**Architecture**: Radius-based Graph Neural Network
- Constructs edges based on spatial proximity
- Processes 3D coordinates and atomic numbers
- Hidden dimension: 128

**Role**: Captures 3D molecular geometry and spatial relationships

### 4. Physics-Informed GNN (PhysicsInformedGNN)

**Architecture**: Specialized GNN for protein dynamics
- Processes protein residue embeddings
- Computes pocket dynamics and correlation features
- Output dimension: 32

**Role**: Models physical interactions and binding pocket characteristics

### 5. ECFP Encoder (ECFP_Encoder)

**Architecture**: Multi-layer perceptron
- Input: 2048-bit Morgan fingerprint
- Output: 128-dimensional embedding

**Role**: Provides traditional cheminformatics features for ensemble

### 6. Conformer Gating Module (ConformerGate)

**Architecture**: Attention-based conformer selection
- Query: Global protein embedding
- Keys/Values: Conformer features
- Outputs: Weighted conformer representation + attention weights

**Role**: Selects relevant conformers based on protein context

### 7. Hierarchical Cross-Attention (HierarchicalCrossAttention)

**Architecture**: Two-level attention mechanism
- Level 1: Atom-residue interactions
- Level 2: Pocket-level importance
- Multi-head attention with 8 heads

**Role**: Models ligand-protein binding interactions

### 8. KAN Regression Head

**Architecture**: Kolmogorov-Arnold Network layers
- Two-layer architecture: 320 → 64 → 1
- Non-linear activation for final prediction

**Role**: Maps fused features to binding affinity

### 9. XGBoost Component

**Architecture**: Gradient boosted decision trees
- Trained on ECFP + DL embeddings
- 200 estimators, early stopping

**Role**: Captures tabular patterns complementary to deep learning

### 10. Ensemble Combiner

**Architecture**: Learnable weighted average
- Sigmoid-activated weight parameter
- Balances DL and XGBoost predictions

**Role**: Optimal combination of model predictions

---

## Interpretability

AURA provides **multi-level interpretability** to understand predictions:

### Level 1: Model Component Analysis
- Individual predictions from DL, XGBoost, and ensemble
- Ensemble weight distribution

### Level 2: Interaction Analysis
- **Ligand-Protein Attention**: Atom-residue interaction heatmap
- **Pocket Importance**: Critical residues for binding
- Shows which protein regions interact with which ligand atoms

### Level 3: Feature Importance (SHAP)
- SHAP values for ECFP fingerprint bits
- Identifies chemical substructures driving predictions
- Separates positive and negative contributions

### Level 4: Atom-Level Attribution
- Per-atom importance scores
- Identifies critical atoms in ligand for binding
- Derived from attention weights

### Level 5: Conformer Selection
- Attention weights across conformers
- Shows which conformations are most relevant for binding

### Visualization Dashboard

The `CompleteAuraInterpreter` generates comprehensive visualization dashboards with:
- Model prediction comparison
- Feature importance plots
- Interaction heatmaps
- Atom attribution graphs
- Pocket residue importance
- Conformer selection weights
- Summary statistics

---

## Results and Outputs

### Model Performance

AURA achieves **competitive results** on all test sets, with performance metrics reported on the original affinity scale:

#### General Set 2020 (113 samples)
| Model | RMSE | MAE | Pearson R | Spearman R | CI |
|-------|------|-----|-----------|------------|-----|
| Deep Learning | 1.4141 | 1.0880 | 0.6423 | 0.6314 | 0.7254 |
| XGBoost | 1.3733 | 1.0604 | 0.6443 | 0.6349 | 0.7270 |
| **Ensemble** | **1.3876** | **1.0683** | **0.6440** | **0.6337** | **0.7263** |

#### Refined Set 2020 (581 samples)
| Model | RMSE | MAE | Pearson R | Spearman R | CI |
|-------|------|-----|-----------|------------|-----|
| Deep Learning | 1.5008 | 1.1660 | 0.6794 | 0.6757 | 0.7447 |
| XGBoost | 1.4338 | 1.1196 | 0.6821 | 0.6784 | 0.7461 |
| **Ensemble** | **1.4569** | **1.1340** | **0.6821** | **0.6777** | **0.7456** |

#### CASF-2016 Benchmark (285 samples)
| Model | RMSE | MAE | Pearson R | Spearman R | CI |
|-------|------|-----|-----------|------------|-----|
| Deep Learning | 1.4452 | 1.1299 | 0.7363 | 0.7231 | 0.7673 |
| XGBoost | 1.4379 | 1.1459 | 0.7392 | 0.7243 | 0.7669 |
| **Ensemble** | **1.4312** | **1.1335** | **0.7386** | **0.7257** | **0.7676** |

**Important Note**: These results are particularly competitive considering that **accuracy for nearly all models in the literature drops significantly when using similarity-aware evaluation splits** compared to traditional random splitting. Our similarity-aware approach prevents data leakage and provides a more realistic assessment of model generalization to novel protein-ligand complexes.


## Project Structure

```
AURA/
├── main.py                      # Main execution script
├── config.py                    # Global configuration and hyperparameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── aura_standalone.py           # Standalone version of the framework
├── src.py                       # Source utilities
├── testing.py                   # Testing and evaluation scripts
│
├── data/                        # Data processing modules
│   ├── __init__.py
│   ├── preprocessing.py         # Conformer generation, ECFP, tokenization
│   ├── normalization.py         # Affinity normalization/denormalization
│   └── dataset.py               # PyTorch dataset and collate function
│
├── models/                      # Model architecture modules
│   ├── __init__.py
│   ├── aura_model.py           # Main AURA model architecture
│   ├── ensemble.py             # Ensemble model with XGBoost
│   ├── encoders.py             # PLM, GNN encoders
│   ├── attention.py            # Attention mechanisms
│   └── kan.py                  # KAN layer implementation
│
├── training/                    # Training utilities
│   ├── __init__.py
│   ├── trainer.py              # Staged training implementation
│   ├── metrics.py              # Evaluation metrics
│   └── utils.py                # Training helper functions
│
└── interpretation/              # Interpretability modules
    ├── __init__.py
    └── explainer.py            # SHAP analysis and visualization
```

---

## Key Innovations

### 1. Similarity-Aware Data Splitting
Traditional random splitting leads to data leakage. Our similarity-aware splits ensure the model is evaluated on genuinely novel protein-ligand pairs, providing realistic assessment of generalization capability.

### 2. Multi-Stream Architecture
Unlike single-modality approaches, AURA integrates five complementary information streams, capturing both molecular and protein perspectives at multiple scales.

### 3. Conformer Attention Mechanism
Rather than averaging conformers, AURA learns to attend to relevant conformations based on protein context, better modeling conformational flexibility.

### 4. Hierarchical Interaction Modeling
Two-level cross-attention captures both fine-grained atom-residue interactions and coarse-grained pocket-level importance.

### 5. Physics-Informed Features
Dedicated GNN processes protein dynamics and correlation patterns, incorporating domain knowledge about binding pocket behavior.

### 6. Hybrid Ensemble Architecture
Combines deep learning (handles complex interactions) with XGBoost (captures tabular patterns), with learnable weight optimization.

### 7. Normalization-Aware Training
Training on normalized scale improves optimization, while evaluation on original scale ensures interpretable metrics.

### 8. Comprehensive Interpretability
Multi-level explanations from ensemble weights down to individual atoms provide scientific insight into predictions.

---

## Acknowledgments


Special acknowledgment for the **similarity-aware-evaluations** by Zhang et. al which address a critical data leakage issue in binding affinity prediction benchmarks.
