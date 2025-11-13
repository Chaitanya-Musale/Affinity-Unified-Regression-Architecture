"""
AURA Framework - Attention Mechanisms Module
Contains cross-attention and hierarchical attention modules for ligand-protein interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Note: Simple CrossAttention class removed - using HierarchicalCrossAttention instead
# If you need the simpler version, it's available in git history


class HierarchicalCrossAttention(nn.Module):
    """
    Hierarchical Cross-Attention module with two levels:
    1. Pocket-level attention: Identifies important binding pocket regions
    2. Interaction-level attention: Captures detailed ligand-protein interactions
    """

    def __init__(self, embed_dim, num_heads=4):
        """
        Initialize hierarchical cross-attention.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads for interaction attention
        """
        super().__init__()
        self.pocket_attention = nn.MultiheadAttention(
            embed_dim, num_heads//2, batch_first=True
        )
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(self, ligand_emb, protein_residues, protein_mask):
        """
        Forward pass through hierarchical attention.

        Args:
            ligand_emb (torch.Tensor): Ligand embeddings [batch, num_atoms, embed_dim]
            protein_residues (torch.Tensor): Protein residue embeddings [batch, seq_len, embed_dim]
            protein_mask (torch.Tensor): Protein attention mask [batch, seq_len]

        Returns:
            tuple: (interaction_output, interaction_weights, pocket_scores)
        """
        # Level 1: Pocket-level attention
        # Use global ligand representation to identify important pocket regions
        ligand_global = ligand_emb.mean(dim=1, keepdim=True)  # [batch, 1, embed_dim]
        pocket_scores, _ = self.pocket_attention(
            ligand_global,
            protein_residues,
            protein_residues,
            key_padding_mask=~protein_mask.bool()
        )

        # Level 2: Interaction-level attention
        # Detailed atom-residue interactions
        interaction_output, interaction_weights = self.interaction_attention(
            ligand_emb,
            protein_residues,
            protein_residues,
            key_padding_mask=~protein_mask.bool()
        )

        return interaction_output, interaction_weights, pocket_scores


class ConformerGate(nn.Module):
    """
    Learned attention mechanism for weighting conformers based on protein alignment.
    This module selects the most relevant conformer(s) for a given protein pocket.
    """

    def __init__(self, hidden_dim, protein_dim):
        """
        Initialize conformer gating module.

        Args:
            hidden_dim (int): Dimension of conformer features
            protein_dim (int): Dimension of protein features
        """
        super().__init__()
        self.conformer_scorer = nn.Sequential(
            nn.Linear(hidden_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, conformer_features, protein_global, n_conformers):
        """
        Compute conformer weights and aggregate conformers.

        Args:
            conformer_features (torch.Tensor): Features from all conformers
                                               [n_conformers, n_atoms, hidden_dim]
            protein_global (torch.Tensor): Global protein embedding [hidden_dim]
            n_conformers (int): Number of conformers

        Returns:
            tuple: (weighted_features, conformer_weights)
                - weighted_features: Aggregated features [n_atoms, hidden_dim]
                - conformer_weights: Attention weights for each conformer [n_conformers, 1]
        """
        # Get conformer-level representations by averaging over atoms
        conformer_global = conformer_features.mean(dim=1)  # [n_conformers, hidden_dim]

        # Expand protein representation to match conformers
        protein_expanded = protein_global.unsqueeze(0).expand(n_conformers, -1)

        # Compute conformer scores by comparing with protein
        scores_input = torch.cat([conformer_global, protein_expanded], dim=-1)
        scores = self.conformer_scorer(scores_input)  # [n_conformers, 1]
        weights = F.softmax(scores, dim=0)  # [n_conformers, 1]

        # Weight and aggregate conformers
        weighted_features = (conformer_features * weights.unsqueeze(1)).sum(dim=0)

        return weighted_features, weights
