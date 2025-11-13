"""
AURA Framework - Main Model Module
Complete AURA deep learning model architecture integrating all components.
"""

import torch
import torch.nn as nn
from .encoders import PLM_Encoder, GNN_2D_Encoder, GNN_3D_Encoder, PhysicsInformedGNN, ECFP_Encoder
from .attention import HierarchicalCrossAttention, ConformerGate
from .kan import KANLinear
import config


class AuraDeepLearningModel(nn.Module):
    """
    AURA (Affinity Unified Regression Architecture) Deep Learning Model.

    Integrates multiple streams:
    - Stream A: 2D molecular topology (GNN)
    - Stream B: 3D molecular geometry (GNN)
    - Stream C: Protein sequence (PLM)
    - Stream D: Physics-informed features
    - Conformer attention for multi-conformer aggregation
    - Hierarchical cross-attention for ligand-protein interactions
    """

    def __init__(self, gnn_2d_node_dim, gnn_2d_edge_dim, hidden_dim, plm_hidden_dim, out_dim, ecfp_dim=2048):
        """
        Initialize AURA model.

        Args:
            gnn_2d_node_dim (int): Node feature dimension for 2D GNN
            gnn_2d_edge_dim (int): Edge feature dimension for 2D GNN
            hidden_dim (int): Hidden dimension for GNNs
            plm_hidden_dim (int): Hidden dimension for protein language model
            out_dim (int): Output dimension (typically 1 for regression)
            ecfp_dim (int): ECFP fingerprint dimension (default: 2048)
        """
        super().__init__()

        # Encoders
        self.plm_encoder = PLM_Encoder()
        self.gnn_2d_encoder = GNN_2D_Encoder(gnn_2d_node_dim, gnn_2d_edge_dim, hidden_dim)
        self.gnn_3d_encoder = GNN_3D_Encoder(hidden_channels=hidden_dim)
        self.physics_gnn = PhysicsInformedGNN(hidden_dim)
        self.ecfp_encoder = ECFP_Encoder(ecfp_dim, hidden_dim)

        # Projection layers to align dimensions
        self.proj_2d = nn.Linear(hidden_dim, plm_hidden_dim)
        self.proj_3d = nn.Linear(hidden_dim, plm_hidden_dim)
        self.proj_physics = nn.Linear(hidden_dim // 4, plm_hidden_dim)
        self.proj_ecfp = nn.Linear(hidden_dim, plm_hidden_dim)

        # Conformer gating module
        self.conformer_gate = ConformerGate(plm_hidden_dim, plm_hidden_dim)

        # Hierarchical attention for ligand-protein interactions
        self.hierarchical_attention = HierarchicalCrossAttention(plm_hidden_dim)

        # Fusion MLP (now with 5 streams: interaction, protein, ligand, physics, ecfp)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(plm_hidden_dim * 5, plm_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(plm_hidden_dim * 2, plm_hidden_dim)
        )

        # Regression head using KAN layers
        self.kan_head = nn.Sequential(
            KANLinear(plm_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            KANLinear(64, out_dim)
        )

    def forward(self, batch):
        """
        Forward pass through AURA model.

        Args:
            batch (dict): Batch dictionary containing:
                - graphs_3d: Batched PyG Data object
                - protein_tokens: Dict with input_ids and attention_mask
                - conformer_batch_idx: Mapping from graph to batch sample

        Returns:
            tuple: (predictions, interaction_weights)
                - predictions: Predicted affinity values [batch_size, 1]
                - interaction_weights: Attention weights for interpretability
        """
        graphs_3d = batch['graphs_3d'].to(config.DEVICE)
        protein_tokens = {k: v.to(config.DEVICE) for k, v in batch['protein_tokens'].items()}
        conformer_batch_idx = batch['conformer_batch_idx'].to(config.DEVICE)
        ecfp = batch['ecfp'].to(config.DEVICE)

        # Encode protein sequence
        protein_residue_embs, protein_global_emb = self.plm_encoder(protein_tokens)

        # Encode ECFP fingerprint
        ecfp_emb = self.ecfp_encoder(ecfp.float())

        # Encode ligand in 2D and 3D
        ligand_2d_nodes = self.gnn_2d_encoder(graphs_3d)
        ligand_3d_nodes = self.gnn_3d_encoder(graphs_3d)

        # Encode physics-informed features
        pocket_dyn, corr_features = self.physics_gnn(protein_residue_embs)

        # Project and fuse 2D and 3D ligand features
        ligand_2d_proj = self.proj_2d(ligand_2d_nodes)
        ligand_3d_proj = self.proj_3d(ligand_3d_nodes)
        ligand_fused_nodes = ligand_2d_proj + ligand_3d_proj

        # Process each batch sample
        batch_size = protein_tokens['input_ids'].size(0)
        ligand_nodes_list = []
        conformer_weights_list = []  # Store conformer weights for interpretability

        for i in range(batch_size):
            # Find which conformer indices belong to this batch sample
            conformer_indices = (conformer_batch_idx == i).nonzero(as_tuple=True)[0]

            if len(conformer_indices) > 0:
                # Collect all nodes belonging to these conformers
                node_indices = []
                for conf_idx in conformer_indices:
                    # graphs_3d.batch tells us which nodes belong to which graph
                    node_mask = (graphs_3d.batch == conf_idx)
                    node_indices.extend(node_mask.nonzero(as_tuple=True)[0].tolist())

                if node_indices:
                    node_indices = torch.tensor(node_indices, device=config.DEVICE)
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
                    conformer_weights_list.append(conformer_weights.squeeze(-1).detach().cpu())  # Save weights

        if not ligand_nodes_list:
            # Return zeros if no valid ligand nodes
            empty_info = {
                'interaction_weights': None,
                'pocket_scores': None,
                'conformer_weights': None
            }
            return torch.zeros(batch_size, 1, device=config.DEVICE), empty_info

        # Pad ligand nodes to same length for batching
        ligand_nodes_padded = nn.utils.rnn.pad_sequence(ligand_nodes_list, batch_first=True)

        # Apply hierarchical cross-attention
        interaction_output, interaction_weights, pocket_scores = self.hierarchical_attention(
            ligand_nodes_padded, protein_residue_embs, protein_tokens['attention_mask']
        )

        # Global pooling
        context_ligand_global = interaction_output.mean(dim=1)
        ligand_global_pool = torch.stack([nodes.mean(dim=0) for nodes in ligand_nodes_list])
        physics_proj = self.proj_physics(corr_features)
        ecfp_proj = self.proj_ecfp(ecfp_emb)

        # Final fusion (5 streams)
        final_fused = self.fusion_mlp(torch.cat([
            context_ligand_global,
            protein_global_emb,
            ligand_global_pool,
            physics_proj,
            ecfp_proj
        ], dim=1))

        # Prediction
        prediction = self.kan_head(final_fused)

        # Return additional interpretability outputs
        interpretability_info = {
            'interaction_weights': interaction_weights,
            'pocket_scores': pocket_scores,
            'conformer_weights': conformer_weights_list if conformer_weights_list else None
        }

        return prediction, interpretability_info
