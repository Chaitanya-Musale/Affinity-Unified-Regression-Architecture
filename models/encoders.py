"""
AURA Framework - Encoder Module
Contains PLM encoder and GNN encoders (2D and 3D) for molecular and protein representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, radius_graph
from transformers.models.esm.modeling_esm import EsmModel


class PLM_Encoder(nn.Module):
    """
    Protein Language Model Encoder using ESM-2.
    Encodes protein sequences into embeddings.
    """

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        """
        Initialize PLM encoder.

        Args:
            model_name (str): HuggingFace model identifier for ESM
        """
        super().__init__()
        self.model = EsmModel.from_pretrained(model_name)

        # Freeze PLM parameters to save memory and computation
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, tokens):
        """
        Forward pass through PLM.

        Args:
            tokens (dict): Dictionary with 'input_ids' and 'attention_mask'

        Returns:
            tuple: (residue_embeddings, global_embedding)
                - residue_embeddings: [batch, seq_len, hidden_dim]
                - global_embedding: [batch, hidden_dim]
        """
        out = self.model(**tokens)
        reps = out.last_hidden_state  # [B, L, H]
        mask = tokens["attention_mask"].unsqueeze(-1)  # [B, L, 1]

        # Mean pooling with attention mask
        summed = (reps * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = summed / denom

        return reps, pooled  # (residue_embs, global_emb)


class GNN_2D_Encoder(nn.Module):
    """
    2D Topology GNN Encoder using Graph Attention Networks.
    Processes molecular graph topology (bonds and atoms).
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=3):
        """
        Initialize 2D GNN encoder.

        Args:
            node_in_dim (int): Input dimension for node features
            edge_in_dim (int): Input dimension for edge features
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of GAT layers
        """
        super().__init__()
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        """
        Forward pass through 2D GNN.

        Args:
            data: PyG Data object with x, edge_index, edge_attr

        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_emb(x)

        # Handle empty edge case
        if edge_attr.size(0) > 0:
            edge_attr = self.edge_emb(edge_attr)
        else:
            # If no edges, create dummy edge attributes
            edge_attr = torch.zeros((0, x.size(1)), device=x.device, dtype=x.dtype)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))

        return x


class GNN_3D_Encoder(nn.Module):
    """
    3D-aware GNN using distance-based message passing.
    Processes molecular 3D geometry using spatial distance information.
    """

    def __init__(self, hidden_channels=128, num_layers=3):
        """
        Initialize 3D GNN encoder.

        Args:
            hidden_channels (int): Hidden dimension
            num_layers (int): Number of GAT layers
        """
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_channels)  # Support atomic numbers 0-99
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
        """
        Expand distances using Gaussian basis functions with numerical stability.

        Args:
            dist (torch.Tensor): Pairwise distances
            start (float): Start of Gaussian centers
            stop (float): End of Gaussian centers
            num_gaussians (int): Number of Gaussian basis functions

        Returns:
            torch.Tensor: Gaussian-expanded distances [num_edges, num_gaussians]
        """
        mu = torch.linspace(start, stop, num_gaussians, device=dist.device)
        mu = mu.view(1, -1)
        sigma = (stop - start) / num_gaussians
        # Add epsilon for numerical stability
        return torch.exp(-((dist.view(-1, 1) - mu) ** 2) / (2 * sigma ** 2 + 1e-8))

    def forward(self, data):
        """
        Forward pass through 3D GNN.

        Args:
            data: PyG Data object with z (atomic numbers), pos (3D coordinates), batch

        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_channels]
        """
        z, pos, batch = data.z, data.pos, data.batch

        # Clamp atomic numbers to valid range
        h = self.embedding(z.clamp(min=0, max=99))

        # Build radius graph (connect atoms within 10 Angstroms)
        edge_index = radius_graph(pos, r=10.0, batch=batch)

        # Handle case with no edges
        if edge_index.size(1) == 0:
            return h

        # Compute distances and expand using Gaussian basis
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1)
        edge_attr = self.distance_expansion(self.gaussian_expansion(dist))

        # Apply GAT layers with residual connections
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h) + h_in  # Residual connection

        return self.final_norm(h)


class PhysicsInformedGNN(nn.Module):
    """
    Physics-informed GNN that processes protein pocket dynamics.
    Encodes pocket features and correlation patterns.
    """

    def __init__(self, hidden_dim=128):
        """
        Initialize physics-informed GNN.

        Args:
            hidden_dim (int): Hidden dimension
        """
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
        """
        Forward pass through physics-informed GNN.

        Args:
            protein_residue_embs (torch.Tensor): Protein residue embeddings [batch, seq_len, 320]

        Returns:
            tuple: (pocket_features, correlation_features)
                - pocket_features: [batch, seq_len, hidden_dim]
                - correlation_features: [batch, hidden_dim//4]
        """
        h_pocket_dyn = self.pocket_encoder(protein_residue_embs)
        corr_features = self.correlation_head(h_pocket_dyn.mean(dim=1))
        return h_pocket_dyn, corr_features


class ECFP_Encoder(nn.Module):
    """
    ECFP (Extended Connectivity Fingerprint) Encoder.
    Processes Morgan fingerprints for molecular representation.
    """

    def __init__(self, ecfp_dim=2048, hidden_dim=128, dropout=0.1):
        """
        Initialize ECFP encoder.

        Args:
            ecfp_dim (int): Dimension of ECFP fingerprint (default: 2048)
            hidden_dim (int): Hidden dimension for output
            dropout (float): Dropout rate for regularization
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, ecfp):
        """
        Forward pass through ECFP encoder.

        Args:
            ecfp (torch.Tensor): ECFP fingerprint [batch, ecfp_dim]

        Returns:
            torch.Tensor: Encoded fingerprint [batch, hidden_dim]
        """
        return self.encoder(ecfp)
