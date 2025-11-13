"""
AURA Framework - KAN (Kolmogorov-Arnold Network) Module
Implements KAN layers using B-spline basis functions for flexible function approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear Layer.

    Uses learnable spline-based transformations instead of fixed activations,
    providing more flexibility in learning complex non-linear mappings.
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        """
        Initialize KAN linear layer.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            grid_size (int): Number of grid points for spline interpolation
            spline_order (int): Order of B-spline basis functions
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create spline grid
        h = (1.0 / grid_size)
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h
        self.register_buffer("grid", grid.float())

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # Initialize parameters
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        with torch.no_grad():
            self.spline_weight.uniform_(-0.1, 0.1)

    def b_splines(self, x):
        """
        Compute B-spline basis functions.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: B-spline basis evaluations
        """
        x = x.unsqueeze(-1)  # [batch, in_features, 1]

        # Degree 0: piecewise constant
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)

        # Recursively compute higher degree basis functions
        for k in range(1, self.spline_order + 1):
            # Avoid division by zero
            denominator1 = self.grid[k:-1] - self.grid[:-(k + 1)]
            denominator2 = self.grid[k + 1:] - self.grid[1:(-k)]

            # Replace zero denominators with 1 to avoid NaN (these terms will be multiplied by 0 anyway)
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
        """
        Forward pass through KAN layer.

        Args:
            x (torch.Tensor): Input tensor [batch_size, in_features]

        Returns:
            torch.Tensor: Output tensor [batch_size, out_features]
        """
        # Base linear transformation
        base_output = F.linear(x, self.base_weight)

        # Spline-based transformation
        spline_basis = self.b_splines(x)  # [batch, in_features, grid_size + spline_order]
        spline_basis_flat = spline_basis.view(x.size(0), -1)  # [batch, in_features * (grid_size + spline_order)]

        spline_weight_flat = self.spline_weight.view(self.out_features, -1)
        spline_output = F.linear(spline_basis_flat, spline_weight_flat)

        return base_output + spline_output

    def extra_repr(self):
        """Return string representation of layer parameters."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'grid_size={self.grid_size}, spline_order={self.spline_order}'
