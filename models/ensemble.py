"""
AURA Framework - Ensemble Model Module
Combines deep learning model with XGBoost for ensemble predictions.
"""

import torch
import torch.nn as nn
import config


class AuraEnsemble(nn.Module):
    """
    AURA Ensemble Model combining deep learning and XGBoost predictions.

    The ensemble uses a learnable weight parameter to balance between:
    - Deep learning model (handles complex interactions)
    - XGBoost model (captures tabular patterns in ECFP features)
    """

    def __init__(self, dl_model, normalizer=None):
        """
        Initialize ensemble model.

        Args:
            dl_model (nn.Module): Deep learning model (AuraDeepLearningModel)
            normalizer (AffinityNormalizer, optional): Normalizer for affinity values
        """
        super().__init__()
        self.dl_model = dl_model
        self.xgb_model = None
        self.ensemble_weight = nn.Parameter(torch.tensor([0.5]))
        self.normalizer = normalizer

        # Store initial weights for L2 regularization during fine-tuning
        self.initial_weights = None

    def set_xgb_model(self, xgb_model):
        """
        Set the XGBoost model for ensemble.

        Args:
            xgb_model: Trained XGBoost model
        """
        self.xgb_model = xgb_model

    def store_initial_weights(self):
        """
        Store initial weights for L2 regularization during fine-tuning.
        This prevents the model from deviating too far from pre-trained weights.
        """
        self.initial_weights = {}
        for name, param in self.dl_model.named_parameters():
            self.initial_weights[name] = param.data.clone()

    def get_l2_regularization_loss(self):
        """
        Calculate L2 regularization loss towards initial weights.
        Used during fine-tuning to prevent catastrophic forgetting.

        Returns:
            torch.Tensor: L2 regularization loss
        """
        if self.initial_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        l2_loss = 0.0
        for name, param in self.dl_model.named_parameters():
            if name in self.initial_weights:
                l2_loss += torch.sum((param - self.initial_weights[name]) ** 2)

        return config.L2_REG_WEIGHT * l2_loss

    def forward(self, batch, xgb_features=None):
        """
        Forward pass through ensemble.

        Args:
            batch (dict): Batch dictionary for deep learning model
            xgb_features (np.ndarray, optional): Pre-extracted features for XGBoost

        Returns:
            tuple: (predictions, interpretability_info)
                - predictions: Ensemble predictions [batch_size, 1]
                - interpretability_info: Dict with interaction_weights, pocket_scores, conformer_weights
        """
        # Get deep learning predictions
        dl_pred, interpretability_info = self.dl_model(batch)

        # If XGBoost model is available and features provided, use ensemble
        if self.xgb_model is not None and xgb_features is not None:
            try:
                # XGBoost expects normalized values if trained on normalized data
                xgb_pred = torch.tensor(
                    self.xgb_model.predict(xgb_features),
                    device=dl_pred.device,
                    dtype=dl_pred.dtype
                ).unsqueeze(1)

                # Weighted ensemble
                w = torch.sigmoid(self.ensemble_weight)
                ensemble_pred = w * dl_pred + (1 - w) * xgb_pred

                return ensemble_pred, interpretability_info
            except Exception as e:
                print(f"Warning: XGBoost prediction failed: {e}")
                print("Falling back to deep learning predictions only")
                return dl_pred, interpretability_info

        # If XGBoost not available, return only DL predictions
        return dl_pred, interpretability_info

    def get_ensemble_weight(self):
        """
        Get the current ensemble weight (DL model weight after sigmoid).

        Returns:
            float: Weight for deep learning model (0-1)
        """
        return torch.sigmoid(self.ensemble_weight).item()

    def freeze_dl_model(self):
        """Freeze all deep learning model parameters."""
        for param in self.dl_model.parameters():
            param.requires_grad = False

    def unfreeze_dl_model(self):
        """Unfreeze all deep learning model parameters."""
        for param in self.dl_model.parameters():
            param.requires_grad = True

    def freeze_except_ensemble_weight(self):
        """Freeze all parameters except ensemble weight."""
        for name, param in self.named_parameters():
            param.requires_grad = (name == 'ensemble_weight')
