"""
AURA Framework - Data Normalization Module
Handles normalization and denormalization of affinity values.
"""

import numpy as np
import pickle
import os


class AffinityNormalizer:
    """
    Handles normalization and denormalization of affinity values.

    Supports two normalization methods:
    - 'zscore': Z-score normalization (mean=0, std=1)
    - 'minmax': Min-max scaling to [0, 1]
    """

    def __init__(self, method='zscore'):
        """
        Initialize normalizer.

        Args:
            method (str): 'zscore' for z-score normalization, 'minmax' for min-max scaling
        """
        if method not in ['zscore', 'minmax']:
            raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'.")

        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False

    def fit(self, values):
        """
        Fit the normalizer on training data.

        Args:
            values: Array-like of affinity values

        Returns:
            self: Returns self for method chaining

        Raises:
            ValueError: If values is empty
        """
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
        """
        Normalize values.

        Args:
            values: Array-like of affinity values

        Returns:
            np.ndarray: Normalized values

        Raises:
            RuntimeError: If normalizer has not been fitted
        """
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
        """
        Denormalize values back to original scale.

        Args:
            values: Array-like of normalized affinity values

        Returns:
            np.ndarray: Denormalized values

        Raises:
            RuntimeError: If normalizer has not been fitted
        """
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
        """
        Save normalizer parameters to disk.

        Args:
            path (str): File path to save the normalizer

        Raises:
            IOError: If save operation fails
        """
        params = {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'fitted': self.fitted
        }

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'wb') as f:
                pickle.dump(params, f)
        except Exception as e:
            raise IOError(f"Failed to save normalizer to {path}: {str(e)}")

    def load(self, path):
        """
        Load normalizer parameters from disk.

        Args:
            path (str): File path to load the normalizer from

        Returns:
            self: Returns self for method chaining

        Raises:
            IOError: If load operation fails
            FileNotFoundError: If file doesn't exist
        """
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
        """
        Get normalization parameters as dict.

        Returns:
            dict: Dictionary containing normalization parameters
        """
        return {
            'method': self.method,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'fitted': self.fitted
        }

    def __repr__(self):
        """String representation of the normalizer."""
        if not self.fitted:
            return f"AffinityNormalizer(method='{self.method}', fitted=False)"

        if self.method == 'zscore':
            return f"AffinityNormalizer(method='zscore', mean={self.mean:.4f}, std={self.std:.4f})"
        else:
            return f"AffinityNormalizer(method='minmax', min={self.min:.4f}, max={self.max:.4f})"
