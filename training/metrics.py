"""
AURA Framework - Metrics Module
Evaluation metrics for regression tasks including RMSE, MAE, CI, Pearson R, and Spearman R.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def concordance_index(y_true, y_pred):
    """
    Calculate the concordance index (C-index).

    The concordance index measures the fraction of pairs where the model's
    prediction ordering matches the true value ordering.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: Concordance index (0-1, higher is better)
    """
    n = len(y_true)
    if n < 2:
        return 0.5  # Not enough data for pairwise comparison

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
    """
    Calculate comprehensive regression evaluation metrics.

    Args:
        y_true (array-like): True values (can be normalized or original)
        y_pred (array-like): Predicted values (should match y_true scale)
        denormalize (bool): If True and normalizer provided, denormalize before computing metrics
        normalizer (AffinityNormalizer, optional): Normalizer for denormalization

    Returns:
        dict: Dictionary containing all metrics
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - MSE: Mean Squared Error
            - CI: Concordance Index
            - Pearson_R: Pearson correlation coefficient
            - Spearman_R: Spearman correlation coefficient
    """
    if len(y_true) == 0:
        return {
            'RMSE': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'CI': 0.0,
            'Pearson_R': 0.0,
            'Spearman_R': 0.0
        }

    # Ensure numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Denormalize if requested
    if denormalize and normalizer and normalizer.fitted:
        y_true = normalizer.inverse_transform(y_true)
        y_pred = normalizer.inverse_transform(y_pred)

    # MSE and RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # Concordance Index (order-based, so normalization doesn't affect it)
    ci = concordance_index(y_true, y_pred)

    # Correlation coefficients (scale-invariant)
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
    """
    Print metrics in a formatted way.

    Args:
        metrics (dict): Dictionary of metrics
        prefix (str): Prefix to add before each line
    """
    if prefix:
        print(f"\n{prefix}:")
    print(f"  RMSE:       {metrics['RMSE']:.4f}")
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  MSE:        {metrics['MSE']:.4f}")
    print(f"  CI:         {metrics['CI']:.4f}")
    print(f"  Pearson R:  {metrics['Pearson_R']:.4f}")
    print(f"  Spearman R: {metrics['Spearman_R']:.4f}")


def compare_metrics(metrics_dict):
    """
    Compare metrics from multiple models side by side.

    Args:
        metrics_dict (dict): Dictionary mapping model names to their metrics
    """
    if not metrics_dict:
        return

    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    # Header
    models = list(metrics_dict.keys())
    metric_names = ['RMSE', 'MAE', 'MSE', 'CI', 'Pearson_R', 'Spearman_R']

    header = f"{'Metric':<12}"
    for model in models:
        header += f"{model:<15}"
    print(header)
    print("-"*80)

    # Metrics
    for metric in metric_names:
        row = f"{metric:<12}"
        for model in models:
            value = metrics_dict[model].get(metric, 0.0)
            row += f"{value:<15.4f}"
        print(row)

    print("="*80)


def best_model_by_metric(metrics_dict, metric_name='Pearson_R'):
    """
    Find the best model based on a specific metric.

    Args:
        metrics_dict (dict): Dictionary mapping model names to their metrics
        metric_name (str): Name of metric to use for comparison

    Returns:
        tuple: (best_model_name, best_metric_value)
    """
    if not metrics_dict:
        return None, None

    best_model = None
    best_value = -float('inf')

    for model_name, metrics in metrics_dict.items():
        value = metrics.get(metric_name, -float('inf'))
        if value > best_value:
            best_value = value
            best_model = model_name

    return best_model, best_value
