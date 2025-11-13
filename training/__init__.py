"""
AURA Framework - Training Package
Contains training utilities, metrics, and trainer classes.
"""

from .metrics import (
    concordance_index,
    regression_metrics,
    print_metrics,
    compare_metrics,
    best_model_by_metric
)
from .utils import (
    train_one_epoch_with_accumulation,
    evaluate_model,
    extract_features_for_xgboost,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping
)
from .trainer import StagedTrainer

__all__ = [
    'concordance_index',
    'regression_metrics',
    'print_metrics',
    'compare_metrics',
    'best_model_by_metric',
    'train_one_epoch_with_accumulation',
    'evaluate_model',
    'extract_features_for_xgboost',
    'save_checkpoint',
    'load_checkpoint',
    'EarlyStopping',
    'StagedTrainer'
]
