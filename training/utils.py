"""
AURA Framework - Training Utilities Module
Training and evaluation utilities including gradient accumulation and feature extraction.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import config


def train_one_epoch_with_accumulation(model, dataloader, optimizer, criterion,
                                     accumulation_steps=4, use_amp=False, scaler=None):
    """
    Training with gradient accumulation and gradient clipping.

    Args:
        model (nn.Module): Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        accumulation_steps (int): Number of steps to accumulate gradients
        use_amp (bool): Whether to use automatic mixed precision
        scaler: GradScaler for mixed precision training

    Returns:
        tuple: (average_loss, scaler)
    """
    model.train()
    total_loss = 0
    n_batches = 0

    if use_amp and scaler is None and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
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
                    labels = batch['label'].to(config.DEVICE).view(-1)
                    loss = criterion(predictions, labels)

                    # Add L2 regularization if available
                    if hasattr(model, 'get_l2_regularization_loss'):
                        loss = loss + model.get_l2_regularization_loss()

                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                predictions, _ = model(batch)
                predictions = predictions.view(-1)
                labels = batch['label'].to(config.DEVICE).view(-1)
                loss = criterion(predictions, labels)

                # Add L2 regularization if available
                if hasattr(model, 'get_l2_regularization_loss'):
                    loss = loss + model.get_l2_regularization_loss()

                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)
            optimizer.step()
        optimizer.zero_grad()

    average_loss = total_loss / max(n_batches, 1)
    return average_loss, scaler


def evaluate_model(model, dataloader, normalizer=None):
    """
    Evaluate regression model - returns both normalized and original predictions.

    Args:
        model (nn.Module): Model to evaluate
        dataloader: Evaluation dataloader
        normalizer (AffinityNormalizer, optional): Normalizer for denormalization

    Returns:
        tuple: (predictions, normalized_labels, original_labels)
    """
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
    """
    Extract features for XGBoost regression.

    Combines ECFP fingerprints with deep learning predictions as features.

    Args:
        model (nn.Module): Deep learning model
        dataloader: Dataloader

    Returns:
        tuple: (features, normalized_labels, original_labels)
    """
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

                # Combine ECFP with DL predictions as features
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

    features_array = np.vstack(features)
    labels_array = np.concatenate(labels)
    original_labels_array = np.concatenate(original_labels) if original_labels else None

    return features_array, labels_array, original_labels_array


def save_checkpoint(model, optimizer, epoch, loss, path, additional_info=None):
    """
    Save training checkpoint.

    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        path (str): Path to save checkpoint
        additional_info (dict, optional): Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if additional_info:
        checkpoint.update(additional_info)

    try:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(model, optimizer, path):
    """
    Load training checkpoint.

    Args:
        model (nn.Module): Model to load weights into
        optimizer: Optimizer to load state into
        path (str): Path to checkpoint

    Returns:
        dict: Checkpoint information (epoch, loss, etc.)
    """
    try:
        checkpoint = torch.load(path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {path}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """

    def __init__(self, patience=10, mode='max', delta=0.0):
        """
        Initialize early stopping.

        Args:
            patience (int): Number of epochs to wait before stopping
            mode (str): 'max' if higher is better, 'min' if lower is better
            delta (float): Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Check if training should stop.

        Args:
            score (float): Current validation metric

        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
