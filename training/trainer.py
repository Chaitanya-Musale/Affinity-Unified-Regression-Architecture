"""
AURA Framework - Trainer Module
Implements staged training strategy for AURA model.
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import xgboost as xgb
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import numpy as np

import config
from .metrics import regression_metrics, print_metrics
from .utils import train_one_epoch_with_accumulation, evaluate_model, extract_features_for_xgboost


class StagedTrainer:
    """
    Implements staged training strategy with normalization support.

    Training stages:
    - Stage B: Train only regression heads with frozen backbone
    - Stage C: Fine-tune entire model with learning rate warmup
    - XGBoost Training: Train XGBoost component for ensemble
    - Ensemble Weight Tuning: Fine-tune ensemble weight parameter
    """

    def __init__(self, ensemble_model, train_loader, val_loader, test_loaders_dict,
                 normalizer=None, device='cuda', use_amp=False, save_dir='./saved_models'):
        """
        Initialize staged trainer.

        Args:
            ensemble_model: AuraEnsemble model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loaders_dict (dict): Dictionary of test dataloaders
            normalizer (AffinityNormalizer, optional): Normalizer for affinity values
            device (str): Device to train on
            use_amp (bool): Whether to use automatic mixed precision
            save_dir (str): Directory to save models and results
        """
        self.ensemble_model = ensemble_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loaders_dict = test_loaders_dict
        self.normalizer = normalizer
        self.device = device
        self.use_amp = use_amp
        self.save_dir = save_dir

        if use_amp and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

        os.makedirs(save_dir, exist_ok=True)

        self.training_history = {
            'stage_b': {'loss': [], 'val_metrics': []},
            'stage_c': {'loss': [], 'val_metrics': []},
            'test_results': {},
            'normalization_params': normalizer.get_params() if normalizer else None
        }

    def freeze_backbone(self):
        """Freeze all layers except regression heads."""
        for name, param in self.ensemble_model.dl_model.named_parameters():
            if 'kan_head' not in name and 'ensemble_weight' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.ensemble_model.parameters():
            param.requires_grad = True

    def stage_b_head_training(self, epochs=15):
        """
        Stage B: Train only heads with frozen backbone.

        Args:
            epochs (int): Number of epochs to train
        """
        print("\n" + "="*70)
        print("STAGE B: Head Training (Backbone Frozen)")
        print("="*70)
        print("Training regression heads on normalized values...")

        if self.normalizer:
            print(f"Normalization: {self.normalizer.method}")
            if self.normalizer.method == 'zscore':
                print(f"  Mean: {self.normalizer.mean:.4f}, Std: {self.normalizer.std:.4f}")
            else:
                print(f"  Min: {self.normalizer.min:.4f}, Max: {self.normalizer.max:.4f}")

        self.freeze_backbone()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.ensemble_model.parameters()),
            lr=5e-4
        )
        criterion = nn.MSELoss()

        best_val_pearson = -1
        patience_counter = 0
        best_path = None

        for epoch in range(epochs):
            train_loss, self.scaler = train_one_epoch_with_accumulation(
                self.ensemble_model, self.train_loader, optimizer, criterion,
                accumulation_steps=config.ACCUMULATION_STEPS,
                use_amp=self.use_amp,
                scaler=self.scaler
            )

            val_preds, val_labels, val_original_labels = evaluate_model(
                self.ensemble_model, self.val_loader, self.normalizer
            )

            if len(val_labels) > 0:
                # Denormalize predictions for metrics
                if self.normalizer and self.normalizer.fitted:
                    val_preds_denorm = self.normalizer.inverse_transform(val_preds)
                else:
                    val_preds_denorm = val_preds

                val_metrics = regression_metrics(val_original_labels, val_preds_denorm)
            else:
                val_metrics = {
                    'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0,
                    'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0
                }

            self.training_history['stage_b']['loss'].append(train_loss)
            self.training_history['stage_b']['val_metrics'].append(val_metrics)

            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f} (normalized scale)")
            print(f"    Val RMSE={val_metrics['RMSE']:.4f}, Pearson R={val_metrics['Pearson_R']:.4f} (original scale)")

            # Early stopping check
            if val_metrics['Pearson_R'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson_R']
                best_path = os.path.join(self.save_dir, 'stage_b_best.pt')
                torch.save(self.ensemble_model.state_dict(), best_path)
                print(f"    ✨ New best model saved! (Pearson R: {val_metrics['Pearson_R']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  ⏹️  Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_path and os.path.exists(best_path):
            self.ensemble_model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best Stage B model from {best_path}")

        # Store initial weights for L2 regularization
        self.ensemble_model.store_initial_weights()

    def stage_c_fine_tuning(self, epochs=25):
        """
        Stage C: Fine-tune entire model with learning rate warmup.

        Args:
            epochs (int): Number of epochs to train
        """
        print("\n" + "="*70)
        print("STAGE C: Fine-tuning (All Layers)")
        print("="*70)
        print("Fine-tuning with normalized values and proper warmup scheduling...")

        self.unfreeze_all()

        optimizer = torch.optim.AdamW(
            self.ensemble_model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-4
        )

        criterion = nn.MSELoss()

        best_val_pearson = -1
        patience_counter = 0
        best_path = None

        global_step = 0

        for epoch in range(epochs):
            self.ensemble_model.train()
            total_loss = 0
            n_batches = 0

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Stage C Epoch {epoch+1}/{epochs}")):
                if batch is None:
                    continue

                try:
                    optimizer.zero_grad()

                    predictions, _ = self.ensemble_model(batch)
                    predictions = predictions.view(-1)
                    labels = batch['label'].to(self.device).view(-1)
                    loss = criterion(predictions, labels)

                    # Add L2 regularization
                    if hasattr(self.ensemble_model, 'get_l2_regularization_loss'):
                        loss = loss + self.ensemble_model.get_l2_regularization_loss()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.ensemble_model.parameters(),
                        config.GRADIENT_CLIP_NORM
                    )
                    optimizer.step()

                    # Learning rate warmup
                    if global_step < config.WARMUP_STEPS:
                        warmup_progress = global_step / config.WARMUP_STEPS
                        lr = config.LEARNING_RATE * (0.01 + 0.99 * warmup_progress)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    global_step += 1
                    total_loss += loss.item()
                    n_batches += 1

                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue

            train_loss = total_loss / max(n_batches, 1)
            val_preds, val_labels, val_original_labels = evaluate_model(
                self.ensemble_model, self.val_loader, self.normalizer
            )

            if len(val_labels) > 0:
                # Denormalize predictions for metrics
                if self.normalizer and self.normalizer.fitted:
                    val_preds_denorm = self.normalizer.inverse_transform(val_preds)
                else:
                    val_preds_denorm = val_preds

                val_metrics = regression_metrics(val_original_labels, val_preds_denorm)
            else:
                val_metrics = {
                    'RMSE': 0.0, 'MAE': 0.0, 'MSE': 0.0,
                    'CI': 0.0, 'Pearson_R': 0.0, 'Spearman_R': 0.0
                }

            self.training_history['stage_c']['loss'].append(train_loss)
            self.training_history['stage_c']['val_metrics'].append(val_metrics)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f} (normalized), LR={current_lr:.6f}")
            print(f"    Val RMSE={val_metrics['RMSE']:.4f}, Pearson R={val_metrics['Pearson_R']:.4f} (original scale)")

            # Early stopping check
            if val_metrics['Pearson_R'] > best_val_pearson:
                best_val_pearson = val_metrics['Pearson_R']
                best_path = os.path.join(self.save_dir, 'aura_final_best.pt')
                torch.save(self.ensemble_model.state_dict(), best_path)
                print(f"    ✨ New best model saved! (Pearson R: {val_metrics['Pearson_R']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"    Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  ⏹️  Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if best_path and os.path.exists(best_path):
            self.ensemble_model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"Loaded best Stage C model from {best_path}")

    def train_xgboost_component(self):
        """Train XGBoost for ensemble regression on normalized values."""
        print("\n" + "="*70)
        print("Training XGBoost Component (on normalized values)")
        print("="*70)

        X_train, y_train, _ = extract_features_for_xgboost(
            self.ensemble_model.dl_model, self.train_loader
        )
        X_val, y_val, _ = extract_features_for_xgboost(
            self.ensemble_model.dl_model, self.val_loader
        )

        if len(X_train) > 0 and len(X_val) > 0:
            xgb_model = xgb.XGBRegressor(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=config.XGB_LEARNING_RATE,
                subsample=config.XGB_SUBSAMPLE,
                colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
                random_state=config.XGB_RANDOM_STATE,
                eval_metric='rmse',
                early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS
            )

            # Train on normalized values
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            self.ensemble_model.set_xgb_model(xgb_model)

            # Save XGBoost model
            xgb_path = os.path.join(self.save_dir, 'xgboost_model.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)
            print(f"Saved XGBoost model to {xgb_path}")

            self.fine_tune_ensemble_weight()
        else:
            print("⚠️  Not enough data to train XGBoost component")

    def fine_tune_ensemble_weight(self, epochs=3):
        """Fine-tune only the ensemble weight parameter."""
        print("\n" + "="*70)
        print("Fine-tuning Ensemble Weight")
        print("="*70)

        for name, param in self.ensemble_model.named_parameters():
            param.requires_grad = (name == 'ensemble_weight')

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.ensemble_model.parameters()),
            lr=1e-3
        )
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.ensemble_model.train()

            for batch in tqdm(self.train_loader, desc=f"Weight Tuning {epoch+1}/{epochs}"):
                if batch is None:
                    continue

                try:
                    with torch.no_grad():
                        dl_pred, _ = self.ensemble_model.dl_model(batch)
                        dl_pred_np = dl_pred.cpu().numpy()
                        ecfp = batch['ecfp'].cpu().numpy()
                        xgb_features = np.hstack([ecfp, dl_pred_np.reshape(-1, 1)])

                    optimizer.zero_grad()
                    ensemble_pred, _ = self.ensemble_model(batch, xgb_features)
                    ensemble_pred = ensemble_pred.view(-1)
                    labels = batch['label'].to(self.device).view(-1)
                    loss = criterion(ensemble_pred, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.ensemble_model.parameters()),
                        config.GRADIENT_CLIP_NORM
                    )

                    optimizer.step()
                except Exception as e:
                    print(f"Error in weight tuning: {e}")
                    continue

            w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
            print(f"  Ensemble weight (DL portion): {w:.3f}")

        # Save final ensemble model
        final_path = os.path.join(self.save_dir, 'aura_ensemble_final.pt')
        torch.save(self.ensemble_model.state_dict(), final_path)
        print(f"Saved final ensemble model to {final_path}")

    def evaluate_on_all_test_sets(self):
        """Comprehensive evaluation on all test sets - reports on original scale."""
        print("\n" + "="*70)
        print("Final Model Evaluation on All Test Sets (Original Scale)")
        print("="*70)

        all_results = {}

        for test_name, test_loader in self.test_loaders_dict.items():
            print(f"\n📊 Evaluating on {test_name}:")
            print("="*50)

            test_preds_dl, test_labels, test_original_labels = evaluate_model(
                self.ensemble_model.dl_model, test_loader, self.normalizer
            )

            if len(test_labels) == 0:
                print(f"No test samples available for {test_name}")
                continue

            # Denormalize predictions for metric computation
            if self.normalizer and self.normalizer.fitted:
                test_preds_dl_denorm = self.normalizer.inverse_transform(test_preds_dl)
            else:
                test_preds_dl_denorm = test_preds_dl

            metrics_results = {}
            metrics_results['Deep Learning'] = regression_metrics(
                test_original_labels, test_preds_dl_denorm
            )

            if self.ensemble_model.xgb_model is not None:
                X_test, _, _ = extract_features_for_xgboost(
                    self.ensemble_model.dl_model, test_loader
                )

                if len(X_test) > 0:
                    test_preds_xgb = self.ensemble_model.xgb_model.predict(X_test)

                    # Denormalize XGBoost predictions
                    if self.normalizer and self.normalizer.fitted:
                        test_preds_xgb_denorm = self.normalizer.inverse_transform(test_preds_xgb)
                    else:
                        test_preds_xgb_denorm = test_preds_xgb

                    w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
                    test_preds_ensemble = w * test_preds_dl_denorm + (1 - w) * test_preds_xgb_denorm

                    metrics_results['XGBoost'] = regression_metrics(
                        test_original_labels, test_preds_xgb_denorm
                    )
                    metrics_results['Ensemble'] = regression_metrics(
                        test_original_labels, test_preds_ensemble
                    )

            # Print results for this test set
            for model_name in metrics_results:
                print_metrics(metrics_results[model_name], prefix=model_name)

            all_results[test_name] = metrics_results

        # Save all results
        self.training_history['test_results'] = all_results

        # Save to JSON
        results_path = os.path.join(self.save_dir, 'all_test_results.json')
        try:
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✅ Saved all test results to {results_path}")
        except Exception as e:
            print(f"Warning: Failed to save results to JSON: {e}")

        return all_results
