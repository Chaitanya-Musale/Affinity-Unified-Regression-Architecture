"""
AURA Framework - Main Execution Script
Complete training pipeline for AURA model with normalization support.
"""

import os
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from data.normalization import AffinityNormalizer
from data.preprocessing import (
    precompute_all_conformers,
    precompute_ecfp_fingerprints,
    preprocess_and_cache_tokens
)
from data.dataset import OptimizedPDBbindDataset, custom_collate
from models.aura_model import AuraDeepLearningModel
from models.ensemble import AuraEnsemble
from training.trainer import StagedTrainer
from interpretation.explainer import CompleteAuraInterpreter


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("AURA FRAMEWORK - MAIN EXECUTION")
    print("="*70)

    # Print configuration
    config.print_config()

    # =========================================================================
    # Setup paths and directories
    # =========================================================================
    print("\n--- Setting up paths ---")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Construct cache paths
    conformer_cache_path = os.path.join(config.OUTPUT_DIR, config.CONFORMER_CACHE)
    ecfp_cache_path = os.path.join(config.OUTPUT_DIR, config.ECFP_CACHE)
    token_cache_path = os.path.join(config.OUTPUT_DIR, config.TOKEN_CACHE)
    normalizer_cache_path = os.path.join(config.OUTPUT_DIR, config.NORMALIZER_CACHE)

    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Structure paths:")
    for i, path in enumerate(config.STRUCTURE_PATHS, 1):
        exists = os.path.exists(path)
        status = '✓ Found' if exists else '✗ Missing'
        print(f"  {i}. {status}: {path}")

    # =========================================================================
    # Load data splits
    # =========================================================================
    print("\n--- Loading data splits ---")
    try:
        train_df = pd.read_csv(config.TRAIN_CSV)
        val_df = pd.read_csv(config.VAL_CSV)
        test_general_df = pd.read_csv(config.TEST_GENERAL_CSV)
        test_refined_df = pd.read_csv(config.TEST_REFINED_CSV)
        test_casf_df = pd.read_csv(config.TEST_CASF_CSV)

        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test General: {len(test_general_df)} samples")
        print(f"Test Refined: {len(test_refined_df)} samples")
        print(f"Test CASF: {len(test_casf_df)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure all CSV files exist and paths are correct.")
        return

    # =========================================================================
    # Initialize and fit normalizer
    # =========================================================================
    print("\n--- Setting up affinity normalizer ---")
    affinity_normalizer = AffinityNormalizer(method=config.NORMALIZATION_METHOD)

    if os.path.exists(normalizer_cache_path):
        try:
            affinity_normalizer.load(normalizer_cache_path)
            print(f"Loaded normalizer from cache")
        except Exception as e:
            print(f"Failed to load normalizer: {e}")
            print("Fitting new normalizer...")
            affinity_normalizer.fit(train_df['affinity'].values)
            affinity_normalizer.save(normalizer_cache_path)
    else:
        affinity_normalizer.fit(train_df['affinity'].values)
        affinity_normalizer.save(normalizer_cache_path)
        print(f"Fitted and saved normalizer")

    print(f"Normalizer: {affinity_normalizer}")

    # =========================================================================
    # Initialize tokenizer
    # =========================================================================
    print("\n--- Initializing protein language model tokenizer ---")
    try:
        plm_tokenizer = AutoTokenizer.from_pretrained(config.PLM_MODEL_NAME)
        plm_pad_id = plm_tokenizer.pad_token_id or 1
        config.update_pad_id(plm_pad_id)
        print(f"Tokenizer initialized: {config.PLM_MODEL_NAME}")
        print(f"Pad token ID: {plm_pad_id}")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return

    # =========================================================================
    # Pre-compute all data
    # =========================================================================
    print("\n" + "="*70)
    print("PRE-COMPUTING ALL DATA (ONE-TIME COST)")
    print("="*70)

    # Combine all unique data
    all_data_df = pd.concat([
        train_df, val_df, test_general_df, test_refined_df, test_casf_df
    ], ignore_index=True).drop_duplicates(subset=['pdb_id'])

    print(f"Total unique PDB entries: {len(all_data_df)}")

    # Pre-compute conformers
    print("\n1️⃣ Pre-computing conformers...")
    all_conformers = precompute_all_conformers(
        all_data_df, config.STRUCTURE_PATHS,
        config.N_CONFORMERS, conformer_cache_path
    )

    # Pre-compute ECFP
    print("\n2️⃣ Pre-computing ECFP fingerprints...")
    all_ecfp = precompute_ecfp_fingerprints(all_data_df, ecfp_cache_path)

    # Pre-compute protein tokens
    print("\n3️⃣ Pre-computing protein tokens...")
    token_cache = preprocess_and_cache_tokens(
        all_data_df, config.STRUCTURE_PATHS, plm_tokenizer, token_cache_path
    )

    print("\n✅ All pre-computation complete!")

    # =========================================================================
    # Create datasets
    # =========================================================================
    print("\n--- Creating optimized datasets ---")
    train_dataset = OptimizedPDBbindDataset(
        train_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
    )
    val_dataset = OptimizedPDBbindDataset(
        val_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
    )
    test_general_dataset = OptimizedPDBbindDataset(
        test_general_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
    )
    test_refined_dataset = OptimizedPDBbindDataset(
        test_refined_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
    )
    test_casf_dataset = OptimizedPDBbindDataset(
        test_casf_df, all_conformers, token_cache, all_ecfp, affinity_normalizer
    )

    # =========================================================================
    # Create dataloaders
    # =========================================================================
    print("\n--- Creating dataloaders ---")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )

    test_loaders_dict = {}
    test_datasets_info = {
        'General Set 2020': test_general_dataset,
        'Refined Set 2020': test_refined_dataset,
        'CASF 2016': test_casf_dataset
    }

    for name, dataset in test_datasets_info.items():
        if len(dataset) > 0:
            test_loaders_dict[name] = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=custom_collate,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            print(f"  {name}: {len(dataset)} samples")
        else:
            print(f"  ⚠️  {name}: No valid samples")

    # =========================================================================
    # Initialize model
    # =========================================================================
    print("\n--- Initializing AURA model ---")
    dl_model = AuraDeepLearningModel(
        gnn_2d_node_dim=config.GNN_2D_NODE_DIM,
        gnn_2d_edge_dim=config.GNN_2D_EDGE_DIM,
        hidden_dim=config.HIDDEN_DIM,
        plm_hidden_dim=config.PLM_HIDDEN_DIM,
        out_dim=config.OUTPUT_DIM
    ).to(config.DEVICE)

    ensemble_model = AuraEnsemble(dl_model, normalizer=affinity_normalizer).to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in ensemble_model.parameters())
    trainable_params = sum(p.numel() for p in ensemble_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # Initialize trainer
    # =========================================================================
    print("\n--- Initializing trainer ---")
    trainer = StagedTrainer(
        ensemble_model,
        train_loader,
        val_loader,
        test_loaders_dict,
        normalizer=affinity_normalizer,
        device=config.DEVICE,
        use_amp=config.USE_AMP,
        save_dir=config.OUTPUT_DIR
    )

    # =========================================================================
    # Run training
    # =========================================================================
    print("\n" + "="*70)
    print("STARTING STAGED TRAINING")
    print("="*70)
    print("Training on normalized scale, evaluating on original scale")
    print("Expected benefits:")
    print("  • Better gradient flow")
    print("  • Faster convergence")
    print("  • Improved stability")

    # Stage B: Head training
    if config.MAX_EPOCHS_STAGE_B > 0:
        trainer.stage_b_head_training(config.MAX_EPOCHS_STAGE_B)
    else:
        print("\nSkipping Stage B (MAX_EPOCHS_STAGE_B = 0)")

    # Stage C: Fine-tuning
    if config.MAX_EPOCHS_STAGE_C > 0:
        trainer.stage_c_fine_tuning(config.MAX_EPOCHS_STAGE_C)
    else:
        print("\nSkipping Stage C (MAX_EPOCHS_STAGE_C = 0)")

    # Train XGBoost and fine-tune ensemble
    trainer.train_xgboost_component()

    # =========================================================================
    # Final evaluation
    # =========================================================================
    if test_loaders_dict:
        all_test_results = trainer.evaluate_on_all_test_sets()
    else:
        print("\n⚠️  No test sets available for evaluation")
        all_test_results = {}

    # =========================================================================
    # Save training history
    # =========================================================================
    print("\n--- Saving training history ---")
    history_path = os.path.join(config.OUTPUT_DIR, 'training_history.pkl')
    try:
        with open(history_path, 'wb') as f:
            pickle.dump(trainer.training_history, f)
        print(f"Training history saved to {history_path}")
    except Exception as e:
        print(f"Warning: Failed to save training history: {e}")

    # =========================================================================
    # Initialize interpreter and generate example visualizations
    # =========================================================================
    print("\n--- Initializing interpreter ---")
    interpreter = CompleteAuraInterpreter(
        ensemble_model, plm_tokenizer, affinity_normalizer
    )
    print("✅ Interpreter ready for generating explanations")

    # Generate example explanations for a few test samples
    print("\n--- Generating Example Interpretability Visualizations ---")
    try:
        # Get a few examples from test set
        num_examples = min(5, len(test_general_df))
        example_indices = np.random.choice(len(test_general_df), num_examples, replace=False)

        os.makedirs(os.path.join(config.OUTPUT_DIR, 'interpretability'), exist_ok=True)

        for idx in example_indices:
            row = test_general_df.iloc[idx]
            pdb_id = row['pdb_id']
            smiles = row['canonical_smiles']

            print(f"\n  Generating explanation for {pdb_id}...")

            # Find protein sequence from cache
            if pdb_id in token_cache:
                protein_sequence = token_cache[pdb_id]['sequence']

                # Generate explanation
                explanation = interpreter.explain(smiles, protein_sequence, pdb_id)

                if explanation is not None:
                    # Visualize and save
                    save_path = os.path.join(
                        config.OUTPUT_DIR,
                        'interpretability',
                        f'{pdb_id}_explanation.png'
                    )
                    interpreter.visualize_explanation(explanation, save_path)
                    print(f"    ✓ Saved visualization to {save_path}")
                else:
                    print(f"    ✗ Failed to generate explanation for {pdb_id}")
            else:
                print(f"    ✗ Protein sequence not found for {pdb_id}")

        print(f"\n✅ Generated {num_examples} example interpretability visualizations")
        print(f"   Location: {os.path.join(config.OUTPUT_DIR, 'interpretability')}")

    except Exception as e:
        print(f"Warning: Failed to generate interpretability visualizations: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("AURA FRAMEWORK TRAINING COMPLETE")
    print("="*70)
    print(f"All models and results saved to: {config.OUTPUT_DIR}")
    print("\nKey improvements from normalization:")
    print("  • Improved gradient flow during backpropagation")
    print("  • Faster convergence and better stability")
    print("  • More consistent loss landscapes")
    print("\nNext steps:")
    print("  • Use interpreter.explain() for model interpretability")
    print("  • Check test results in all_test_results.json")
    print("  • Review training history in training_history.pkl")
    print("="*70)


if __name__ == "__main__":
    main()
