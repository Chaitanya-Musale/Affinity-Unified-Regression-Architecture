"""
AURA Framework - Interpretability Module
Complete interpretability suite for AURA with SHAP analysis and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap

import config
from data.preprocessing import generate_conformer_ensemble, mol_to_3d_graph, smiles_to_ecfp
from data.dataset import custom_collate


class CompleteAuraInterpreter:
    """
    Complete interpretability suite for AURA model.

    Provides multi-level explanations:
    - Level 1: Model predictions (DL, XGBoost, Ensemble)
    - Level 2: Ligand-protein interaction patterns
    - Level 3: Feature importance via SHAP
    - Level 4: Atom-level attributions
    """

    def __init__(self, ensemble_model, plm_tokenizer, normalizer=None):
        """
        Initialize interpreter.

        Args:
            ensemble_model: Trained AuraEnsemble model
            plm_tokenizer: HuggingFace tokenizer for protein sequences
            normalizer (AffinityNormalizer, optional): Normalizer for denormalization
        """
        self.ensemble_model = ensemble_model.to(config.DEVICE)
        self.plm_tokenizer = plm_tokenizer
        self.normalizer = normalizer
        self.ensemble_model.eval()

        # Initialize SHAP explainer if XGBoost model is available
        if ensemble_model.xgb_model is not None:
            try:
                self.shap_explainer = shap.TreeExplainer(ensemble_model.xgb_model)
            except Exception as e:
                print(f"Warning: Failed to initialize SHAP explainer: {e}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None

    def explain(self, smiles, protein_sequence, pdb_id):
        """
        Generate comprehensive multi-level explanations.

        Args:
            smiles (str): SMILES string of ligand
            protein_sequence (str): Protein sequence
            pdb_id (str): PDB identifier

        Returns:
            dict: Dictionary containing all explanation levels
        """
        print(f"\n=== AURA Explanation for {pdb_id} ===")

        # Generate conformers
        mol_conf_data = generate_conformer_ensemble(smiles, config.N_CONFORMERS)
        if mol_conf_data is None:
            print("Failed to generate conformers")
            return None

        mol, conf_ids = mol_conf_data
        graphs_3d = [mol_to_3d_graph(mol, cid) for cid in conf_ids[:config.N_CONFORMERS]]
        graphs_3d = [g for g in graphs_3d if g is not None]

        if not graphs_3d:
            print("No valid graphs generated")
            return None

        # Generate ECFP
        ecfp = smiles_to_ecfp(smiles)

        # Tokenize protein
        protein_tokens = self.plm_tokenizer(
            protein_sequence, return_tensors='pt',
            max_length=config.MAX_PROTEIN_LENGTH, truncation=True
        )

        # Create batch (need both label and original_label for custom_collate)
        batch = custom_collate([{
            'graphs_3d': graphs_3d,
            'ecfp': torch.from_numpy(ecfp),
            'protein_tokens': {k: v.squeeze(0) for k, v in protein_tokens.items()},
            'label': torch.tensor([0.0]),
            'original_label': torch.tensor([0.0]),
            'pdb_id': pdb_id,
            'smiles': smiles
        }])

        if batch is None:
            print("Failed to create batch")
            return None

        # Get predictions
        with torch.no_grad():
            dl_pred, interpretability_info = self.ensemble_model.dl_model(batch)
            dl_affinity_norm = dl_pred.squeeze().item()

            # Denormalize for display
            if self.normalizer and self.normalizer.fitted:
                dl_affinity = self.normalizer.inverse_transform(np.array([dl_affinity_norm]))[0]
            else:
                dl_affinity = dl_affinity_norm

            # Extract interpretability information
            attn_weights = interpretability_info.get('interaction_weights')
            pocket_scores = interpretability_info.get('pocket_scores')
            conformer_weights = interpretability_info.get('conformer_weights')

            # Extract interaction matrix
            # attn_weights shape: [batch, num_atoms, seq_len]
            if attn_weights is not None:
                interaction_matrix = attn_weights.squeeze(0).cpu().numpy()  # [num_atoms, seq_len]
            else:
                interaction_matrix = np.zeros((10, 10))

            # Extract pocket importance scores
            if pocket_scores is not None:
                pocket_importance = pocket_scores.squeeze().cpu().numpy()
            else:
                pocket_importance = None

            # XGBoost and ensemble predictions
            feature_importance = None
            if self.ensemble_model.xgb_model is not None:
                xgb_features = np.hstack([ecfp.reshape(1, -1), [[dl_affinity_norm]]])
                xgb_affinity_norm = self.ensemble_model.xgb_model.predict(xgb_features)[0]

                # Denormalize
                if self.normalizer and self.normalizer.fitted:
                    xgb_affinity = self.normalizer.inverse_transform(np.array([xgb_affinity_norm]))[0]
                else:
                    xgb_affinity = xgb_affinity_norm

                # SHAP analysis
                if self.shap_explainer is not None:
                    try:
                        shap_values = self.shap_explainer(xgb_features)
                        feature_importance = self._extract_top_features(shap_values, ecfp)
                    except Exception as e:
                        print(f"Warning: SHAP analysis failed: {e}")
                        feature_importance = None

                w = torch.sigmoid(self.ensemble_model.ensemble_weight).item()
                final_affinity = w * dl_affinity + (1 - w) * xgb_affinity
            else:
                xgb_affinity = None
                final_affinity = dl_affinity

            # Atom attributions
            if attn_weights is not None:
                atom_attributions = self._compute_atom_attributions(
                    graphs_3d[0], attn_weights
                )
            else:
                atom_attributions = np.zeros(graphs_3d[0].num_nodes)

        return {
            'predictions': {
                'deep_learning': float(dl_affinity),
                'xgboost': float(xgb_affinity) if xgb_affinity is not None else None,
                'ensemble': float(final_affinity),
                'ensemble_weight_dl': torch.sigmoid(self.ensemble_model.ensemble_weight).item()
            },
            'level2_interactions': interaction_matrix,
            'level3_features': feature_importance,
            'level4_atoms': atom_attributions,
            'pocket_importance': pocket_importance,
            'conformer_weights': [w.numpy() if w is not None else None for w in conformer_weights] if conformer_weights else None,
            'conformer_count': len(graphs_3d),
            'pdb_id': pdb_id,
            'smiles': smiles
        }

    def _extract_top_features(self, shap_values, ecfp, top_k=10):
        """
        Extract top chemical features from SHAP values.

        Args:
            shap_values: SHAP values from explainer
            ecfp (np.ndarray): ECFP fingerprint
            top_k (int): Number of top features to extract

        Returns:
            dict: Top positive and negative features
        """
        shap_array = shap_values.values[0]
        ecfp_shap = shap_array[:-1]  # Exclude DL prediction feature

        top_positive = np.argsort(ecfp_shap)[-top_k:]
        top_negative = np.argsort(ecfp_shap)[:top_k]

        return {
            'positive_bits': [(int(i), float(ecfp_shap[i])) for i in top_positive if ecfp[i] == 1],
            'negative_bits': [(int(i), float(ecfp_shap[i])) for i in top_negative if ecfp[i] == 1]
        }

    def _compute_atom_attributions(self, graph, attn_weights):
        """
        Compute per-atom importance scores from attention weights.

        Args:
            graph: PyG Data object
            attn_weights (torch.Tensor): Attention weights [batch, num_atoms, seq_len]

        Returns:
            np.ndarray: Normalized atom importance scores
        """
        # attn_weights shape: [batch, num_atoms, seq_len]
        # Sum over protein dimension to get per-atom importance
        atom_scores = attn_weights.sum(dim=2)  # [batch, num_atoms]
        atom = atom_scores[0].cpu().numpy()  # Take first batch item

        # Normalize to [0, 1]
        atom_range = atom.max() - atom.min()
        if atom_range > 0:
            atom = (atom - atom.min()) / atom_range
        else:
            atom = np.zeros_like(atom)

        return atom[:graph.num_nodes]

    def visualize_explanation(self, explanation_dict, save_path=None):
        """
        Create comprehensive visualization dashboard.

        Args:
            explanation_dict (dict): Explanation dictionary from explain()
            save_path (str, optional): Path to save figure

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig = plt.figure(figsize=(18, 12))  # Larger for 3x3 grid

        # 1. Prediction comparison (top left)
        ax1 = plt.subplot(3, 3, 1)
        preds = explanation_dict['predictions']
        names = ['DL', 'XGB', 'Ensemble']
        values = [
            preds['deep_learning'],
            preds.get('xgboost', 0) if preds.get('xgboost') is not None else 0,
            preds['ensemble']
        ]
        bars = ax1.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Model Predictions (Original Scale)', fontweight='bold')
        ax1.set_ylabel('Predicted Affinity (-logKd/Ki)')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Feature importance (top middle)
        ax2 = plt.subplot(3, 3, 2)
        if explanation_dict['level3_features'] is not None:
            feat_imp = explanation_dict['level3_features']
            if feat_imp['positive_bits'] or feat_imp['negative_bits']:
                pos_indices = [x[0] for x in feat_imp['positive_bits'][:5]]
                pos_values = [x[1] for x in feat_imp['positive_bits'][:5]]
                neg_indices = [x[0] for x in feat_imp['negative_bits'][:5]]
                neg_values = [x[1] for x in feat_imp['negative_bits'][:5]]

                all_indices = pos_indices + neg_indices
                all_values = pos_values + neg_values
                colors = ['g'] * len(pos_indices) + ['r'] * len(neg_indices)

                ax2.barh(range(len(all_values)), all_values, color=colors)
                ax2.set_yticks(range(len(all_values)))
                ax2.set_yticklabels([f'Bit {i}' for i in all_indices], fontsize=8)
                ax2.set_xlabel('SHAP Value')
                ax2.set_title('Top ECFP Feature Importance', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No features available', ha='center', va='center')
                ax2.set_title('Top ECFP Feature Importance', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'XGBoost not available', ha='center', va='center')
            ax2.set_title('Top ECFP Feature Importance', fontweight='bold')
        ax2.axis('off') if explanation_dict['level3_features'] is None else None

        # 3. Ligand-Protein interactions (top right)
        ax3 = plt.subplot(3, 3, 3)
        interaction_data = explanation_dict['level2_interactions']
        if interaction_data.size > 0:
            im = ax3.imshow(
                interaction_data[:min(20, interaction_data.shape[0]),
                                :min(50, interaction_data.shape[1])],
                cmap='viridis', aspect='auto'
            )
            plt.colorbar(im, ax=ax3)
            ax3.set_title('Ligand-Protein Interactions', fontweight='bold')
            ax3.set_xlabel('Protein Residues')
            ax3.set_ylabel('Ligand Atoms')
        else:
            ax3.text(0.5, 0.5, 'No interaction data', ha='center', va='center')
            ax3.axis('off')

        # 4. Ensemble weight breakdown (middle left)
        ax4 = plt.subplot(3, 3, 4)
        if explanation_dict['predictions']['xgboost'] is not None:
            w_dl = explanation_dict['predictions']['ensemble_weight_dl']
            w_xgb = 1 - w_dl
            wedges, texts, autotexts = ax4.pie(
                [w_dl, w_xgb],
                labels=['Deep Learning', 'XGBoost'],
                autopct='%1.1f%%',
                colors=['#1f77b4', '#ff7f0e'],
                startangle=90
            )
            ax4.set_title('Ensemble Weight Distribution', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '100% Deep Learning', ha='center', va='center')
            ax4.axis('off')

        # 5. Atom-level attributions (middle center)
        ax5 = plt.subplot(3, 3, 5)
        atom_scores = explanation_dict['level4_atoms']
        if len(atom_scores) > 0:
            ax5.plot(atom_scores, 'o-', color='#2ca02c')
            ax5.set_title('Atom-level Attributions', fontweight='bold')
            ax5.set_xlabel('Atom Index')
            ax5.set_ylabel('Importance Score')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No atom data', ha='center', va='center')
            ax5.axis('off')

        # 6. Pocket importance scores (middle right) - NEW!
        ax6 = plt.subplot(3, 3, 6)
        pocket_scores_data = explanation_dict.get('pocket_importance')
        if pocket_scores_data is not None and len(pocket_scores_data) > 0:
            top_residues = min(20, len(pocket_scores_data))
            ax6.barh(range(top_residues), pocket_scores_data[:top_residues], color='#9467bd')
            ax6.set_xlabel('Importance Score')
            ax6.set_ylabel('Residue Index')
            ax6.set_title('Pocket Residue Importance', fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
            ax6.invert_yaxis()
        else:
            ax6.text(0.5, 0.5, 'No pocket data', ha='center', va='center')
            ax6.axis('off')

        # 7. Conformer weights (bottom left) - NEW!
        ax7 = plt.subplot(3, 3, 7)
        conformer_weights_data = explanation_dict.get('conformer_weights')
        if conformer_weights_data is not None and len(conformer_weights_data) > 0:
            weights = conformer_weights_data[0]  # First batch sample
            if weights is not None:
                n_conf = len(weights)
                ax7.bar(range(n_conf), weights, color='#8c564b')
                ax7.set_xlabel('Conformer Index')
                ax7.set_ylabel('Attention Weight')
                ax7.set_title('Conformer Selection Weights', fontweight='bold')
                ax7.grid(axis='y', alpha=0.3)
                ax7.set_xticks(range(n_conf))
            else:
                ax7.text(0.5, 0.5, 'No conformer weights', ha='center', va='center')
                ax7.axis('off')
        else:
            ax7.text(0.5, 0.5, 'No conformer data', ha='center', va='center')
            ax7.axis('off')

        # 8. Summary information (bottom center)
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        info_text = f"""
PDB ID: {explanation_dict['pdb_id']}
SMILES: {explanation_dict['smiles'][:40]}...
Conformers: {explanation_dict['conformer_count']}
DL Weight: {explanation_dict['predictions']['ensemble_weight_dl']:.3f}

Predicted Affinity: {explanation_dict['predictions']['ensemble']:.2f}

Model Components:
  • 2D GNN (Topology)
  • 3D GNN (Geometry)
  • PLM (Protein)
  • Physics-informed
        """
        ax8.text(0.1, 0.5, info_text, fontsize=9, family='monospace',
                verticalalignment='center')
        ax8.set_title('Summary', fontweight='bold')

        # 9. Leave bottom right empty for now (future use)
        ax9 = plt.subplot(3, 3, 9)
        ax9.text(0.5, 0.5, 'Reserved\nfor future\nvisualization',
                ha='center', va='center', fontsize=10, alpha=0.5)
        ax9.axis('off')

        plt.suptitle('AURA Regression Explanation Dashboard',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=config.VISUALIZATION_DPI, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            except Exception as e:
                print(f"Warning: Failed to save visualization: {e}")

        return fig
