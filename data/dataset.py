"""
AURA Framework - Dataset Module
PyTorch Dataset and collate functions for AURA model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import config


class OptimizedPDBbindDataset(Dataset):
    """
    Ultra-fast dataset using only pre-computed data with normalization support.

    This dataset loads pre-computed conformers, ECFP fingerprints, and protein tokens
    to avoid expensive on-the-fly computation during training.
    """

    def __init__(self, dataframe, conformer_dict, token_cache, ecfp_dict, normalizer=None):
        """
        Initialize dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame with PDB IDs, SMILES, and affinity values
            conformer_dict (dict): Pre-computed conformer graphs
            token_cache (dict): Pre-computed protein tokens
            ecfp_dict (dict): Pre-computed ECFP fingerprints
            normalizer (AffinityNormalizer, optional): Normalizer for affinity values
        """
        self.conformer_dict = conformer_dict
        self.token_cache = token_cache
        self.ecfp_dict = ecfp_dict
        self.normalizer = normalizer

        # Filter to valid entries only
        valid_entries = []
        skipped_no_conformer = 0
        skipped_no_token = 0
        skipped_no_ecfp = 0

        for _, row in dataframe.iterrows():
            pdb_id = row['pdb_id']
            smiles = row['canonical_smiles']
            key = f"{pdb_id}_{smiles}"

            # Check if all required data is available
            has_conformer = key in self.conformer_dict
            has_token = pdb_id in self.token_cache
            has_ecfp = smiles in self.ecfp_dict

            if has_conformer and has_token and has_ecfp:
                entry = row.to_dict()
                # Normalize affinity if normalizer is provided
                if self.normalizer and self.normalizer.fitted:
                    entry['normalized_affinity'] = self.normalizer.transform(
                        [entry['affinity']]
                    )[0]
                else:
                    entry['normalized_affinity'] = entry['affinity']
                valid_entries.append(entry)
            else:
                # Track why entries were skipped
                if not has_conformer:
                    skipped_no_conformer += 1
                if not has_token:
                    skipped_no_token += 1
                if not has_ecfp:
                    skipped_no_ecfp += 1

        self.data = valid_entries

        # Print summary
        total_requested = len(dataframe)
        total_valid = len(self.data)
        print(f"Dataset initialized: {total_valid}/{total_requested} valid entries")
        if total_valid < total_requested:
            print(f"  Skipped - Missing conformers: {skipped_no_conformer}")
            print(f"  Skipped - Missing tokens: {skipped_no_token}")
            print(f"  Skipped - Missing ECFP: {skipped_no_ecfp}")

    def __len__(self):
        """Return number of valid samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Args:
            idx (int): Index of sample

        Returns:
            dict: Dictionary containing graphs, ECFP, protein tokens, and labels
        """
        row = self.data[idx]
        pdb_id = row['pdb_id']
        smiles = row['canonical_smiles']
        key = f"{pdb_id}_{smiles}"

        return {
            'graphs_3d': self.conformer_dict[key],
            'ecfp': torch.from_numpy(self.ecfp_dict[smiles]),
            'protein_tokens': self.token_cache[pdb_id],
            'label': torch.tensor(row['normalized_affinity'], dtype=torch.float32),
            'original_label': torch.tensor(row['affinity'], dtype=torch.float32),
            'pdb_id': pdb_id,
            'smiles': smiles
        }


def custom_collate(batch):
    """
    Collate function handling conformer ensembles and variable-length sequences.

    Args:
        batch (list): List of samples from Dataset

    Returns:
        dict: Batched data with proper padding and batching
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Collect all graphs and track which batch they belong to
    all_graphs = []
    conformer_batch_idx = []
    for i, item in enumerate(batch):
        graphs = item.get('graphs_3d', [])
        if not graphs:
            continue
        for graph in graphs:
            all_graphs.append(graph)
            conformer_batch_idx.append(i)

    if not all_graphs:
        print("Warning: No valid graphs in batch")
        return None

    # Handle variable-length protein sequences by padding
    try:
        max_seq_len = max(item['protein_tokens']['input_ids'].size(0) for item in batch)
    except Exception as e:
        print(f"Error getting max sequence length: {e}")
        return None

    padded_input_ids = []
    padded_attention_masks = []

    for item in batch:
        try:
            input_ids = item['protein_tokens']['input_ids']
            attention_mask = item['protein_tokens']['attention_mask']

            # Calculate padding needed
            pad_len = max_seq_len - input_ids.size(0)

            if pad_len > 0:
                # Use PLM_PAD_ID from config
                input_ids_padded = F.pad(input_ids, (0, pad_len), value=config.PLM_PAD_ID)
                # Pad attention_mask with 0s
                attention_mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
            else:
                input_ids_padded = input_ids
                attention_mask_padded = attention_mask

            padded_input_ids.append(input_ids_padded)
            padded_attention_masks.append(attention_mask_padded)
        except Exception as e:
            print(f"Error padding sequence: {e}")
            return None

    # Create collated batch
    try:
        collated_batch = {
            'graphs_3d': Batch.from_data_list(all_graphs),
            'conformer_batch_idx': torch.tensor(conformer_batch_idx, dtype=torch.long),
            'ecfp': torch.stack([item['ecfp'] for item in batch]),
            'protein_tokens': {
                'input_ids': torch.stack(padded_input_ids),
                'attention_mask': torch.stack(padded_attention_masks)
            }
        }

        # Only add optional fields if they exist
        if 'label' in batch[0]:
            collated_batch['label'] = torch.stack([item['label'] for item in batch])
            collated_batch['original_label'] = torch.stack([item['original_label'] for item in batch])
        if 'pdb_id' in batch[0]:
            collated_batch['pdb_id'] = [item['pdb_id'] for item in batch]
        if 'smiles' in batch[0]:
            collated_batch['smiles'] = [item['smiles'] for item in batch]

        return collated_batch

    except Exception as e:
        print(f"Error creating collated batch: {e}")
        return None
