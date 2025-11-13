"""
AURA Framework - Data Preprocessing Module
Functions for molecular conformer generation, protein sequence extraction,
and feature computation with robust error handling.
"""

import os
import pickle
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from Bio import PDB
from tqdm import tqdm
from torch_geometric.data import Data


def extract_protein_sequence(pdb_file_path):
    """
    Extract protein sequence from PDB file - handles multiple chains correctly.

    Args:
        pdb_file_path (str): Path to PDB file

    Returns:
        str: Protein sequence in one-letter code, or None if extraction fails
    """
    parser = PDB.PDBParser(QUIET=True)

    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file not found: {pdb_file_path}")
        return None

    try:
        structure = parser.get_structure('protein', pdb_file_path)

        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard amino acid
                        res_name = residue.resname
                        if res_name in three_to_one:
                            sequence.append(three_to_one[res_name])

        if not sequence:
            print(f"Warning: No valid residues found in {pdb_file_path}")
            return None

        return ''.join(sequence)

    except Exception as e:
        print(f"Error extracting sequence from {pdb_file_path}: {e}")
        return None


def generate_conformer_ensemble_with_crystal(smiles_string, crystal_mol=None, n_conformers=5):
    """
    Generates conformer ensemble, optionally using crystal structure as first conformer.

    Args:
        smiles_string (str): SMILES representation of molecule
        crystal_mol (Chem.Mol, optional): Crystal structure molecule
        n_conformers (int): Number of conformers to generate

    Returns:
        tuple: (mol, conf_ids) or None if generation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            print(f"Warning: Failed to parse SMILES: {smiles_string}")
            return None

        mol = Chem.AddHs(mol)

        # Try to use crystal structure as first conformer
        use_crystal = False
        if crystal_mol is not None:
            try:
                crystal_mol_h = Chem.AddHs(crystal_mol, addCoords=True)

                # Check if number of atoms match
                if mol.GetNumAtoms() == crystal_mol_h.GetNumAtoms():
                    # Try to match atom ordering
                    match = mol.GetSubstructMatch(crystal_mol_h)
                    if match and len(match) == mol.GetNumAtoms():
                        # Create conformer with crystal coordinates
                        conf = Chem.Conformer(mol.GetNumAtoms())
                        for i, j in enumerate(match):
                            pos = crystal_mol_h.GetConformer().GetAtomPosition(j)
                            conf.SetAtomPosition(i, pos)
                        mol.AddConformer(conf)
                        use_crystal = True
            except Exception as e:
                # Crystal structure couldn't be used, will generate all conformers
                pass

        # Generate remaining conformers
        if use_crystal:
            # We already have the crystal as conformer 0, generate n_conformers-1 more
            if n_conformers > 1:
                try:
                    additional_conf_ids = AllChem.EmbedMultipleConfs(
                        mol,
                        numConfs=n_conformers-1,
                        pruneRmsThresh=0.5,
                        randomSeed=42,
                        clearConfs=False  # Don't clear existing conformers
                    )
                except Exception as e:
                    additional_conf_ids = []
            conf_ids = list(range(mol.GetNumConformers()))
        else:
            # Generate all conformers from scratch
            try:
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers,
                    pruneRmsThresh=0.5,
                    randomSeed=42,
                    useRandomCoords=True
                )
            except Exception as e:
                conf_ids = []

            if len(conf_ids) == 0:
                # Fallback: try single conformer generation
                try:
                    res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    if res == -1:
                        print(f"Warning: Failed to embed molecule: {smiles_string}")
                        return None
                    conf_ids = [0]
                except Exception as e:
                    print(f"Error embedding molecule {smiles_string}: {e}")
                    return None

        # Optimize non-crystal conformers only
        for i, conf_id in enumerate(conf_ids):
            # Don't optimize the crystal pose (first conformer if use_crystal)
            if use_crystal and i == 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
            except Exception as e:
                pass  # Continue even if optimization fails

        # Validate conformer IDs
        valid_conf_ids = []
        for conf_id in conf_ids:
            if conf_id < mol.GetNumConformers():
                valid_conf_ids.append(conf_id)

        if not valid_conf_ids:
            print(f"Warning: No valid conformers generated for {smiles_string}")
            return None

        return mol, valid_conf_ids

    except Exception as e:
        print(f"Error generating conformers for {smiles_string}: {e}")
        return None


def generate_conformer_ensemble(smiles_string, n_conformers=5):
    """
    Simple conformer generation without crystal structure.

    Args:
        smiles_string (str): SMILES representation of molecule
        n_conformers (int): Number of conformers to generate

    Returns:
        tuple: (mol, conf_ids) or None if generation fails
    """
    return generate_conformer_ensemble_with_crystal(smiles_string, None, n_conformers)


def mol_to_3d_graph(mol, conf_id=0):
    """
    Converts a molecule with 3D coordinates to a PyG Data object with validation.

    Args:
        mol (Chem.Mol): RDKit molecule with conformers
        conf_id (int): Conformer ID to use

    Returns:
        Data: PyTorch Geometric Data object, or None if conversion fails
    """
    if mol is None:
        return None

    # Validate conformer ID
    if conf_id >= mol.GetNumConformers():
        print(f"Warning: Invalid conformer ID {conf_id} (molecule has {mol.GetNumConformers()} conformers)")
        return None

    try:
        conf = mol.GetConformer(conf_id)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

        atom_features = []
        atomic_numbers = []
        for atom in mol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
            atom_features.append([
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs()
            ])

        z = torch.tensor(atomic_numbers, dtype=torch.long)
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_indices, edge_features = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = [
                int(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_type, bond_type])

        if not edge_indices:
            # Molecule has no bonds (single atom or disconnected)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as e:
        print(f"Error converting molecule to graph (conf_id={conf_id}): {e}")
        return None


def smiles_to_ecfp(smiles_string, n_bits=2048):
    """
    Converts a SMILES string to an ECFP (Morgan) fingerprint.

    Args:
        smiles_string (str): SMILES representation of molecule
        n_bits (int): Number of bits in fingerprint

    Returns:
        np.ndarray: ECFP fingerprint array, or zeros if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            print(f"Warning: Failed to parse SMILES for ECFP: {smiles_string}")
            return np.zeros(n_bits, dtype=np.float32)

        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    except Exception as e:
        print(f"Error generating ECFP for {smiles_string}: {e}")
        return np.zeros(n_bits, dtype=np.float32)


def precompute_all_conformers(all_data_df, structure_paths, n_conformers=5, cache_path='conformers_precomputed.pkl'):
    """
    Pre-compute all conformers before training starts - supports multiple structure paths.

    Args:
        all_data_df (pd.DataFrame): DataFrame containing PDB IDs and SMILES
        structure_paths (list): List of paths to search for structure files
        n_conformers (int): Number of conformers to generate
        cache_path (str): Path to save/load cache

    Returns:
        dict: Dictionary mapping (pdb_id, smiles) to list of graph Data objects
    """
    if os.path.exists(cache_path):
        try:
            print(f"Loading pre-computed conformers from {cache_path}")
            with open(cache_path, 'rb') as f:
                conformer_dict = pickle.load(f)
            print(f"Loaded {len(conformer_dict)} pre-computed conformer sets")
            return conformer_dict
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            print("Will regenerate conformers...")

    print("Pre-computing all conformers (this only happens once)...")
    conformer_dict = {}
    failed_entries = []

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Pre-computing conformers"):
        pdb_id = row['pdb_id']
        smiles = row['canonical_smiles']

        # Try to find the structure in any of the provided paths
        crystal_mol = None
        ligand_found = False

        for structure_path in structure_paths:
            ligand_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_ligand.sdf")

            # Try alternative naming for CASF
            if not os.path.exists(ligand_path):
                ligand_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_ligand.mol2")

            if os.path.exists(ligand_path):
                try:
                    if ligand_path.endswith('.sdf'):
                        ligand_supplier = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)
                        if ligand_supplier and len(ligand_supplier) > 0:
                            crystal_mol = ligand_supplier[0]
                    elif ligand_path.endswith('.mol2'):
                        crystal_mol = Chem.MolFromMol2File(ligand_path, removeHs=False, sanitize=True)
                    ligand_found = True
                    break
                except Exception as e:
                    pass  # Try next path

        # Generate conformers
        mol_conf_data = generate_conformer_ensemble_with_crystal(smiles, crystal_mol, n_conformers)

        if mol_conf_data:
            mol, conf_ids = mol_conf_data
            graphs = []
            for conf_id in conf_ids[:n_conformers]:
                graph = mol_to_3d_graph(mol, conf_id)
                if graph:
                    graphs.append(graph)

            if graphs:
                key = f"{pdb_id}_{smiles}"
                conformer_dict[key] = graphs
            else:
                failed_entries.append(pdb_id)
        else:
            failed_entries.append(pdb_id)

    # Save the pre-computed conformers
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(conformer_dict, f)
        print(f"💾 Saved to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save conformer cache: {e}")

    print(f"✅ Pre-computed {len(conformer_dict)} conformer sets")
    if failed_entries:
        print(f"⚠️  Failed to compute conformers for {len(failed_entries)} entries")

    return conformer_dict


def precompute_ecfp_fingerprints(all_data_df, cache_path='ecfp_precomputed.pkl'):
    """
    Pre-compute all ECFP fingerprints.

    Args:
        all_data_df (pd.DataFrame): DataFrame containing SMILES strings
        cache_path (str): Path to save/load cache

    Returns:
        dict: Dictionary mapping SMILES to ECFP fingerprint arrays
    """
    if os.path.exists(cache_path):
        try:
            print(f"Loading pre-computed ECFP fingerprints from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load ECFP cache from {cache_path}: {e}")
            print("Will regenerate ECFP fingerprints...")

    print("Pre-computing ECFP fingerprints...")
    ecfp_dict = {}

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Computing ECFP"):
        smiles = row['canonical_smiles']
        if smiles not in ecfp_dict:
            ecfp_dict[smiles] = smiles_to_ecfp(smiles)

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(ecfp_dict, f)
    except Exception as e:
        print(f"Warning: Failed to save ECFP cache: {e}")

    print(f"✅ Pre-computed {len(ecfp_dict)} ECFP fingerprints")
    return ecfp_dict


def preprocess_and_cache_tokens(all_data_df, structure_paths, plm_tokenizer, cache_path):
    """
    Pre-tokenize all protein sequences for efficiency - supports multiple structure paths.

    Args:
        all_data_df (pd.DataFrame): DataFrame containing PDB IDs
        structure_paths (list): List of paths to search for protein files
        plm_tokenizer: HuggingFace tokenizer for protein sequences
        cache_path (str): Path to save/load cache

    Returns:
        dict: Dictionary mapping PDB ID to tokenized sequences
    """
    if os.path.exists(cache_path):
        try:
            print(f"Loading cached protein tokens from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load token cache from {cache_path}: {e}")
            print("Will regenerate tokens...")

    print("Pre-tokenizing protein sequences (this will only happen once)...")
    token_cache = {}

    for _, row in tqdm(all_data_df.iterrows(), total=len(all_data_df), desc="Tokenizing proteins"):
        pdb_id = row['pdb_id']

        # Try to find the protein in any of the provided paths
        sequence = None
        for structure_path in structure_paths:
            protein_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_protein.pdb")

            # Try alternative naming for CASF (pocket instead of protein)
            if not os.path.exists(protein_path):
                protein_path = os.path.join(structure_path, pdb_id, f"{pdb_id}_pocket.pdb")

            if os.path.exists(protein_path):
                sequence = extract_protein_sequence(protein_path)
                if sequence:
                    break

        if sequence:
            try:
                tokens = plm_tokenizer(
                    sequence, return_tensors='pt', padding='longest',
                    truncation=True, max_length=1024
                )
                token_cache[pdb_id] = {
                    'input_ids': tokens['input_ids'].squeeze(0),
                    'attention_mask': tokens['attention_mask'].squeeze(0),
                    'sequence': sequence
                }
            except Exception as e:
                print(f"Error tokenizing protein {pdb_id}: {e}")

    # Save cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(token_cache, f)
        print(f"Saved token cache with {len(token_cache)} proteins to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save token cache: {e}")

    return token_cache
