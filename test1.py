import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_scatter import scatter
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters (should match training)
hidden_channels = 64

# Define GRL layer (same as training)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.05, None

class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)

# Define GNN model (same as training)
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(15, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index, batch):
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return x

# Molecular processing functions (same as training)
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
    return list(map(lambda s: x == s, allowable_set))

def get_mol_nodes_edges(mol):
    """Convert RDKit molecule to graph representation"""
    if mol is None:
        raise ValueError("Invalid molecule")
    
    # Read node features
    N = mol.GetNumAtoms()
    atom_type = []
    atomic_number = []
    aromatic = []
    hybridization = []
    
    for atom in mol.GetAtoms():
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())

    # Read edge features
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    
    if len(row) == 0:  # Handle molecules with no bonds
        edge_index = torch.LongTensor([[], []])
        edge_attr = torch.FloatTensor([])
    else:
        edge_index = torch.LongTensor([row, col])
        edge_type = [one_of_k_encoding(t, [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) for t in edge_type]
        edge_attr = torch.FloatTensor(edge_type)
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

    row, col = edge_index if len(row) > 0 else (torch.tensor([]), torch.tensor([]))
    
    # Concat node features
    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    if len(row) > 0:
        num_hs = scatter(hs[row], col, dim_size=N).tolist()
    else:
        num_hs = [0] * N
    
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3]) for h in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)
    
    return x, edge_index, edge_attr

def is_valid_smiles(smi):
    """Check if SMILES string is valid"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        Chem.MolToSmiles(mol, isomericSmiles=True)
        return True
    except:
        return False

def smiles_to_data(smiles):
    """Convert SMILES string to PyG Data object"""
    if not is_valid_smiles(smiles):
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.MolFromSmiles(smiles)
    x, edge_index, edge_attr = get_mol_nodes_edges(mol)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

class MolecularPredictor:
    """Main class for molecular property prediction"""
    
    def __init__(self, model_path="saved_models/best_model.pth"):
        self.device = device
        self.model_path = model_path
        
        # Initialize models
        self.encoder = GNN().to(self.device)
        self.reg_model = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, 3),
            nn.Softplus(),
        ).to(self.device)
        
        # Load trained model
        self.load_model()
    
    def load_model(self):
        """Load trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.reg_model.load_state_dict(checkpoint['reg_model_state_dict'])
        
        # Set to evaluation mode
        self.encoder.eval()
        self.reg_model.eval()
        
        print("Model loaded successfully!")
        
        # Print model info if available
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'best_target_rmse_D' in checkpoint:
            print(f"Best RMSE - D: {checkpoint['best_target_rmse_D']:.4f}, "
                  f"P: {checkpoint['best_target_rmse_P']:.4f}, "
                  f"H: {checkpoint['best_target_rmse_H']:.4f}")
    
    def predict_single(self, smiles):
        """Predict HSP parameters for a single SMILES string"""
        try:
            # Convert SMILES to data
            data = smiles_to_data(smiles)
            
            # Create batch
            batch = torch.zeros(data.x.size(0), dtype=torch.long).to(self.device)
            
            # Predict
            with torch.no_grad():
                # Encode
                encoded = self.encoder(data.x, data.edge_index, batch)
                # Predict
                prediction = self.reg_model(encoded)
                
            prediction = prediction.cpu().numpy()[0]
            
            return {
                'smiles': smiles,
                'D': float(prediction[0]),
                'P': float(prediction[1]),
                'H': float(prediction[2])
            }
            
        except Exception as e:
            print(f"Error predicting {smiles}: {str(e)}")
            return None
    
    def predict_batch(self, smiles_list):
        """Predict HSP parameters for a list of SMILES strings"""
        results = []
        
        print(f"Predicting for {len(smiles_list)} molecules...")
        
        for smiles in tqdm(smiles_list):
            result = self.predict_single(smiles)
            if result is not None:
                results.append(result)
        
        return results
    
    def predict_from_file(self, file_path, smiles_column='SMILES', output_file=None):
        """Predict from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            if smiles_column not in df.columns:
                raise ValueError(f"Column '{smiles_column}' not found in file")
            
            smiles_list = df[smiles_column].tolist()
            results = self.predict_batch(smiles_list)
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Merge with original data
            df_merged = df.merge(results_df, left_on=smiles_column, right_on='smiles', how='left')
            
            if output_file:
                df_merged.to_csv(output_file, index=False)
                print(f"Results saved to: {output_file}")
            
            return df_merged
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None

# Test examples
def test_examples():
    """Test the model with example molecules"""
    
    # Initialize predictor
    predictor = MolecularPredictor()
    
    # Example molecules
    test_molecules = [
        "S1C=CC=C1C1C2N=CC(OCC(CCCCCC)CCCCCCCC)=NC2C=C(F)C=1F",  # Ethanol
        "S1C=CC=C1C1C2N=CC(OCCOCCOCCOCCOCCOCCOC)=NC2C=C(F)C=1F",  # Isopropanol
        "S1C=CC=C1C1C2N=CC(OC(COCCOCCOCCOC)COCCOCCOCCOC)=NC2C=C(F)C=1F",  # Stearic acid
        "S1C=CC=C1C1C2N=CC(OCC(COCCOCCOCCOC)COCCOCCOCCOC)=NC2C=C(F)C=1F",  # Benzene
        "S1C=C2C(F)=C(C(OCC(CC)CCCC)=O)SC2=C1C1SC2C(C3SC(CC(CCCC)CC)=CC=3)=C3C=CSC3=C(C3SC(CC(CCCC)CC)=CC=3)C=2C=1",  # Ethylbenzene
        "C(C1=CC(F)=C(CC(CC)CCCC)S1)1=C2C=CSC2=C(C2=CC(F)=C(CC(CCCC)CC)S2)C2C=C(C3=CC=C(C4SC(C5=CC=CS5)=C5C(=O)C6=C(C(CCCC)CC)SC(CC(CCCC)CC)=C6C(=O)C=45)S3)SC1=2",  # 4-tert-butylphenol
    ]
    
    print("="*60)
    print("Testing molecular property prediction")
    print("="*60)
    
    # Test single predictions
    print("\n1. Single molecule predictions:")
    print("-" * 50)
    
    for smiles in test_molecules:
        result = predictor.predict_single(smiles)
        if result:
            print(f"SMILES: {result['smiles']}")
            print(f"HSP Parameters - D: {result['D']:.2f}, P: {result['P']:.2f}, H: {result['H']:.2f}")
            print("-" * 50)
    
    # Test batch prediction
    print("\n2. Batch prediction:")
    print("-" * 50)
    
    batch_results = predictor.predict_batch(test_molecules)
    results_df = pd.DataFrame(batch_results)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('prediction_results.csv', index=False)
    print(f"\nResults saved to: prediction_results.csv")
    
    return results_df

if __name__ == "__main__":
    test_examples()
