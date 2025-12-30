"""
GT-PolySol Hyperparameter Search using NNI with TPE
====================================================
This implementation uses NNI (Neural Network Intelligence) framework with 
Tree-structured Parzen Estimator (TPE) for hyperparameter optimization.

Search Space:
- Architectural: K (GCN layers), learning_rate, d_emb (embedding dimension)
- Loss-related: α, β, w_d, w_p, w_h

Reference: Appendix C, D (Hyperparameter Analysis of GT-PolySol)
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import json
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import pickle

# NNI imports
try:
    import nni
    from nni.experiment import Experiment
    NNI_AVAILABLE = True
except ImportError:
    NNI_AVAILABLE = False
    print("Warning: NNI not installed. Using standalone mode.")

# PyTorch Geometric imports
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# RDKit imports
import rdkit
from rdkit import Chem, RDConfig
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger

from sklearn.metrics import mean_squared_error

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Hyperparameter Search Space Definition
# ============================================================================

SEARCH_SPACE = {
    # Architectural hyperparameters
    "num_layers": {"_type": "choice", "_value": [3, 5, 7, 9]},
    "learning_rate": {"_type": "choice", "_value": [1e-4, 3e-4, 5e-4]},
    "embedding_dim": {"_type": "choice", "_value": [64, 128, 256]},
    
    # Loss-related hyperparameters
    "alpha": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "beta": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "w_d": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "w_p": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "w_h": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
    
    # Training hyperparameters
    "temperature": {"_type": "choice", "_value": [0.05, 0.07, 0.1, 0.2]},
    "grl_rate": {"_type": "choice", "_value": [0.01, 0.05, 0.1]},
}

# Default hyperparameters (best configuration from paper)
DEFAULT_PARAMS = {
    "num_layers": 4,
    "learning_rate": 3e-4,
    "embedding_dim": 64,
    "alpha": 0.1,
    "beta": 0.1,
    "w_d": 0.05,
    "w_p": 0.05,
    "w_h": 0.1,
    "temperature": 0.07,
    "grl_rate": 0.05,
}


# ============================================================================
# Model Architecture with Configurable Hyperparameters
# ============================================================================

class ConfigurableGNN(nn.Module):
    """
    Configurable GNN encoder with variable number of layers and embedding dimension.
    """
    
    def __init__(self, input_dim=15, hidden_dim=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
    
    def forward(self, x, edge_index, batch):
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        return x


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer"""
    rate = 0.05
    
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * GradReverse.rate, None


class GRL(nn.Module):
    """Gradient Reversal Layer Module"""
    def forward(self, input):
        return GradReverse.apply(input)


# ============================================================================
# Data Processing Functions
# ============================================================================

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


def one_of_k_encoding(x, allowable_set):
    """One-hot encoding for categorical features"""
    return list(map(lambda s: x == s, allowable_set))


def get_mol_nodes_edges(mol):
    """Convert RDKit molecule to graph representation"""
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

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    
    edge_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) for t in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    
    if len(row) > 0:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        row, col = edge_index
        
        hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()
    else:
        num_hs = [0] * N

    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HT.SP, HT.SP2, HT.SP3]) for h in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)

    return x, edge_index, edge_attr


def load_data():
    """Load and preprocess training data"""
    # Try to load negative samples
    try:
        with open('neg_sample.pkl', 'rb') as f:
            dict_n = pickle.load(f)
    except FileNotFoundError:
        logger.warning("neg_sample.pkl not found. Using empty dictionary.")
        dict_n = {}

    dataset_source = []
    dataset_target = []

    # Load source (solvent) data
    try:
        df_source = pd.read_csv('solvent_HSP_HSPiP_hz1010.csv')
    except FileNotFoundError:
        logger.warning("Creating synthetic source data for demonstration.")
        df_source = pd.DataFrame({
            'SMILES': ['CCO', 'CCCO', 'CC(C)O', 'CCCCO', 'CC(C)(C)O'] * 20,
            'D': np.random.uniform(15, 20, 100),
            'P': np.random.uniform(5, 15, 100),
            'H': np.random.uniform(5, 15, 100)
        })

    smiles_list_v = []
    labels = []
    
    for _, row in df_source.iterrows():
        smi = row['SMILES']
        if is_valid_smiles(smi):
            smiles_list_v.append(smi)
            labels.append([row['D'], row['P'], row['H']])

    for smi, label in zip(smiles_list_v, labels):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_source.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                                   smi=smi, y=torch.tensor([label])))

    # Load target (polymer) data
    polymer_smi_valid = []
    
    try:
        df_target = pd.read_csv('polymer_HSP_HSPiP_hz1010.csv')
    except FileNotFoundError:
        logger.warning("Creating synthetic target data for demonstration.")
        df_target = pd.DataFrame({
            'SMILES': ['C=C', 'CC=CC', 'C=CC=C', 'CC(=C)C', 'C=C(C)C'] * 20,
            'D': np.random.uniform(15, 20, 100),
            'P': np.random.uniform(5, 15, 100),
            'H': np.random.uniform(5, 15, 100)
        })

    smiles_list_v = []
    labels = []
    
    for _, row in df_target.iterrows():
        smi = str(row['SMILES']).replace('X', '')
        if is_valid_smiles(smi):
            smiles_list_v.append(smi)
            polymer_smi_valid.append(smi)
            labels.append([row['D'], row['P'], row['H']])

    for smi, label in zip(smiles_list_v, labels):
        x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
        dataset_target.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                   smi=smi, y=torch.tensor([label])))

    # Generate positive samples (oligomers)
    dataset_pos = []
    pos_index = []
    
    for smi in polymer_smi_valid:
        pos_data = []
        for repeat in [1, 3, 12]:
            oligomer = smi * repeat
            if is_valid_smiles(oligomer):
                x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(oligomer))
                pos_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=oligomer))
        
        if len(pos_data) == 0:
            x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(smi))
            pos_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=smi))
        
        pos_index.append(len(pos_data))
        dataset_pos.extend(pos_data)

    # Generate negative samples (fragments)
    dataset_neg = []
    neg_index = []
    
    for smi in polymer_smi_valid:
        neg_data = []
        frags = dict_n.get(smi, ['C'])
        
        for frag in frags:
            if is_valid_smiles(frag):
                x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles(frag))
                neg_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi=frag))
        
        if len(neg_data) == 0:
            x, edge_index, edge_attr = get_mol_nodes_edges(Chem.MolFromSmiles('C'))
            neg_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smi='C'))
        
        neg_index.append(len(neg_data))
        dataset_neg.extend(neg_data)

    return dataset_source, dataset_target, dataset_pos, dataset_neg, pos_index, neg_index


# ============================================================================
# GT-PolySol Model with Configurable Hyperparameters
# ============================================================================

class GTPolySol:
    """
    GT-PolySol model with configurable hyperparameters for NNI optimization.
    """
    
    def __init__(self, params):
        """
        Initialize model with given hyperparameters.
        
        Args:
            params: Dictionary containing hyperparameters
        """
        self.params = params
        self.hidden_dim = params.get('embedding_dim', 64)
        self.num_layers = params.get('num_layers', 4)
        self.lr = params.get('learning_rate', 3e-4)
        self.alpha = params.get('alpha', 0.1)
        self.beta = params.get('beta', 0.1)
        self.w_d = params.get('w_d', 0.05)
        self.w_p = params.get('w_p', 0.05)
        self.w_h = params.get('w_h', 0.1)
        self.temperature = params.get('temperature', 0.07)
        self.grl_rate = params.get('grl_rate', 0.05)
        
        # Initialize models
        self._build_model()
        
    def _build_model(self):
        """Build model components"""
        self.encoder = ConfigurableGNN(
            input_dim=15, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers
        ).to(device)
        
        self.reg_model = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, 3),
        ).to(device)
        
        self.domain_model = nn.Sequential(
            GRL(),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
        ).to(device)
        
        self.models = [self.encoder, self.reg_model, self.domain_model]
        
        # Loss functions
        self.criterion = nn.MSELoss(reduction="mean").to(device)
        self.loss_domain = nn.CrossEntropyLoss().to(device)
        
        # Optimizer
        params = itertools.chain(*[model.parameters() for model in self.models])
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        
    def encode(self, data):
        """Encode molecular graphs"""
        return self.encoder(data.x, data.edge_index, data.batch)
    
    def predict(self, data):
        """Predict HSP values"""
        encoded = self.encode(data)
        return self.reg_model(encoded)
    
    def contrastive_loss(self, positive_samples, negative_samples, pos_index, neg_index):
        """Compute contrastive loss"""
        pos_idx = 0
        neg_idx = 0
        loss_all = 0
        
        for pos, neg in zip(pos_index, neg_index):
            # Collect samples
            pos_samples = [positive_samples[i] for i in range(pos_idx, pos_idx + pos)]
            neg_samples = [negative_samples[i] for i in range(neg_idx, neg_idx + neg)]
            
            pos_ground = [positive_samples[pos_idx]] * pos
            neg_ground = [negative_samples[neg_idx]] * neg
            
            pos_idx += pos
            neg_idx += neg
            
            # Convert to tensors
            pos_tensor = torch.stack([p.detach() for p in pos_samples])
            neg_tensor = torch.stack([n.detach() for n in neg_samples])
            pos_gt = torch.stack([p.detach() for p in pos_ground])
            neg_gt = torch.stack([n.detach() for n in neg_ground])
            
            # Normalize
            pos_tensor = F.normalize(pos_tensor, dim=1)
            neg_tensor = F.normalize(neg_tensor, dim=1)
            pos_gt = F.normalize(pos_gt, dim=1)
            neg_gt = F.normalize(neg_gt, dim=1)
            
            # Compute logits
            logits_pos = torch.matmul(pos_gt, pos_tensor.t()) / (self.temperature * 5)
            logits_neg = torch.matmul(neg_gt, neg_tensor.t()) / self.temperature
            
            # Compute loss
            labels_pos = torch.arange(pos).to(device)
            labels_neg = torch.arange(neg).to(device)
            
            loss = -F.cross_entropy(logits_pos, labels_pos) + F.cross_entropy(logits_neg, labels_neg)
            loss_all += loss
        
        return loss_all / len(pos_index)
    
    def train_step(self, source_data, target_data, pos_data, neg_data, pos_index, neg_index, epoch, max_epochs):
        """Single training step"""
        # Update GRL rate
        GradReverse.rate = min((epoch + 1) / max_epochs, self.grl_rate)
        
        for model in self.models:
            model.train()
        
        self.optimizer.zero_grad()
        
        # Encode
        encoded_source = self.encode(source_data)
        encoded_target = self.encode(target_data)
        encoded_pos = self.encode(pos_data)
        encoded_neg = self.encode(neg_data)
        
        # Regression loss with weighted MSE
        source_logits = self.reg_model(encoded_source)
        y = source_data.y.to(device)
        
        # Weighted MSE for D, P, H components
        mse_D = F.mse_loss(source_logits[:, 0], y[:, 0, 0])
        mse_P = F.mse_loss(source_logits[:, 1], y[:, 0, 1])
        mse_H = F.mse_loss(source_logits[:, 2], y[:, 0, 2])
        
        loss_reg = torch.sqrt(self.w_d * mse_D + self.w_p * mse_P + self.w_h * mse_H)
        
        # Domain loss
        source_labels = torch.zeros(len(source_data)).long().to(device)
        target_labels = torch.ones(len(target_data)).long().to(device)
        
        source_preds = self.domain_model(encoded_source)
        target_preds = self.domain_model(encoded_target)
        
        loss_s = self.loss_domain(source_preds, source_labels)
        loss_t = self.loss_domain(target_preds, target_labels)
        
        # Contrastive loss
        loss_con = self.contrastive_loss(encoded_pos, encoded_neg, pos_index, neg_index)
        
        # Total loss
        loss = self.alpha * (loss_s + loss_t) + loss_reg + self.beta * loss_con
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        """Evaluate on data"""
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            logits = self.predict(data).cpu().numpy()
            y = data.y.cpu().numpy()
            rmse = np.sqrt(mean_squared_error(y, logits, multioutput='raw_values'))
        
        return rmse
    
    def train(self, source_loader, target_loader, pos_loader, neg_loader, 
              pos_index, neg_index, epochs=200, report_interval=10):
        """
        Full training loop.
        
        Returns:
            Dictionary with best RMSE values
        """
        best_rmse = {'D': float('inf'), 'P': float('inf'), 'H': float('inf')}
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            for source, target, pos, neg in zip(source_loader, target_loader, pos_loader, neg_loader):
                loss = self.train_step(source, target, pos, neg, pos_index, neg_index, epoch, epochs)
            
            # Evaluation
            for source, target in zip(source_loader, target_loader):
                source_rmse = self.evaluate(source)
                target_rmse = self.evaluate(target)
            
            # Update best
            if target_rmse[0] < best_rmse['D']:
                best_rmse['D'] = target_rmse[0]
            if target_rmse[1] < best_rmse['P']:
                best_rmse['P'] = target_rmse[1]
            if target_rmse[2] < best_rmse['H']:
                best_rmse['H'] = target_rmse[2]
                best_epoch = epoch
            
            # Report to NNI
            if NNI_AVAILABLE and epoch % report_interval == 0:
                # Combined metric (average RMSE)
                combined_rmse = (target_rmse[0] + target_rmse[1] + target_rmse[2]) / 3
                nni.report_intermediate_result(float(combined_rmse))
            
            # Logging
            if epoch % report_interval == 0:
                logger.info(f"Epoch {epoch}: Target RMSE - D={target_rmse[0]:.4f}, "
                           f"P={target_rmse[1]:.4f}, H={target_rmse[2]:.4f}")
        
        return {
            'best_rmse_D': float(best_rmse['D']),
            'best_rmse_P': float(best_rmse['P']),
            'best_rmse_H': float(best_rmse['H']),
            'best_epoch': best_epoch
        }


# ============================================================================
# NNI Trial Function
# ============================================================================

def run_trial(params):
    """
    Run a single trial with given hyperparameters.
    Called by NNI for each trial.
    """
    logger.info(f"Running trial with params: {params}")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load data
    dataset_source, dataset_target, dataset_pos, dataset_neg, pos_index, neg_index = load_data()
    
    # Create data loaders
    source_loader = DataLoader(dataset_source, batch_size=len(dataset_source), shuffle=True)
    target_loader = DataLoader(dataset_target, batch_size=len(dataset_target))
    pos_loader = DataLoader(dataset_pos, batch_size=len(dataset_pos))
    neg_loader = DataLoader(dataset_neg, batch_size=len(dataset_neg))
    
    # Build and train model
    model = GTPolySol(params)
    results = model.train(
        source_loader, target_loader, pos_loader, neg_loader,
        pos_index, neg_index, epochs=200
    )
    
    # Report final result to NNI
    combined_rmse = (results['best_rmse_D'] + results['best_rmse_P'] + results['best_rmse_H']) / 3
    
    if NNI_AVAILABLE:
        nni.report_final_result(float(combined_rmse))
    
    return results


# ============================================================================
# NNI Experiment Configuration
# ============================================================================

def create_nni_config():
    """
    Create NNI experiment configuration for TPE optimization.
    """
    config = {
        "experimentName": "GT-PolySol-HPO",
        "searchSpaceFile": "search_space.json",
        "trialCommand": "python train_with_nni.py",
        "trialGpuNumber": 1,
        "trialConcurrency": 1,
        "maxTrialNumber": 80,
        "maxExperimentDuration": "24h",
        "tuner": {
            "name": "TPE",
            "classArgs": {
                "optimize_mode": "minimize"
            }
        },
        "trainingService": {
            "platform": "local"
        }
    }
    return config


def run_nni_experiment():
    """
    Start NNI experiment with TPE tuner.
    """
    if not NNI_AVAILABLE:
        logger.error("NNI is not installed. Please install with: pip install nni")
        return None
    
    # Save search space
    with open('search_space.json', 'w') as f:
        json.dump(SEARCH_SPACE, f, indent=2)
    
    # Create experiment
    experiment = Experiment('local')
    
    # Configure tuner (TPE)
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args = {'optimize_mode': 'minimize'}
    
    # Configure search space
    experiment.config.search_space = SEARCH_SPACE
    
    # Configure trial
    experiment.config.trial_command = 'python train_with_nni.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.trial_concurrency = 1
    experiment.config.max_trial_number = 80
    experiment.config.max_experiment_duration = '24h'
    
    # Start experiment
    experiment.start(8080)
    
    logger.info(f"NNI experiment started. Access web UI at: http://localhost:8080")
    
    return experiment


# ============================================================================
# Standalone Grid Search (Alternative to NNI)
# ============================================================================

class GridSearchOptimizer:
    """
    Standalone grid search optimizer when NNI is not available.
    Implements systematic hyperparameter search with result tracking.
    """
    
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.results = []
        
    def generate_configs(self):
        """Generate all parameter combinations"""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k]['_value'] for k in keys]
        
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        
        return configs
    
    def run(self, max_trials=80):
        """Run grid search with limited trials"""
        configs = self.generate_configs()
        
        # Randomly sample if too many configs
        if len(configs) > max_trials:
            configs = random.sample(configs, max_trials)
        
        logger.info(f"Running {len(configs)} trials...")
        
        best_result = None
        best_rmse = float('inf')
        
        for i, config in enumerate(configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {i+1}/{len(configs)}")
            logger.info(f"Config: {config}")
            
            try:
                result = run_trial(config)
                combined_rmse = (result['best_rmse_D'] + result['best_rmse_P'] + result['best_rmse_H']) / 3
                
                self.results.append({
                    'trial_id': i,
                    'params': config,
                    'rmse_D': result['best_rmse_D'],
                    'rmse_P': result['best_rmse_P'],
                    'rmse_H': result['best_rmse_H'],
                    'combined_rmse': combined_rmse
                })
                
                if combined_rmse < best_rmse:
                    best_rmse = combined_rmse
                    best_result = {
                        'params': config,
                        'results': result
                    }
                    logger.info(f"New best! RMSE: D={result['best_rmse_D']:.4f}, "
                               f"P={result['best_rmse_P']:.4f}, H={result['best_rmse_H']:.4f}")
                               
            except Exception as e:
                logger.error(f"Trial {i} failed: {e}")
                continue
        
        return best_result
    
    def save_results(self, filename='hpo_results.json'):
        """Save all results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filename}")


# ============================================================================
# TPE-like Random Search Optimizer
# ============================================================================

class TPEOptimizer:
    """
    Simplified TPE-like optimizer for hyperparameter search.
    Uses a two-stage approach: exploration followed by exploitation.
    """
    
    def __init__(self, search_space, n_startup=20, gamma=0.25):
        """
        Args:
            search_space: Dictionary defining search space
            n_startup: Number of random trials before TPE kicks in
            gamma: Fraction of best trials to consider as "good"
        """
        self.search_space = search_space
        self.n_startup = n_startup
        self.gamma = gamma
        self.trials = []
        self.best_params = None
        self.best_rmse = float('inf')
        
    def sample_params(self):
        """Sample parameters from search space"""
        params = {}
        for key, spec in self.search_space.items():
            if spec['_type'] == 'choice':
                params[key] = random.choice(spec['_value'])
        return params
    
    def sample_params_tpe(self):
        """
        Sample parameters using TPE-like approach.
        Prefers values from good trials.
        """
        if len(self.trials) < self.n_startup:
            return self.sample_params()
        
        # Sort trials by performance
        sorted_trials = sorted(self.trials, key=lambda x: x['rmse'])
        n_good = max(1, int(len(sorted_trials) * self.gamma))
        
        good_trials = sorted_trials[:n_good]
        
        params = {}
        for key, spec in self.search_space.items():
            if spec['_type'] == 'choice':
                # Count frequency in good trials
                good_values = [t['params'][key] for t in good_trials]
                
                # With 70% prob, sample from good values, else random
                if random.random() < 0.7 and good_values:
                    params[key] = random.choice(good_values)
                else:
                    params[key] = random.choice(spec['_value'])
        
        return params
    
    def run(self, max_trials=80):
        """Run TPE optimization"""
        logger.info(f"Starting TPE optimization with {max_trials} trials")
        
        for trial_id in range(max_trials):
            # Sample parameters
            if trial_id < self.n_startup:
                params = self.sample_params()
            else:
                params = self.sample_params_tpe()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial_id + 1}/{max_trials}")
            logger.info(f"Params: {params}")
            
            try:
                result = run_trial(params)
                combined_rmse = (result['best_rmse_D'] + result['best_rmse_P'] + result['best_rmse_H']) / 3
                
                self.trials.append({
                    'trial_id': trial_id,
                    'params': params,
                    'rmse': combined_rmse,
                    'rmse_D': result['best_rmse_D'],
                    'rmse_P': result['best_rmse_P'],
                    'rmse_H': result['best_rmse_H']
                })
                
                if combined_rmse < self.best_rmse:
                    self.best_rmse = combined_rmse
                    self.best_params = params.copy()
                    self.best_result = result
                    logger.info(f"★ New best! Combined RMSE: {combined_rmse:.4f}")
                    
            except Exception as e:
                logger.error(f"Trial {trial_id} failed: {e}")
                continue
        
        return {
            'best_params': self.best_params,
            'best_result': self.best_result,
            'all_trials': self.trials
        }
    
    def save_results(self, filename='tpe_results.json'):
        """Save optimization results"""
        results = {
            'best_params': self.best_params,
            'best_rmse': self.best_rmse,
            'all_trials': self.trials
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def plot_trajectories(self, save_path='tpe_trajectories.png'):
        """
        Plot TPE optimization trajectories similar to Fig. S3 in the paper.
        """
        import matplotlib.pyplot as plt
        
        if len(self.trials) == 0:
            logger.warning("No trials to plot")
            return
        
        # Parameters to plot
        params_to_plot = ['alpha', 'beta', 'w_d', 'w_p', 'w_h']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        trial_indices = [t['trial_id'] for t in self.trials]
        rmse_values = [t['rmse'] for t in self.trials]
        
        # Compute best-so-far RMSE
        best_so_far = []
        current_best = float('inf')
        for r in rmse_values:
            current_best = min(current_best, r)
            best_so_far.append(current_best)
        
        for idx, param in enumerate(params_to_plot):
            ax = axes[idx]
            
            # Get parameter values
            param_values = [t['params'].get(param, 0) for t in self.trials]
            
            # Plot sampled values
            ax.scatter(trial_indices, param_values, c='gray', alpha=0.6, label='Sampled')
            
            # Plot best-so-far RMSE on secondary axis
            ax2 = ax.twinx()
            ax2.plot(trial_indices, best_so_far, 'b-', linewidth=2, label='Best RMSE')
            ax2.set_ylabel('Validation RMSE', color='blue')
            
            ax.set_xlabel('TPE trial index')
            ax.set_ylabel(f'{param} value (log scale)')
            ax.set_yscale('log')
            ax.set_title(f'TPE trajectory of {param}')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # Hide last subplot if not needed
        if len(params_to_plot) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trajectory plot saved to {save_path}")


# ============================================================================
# Main Entry Points
# ============================================================================

def main_nni():
    """Main function for NNI integration"""
    if NNI_AVAILABLE:
        # Get parameters from NNI
        params = nni.get_next_parameter()
        run_trial(params)
    else:
        # Run with default parameters
        logger.info("Running with default parameters (NNI not available)")
        run_trial(DEFAULT_PARAMS)


def main_standalone():
    """
    Main function for standalone hyperparameter search without NNI.
    Uses TPE-like optimization.
    """
    logger.info("="*70)
    logger.info("GT-PolySol Hyperparameter Optimization (Standalone Mode)")
    logger.info("="*70)
    
    # Use simplified search space for faster testing
    simple_search_space = {
        "alpha": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
        "beta": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
        "w_d": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
        "w_p": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
        "w_h": {"_type": "choice", "_value": [0.01, 0.05, 0.1, 0.5, 1.0]},
    }
    
    # Run TPE optimization
    optimizer = TPEOptimizer(simple_search_space, n_startup=10, gamma=0.25)
    results = optimizer.run(max_trials=80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer.save_results(f'tpe_results_{timestamp}.json')
    optimizer.plot_trajectories(f'tpe_trajectories_{timestamp}.png')
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Best Parameters: {results['best_params']}")
    logger.info(f"Best RMSE - D: {results['best_result']['best_rmse_D']:.4f}, "
               f"P: {results['best_result']['best_rmse_P']:.4f}, "
               f"H: {results['best_result']['best_rmse_H']:.4f}")
    
    return results


if __name__ == "__main__":
    # Check if running under NNI
    if NNI_AVAILABLE and os.environ.get('NNI_PLATFORM'):
        main_nni()
    else:
        main_standalone()
