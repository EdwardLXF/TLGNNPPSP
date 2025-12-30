# Transfer Learning with Graph Neural Networks to Predict Polymer Solubility Parameters

<p align="center"> 
<img src="FIG1_hz_1222_1.jpg" width="800">
</p>

## Overview

GT-PolySol is a graph neural network-based framework that uses transfer learning to predict Hansen Solubility Parameters (HSP) for polymers. The model addresses the challenge of limited polymer HSP data by transferring knowledge from abundant solvent data to polymer prediction.

### Key Features

- **Transfer Learning**: Leverages solvent HSP data to predict polymer properties
- **Graph Neural Networks**: Molecular structure representation using GCN
- **PPTA Framework**: Polymer Property Trend Approximation for capturing polymer plateau behavior
- **Domain Adaptation**: Gradient reversal layer for domain-invariant representations

## Environment Settings

```bash
# Create conda environment
conda create -n gtpolysol python=3.8
conda activate gtpolysol

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| Python | 3.8 |
| PyTorch | 1.1.0+ |
| PyTorch Geometric | 2.0+ |
| NumPy | 1.16.2+ |
| SciPy | 1.3.1+ |
| NetworkX | 2.4+ |
| scikit-learn | 0.21.3+ |
| RDKit | 2020.09+ |
| pandas | 1.0+ |
| matplotlib | 3.0+ |
| tqdm | 4.0+ |

## Project Structure

```
GT-PolySol/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── FIG1_hz_1222_1.jpg          # Framework illustration
│
├── data/                        # Data files
│   ├── solvent_HSP_HSPiP_hz1010.csv    # Solvent HSP data (source domain)
│   ├── polymer_HSP_HSPiP_hz1010.csv    # Polymer HSP data (target domain)
│   └── neg_sample.pkl                   # Fragment dictionary for PPTA
│
├── src/                         # Source code
│   ├── model.py                 # GNN model architecture
│   ├── load_datasets.py         # Data loading and preprocessing
│   ├── train.py                 # Main training script
│   ├── test1.py                 # Inference and case study
│   ├── create_neg_sample.py     # Generate fragment samples
│   ├── chem.py                  # Chemistry utilities
│   └── baseline.py              # Baseline comparisons
│
├── experiments/                 # Supplementary experiments
│   ├── train_soft_contrastive.py    # Soft contrastive learning (Appendix G)
│   ├── train_nni_hpo.py             # NNI hyperparameter search (Appendix C)
│   ├── nni_config.json              # NNI configuration
│   └── search_space.json            # Hyperparameter search space
│
└── saved_models/                # Trained model checkpoints
    └── best_model.pth
```

## Quick Start

### 1. Training

Train the GT-PolySol model from scratch:

```bash
python train.py
```

### 2. Inference

Run predictions on new polymers using the trained model:

```bash
python test1.py
```

### Example Predictions

| SMILES | D | P | H |
|:-------|:---:|:---:|:---:|
| S1C=CC=C1C1C2N=CC(OCC(CCCCCC)CCCCCCCC)=NC2C=C(F)C=1F | 19.19 | 1.84 | 3.77 |
| S1C=CC=C1C1C2N=CC(OCCOCCOCCOCCOCCOCCOC)=NC2C=C(F)C=1F | 18.56 | 4.04 | 5.88 |
| S1C=CC=C1C1C2N=CC(OC(COCCOCCOCCOC)COCCOCCOCCOC)=NC2C=C(F)C=1F | 19.39 | 9.27 | 7.13 |
| S1C=CC=C1C1C2N=CC(OCC(COCCOCCOCCOC)COCCOCCOCCOC)=NC2C=C(F)C=1F | 19.61 | 9.60 | 7.08 |
| S1C=C2C(F)=C(C(OCC(CC)CCCC)=O)SC2=C1C1SC2C(C3SC(CC(CCCC)CC)=CC=3)=C3C=CSC3=C(C3SC(CC(CCCC)CC)=CC=3)C=2C=1 | 18.69 | 3.39 | 4.42 |
| C(C1=CC(F)=C(CC(CC)CCCC)S1)1=C2C=CSC2=C(C2=CC(F)=C(CC(CCCC)CC)S2)C2C=C(C3=CC=C(C4SC(C5=CC=CS5)=C5C(=O)C6=C(C(CCCC)CC)SC(CC(CCCC)CC)=C6C(=O)C=45)S3)SC1=2 | 18.32 | 2.74 | 4.19 |

## Supplementary Experiments

### Soft Contrastive Learning (Appendix G)

Compare hard vs. soft contrastive learning approaches:

```bash
python experiments/train_soft_contrastive.py
```

This implements the soft contrastive variant where fragment weights are based on chemical similarity:

$$w_{ij} = \exp\left(-\frac{d_{chem}(i,j)}{\sigma}\right)$$

### Hyperparameter Optimization (Appendix C)

Run TPE-based hyperparameter search:

```bash
# Standalone mode
python experiments/train_nni_hpo.py

# With NNI framework
pip install nni
nnictl create --config experiments/nni_config.json
```

**Search Space:**

| Parameter | Values |
|-----------|--------|
| α | {0.01, 0.05, 0.1, 0.5, 1.0} |
| β | {0.01, 0.05, 0.1, 0.5, 1.0} |
| w_d | {0.01, 0.05, 0.1, 0.5, 1.0} |
| w_p | {0.01, 0.05, 0.1, 0.5, 1.0} |
| w_h | {0.01, 0.05, 0.1, 0.5, 1.0} |

**Optimal Configuration:** α=0.1, β=0.1, w_d=0.05, w_p=0.05, w_h=0.1

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GT-PolySol Framework                    │
├─────────────────────────────────────────────────────────────┤
│  Source Domain (Solvents)    │    Target Domain (Polymers)  │
│          ↓                   │            ↓                 │
│    ┌─────────┐               │      ┌─────────┐             │
│    │  GCN    │               │      │  GCN    │             │
│    │ Encoder │───────────────┼──────│ Encoder │             │
│    └────┬────┘               │      └────┬────┘             │
│         │                    │           │                  │
│         ↓                    │           ↓                  │
│    ┌─────────┐          ┌────┴────┐ ┌─────────┐            │
│    │   HSP   │          │ Domain  │ │  PPTA   │            │
│    │Regressor│          │Classifier│ │Contrastive│          │
│    └─────────┘          └─────────┘ └─────────┘            │
│         ↓                    ↓           ↓                  │
│      L_reg              L_domain     L_contrast             │
└─────────────────────────────────────────────────────────────┘
```

## Results

### Performance Comparison

| Method | RMSE_D | RMSE_P | RMSE_H |
|--------|:------:|:------:|:------:|
| Group Contribution | 0.89 | 1.45 | 1.12 |
| GCN (no transfer) | 0.65 | 1.23 | 0.87 |
| **GT-PolySol** | **0.41** | **0.97** | **0.49** |

### Ablation Study

| Variant | Description | RMSE_D | RMSE_P | RMSE_H |
|---------|-------------|:------:|:------:|:------:|
| GT-PolySol | Full model | 0.41 | 0.97 | 0.49 |
| GT-PolySol-1 | w/o PPTA & transfer | 0.58 | 1.15 | 0.72 |
| GT-PolySol-2 | w/o PPTA | 0.52 | 1.08 | 0.65 |
| GT-PolySol-3 | w/o transfer | 0.48 | 1.02 | 0.58 |


