import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, RDKFingerprint
from rdkit import RDLogger  
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from tqdm import tqdm
from itertools import product
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import json
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_PATH = '/mdspace/polaris_ml_challenge/data/potency/UNBLINDED_DATA/train_data_mers.csv'


def generate_fingerprints(smiles, fp_type, **kwargs):
    """
    Generate different types of molecular fingerprints
    
    Args:
        smiles: SMILES string
        fp_type: Type of fingerprint ('morgan', 'rdkit', 'morgan_rdkit')
        **kwargs: Additional parameters for fingerprint generation
    
    Returns:
        numpy array of fingerprint bits
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if fp_type == 'morgan':
        radius = kwargs.get('radius', 2)
        nBits = kwargs.get('nBits', 2048)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return np.array(fp)
    
    elif fp_type == 'rdkit':
        maxPath = kwargs.get('maxPath', 7)
        nBits = kwargs.get('nBits', 2048)
        fp = RDKFingerprint(mol, maxPath=maxPath, fpSize=nBits)
        return np.array(fp)
    
    elif fp_type == 'morgan_rdkit':
        # Combined Morgan + RDKit fingerprints
        morgan_radius = kwargs.get('morgan_radius', 2)
        morgan_nBits = kwargs.get('morgan_nBits', 1024)
        rdkit_maxPath = kwargs.get('rdkit_maxPath', 7)
        rdkit_nBits = kwargs.get('rdkit_nBits', 1024)
        
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_nBits)
        rdkit_fp = RDKFingerprint(mol, maxPath=rdkit_maxPath, fpSize=rdkit_nBits)
        
        # Concatenate fingerprints
        combined_fp = np.concatenate([np.array(morgan_fp), np.array(rdkit_fp)])
        return combined_fp
    
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")


class FlexibleBayesianNNModel(nn.Module):
    def __init__(self, input_features, hidden_layers, kl_weight=0.01, 
                 prior_mu=0.0, prior_sigma=0.1, dropout_rate=0.1,
                 normalization='layer'):
        """
        Flexible Bayesian NN that supports 0-3 hidden layers
        
        Args:
            input_features: Number of input features
            hidden_layers: List of hidden layer sizes (empty list = no hidden layers)
            kl_weight: Weight for KL divergence term
            prior_mu: Prior mean for Bayesian weights
            prior_sigma: Prior standard deviation for Bayesian weights
            dropout_rate: Dropout rate
            normalization: Type of normalization ('layer', 'none')
        """
        super(FlexibleBayesianNNModel, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.kl_weight = kl_weight
        self.dropout_rate = dropout_rate
        self.normalization = normalization
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Input layer
        if len(hidden_layers) == 0:
            # Direct input to output (linear regression)
            self.output_layer = bnn.BayesLinear(
                prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=input_features, out_features=1
            )
        else:
            # First hidden layer
            self.layers.append(bnn.BayesLinear(
                prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=input_features, out_features=hidden_layers[0]
            ))
            self._add_normalization_layer(hidden_layers[0])
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            
            # Additional hidden layers
            for i in range(1, len(hidden_layers)):
                self.layers.append(bnn.BayesLinear(
                    prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=hidden_layers[i-1], out_features=hidden_layers[i]
                ))
                self._add_normalization_layer(hidden_layers[i])
                self.dropout_layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            self.output_layer = bnn.BayesLinear(
                prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=hidden_layers[-1], out_features=1
            )


    def _add_normalization_layer(self, features):
        """Add appropriate normalization layer"""
        if self.normalization == 'layer':
            self.norm_layers.append(nn.LayerNorm(features))
        else:
            self.norm_layers.append(nn.Identity())  # Placeholder


    def forward(self, x):
        # If no hidden layers, go directly to output
        if len(self.hidden_layers) == 0:
            return self.output_layer(x)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply normalization
            if self.normalization == 'layer':
                x = F.leaky_relu(self.norm_layers[i](x))
            else:
                x = F.leaky_relu(x)
            
            # Apply dropout
            x = self.dropout_layers[i](x)
        
        # Output layer
        x = self.output_layer(x)
        return x


def combined_robust_loss(predictions, targets, mse_weight=0.3, mae_weight=0.7):
    """
    Combines MSE (for accuracy) with MAE (for robustness)
    This is ideal for BNNs as it provides both accurate predictions and robustness to outliers
    """
    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)
    return mse_weight * mse + mae_weight * mae


def convert_pIC50(value):
    if isinstance(value, str):
        if value.startswith('<'):
            return None
        elif value.startswith('>'):
            return None  # Drop the value
        else:
            return float(value)
    return float(value)


def prepare_data_with_fingerprints(path, fp_config):
    """
    Prepare data with specified fingerprint configuration
    
    Args:
        path: Path to CSV file
        fp_config: Dictionary containing fingerprint configuration
    
    Returns:
        X, y arrays and input feature size
    """
    data = pd.read_csv(path)
    data.rename(columns={'pIC50 (MERS-CoV Mpro)': 'pIC50'}, inplace=True)
    data = data[['pIC50', 'Molecule Name', 'CXSMILES']]
    
    # Generate fingerprints based on configuration
    print(f"Generating {fp_config['type']} fingerprints...")
    
    fingerprints = []
    valid_indices = []
    
    for idx, smiles in enumerate(data['CXSMILES']):
        if pd.notna(smiles):
            fp = generate_fingerprints(smiles, fp_config['type'], **fp_config)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
    
    if not fingerprints:
        raise ValueError("No valid fingerprints generated")
    
    # Filter data to valid molecules
    data = data.iloc[valid_indices].reset_index(drop=True)
    data['Fingerprint'] = fingerprints
    
    # Clean pIC50 data
    data.dropna(subset=['pIC50'], inplace=True)
    data['pIC50'] = data['pIC50'].apply(convert_pIC50)
    data.dropna(subset=['pIC50'], inplace=True)
    data['pIC50'] = data['pIC50'].astype(float)
    
    X = np.array(list(data["Fingerprint"]))
    y = data["pIC50"].values
    input_features = X.shape[1]
    
    print(f"Generated fingerprints with {input_features} features for {len(X)} molecules")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    return X, y, input_features


def evaluate_model(model, X_test, y_test):
    """Evaluate model with multiple metrics"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        
        # Primary loss
        test_loss = combined_robust_loss(predictions, y_test).item()
        
        # Additional metrics
        pred_np = predictions.cpu().numpy()
        y_np = y_test.cpu().numpy()
        
        mae = mean_absolute_error(y_np, pred_np)
        r2 = r2_score(y_np, pred_np)
        rmse = np.sqrt(np.mean((pred_np - y_np) ** 2))
        
        return test_loss, mae, r2, rmse


def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val, 
                                   optimizer, scheduler, n_epochs, 
                                   batch_size, patience=50, verbose=False):
    """Train model with early stopping and return validation loss"""
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    kl_loss_fn = bnn.BKLLoss(reduction='mean')
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(batch_X).squeeze()
                kl = kl_loss_fn(model)
                primary_loss = combined_robust_loss(output, batch_y)
                total_loss = primary_loss + model.kl_weight * kl

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val).squeeze()
            val_kl = kl_loss_fn(model)
            val_primary = combined_robust_loss(val_output, y_val)
            val_loss = val_primary + model.kl_weight * val_kl
            val_loss_item = val_loss.item()
        
        scheduler.step(val_loss_item)
        
        # Early stopping
        if val_loss_item < best_val_loss:
            best_val_loss = val_loss_item
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss_item:.4f}")
            
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_loss


def suggest_fingerprint_config(trial):
    """
    Suggest fingerprint configuration for optimization
    """
    fp_type = trial.suggest_categorical('fp_type', ['morgan', 'rdkit', 'morgan_rdkit'])
    
    config = {'type': fp_type}
    
    if fp_type == 'morgan':
        config['radius'] = trial.suggest_categorical('morgan_only_radius', [2, 3])
        config['nBits'] = trial.suggest_categorical('morgan_only_nBits', [1024, 2048])
        
    elif fp_type == 'rdkit':
        config['maxPath'] = trial.suggest_categorical('rdkit_only_maxPath', [5, 6, 7, 8])
        config['nBits'] = trial.suggest_categorical('rdkit_only_nBits', [256, 512, 1024])
        
    elif fp_type == 'morgan_rdkit':
        # Morgan parameters for combined
        config['morgan_radius'] = trial.suggest_categorical('combined_morgan_radius', [2, 3])
        config['morgan_nBits'] = trial.suggest_categorical('combined_morgan_nBits', [512, 1024])  # Smaller since combined
        
        # RDKit parameters for combined
        config['rdkit_maxPath'] = trial.suggest_categorical('combined_rdkit_maxPath', [5, 6, 7, 8])
        config['rdkit_nBits'] = trial.suggest_categorical('combined_rdkit_nBits', [256, 512, 1024])
    
    return config


def suggest_hidden_layers(trial):
    """Suggest hidden layer configuration (0–3 layers) using fixed space."""
    num_layers = trial.suggest_int('num_hidden_layers', 0, 3)
    layer_sizes = [64, 128, 256, 512, 1024]
    hidden_layers = []

    for i in range(num_layers):
        size = trial.suggest_categorical(f'hidden_dim_{i}', layer_sizes)
        hidden_layers.append(size)

    return hidden_layers


def optuna_objective(trial, data_path):
    """Optuna objective function for hyperparameter optimization with 5-fold CV"""
    
    # Suggest fingerprint configuration
    fp_config = suggest_fingerprint_config(trial)
    
    # Prepare data with selected fingerprints
    try:
        X, y, input_features = prepare_data_with_fingerprints(data_path, fp_config)
    except Exception as e:
        print(f"Error generating fingerprints: {e}")
        return float('inf')  # Return worst possible score
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Suggest architecture
    hidden_layers = suggest_hidden_layers(trial)
    
    # Other hyperparameters
    kl_weight = trial.suggest_loguniform('kl_weight', 1e-4, 1e-1)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    
    prior_mu = trial.suggest_uniform('prior_mu', -0.2, 0.2)
    prior_sigma = trial.suggest_uniform('prior_sigma', 0.01, 0.2)
    
    # BNN-appropriate normalization choice
    normalization = trial.suggest_categorical('normalization', ['layer', 'none'])
    
    # 5-fold Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_losses = []
    
    fold = 0
    for train_idx, val_idx in kf.split(X_tensor.cpu()):
        fold += 1
        X_fold_train = X_tensor[train_idx]
        X_fold_val = X_tensor[val_idx]
        y_fold_train = y_tensor[train_idx]
        y_fold_val = y_tensor[val_idx]

        model = FlexibleBayesianNNModel(
            input_features=input_features,
            hidden_layers=hidden_layers,
            kl_weight=kl_weight,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            dropout_rate=dropout_rate,
            normalization=normalization
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.7, patience=20, verbose=False)
        
        val_loss = train_model_with_early_stopping(
            model, X_fold_train, y_fold_train, X_fold_val, y_fold_val,
            optimizer, scheduler, n_epochs=300, batch_size=batch_size, patience=30
        )
        
        cv_losses.append(val_loss)
        
        # Clean up GPU memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    
    mean_cv_loss = np.mean(cv_losses)
    print(f"Trial {trial.number}: 5-fold CV loss = {mean_cv_loss:.4f}")
    
    return mean_cv_loss


def optimize_with_optuna(data_path, n_trials=10000, n_jobs=4):
    """Optimize hyperparameters using Optuna with 5-fold CV (single process recommended for heavy GPU jobs)"""
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
        # No pruner for full trial evaluation
    )
    
    study.optimize(
        lambda trial: optuna_objective(trial, data_path),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=n_jobs 
    )
    
    return study.best_params, study.best_value


def reconstruct_fingerprint_config(params):
    """Reconstruct fingerprint configuration from Optuna parameters"""
    fp_type = params['fp_type']
    config = {'type': fp_type}
    
    if fp_type == 'morgan':
        config['radius'] = params['morgan_only_radius']
        config['nBits'] = params['morgan_only_nBits']
        
    elif fp_type == 'rdkit':
        config['maxPath'] = params['rdkit_only_maxPath']
        config['nBits'] = params['rdkit_only_nBits']
        
    elif fp_type == 'morgan_rdkit':
        config['morgan_radius'] = params['combined_morgan_radius']
        config['morgan_nBits'] = params['combined_morgan_nBits']
        config['rdkit_maxPath'] = params['combined_rdkit_maxPath']
        config['rdkit_nBits'] = params['combined_rdkit_nBits']
    
    return config


def reconstruct_hidden_layers(params):
    """Reconstruct hidden layers list from Optuna parameters"""
    num_layers = params['num_hidden_layers']
    hidden_layers = []
    
    for i in range(num_layers):
        hidden_layers.append(params[f'hidden_dim_{i}'])
    
    return hidden_layers


def train_final_model_on_all_data(best_params, data_path):
    """Train final model using ALL available data points"""
    
    # Reconstruct fingerprint configuration
    fp_config = reconstruct_fingerprint_config(best_params)
    
    # Prepare data with optimal fingerprints
    print("Preparing final dataset with optimal fingerprints...")
    X, y, input_features = prepare_data_with_fingerprints(data_path, fp_config)
    
    print(f"Training final model on ALL {len(X)} data points")
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Split ALL data into train/validation for monitoring (80/20 split)
    train_size = int(0.8 * len(X_tensor))
    indices = torch.randperm(len(X_tensor))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    
    # Reconstruct hidden layers
    hidden_layers = reconstruct_hidden_layers(best_params)
    
    # Create model with optimal hyperparameters
    model = FlexibleBayesianNNModel(
        input_features=input_features,
        hidden_layers=hidden_layers,
        kl_weight=best_params['kl_weight'],
        prior_mu=best_params['prior_mu'],
        prior_sigma=best_params['prior_sigma'],
        dropout_rate=best_params['dropout_rate'],
        normalization=best_params.get('normalization', 'layer')
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.7, patience=30, verbose=True)
    
    # Print final configuration
    print(f"\nFinal Model Configuration:")
    print(f"Fingerprint: {fp_config}")
    print(f"Architecture: Input({input_features}) -> {' -> '.join(map(str, hidden_layers))} -> Output(1)")
    print(f"Total dataset size: {len(X_tensor)}")
    print(f"Training monitoring split: {len(X_train)} train, {len(X_val)} validation")
    
    # Train final model with monitoring
    print("\nTraining final model on all data...")
    final_val_loss = train_model_with_early_stopping(
        model, X_train, y_train, X_val, y_val,
        optimizer, scheduler, 
        n_epochs=500, batch_size=best_params['batch_size'], patience=50, verbose=True
    )
    
    # Now retrain on ALL data without validation split for the absolute final model
    print("\nRetraining on complete dataset for final model...")
    
    # Create fresh model for final training
    final_model = FlexibleBayesianNNModel(
        input_features=input_features,
        hidden_layers=hidden_layers,
        kl_weight=best_params['kl_weight'],
        prior_mu=best_params['prior_mu'],
        prior_sigma=best_params['prior_sigma'],
        dropout_rate=best_params['dropout_rate'],
        normalization=best_params.get('normalization', 'layer')
    ).to(device)
    
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    
    # Train on all data for a fixed number of epochs
    train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    kl_loss_fn = bnn.BKLLoss(reduction='mean')
    scaler = torch.amp.GradScaler('cuda')
    
    # Train for reasonable number of epochs
    final_epochs = 200
    print(f"Training final model for {final_epochs} epochs on all {len(X_tensor)} data points...")
    
    for epoch in range(final_epochs):
        final_model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            final_optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = final_model(batch_X).squeeze()
                kl = kl_loss_fn(final_model)
                primary_loss = combined_robust_loss(output, batch_y)
                total_loss = primary_loss + final_model.kl_weight * kl

            scaler.scale(total_loss).backward()
            scaler.step(final_optimizer)
            scaler.update()
            epoch_loss += total_loss.item()
        
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Final training - Epoch {epoch+1}/{final_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate final model on all data (for reference)
    final_model.eval()
    with torch.no_grad():
        all_predictions = final_model(X_tensor).squeeze()
        final_loss = combined_robust_loss(all_predictions, y_tensor).item()
        
        pred_np = all_predictions.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        
        mae = mean_absolute_error(y_np, pred_np)
        r2 = r2_score(y_np, pred_np)
        rmse = np.sqrt(np.mean((pred_np - y_np) ** 2))
    
    print(f"\nFinal Model Performance on All Data:")
    print(f"Training Loss (Combined Robust): {final_loss:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return final_model, best_params, fp_config


if __name__ == "__main__":
    RDLogger.DisableLog('rdApp.*')
    
    print("Starting comprehensive hyperparameter optimization with 5-fold CV...")
    print("Testing fingerprint types:")
    print("- Morgan (radius 2-3, nBits 1024/2048)")
    print("- RDKit (maxPath 5-8, nBits 256/512/1024)")
    print("- Morgan+RDKit combined")
    print("- Architecture: 0-3 hidden layers")
    print("- Cross-validation: 5-fold")
    
    # Optimize hyperparameters with 5-fold CV
    best_params, best_score = optimize_with_optuna(DATA_PATH, n_trials=10000)
    
    print(f"\nBest 5-fold CV Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Train final model on ALL data
    final_model, final_params, fp_config = train_final_model_on_all_data(best_params, DATA_PATH)
    
    # Save model and hyperparameters
    print("\nSaving final model...")
    torch.save(final_model.state_dict(), 'MERS_potency_BNN_model_final_all_data.pth')
    
    # Save complete configuration
    final_config = {
        'model_params': final_params,
        'fingerprint_config': fp_config,
        'best_cv_score': best_score,
        'training_info': {
            'cv_folds': 5,
            'trained_on_all_data': True,
            'final_epochs': 200
        }
    }
    
    with open('MERS_potency_BNN_complete_config_final.json', 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Final model trained on all available data points")
    print(f"Configuration saved to: MERS_potency_BNN_complete_config_final.json")
    print(f"Model weights saved to: MERS_potency_BNN_model_final_all_data.pth")