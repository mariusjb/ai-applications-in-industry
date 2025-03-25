from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    num_epochs: int = 30,
    device: str = "cpu",
    patience: int = 5,
    verbose: bool = True,
    save_best: Optional[str] = None,
) -> nn.Module:
    """
    Generic training loop with early stopping.

    Parameters:
    - model: PyTorch model to train
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - optimizer: Optimizer instance (e.g., Adam, SGD)
    - loss_fn: Loss function (e.g., nn.MSELoss())
    - num_epochs: Total number of training epochs
    - device: 'cpu', 'cuda', or 'mps'
    - patience: Early stopping patience (epochs)
    - verbose: If True, print training progress
    - save_best: Path to save best model (optional)

    Returns:
    - The trained model (best version if early stopping triggered)
    """

    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        # Training
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.unsqueeze(1) if len(y_batch.shape) == 1 else y_batch

            optimizer.zero_grad()
            preds = model(X_batch)
            preds = preds.squeeze()
            y_batch = y_batch.squeeze()

            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.unsqueeze(1) if len(y_batch.shape) == 1 else y_batch

                preds = model(X_batch)
                preds = preds.squeeze()
                y_batch = y_batch.squeeze()
                val_loss += loss_fn(preds, y_batch).item()


        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if verbose:
            print(
                f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            if save_best:
                torch.save(best_model_state, save_best)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a trained model on a DataLoader and returns flattened predictions and targets.

    Parameters:
    - model: Trained PyTorch model
    - data_loader: DataLoader containing evaluation data
    - device: Device to run evaluation on ('cpu', 'cuda', or 'mps')

    Returns:
    - Tuple of (y_true, y_pred): Flattened NumPy arrays
    """
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if len(y_batch.shape) == 1:
                y_batch = y_batch.unsqueeze(1)

            outputs = model(X_batch)

            preds.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    y_pred = np.vstack(preds).flatten()
    y_true = np.vstack(actuals).flatten()
    return y_true, y_pred


def evaluate_multi_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a trained model on a DataLoader and returns multi-step predictions and targets.

    Parameters:
    - model: Trained PyTorch model with multi-step output
    - data_loader: DataLoader containing evaluation data
    - device: Device to run evaluation on ('cpu', 'cuda', or 'mps')

    Returns:
    - Tuple of (y_true, y_pred): NumPy arrays of shape [samples, forecast_steps]
    """
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            
            outputs = model(X_batch)
            
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(1) 
                
            preds.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(actuals)
    
    return y_true, y_pred