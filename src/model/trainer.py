"""
Training utilities for the rMD autoencoder.
"""
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from .autoencoder import rmsd_loss


class Trainer:
    """
    Trainer for the rMD autoencoder.
    """
    
    def __init__(self, model, optimizer, criterion, device=None):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        model : InformedAutoencoder
            The autoencoder model.
        optimizer : torch.optim.Optimizer
            The optimizer for training.
        criterion : RMDLoss
            The loss function.
        device : torch.device, optional
            The device to use for training.
            If None, will use CUDA if available.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_total_loss': [],
            'train_loss1': [],
            'train_loss2': [],
            'train_rmsd': [],
            'val_total_loss': [],
            'val_loss1': [],
            'val_loss2': [],
            'val_rmsd': []
        }
        
        # Best validation loss and corresponding epoch
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for training data.
            
        Returns:
        --------
        dict
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_rmsd = 0.0
        
        for coords, cv_coords in dataloader:
            # Move data to device
            coords = coords.to(self.device)
            cv_coords = cv_coords.to(self.device)
            
            # Forward pass
            latent, reconstructed = self.model(coords)
            
            # Calculate loss
            loss, loss1, loss2 = self.criterion(latent, cv_coords, reconstructed, coords)
            
            # Calculate RMSD
            with torch.no_grad():
                rmsd = rmsd_loss(reconstructed, coords)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_rmsd += rmsd.item()
        
        # Calculate average metrics
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_loss1 = total_loss1 / num_batches
        avg_loss2 = total_loss2 / num_batches
        avg_rmsd = total_rmsd / num_batches
        
        return {
            'total_loss': avg_loss,
            'loss1': avg_loss1,
            'loss2': avg_loss2,
            'rmsd': avg_rmsd
        }
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for validation data.
            
        Returns:
        --------
        dict
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_rmsd = 0.0
        
        with torch.no_grad():
            for coords, cv_coords in dataloader:
                # Move data to device
                coords = coords.to(self.device)
                cv_coords = cv_coords.to(self.device)
                
                # Forward pass
                latent, reconstructed = self.model(coords)
                
                # Calculate loss
                loss, loss1, loss2 = self.criterion(latent, cv_coords, reconstructed, coords)
                
                # Calculate RMSD
                rmsd = rmsd_loss(reconstructed, coords)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_rmsd += rmsd.item()
        
        # Calculate average metrics
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_loss1 = total_loss1 / num_batches
        avg_loss2 = total_loss2 / num_batches
        avg_rmsd = total_rmsd / num_batches
        
        return {
            'total_loss': avg_loss,
            'loss1': avg_loss1,
            'loss2': avg_loss2,
            'rmsd': avg_rmsd
        }
    
    def train(self, train_dataloader, val_dataloader, num_epochs, checkpoint_dir=None, patience=10):
        """
        Train the model for multiple epochs.
        
        Parameters:
        -----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for training data.
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for validation data.
        num_epochs : int
            Number of epochs to train.
        checkpoint_dir : str, optional
            Directory to save checkpoints.
            If None, checkpoints will not be saved.
        patience : int, optional
            Number of epochs to wait for improvement in validation loss before early stopping.
            
        Returns:
        --------
        dict
            Dictionary of training history.
        """
        # Create checkpoint directory if it doesn't exist
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        epochs_without_improvement = 0
        start_time = time.time()
        
        print(f"Training on {self.device}")
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate
            val_metrics = self.validate(val_dataloader)
            
            # Update history
            self.history['train_total_loss'].append(train_metrics['total_loss'])
            self.history['train_loss1'].append(train_metrics['loss1'])
            self.history['train_loss2'].append(train_metrics['loss2'])
            self.history['train_rmsd'].append(train_metrics['rmsd'])
            self.history['val_total_loss'].append(val_metrics['total_loss'])
            self.history['val_loss1'].append(val_metrics['loss1'])
            self.history['val_loss2'].append(val_metrics['loss2'])
            self.history['val_rmsd'].append(val_metrics['rmsd'])
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Loss1: {train_metrics['loss1']:.4f}, Loss2: {train_metrics['loss2']:.4f}, RMSD: {train_metrics['rmsd']:.4f} Å) - "
                  f"Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(Loss1: {val_metrics['loss1']:.4f}, Loss2: {val_metrics['loss2']:.4f}, RMSD: {val_metrics['rmsd']:.4f} Å) - "
                  f"Time: {elapsed:.0f}s")
            
            # Check for improvement
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save checkpoint
                if checkpoint_dir is not None:
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_best.pth")
                    self.save_checkpoint(checkpoint_path)
            else:
                epochs_without_improvement += 1
            
            # Save periodic checkpoint
            if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                self.save_checkpoint(checkpoint_path)
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break
        
        # Final message
        print(f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch+1})")
        
        return self.history
    
    def save_checkpoint(self, path):
        """
        Save a checkpoint of the model.
        
        Parameters:
        -----------
        path : str
            Path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        """
        Load a checkpoint.
        
        Parameters:
        -----------
        path : str
            Path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']
        
    def plot_history(self, save_path=None):
        """
        Plot training and validation metrics.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot.
            If None, the plot will not be saved.
        """
        plt.figure(figsize=(16, 12))
        
        # Plot total loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_total_loss'], label='Train')
        plt.plot(self.history['val_total_loss'], label='Validation')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Loss1 (latent to CV)
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_loss1'], label='Train')
        plt.plot(self.history['val_loss1'], label='Validation')
        plt.title('Loss1 (Latent to CV)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Loss2 (reconstruction)
        plt.subplot(2, 2, 3)
        plt.plot(self.history['train_loss2'], label='Train')
        plt.plot(self.history['val_loss2'], label='Validation')
        plt.title('Loss2 (Reconstruction)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot RMSD
        plt.subplot(2, 2, 4)
        plt.plot(self.history['train_rmsd'], label='Train')
        plt.plot(self.history['val_rmsd'], label='Validation')
        plt.title('RMSD (Å)')
        plt.xlabel('Epoch')
        plt.ylabel('RMSD')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            
        plt.show()


def create_dataloaders(coords_data, cv_coords_data, batch_size=64, train_ratio=0.8, shuffle=True):
    """
    Create training and validation dataloaders.
    
    Parameters:
    -----------
    coords_data : numpy.ndarray
        Array of flattened protein coordinates with shape (n_samples, n_atoms * 3).
    cv_coords_data : numpy.ndarray
        Array of collective variable coordinates with shape (n_samples, cv_dim).
    batch_size : int, optional
        Batch size for dataloaders.
    train_ratio : float, optional
        Ratio of data to use for training (remainder used for validation).
    shuffle : bool, optional
        Whether to shuffle the data before splitting.
        
    Returns:
    --------
    tuple
        Tuple containing:
        - train_dataloader: DataLoader for training data.
        - val_dataloader: DataLoader for validation data.
    """
    # Convert to torch tensors
    coords_tensor = torch.tensor(coords_data, dtype=torch.float32)
    cv_coords_tensor = torch.tensor(cv_coords_data, dtype=torch.float32)
    
    # Create dataset
    dataset = TensorDataset(coords_tensor, cv_coords_tensor)
    
    # Split into training and validation
    n_samples = len(dataset)
    n_train = int(train_ratio * n_samples)
    n_val = n_samples - n_train
    
    if shuffle:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    else:
        train_dataset = torch.utils.data.Subset(dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_samples))
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader