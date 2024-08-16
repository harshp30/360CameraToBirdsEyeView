import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model import UNetXST
from dataloader import get_dataloader
import os
import matplotlib.pyplot as plt

def visualize_prediction(inputs, targets, outputs, epoch, batch_idx, save_dir):
    """
    Visualize and save the input, target, and predicted BEV images.

    Args:
        inputs (torch.Tensor): Input tensor (batch of images).
        targets (torch.Tensor): Target BEV tensor.
        outputs (torch.Tensor): Predicted BEV tensor.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
        save_dir (str): Directory to save the visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot input (front view)
    axes[0].imshow(inputs[0, :3].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Input (Front View)')
    
    # Plot target BEV
    axes[1].imshow(targets[0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Target BEV')
    
    # Plot predicted BEV
    axes[2].imshow(outputs[0].permute(1, 2, 0).detach().cpu().numpy())
    axes[2].set_title('Predicted BEV')
    
    # Remove axes for cleaner visualization
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'prediction_epoch_{epoch}_batch_{batch_idx}.png'))
    plt.close()

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    """
    Train the model.

    Args:
        model (nn.Module): The UNetXST model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training.
        scheduler (lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to train on (cuda/cpu).
        save_dir (str): Directory to save model checkpoints and visualizations.
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            inputs = batch['inputs'].to(device)
            targets = batch['bev_target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Visualize predictions every 100 batches
            if batch_idx % 100 == 0:
                visualize_prediction(inputs, targets, outputs, epoch, batch_idx, save_dir)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                targets = batch['bev_target'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    print(f'Training completed. Best validation loss: {best_val_loss:.4f}')

def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.001
    img_size = (256, 256)
    
    # Paths
    root_dir = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR'
    save_dir = '/home/paperspace/Projects/360CameraToBirdsEyeView/models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    train_loader = get_dataloader(root_dir, 'train', batch_size, num_workers=2, img_size=img_size)
    val_loader = get_dataloader(root_dir, 'val', batch_size, num_workers=2, img_size=img_size)
    
    # Model initialization
    model = UNetXST(in_channels=12, out_channels=3).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir)

if __name__ == '__main__':
    main()