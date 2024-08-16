import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from model import UNetXST
from dataloader import BEVDataset

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def evaluate_model(model, test_loader, device, save_dir):
    """Evaluate the model on the test set."""
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['inputs'].to(device)
            targets = batch['bev_target'].to(device)
            
            outputs = model(inputs)
            
            # Calculate MSE
            mse = mse_loss(outputs, targets).item()
            total_mse += mse
            
            # Calculate PSNR and SSIM
            for j in range(outputs.size(0)):
                pred = outputs[j].permute(1, 2, 0).cpu().numpy()
                target = targets[j].permute(1, 2, 0).cpu().numpy()
                
                psnr = calculate_psnr(pred, target)
                total_psnr += psnr
                
                ssim_value = ssim(pred, target, multichannel=True)
                total_ssim += ssim_value
            
            # Visualize predictions (for first batch only)
            if i == 0:
                visualize_predictions(inputs, targets, outputs, save_dir)
    
    num_samples = len(test_loader.dataset)
    avg_mse = total_mse / len(test_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

def visualize_predictions(inputs, targets, outputs, save_dir):
    """Visualize and save model predictions."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    for i in range(4):
        # Input (front view)
        axes[i, 0].imshow(inputs[i, :3].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title('Input (Front View)')
        axes[i, 0].axis('off')
        
        # Ground Truth BEV
        axes[i, 1].imshow(targets[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_title('Ground Truth BEV')
        axes[i, 1].axis('off')
        
        # Predicted BEV
        axes[i, 2].imshow(outputs[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title('Predicted BEV')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_predictions.png'))
    plt.close()

def main():
    # Parameters
    model_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/models/best_model.pth'
    test_data_dir = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/test'
    save_dir = '/home/paperspace/Projects/360CameraToBirdsEyeView/output'
    batch_size = 4
    img_size = (256, 256)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNetXST(in_channels=12, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Prepare test dataset and dataloader
    test_dataset = BEVDataset(test_data_dir, split='test', img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Evaluate the model
    evaluate_model(model, test_loader, device, save_dir)

if __name__ == '__main__':
    main()


'''
Average MSE: below 0.05
Average PSNR: above 25 dB
Average SSIM: above 0.8
'''