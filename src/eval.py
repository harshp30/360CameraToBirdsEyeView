import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

def calculate_miou(pred, target, num_classes):
    """
    Calculate Mean Intersection over Union (MIoU).
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious)

def pixel_accuracy(pred, target):
    """
    Calculate Pixel Accuracy.
    """
    correct = (pred == target).sum().item()
    total = pred.numel()
    return correct / total

def fw_iou(pred, target, num_classes):
    """
    Calculate Frequency Weighted Intersection over Union (FWIoU).
    """
    ious = []
    frequencies = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            iou = intersection / union
            freq = target_mask.sum().item() / target.numel()
            ious.append(iou)
            frequencies.append(freq)
    return np.sum(np.array(ious) * np.array(frequencies))

def evaluate_model(model, test_loader, device, num_classes=3):
    """
    Evaluate the model on the test set using multiple metrics.
    """
    model.eval()
    total_miou = 0
    total_pixel_accuracy = 0
    total_fw_iou = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(device)
            targets = batch['bev_target'].to(device)
            
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            targets = torch.argmax(targets, dim=1)  # Assuming targets are one-hot encoded
            
            total_samples += inputs.size(0)
            
            miou = calculate_miou(preds, targets, num_classes)
            pixel_acc = pixel_accuracy(preds, targets)
            fw_iou_value = fw_iou(preds, targets, num_classes)
            
            total_miou += miou * inputs.size(0)
            total_pixel_accuracy += pixel_acc * inputs.size(0)
            total_fw_iou += fw_iou_value * inputs.size(0)
    
    avg_miou = total_miou / total_samples
    avg_pixel_accuracy = total_pixel_accuracy / total_samples
    avg_fw_iou = total_fw_iou / total_samples
    
    print(f"Average MIoU: {avg_miou:.4f}")
    print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")
    print(f"Average FWIoU: {avg_fw_iou:.4f}")
    
    return avg_miou, avg_pixel_accuracy, avg_fw_iou

def main():
    # Parameters
    model_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/models/best_model.pth'
    test_data_dir = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/test'
    batch_size = 4
    img_size = (256, 256)
    num_classes = 3  # Assuming 3 classes in the segmentation task
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNetXST(in_channels=12, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Prepare test dataset and dataloader
    test_loader = get_dataloader(test_data_dir, split='test', batch_size=batch_size, num_workers=2, img_size=img_size)
    
    # Evaluate the model
    evaluate_model(model, test_loader, device, num_classes)

if __name__ == '__main__':
    main()
