import os
import torch
from torchvision import transforms
from PIL import Image
from model import UNetXST

def load_image(image_path, img_size=(256, 256)):
    """
    Load and preprocess an image from a file path.

    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Size to resize the image to.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    return transform(Image.open(image_path)).unsqueeze(0)  # Add batch dimension

def inference(model, front_path, rear_path, left_path, right_path, output_path, device):
    """
    Perform inference using the BEV model on four input views.

    Args:
        model (nn.Module): The trained UNetXST model.
        front_path (str): Path to the front view image.
        rear_path (str): Path to the rear view image.
        left_path (str): Path to the left view image.
        right_path (str): Path to the right view image.
        output_path (str): Path to save the output BEV image.
        device (torch.device): Device to perform inference on (cuda/cpu).
    """
    # Load and process input views
    front = load_image(front_path)
    rear = load_image(rear_path)
    left = load_image(left_path)
    right = load_image(right_path)
    
    # Concatenate inputs along the channel dimension
    inputs = torch.cat([front, rear, left, right], dim=1).to(device)
    
    # Perform inference
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        output = model(inputs)
    
    # Convert output tensor to image and save
    output_image = transforms.ToPILImage()(output.squeeze().cpu())
    output_image.save(output_path)
    print(f"Saved output image to {output_path}")

def main():
    # Parameters
    model_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/models/best_model.pth'
    img_size = (256, 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNetXST(in_channels=12, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Input image paths
    front_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/val/front/front/v_0_0020000.png'
    rear_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/val/rear/rear/v_0_0020000.png'
    left_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/val/left/left/v_0_0020000.png'
    right_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/data/1_FRLR/val/right/right/v_0_0020000.png'
    
    # Output image path
    output_path = '/home/paperspace/Projects/360CameraToBirdsEyeView/output/model_prediction.png'
    
    # Perform inference
    inference(model, front_path, rear_path, left_path, right_path, output_path, device)

if __name__ == '__main__':
    main()