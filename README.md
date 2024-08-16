# Cam2BEV: Multi-View Camera to Bird's Eye View Transformation

## 1. Project Overview

Cam2BEV is an advanced computer vision project that transforms multi-view vehicle-mounted camera images into a semantically segmented bird's eye view (BEV) representation. It employs a custom UNetXST architecture to process inputs from four cameras (front, rear, left, right) and generate an accurate BEV output.

## 2. Technical Details

**Technical Topics:** *Computer Vision, UNet, Spatial Transformer Networks, Data Augmentation, Image Segmentation, Multi-View Fusion, Sim2Real Transfer Learning*

**Tools / Technologies:** *Python, PyTorch, OpenCV, PIL, Torchvision*

**Use Case:** *Environment perception for autonomous vehicles, particularly transforming multiple camera views into a unified bird's eye view for improved situational awareness and decision-making in complex traffic scenarios.*

## 3. Data Handling

The `BEVDataset` class in `dataloader.py` manages data loading and preprocessing:

Key features and design decisions:
- Supports different data splits (train, val, test) for robust model evaluation
- Resizes images to a specified size (default 256x256) to ensure consistent input dimensions and reduce computational load
- Concatenates four view inputs along the channel dimension, allowing the model to process all views simultaneously
- Returns input tensor of shape (12, 256, 256) and target of shape (3, 256, 256), facilitating efficient batch processing

## 4. Model Architecture

The `UNetXST` class implements a custom U-Net architecture with cross-view spatial transformer modules:

Architecture details and rationale:

1. **Input Layer**: Accepts tensor of shape (B, 12, 256, 256) where B is batch size. The 12 channels correspond to 4 views with 3 color channels each, allowing for simultaneous processing of all views.

2. **Encoder**: 4 convolutional blocks with max pooling, doubling channels each time
   - ConvBlock1: 12 → 64 channels
   - ConvBlock2: 64 → 128 channels
   - ConvBlock3: 128 → 256 channels
   - ConvBlock4: 256 → 512 channels
   
   Rationale: The progressive increase in channel depth allows the network to capture increasingly complex features while reducing spatial dimensions.

3. **Bottleneck**: ConvBlock 512 → 1024 channels
   
   Rationale: The bottleneck layer compresses spatial information and forces the network to encode the most salient features.

4. **Decoder**: 4 transposed convolutions with skip connections
   - UpConv4 + Skip4: 1024 → 512 channels
   - UpConv3 + Skip3: 512 → 256 channels
   - UpConv2 + Skip2: 256 → 128 channels
   - UpConv1 + Skip1: 128 → 64 channels
   
   Rationale: Transposed convolutions upsample the feature maps, while skip connections preserve fine-grained spatial information from the encoder, crucial for accurate segmentation.

5. **Output**: Final 1x1 convolution, 64 → 3 channels
   
   Rationale: The 1x1 convolution reduces the channel dimension to match the desired output classes, producing the final segmentation map.

Each `ConvBlock` employs a double 3x3 convolution with batch normalization and ReLU activation:

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        # ... (initialization code)

    def forward(self, x):
        return self.conv(x)
```

Key design decisions:
- Double 3x3 convolution increases the receptive field without losing spatial resolution
- Batch normalization stabilizes training and mitigates internal covariate shift
- ReLU activation introduces non-linearity without the vanishing gradient problem

## 5. Evaluation Metrics

The Cam2BEV system uses the following metrics for evaluation:
```
Average MSE: below 0.05
Average PSNR: above 25 dB
Average SSIM: above 0.8
```

**Sample Prediction:**

First image is front view, second is true BEV, third is the prediction (model output)

<a>
    <img src="assets/Sample1.png" alt="Inference Image" width="800" height="200">
</a>


1. **Mean Intersection over Union (MIoU)**:
   ```python
   def calculate_miou(pred, target, num_classes):
       ious = []
       for cls in range(num_classes):
           pred_mask = (pred == cls)
           target_mask = (target == cls)
           intersection = (pred_mask & target_mask).sum()
           union = (pred_mask | target_mask).sum()
           iou = intersection / (union + 1e-6)
           ious.append(iou)
       return np.mean(ious)
   ```
   MIoU provides a comprehensive measure of segmentation accuracy across all classes, balancing precision and recall.

2. **Per-class IoU**:
   Calculated similarly to MIoU but reported for each class separately, allowing for detailed analysis of model performance on different semantic categories.

3. **Pixel Accuracy**:
   ```python
   def pixel_accuracy(pred, target):
       correct = (pred == target).sum()
       total = pred.numel()
       return correct / total
   ```
   Provides a simple measure of overall segmentation correctness but can be biased by class imbalance.

4. **Frequency Weighted IoU (FWIoU)**:
   ```python
   def fw_iou(pred, target, num_classes):
       ious = []
       frequencies = []
       for cls in range(num_classes):
           pred_mask = (pred == cls)
           target_mask = (target == cls)
           intersection = (pred_mask & target_mask).sum()
           union = (pred_mask | target_mask).sum()
           iou = intersection / (union + 1e-6)
           freq = target_mask.sum() / target_mask.numel()
           ious.append(iou)
           frequencies.append(freq)
       return np.sum(np.array(ious) * np.array(frequencies))
   ```
   FWIoU weights the IoU of each class by its pixel frequency, providing a balanced measure for datasets with class imbalance.


## 6. Next Steps

Future work to enhance the Cam2BEV project could include:

1. Implementation of attention mechanisms to improve feature fusion across different views
2. Exploration of more advanced architectures like Transformer-based models for improved spatial reasoning
3. Integration of temporal information for video sequence processing
4. Development of a more sophisticated data augmentation pipeline to improve model generalization
5. Investigation of domain adaptation techniques for better real-world performance
6. Optimization of the model for real-time inference on embedded systems

## 7. Citations

```
Reiher, L., Lampe, B., & Eckstein, L. (2020). A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird's Eye View. arXiv:2005.04078v1 [cs.CV].
```