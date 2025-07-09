import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2

from config import config
from model import VisionTransformerWithGradCAM

def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    model = VisionTransformerWithGradCAM(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout
    ).to(config.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(config.device)

def generate_gradcam(model, fundus_img, oct_img, class_idx=None):
    """Generate Grad-CAM visualization for a single sample"""
    # Forward pass
    output = model(fundus_img, oct_img)
    
    # If class_idx is None, use the predicted class
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Get the gradient of the output with respect to the model's last layer
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Get the gradients and activations for both modalities
    gradients_fundus = model.gradients_x1
    activations_fundus = model.last_x1.detach()
    gradients_oct = model.gradients_x2
    activations_oct = model.last_x2.detach()
    
    # Function to compute Grad-CAM for a single modality
    def compute_cam(gradients, activations):
        weights = torch.mean(gradients, dim=[1, 2], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=(config.image_size, config.image_size), 
            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    # Compute Grad-CAM for both modalities
    cam_fundus = compute_cam(gradients_fundus, activations_fundus)
    cam_oct = compute_cam(gradients_oct, activations_oct)
    
    return cam_fundus, cam_oct, class_idx, output.softmax(dim=1).squeeze().detach().cpu().numpy()

def visualize_results(fundus_img, oct_img, cam_fundus, cam_oct, class_idx, probs):
    """Visualize original images with Grad-CAM overlays"""
    # Convert tensors to numpy arrays
    fundus_img = fundus_img.squeeze().permute(1, 2, 0).cpu().numpy()
    oct_img = oct_img.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    fundus_img = std * fundus_img + mean
    oct_img = std * oct_img + mean
    fundus_img = np.clip(fundus_img, 0, 1)
    oct_img = np.clip(oct_img, 0, 1)
    
    # Create heatmaps
    heatmap_fundus = cv2.applyColorMap(np.uint8(255 * cam_fundus), cv2.COLORMAP_JET)
    heatmap_oct = cv2.applyColorMap(np.uint8(255 * cam_oct), cv2.COLORMAP_JET)
    
    # Superimpose heatmaps on images
    superimposed_fundus = heatmap_fundus * 0.4 + fundus_img * 255 * 0.6
    superimposed_oct = heatmap_oct * 0.4 + oct_img * 255 * 0.6
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original images
    axes[0, 0].imshow(fundus_img)
    axes[0, 0].set_title('Original Fundus')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(oct_img)
    axes[1, 0].set_title('Original OCT')
    axes[1, 0].axis('off')
    
    # Plot Grad-CAM results
    axes[0, 1].imshow(superimposed_fundus.astype('uint8'))
    axes[0, 1].set_title(f'Fundus Grad-CAM\nPredicted: {config.class_names[class_idx]} ({probs[class_idx]:.2f})')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(superimposed_oct.astype('uint8'))
    axes[1, 1].set_title(f'OCT Grad-CAM\nPredicted: {config.class_names[class_idx]} ({probs[class_idx]:.2f})')
    axes[1, 1].axis('off')
    
    # Add probability distribution
    plt.tight_layout()
    
    # Add probability distribution as text
    prob_text = "Class Probabilities:\n"
    for i, prob in enumerate(probs):
        prob_text += f"{config.class_names[i]}: {prob:.3f}\n"
    
    plt.figtext(0.5, 0.01, prob_text, ha='center', fontsize=10,
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    return fig

def main():
    # Load trained model
    checkpoint_path = os.path.join(config.output_dir, 'model_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
        return
    
    model = load_model(checkpoint_path)
    
    # Create a test dataset
    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MultiModalSingleImageFolder(
        fundus_root=os.path.join(config.data_dir, "assemble/test"),
        oct_root=os.path.join(config.data_dir, "assemble_oct/test"),
        cls_num=config.num_classes,
        mode='test',
        transform=test_transform,
        transform_oct=test_transform
    )
    
    # Get a sample from the test dataset
    (fundus_img, oct_img), (label, _), img_name = test_dataset[0]
    fundus_img = fundus_img.unsqueeze(0).to(config.device)
    oct_img = oct_img.unsqueeze(0).to(config.device)
    
    # Generate Grad-CAM
    cam_fundus, cam_oct, class_idx, probs = generate_gradcam(model, fundus_img, oct_img)
    
    # Visualize results
    fig = visualize_results(fundus_img, oct_img, cam_fundus, cam_oct, class_idx, probs)
    plt.show()
    
    # Save visualization
    output_path = os.path.join(config.output_dir, 'gradcam_visualization.png')
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Grad-CAM visualization saved to {output_path}")

if __name__ == "__main__":
    main()
