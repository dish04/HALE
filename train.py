import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from config import config
from dataset import MultiModalSingleImageFolder
from model import VisionTransformerWithGradCAM

def get_transforms(is_training=False):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_data_loaders(subset_ratio=1.0):
    """Create train and validation data loaders
    
    Args:
        subset_ratio: float, fraction of the dataset to use (0.0 to 1.0)
    """
    print("Creating datasets...")
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Check if validation directories exist
    fundus_val_path = os.path.join(config.data_dir, "assemble/dev")
    oct_val_path = os.path.join(config.data_dir, "assemble_oct/dev")
    
    # Create full training dataset first
    full_dataset = MultiModalSingleImageFolder(
        fundus_root=os.path.join(config.data_dir, "assemble/train"),
        oct_root=os.path.join(config.data_dir, "assemble_oct/train"),
        cls_num=config.num_classes,
        mode='train',
        transform=train_transform,
        transform_oct=train_transform
    )
    
    # If subset_ratio is less than 1.0, take a subset of the data
    if subset_ratio < 1.0:
        subset_size = int(len(full_dataset) * subset_ratio)
        indices = torch.randperm(len(full_dataset))[:subset_size]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Using subset of {subset_size} samples ({subset_ratio*100:.1f}% of full dataset)")
    
    # If validation directories exist, use them
    if os.path.exists(fundus_val_path) and os.path.exists(oct_val_path):
        print("Using separate validation set...")
        train_dataset = full_dataset
        val_dataset = MultiModalSingleImageFolder(
            fundus_root=fundus_val_path,
            oct_root=oct_val_path,
            cls_num=config.num_classes,
            mode='val',
            transform=val_transform,
            transform_oct=val_transform
        )
    else:
        print("Validation directories not found. Splitting training data...")
        # Split the training data into train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        # Apply validation transforms to validation set
        val_dataset.dataset.transform = val_transform
        val_dataset.dataset.transform_oct = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for (fundus_imgs, oct_imgs), (labels, _), _ in tqdm(dataloader, desc="Training"):
        fundus_imgs = fundus_imgs.to(device)
        oct_imgs = oct_imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(fundus_imgs, oct_imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (fundus_imgs, oct_imgs), (labels, _), _ in tqdm(dataloader, desc="Validating"):
            fundus_imgs = fundus_imgs.to(device)
            oct_imgs = oct_imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(fundus_imgs, oct_imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, is_best=False):
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    filename = os.path.join(config.output_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(config.output_dir, 'model_best.pth')
        torch.save(state, best_filename)

def visualize_gradcam(model, fundus_img, oct_img, class_idx=None):
    """Generate Grad-CAM visualization for a single sample"""
    model.eval()
    
    # Forward pass
    output = model(fundus_img.unsqueeze(0), oct_img.unsqueeze(0))
    
    # If class_idx is None, use the predicted class
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    
    # Get the gradient of the output with respect to the model's last layer
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Get the gradients and activations
    gradients = model.gradients_x1  # For fundus
    activations = model.last_x1.detach()  # For fundus
    
    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=[1, 2], keepdim=True)
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, size=(config.image_size, config.image_size), 
        mode='bilinear', align_corners=False)
    
    # Normalize the heatmap
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Prepare original image
    img = fundus_img.permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # Superimpose heatmap on image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(superimposed_img)
    ax2.set_title(f'Grad-CAM (Class: {config.class_names[class_idx]})')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Train a multi-modal model on OCT and Fundus images')
    parser.add_argument('--subset_ratio', type=float, default=1.0,
                        help='Fraction of the dataset to use (0.0 to 1.0, default: 1.0)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model
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
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Data loaders
    train_loader, val_loader = create_data_loaders(subset_ratio=args.subset_ratio)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, is_best=True)
        else:
            save_checkpoint(model, optimizer, epoch, is_best=False)
    
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
