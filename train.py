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
from pathlib import Path

from config import config
from dataset import MultiModalEyeDataset, UnpairedMultiModalEyeDataset
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

def create_data_loaders(subset_ratio=1.0, unpaired=False):
    """Create train and validation data loaders from processed dataset
    
    Args:
        subset_ratio: float, fraction of the dataset to use (0.0 to 1.0)
        unpaired: bool, whether to use unpaired dataset (default: False)
    """
    print("="*80)
    print("Creating datasets...")
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Verify dataset directory exists
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir.absolute()}")
    
    print(f"Using dataset directory: {dataset_dir.absolute()}")
    
    # Check directory structure
    def check_dir_structure():
        print("\nChecking dataset directory structure...")
        required_dirs = [
            'train/fundus', 'train/oct',
            'val/fundus', 'val/oct',
            'test/fundus', 'test/oct'
        ]
        
        for dir_path in required_dirs:
            full_path = dataset_dir / dir_path
            exists = full_path.exists()
            print(f"  {dir_path}: {'✓' if exists else '✗'}")
            
            # If directory exists, list its contents
            if exists:
                try:
                    class_dirs = [d.name for d in full_path.iterdir() if d.is_dir()]
                    print(f"    Classes: {', '.join(class_dirs[:5])}{'...' if len(class_dirs) > 5 else ''} ({len(class_dirs)} total)")
                    # Count files in first class as sample
                    if class_dirs:
                        sample_class = full_path / class_dirs[0]
                        file_count = sum(1 for _ in sample_class.glob('*.*'))
                        print(f"    Sample class '{class_dirs[0]}' has {file_count} files")
                except Exception as e:
                    print(f"    Error checking contents: {e}")
    
    check_dir_structure()
    
    # Create training dataset
    print("\nLoading training dataset...")
    dataset_class = UnpairedMultiModalEyeDataset if unpaired else MultiModalEyeDataset
    print(f"Using {'unpaired' if unpaired else 'paired'} dataset")
    
    train_dataset = dataset_class(
        split='train',
        transform=train_transform
    )
    
    if len(train_dataset) == 0:
        print("\nERROR: Training dataset is empty. Please check:")
        print("1. The dataset has been prepared using prepare_dataset.py")
        print("2. The directory structure matches the expected format")
        print("3. There are image files in the class directories")
        print("4. File permissions allow reading the files")
        print("\nExpected structure:")
        print("dataset/")
        print("├── train/")
        print("│   ├── fundus/")
        print("│   │   ├── normal/     # Contains .jpg, .png, etc.")
        print("│   │   ├── damd/")
        print("│   │   └── ...")
        print("│   └── oct/")
        print("│       ├── normal/     # Contains .jpg, .png, etc.")
        print("│       ├── damd/")
        print("│       └── ...")
        print("└── val/...")  # Similar structure for val and test
        
        # Try to find any image files in the dataset directory
        print("\nSearching for image files...")
        img_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        found_files = []
        for ext in img_exts:
            found_files.extend(dataset_dir.rglob(f'**/{ext}'))
            found_files.extend(dataset_dir.rglob(f'**/{ext.upper()}'))
        
        if found_files:
            print(f"\nFound {len(found_files)} image files in unexpected locations:")
            for f in found_files[:5]:  # Show first 5 files as examples
                print(f"  {f}")
            if len(found_files) > 5:
                print(f"  ... and {len(found_files) - 5} more")
        else:
            print("No image files found in the dataset directory.")
        
        raise RuntimeError("Failed to load training dataset")
    else:
        print(f"\nSuccessfully loaded training dataset with {len(train_dataset)} samples")
    
    # If subset_ratio is less than 1.0, take a subset of the data
    if subset_ratio < 1.0 and len(train_dataset) > 0:
        subset_size = int(len(train_dataset) * subset_ratio)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using subset of {subset_size} training samples ({subset_ratio*100:.1f}% of full dataset)")
    
    # Try to load validation set, if it exists
    print("\nLoading validation dataset...")
    val_dataset = dataset_class(
        split='val',
        transform=val_transform
    )
    
    if len(val_dataset) == 0:
        print(f"Warning: Validation dataset is empty at {dataset_dir}/val")
        print("Splitting training data into train/val (80/20) instead...")
        
        if len(train_dataset) == 0:
            raise RuntimeError("Cannot split empty training dataset. Please check your dataset directory.")
            
        # Split the training data into train/val (80/20)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        print(f"Split training data into {train_size} training and {val_size} validation samples")
    else:
        print(f"Successfully loaded validation dataset with {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created dataloaders - Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)} samples")
    
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
    parser.add_argument('--unpaired', action='store_true',
                        help='Use unpaired dataset (images from same class but not necessarily from same patient)')
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
    train_loader, val_loader = create_data_loaders(
        subset_ratio=args.subset_ratio,
        unpaired=args.unpaired
    )
    
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
