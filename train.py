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

# Set up CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    # Additional CUDA optimizations
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for faster training

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

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()  # Set model to training mode
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # Clear CUDA cache before training
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create a progress bar
    dataloader = tqdm(dataloader, desc="Training", unit="batch")
    
    for batch_idx, ((fundus, oct_img), (labels, _), _) in enumerate(dataloader):
        # Move data to the specified device
        fundus = fundus.to(device, non_blocking=True)
        oct_img = oct_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            # Forward pass - always pass inputs as separate arguments
            outputs = model(fundus, oct_img)
            loss = criterion(outputs, labels)
        
        # Backward pass and optimize with gradient scaling for mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss = 0.9 * running_loss + 0.1 * loss.item()  # Smooth loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        dataloader.set_postfix({
            'loss': f"{running_loss:.4f}",
            'acc': f"{(100. * correct / total):.2f}%"
        })
        
        # Free up GPU memory
        if batch_idx % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    epoch_acc = 100. * correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # Clear CUDA cache before validation
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        # Create a progress bar for validation
        dataloader = tqdm(dataloader, desc="Validating", unit="batch", leave=False)
        
        for (fundus, oct_img), (labels, _), _ in dataloader:
            # Move data to the specified device
            fundus = fundus.to(device, non_blocking=True)
            oct_img = oct_img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                outputs = model(fundus, oct_img)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * fundus.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            dataloader.set_postfix({
                'val_loss': f"{running_loss/total:.4f}",
                'val_acc': f"{(100. * correct / total):.2f}%"
            })
    
    # Calculate final metrics
    avg_loss = running_loss / total if total > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    # Free up GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return avg_loss, accuracy

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
    
    # Set device to use first available GPU or CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print GPU information if available
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Set some performance flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
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
    )
    
    # Move model to the specified device
    model = model.to(device)
    
    # For now, let's use a single GPU to avoid DataParallel issues
    # You can re-enable multi-GPU later once we confirm single-GPU works
    if torch.cuda.device_count() > 1:
        print(f"Warning: Multi-GPU support is temporarily disabled. Using only 1 GPU.")
    
    # Use the first available GPU or fall back to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Data loaders with pin_memory for faster GPU transfer
    train_loader, val_loader = create_data_loaders(
        subset_ratio=args.subset_ratio,
        unpaired=args.unpaired
    )
    
    # Move data to device in the training loop for better memory management
    
    # Training loop
    best_acc = 0.0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, is_best=True)
        else:
            save_checkpoint(model, optimizer, epoch, is_best=False)
            
        # Clear CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
