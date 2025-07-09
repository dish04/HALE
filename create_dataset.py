import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SingleLabelImageFolder
import matplotlib.pyplot as plt
import numpy as np

def get_transforms(img_size=224, is_training=False):
    """Create data transformations"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def visualize_sample(images, labels, class_names, num_samples=4):
    """Visualize a few samples from the dataset"""
    plt.figure(figsize=(15, 10))
    
    if isinstance(images, tuple):  # If we have both fundus and OCT images
        fundus_imgs, oct_imgs = images
        num_samples = min(num_samples, len(fundus_imgs))
        
        for i in range(num_samples):
            # Denormalize
            fundus_img = fundus_imgs[i].numpy().transpose((1, 2, 0))
            fundus_img = np.clip((fundus_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]), 0, 1)
            
            oct_img = oct_imgs[i].numpy().transpose((1, 2, 0))
            oct_img = np.clip((oct_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]), 0, 1)
            
            # Plot fundus image
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(fundus_img)
            plt.title(f"Fundus\n{class_names[labels[0][i]]}")
            plt.axis('off')
            
            # Plot OCT image
            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(oct_img)
            plt.title(f"OCT\n{class_names[labels[1][i]]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Define class names based on your README
    class_names = {
        0: "Normal",
        1: "Dry_AMD",
        2: "CSC",
        3: "DR",
        4: "Glaucoma",
        5: "MEM",
        6: "RVO",
        7: "Wet_AMD"
    }
    
    # Set paths (update these according to your directory structure)
    data_dir = "/Users/dishantharya/Downloads/multieye_data"
    
    # Create transforms
    train_transform = get_transforms(img_size=224, is_training=True)
    val_transform = get_transforms(img_size=224, is_training=False)
    
    # Create dataset
    print("Creating training dataset...")
    train_dataset = SingleLabelImageFolder(
        root=os.path.join(data_dir, "assemble/train"),
        cls_num=8,  # Number of classes
        transform=train_transform,
        modality='fundus',
        if_semi=False
    )
    
    # Create data loader
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get a batch of data
    print("Loading a batch of data...")
    batch = next(iter(train_loader))
    
    if isinstance(batch, tuple) and len(batch) == 3:
        (fundus_imgs, oct_imgs), (fundus_labels, oct_labels), (f_names, o_names) = batch
        print(f"Batch size: {len(fundus_imgs)}")
        print(f"Fundus images shape: {fundus_imgs.shape}")
        print(f"OCT images shape: {oct_imgs.shape}")
        print(f"Fundus labels: {fundus_labels}")
        print(f"OCT labels: {oct_labels}")
        
        # Visualize samples
        visualize_sample((fundus_imgs, oct_imgs), (fundus_labels, oct_labels), class_names)
    else:
        print("Unexpected batch format:", [type(x) for x in batch])

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Only needed on some macOS systems
    main()
