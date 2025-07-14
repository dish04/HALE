import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MultiModalEyeDataset
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
            
            # Get label and class name
            label_idx = labels[0][i]  # labels is a tuple of (tensor, class_names)
            class_name = labels[1][i]  # Get the actual class name string
            
            # Plot fundus image
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(fundus_img)
            plt.title(f"Fundus\n{class_name}")
            plt.axis('off')
            
            # Plot OCT image
            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(oct_img)
            plt.title(f"OCT\n{class_name}")
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
    
    # Set base data directory
    base_data_dir = "/Users/dishantharya/Downloads/multieye_data"
    
    # Data loading parameters
    batch_size = 8
    num_workers = 4
    
    # Create transforms
    transform = get_transforms(img_size=224, is_training=True)
    
    try:
        # Create dataset
        print("Loading dataset...")
        dataset = MultiModalEyeDataset(
            split='train',  # or 'val' if you want to visualize validation data
            transform=transform
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Get a batch of data
        batch = next(iter(dataloader))
        (fundus_images, oct_images), (labels, class_names_list), img_paths = batch
        
        # Print dataset information
        print(f"Number of training samples: {len(dataset)}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Class names: {', '.join(dataset.classes)}")
        print(f"Batch size: {batch_size}")
        
        # Visualize samples
        print("Visualizing samples...")
        visualize_sample(
            (fundus_images, oct_images),
            (labels, class_names_list),
            class_names
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the dataset is properly prepared and the paths are correct.")
        print("You may need to run prepare_dataset.py first.")

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Only needed on some macOS systems
    main()
