import os
import torch
from pathlib import Path

class Config:
    # Directory structure
    base_dir = Path(__file__).parent.absolute()
    
    # Data paths
    raw_data_dir = base_dir / "raw_dataset"  # Where raw dataset should be placed
    processed_data_dir = base_dir / "dataset"  # Where processed dataset will be saved
    
    # Dataset parameters
    num_classes = 8
    class_names = [
        'normal', 'damd', 'csc', 'dr', 
        'glc', 'mem', 'rvo', 'wamd'
    ]
    modalities = ['fundus', 'oct']
    
    # Training parameters
    batch_size = 16
    num_workers = 4
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # Model parameters
    image_size = 224
    patch_size = 16
    dim = 768
    depth = 6
    heads = 12
    mlp_dim = 3072
    dropout = 0.1
    
    # Output directories
    output_dir = base_dir / "outputs"
    log_dir = base_dir / "logs"
    
    # Create necessary directories
    for directory in [output_dir, log_dir, processed_data_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data splits
    test_size = 0.2
    val_size = 0.1
    random_state = 42
    
    # Data augmentation
    use_augmentation = True
    
    # Checkpoint
    checkpoint_path = None  # Path to checkpoint to resume training
    
    def get_data_paths(self, split='train'):
        """Get paths for data based on split (train/val/test)."""
        return {
            'fundus': self.processed_data_dir / split / 'fundus',
            'oct': self.processed_data_dir / split / 'oct'
        }

# Create config instance
config = Config()
