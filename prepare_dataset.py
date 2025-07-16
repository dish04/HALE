import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

class DatasetPreprocessor:
    def __init__(self, raw_data_dir='raw_dataset', processed_dir='dataset'):
        """
        Initialize the dataset preprocessor.
        
        Args:
            raw_data_dir: Directory containing the raw dataset
            processed_dir: Directory where processed dataset will be saved
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create necessary directories
        self.processed_dir.mkdir(exist_ok=True)
        
        # Define class names based on your labels
        self.class_names = [
            'normal', 'damd', 'csc', 'dr', 
            'glc', 'mem', 'rvo', 'wamd'
        ]
        
    def create_directory_structure(self):
        """Create the required directory structure for the processed dataset."""
        # Create main splits
        for split in ['train', 'val', 'test']:
            for modality in ['fundus', 'oct']:
                split_dir = self.processed_dir / split / modality
                
                # Create class directories with 'images' subdirectory
                for cls_name in self.class_names:
                    class_dir = split_dir / cls_name / 'images'
                    class_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Process the raw dataset into the required structure.
        
        Args:
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
        """
        print("Creating directory structure...")
        self.create_directory_structure()
        
        # Process each modality
        for modality, src_dir in [('fundus', 'assemble'), ('oct', 'assemble_oct')]:
            print(f"\nProcessing {modality} images...")
            
            # Set up paths based on modality
            if modality == 'fundus':
                image_dir = self.raw_data_dir / src_dir / 'train' / 'ImageData' / 'images'
                label_file = self.raw_data_dir / src_dir / 'train' / 'large9cls.txt'
            else:
                image_dir = self.raw_data_dir / src_dir / 'train' / 'ImageData' / 'images'
                label_file = self.raw_data_dir / src_dir / 'train' / 'large9cls.txt'
            
            # Check if paths exist
            if not image_dir.exists():
                print(f"Image directory not found: {image_dir}")
                continue
                
            if not label_file.exists():
                print(f"Label file not found: {label_file}")
                continue
            
            # Read all image paths and labels
            data = []
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            
            # First, read all labels
            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                            
                        img_name = parts[0]
                        label = int(parts[1])
                        
                        # Try different extensions
                        found = False
                        for ext in image_extensions:
                            img_path = image_dir / f"{img_name}{ext}"
                            if img_path.exists():
                                data.append({
                                    'path': img_path,
                                    'label': label,
                                    'split': 'train'  # All data is in train, we'll split it
                                })
                                found = True
                                break
                                
                        if not found:
                            print(f"Warning: Image not found for {img_name} with any extension")
                            
                    except Exception as e:
                        print(f"Error processing line: {line.strip()}. Error: {e}")
            
            if not data:
                print(f"No data found for {modality}, skipping...")
                continue
            
            # Split into train/val/test
            train_data = [d for d in data if d['split'] == 'train']
            val_data = [d for d in data if d['split'] == 'dev']
            test_data = [d for d in data if d['split'] == 'test']
            
            # If using dev as validation, we need to split train into train/val
            if not val_data and train_data:
                train_data, val_data = train_test_split(
                    train_data, 
                    test_size=val_size,
                    random_state=random_state,
                    stratify=[d['label'] for d in train_data]
                )
            
            # Copy files to their respective directories
            for split_name, split_data in [('train', train_data), 
                                         ('val', val_data), 
                                         ('test', test_data)]:
                if not split_data:
                    continue
                    
                print(f"Processing {len(split_data)} {modality} images for {split_name}...")
                
                for item in tqdm(split_data):
                    src_path = item['path']
                    label = item['label']
                    
                    # Skip invalid labels
                    if label >= len(self.class_names):
                        continue
                        
                    cls_name = self.class_names[label]
                    dst_dir = self.processed_dir / split_name / modality / cls_name / 'images'
                    
                    # Copy image
                    try:
                        shutil.copy2(src_path, dst_dir / src_path.name)
                    except Exception as e:
                        print(f"Error copying {src_path}: {e}")
        
        print("\nDataset preparation complete!")
        print(f"Processed dataset saved to: {self.processed_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare MultiEye dataset for training')
    parser.add_argument('--raw_dir', type=str, default='raw_dataset',
                       help='Directory containing the raw dataset')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Directory to save processed dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Fraction of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(
        raw_data_dir=args.raw_dir,
        processed_dir=args.output_dir
    )
    
    preprocessor.process_dataset(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
