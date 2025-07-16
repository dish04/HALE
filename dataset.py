import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
from config import config

# ------ Single-Label Image Datasets ------
class SingleLabelImageFolder(Dataset):
    def __init__(self, root, cls_num, transform=None, target_transform=None, modality='fundus', if_semi=False, test_file=None):
        super(SingleLabelImageFolder, self).__init__()
        self.cls_num = cls_num
        self.modality = modality.lower()
        
        # Set up paths based on modality
        if self.modality == 'fundus':
            self.image_dir = os.path.join(root, 'ImageData/images')
            self.label_file = os.path.join(root, 'large9cls.txt')
        elif self.modality == 'oct':
            root = root.replace('/assemble/', '/assemble_oct/')
            self.image_dir = os.path.join(root, 'ImageData/images')
            self.label_file = os.path.join(root, 'large9cls.txt')
        else:
            raise ValueError(f"Unsupported modality: {modality}. Choose 'fundus' or 'oct'")
            
        # Verify required files exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"Label file not found: {self.label_file}")
            
        self.transform = transform
        self.target_transform = target_transform
        self.img2labels = {}
        self.imgs = []
        
        # Load image paths and labels
        self._load_labels()
        
    def _load_labels(self):
        """Load image paths and labels from the label file."""
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle different possible delimiters (space or tab)
            if '\t' in line:
                img_name, label = line.split('\t')[:2]
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_name, label = parts[0], parts[1]
                
            # Remove file extension if present
            img_name = os.path.splitext(img_name)[0]
            
            try:
                label = int(label)
                if 0 <= label < self.cls_num:
                    self.imgs.append(img_name)
                    self.img2labels[img_name] = label
            except (ValueError, IndexError):
                print(f"Warning: Invalid label format in {self.label_file}: {line}")

    def __getitem__(self, index):
        img_name = self.imgs[index]
        base_name = os.path.basename(img_name)  # In case full path is stored
        
        # Construct the image path
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        img_path = None
        
        # Try different extensions if needed
        for ext in img_extensions:
            test_path = os.path.join(self.image_dir, f"{base_name}{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image {base_name} not found in {self.image_dir} with any common extension")
        
        try:
            # Open and convert image
            img = Image.open(img_path).convert('RGB')
            
            # Get the label for the corresponding image
            label = self.img2labels[img_name]
            
            # Apply transforms if specified
            if self.transform is not None:
                img = self.transform(img)
                
            if self.target_transform is not None:
                label = self.target_transform(label)
            
            return img, label, base_name
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # If there's an error, try the next image in the dataset
            next_index = (index + 1) % len(self)
            if next_index == index:  # If there's only one image
                raise
            return self.__getitem__(next_index)
    
    def __len__(self):
        return len(self.imgs)
    
    def label_statistics(self):
        cls_count = np.zeros(self.cls_num).astype(np.int64)

        for i, label in self.img2labels.items():
            cls_count[label] += 1
        return cls_count
            
    def label_weights_for_balance(self):
        cls_count = self.label_statistics()
        labels_weight_list = []
        for i, label in self.img2labels.items():
            weight = 1 / cls_count[label]
            labels_weight_list.append(weight)
        return labels_weight_list
    

class MultiModalEyeDataset(Dataset):
    """
    Dataset class for multi-modal eye disease classification.
    Loads paired fundus and OCT images from the processed dataset directory.
    """
    
    def __init__(self, split='train', transform=None):
        """
        Args:
            split: One of 'train', 'val', or 'test'
            transform: Transform to be applied to fundus images (OCT images will use the same transform)
        """
        self.split = split
        self.transform = transform
        
        # Get paths for the specified split - using the new directory structure
        dataset_root = Path('dataset')  # Assuming the dataset is in a 'dataset' directory
        self.fundus_dir = dataset_root / split / 'fundus'
        self.oct_dir = dataset_root / split / 'oct'
        
        # Verify directories exist
        if not self.fundus_dir.exists() or not self.oct_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found. Please run prepare_dataset.py first.\n"
                f"Expected directories:\n"
                f"- {self.fundus_dir}\n"
                f"- {self.oct_dir}"
            )
        
        # Define class names based on folder structure
        self.class_names = [
            'normal', 'damd', 'csc', 'dr',
            'glc', 'mem', 'rvo', 'wamd'
        ]
        
        # Get list of classes from both fundus and oct directories (convert to lowercase)
        fundus_classes = set(d.name.lower() for d in self.fundus_dir.iterdir() if d.is_dir())
        oct_classes = set(d.name.lower() for d in self.oct_dir.iterdir() if d.is_dir())
        
        # Only use classes that exist in both directories
        common_classes = fundus_classes.intersection(oct_classes)
        
        # Define the expected class names (lowercase only)
        expected_classes = {'normal', 'damd', 'csc', 'dr', 'glc', 'mem', 'rvo', 'wamd'}
        
        # Only keep classes that are in our expected set
        self.classes = sorted(common_classes.intersection(expected_classes))
        
        if not self.classes:
            print(f"Warning: No common classes found between {self.fundus_dir} and {self.oct_dir}")
            print(f"Fundus classes: {sorted(fundus_classes)}")
            print(f"OCT classes: {sorted(oct_classes)}")
            # Initialize with empty samples
            self.samples = []
            self.class_to_idx = {}
            print(f"Initialized empty dataset for {split} split")
            return
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load samples
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            print(f"Warning: No valid samples found in {self.fundus_dir} and {self.oct_dir}")
            print(f"Checked classes: {self.classes}")
            # Initialize with empty samples
            self.samples = []
            print(f"Initialized empty dataset for {split} split")
            return
            
        print(f"Loaded {len(self.samples)} {split} samples with {len(self.classes)} classes")
    
    def _get_lowercase_class_mapping(self, class_name):
        """Map a class name to its lowercase version."""
        # Convert to lowercase and handle any special cases
        lower_name = class_name.lower()
        # Map any variations to standard names
        name_map = {
            'dry_amd': 'damd',
            'wet_amd': 'wamd',
            'glaucoma': 'glc',
            'normal': 'normal',
            'csc': 'csc',
            'dr': 'dr',
            'mem': 'mem',
            'rvo': 'rvo'
        }
        return name_map.get(lower_name, lower_name)
    
    def _load_samples(self):
        """Load samples from the dataset directory."""
        samples = []
        
        # Supported image extensions
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Track statistics
        total_pairs = 0
        
        print("\n" + "="*80)
        print(f"Starting to load samples from:")
        print(f"- Fundus dir: {self.fundus_dir}")
        print(f"- OCT dir: {self.oct_dir}")
        print(f"Classes to process: {self.classes}")
        print("="*80 + "\n")
        
        # First, verify the directory structure
        print("Directory structure check:")
        print(f"Fundus directory exists: {self.fundus_dir.exists()}")
        print(f"OCT directory exists: {self.oct_dir.exists()}")
        
        if self.fundus_dir.exists():
            print(f"Subdirectories in fundus: {[d.name for d in self.fundus_dir.iterdir() if d.is_dir()]}")
        if self.oct_dir.exists():
            print(f"Subdirectories in OCT: {[d.name for d in self.oct_dir.iterdir() if d.is_dir()]}")
        
        # Iterate through each class directory
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            
            print("\n" + "-"*50)
            print(f"Processing class: {class_name} (index: {class_idx})")
            
            # Find the actual directory name (case-insensitive)
            def find_case_insensitive_dir(base_dir, target_name):
                target_lower = target_name.lower()
                for d in base_dir.iterdir():
                    if d.is_dir() and d.name.lower() == target_lower:
                        return d
                return base_dir / target_name  # Return expected path if not found
            
            # Get the actual directory paths (case-insensitive)
            fundus_class_dir = find_case_insensitive_dir(self.fundus_dir, class_name)
            oct_class_dir = find_case_insensitive_dir(self.oct_dir, class_name)
            
            print(f"Fundus class dir: {fundus_class_dir}")
            print(f"OCT class dir: {oct_class_dir}")
            
            # Check if directories exist
            fundus_exists = fundus_class_dir.exists()
            oct_exists = oct_class_dir.exists()
            
            print(f"Fundus dir exists: {fundus_exists}")
            print(f"OCT dir exists: {oct_exists}")
            
            if not fundus_exists or not oct_exists:
                if not fundus_exists:
                    print(f"Error: Fundus directory not found: {fundus_class_dir}")
                if not oct_exists:
                    print(f"Error: OCT directory not found: {oct_class_dir}")
                continue
            
            # Debug: List all files in the directories
            print("\nScanning for image files...")
            
            # Get all fundus images
            fundus_images = []
            for ext in img_extensions:
                # Try both lowercase and uppercase extensions
                for ext_variant in [ext, ext.upper()]:
                    pattern = f'*{ext_variant}'
                    files = list(fundus_class_dir.glob(pattern))
                    if files:
                        print(f"  Found {len(files)} {ext_variant} files in {fundus_class_dir}")
                        fundus_images.extend(files)
            
            # Get all OCT images
            oct_images = []
            for ext in img_extensions:
                for ext_variant in [ext, ext.upper()]:
                    pattern = f'*{ext_variant}'
                    files = list(oct_class_dir.glob(pattern))
                    if files:
                        print(f"  Found {len(files)} {ext_variant} files in {oct_class_dir}")
                        oct_images.extend(files)
            
            print(f"\nFound {len(fundus_images)} fundus images and {len(oct_images)} OCT images")
            
            if not fundus_images or not oct_images:
                print("Skipping class due to missing images in one or both modalities")
                continue
            
            # Debug: Print some sample filenames
            print("\nSample fundus filenames:")
            for f in fundus_images[:3]:
                print(f"  {f.name}")
            if len(fundus_images) > 3:
                print(f"  ... and {len(fundus_images) - 3} more")
                
            print("\nSample OCT filenames:")
            for f in oct_images[:3]:
                print(f"  {f.name}")
            if len(oct_images) > 3:
                print(f"  ... and {len(oct_images) - 3} more")
            
            # Create dictionaries mapping image IDs to paths
            print("\nMatching image pairs...")
            
            # Function to extract base ID from filename (without extension)
            def get_base_id(path):
                # Try to handle different filename patterns
                stem = path.stem
                # Remove any suffixes like _left, _right, _fundus, _oct, etc.
                for suffix in ['_left', '_right', '_fundus', '_oct', '_OCT', '_FUNDUS']:
                    if stem.lower().endswith(suffix.lower()):
                        stem = stem[:-len(suffix)]
                return stem.upper()
            
            fundus_dict = {get_base_id(f): f for f in fundus_images}
            oct_dict = {get_base_id(o): o for o in oct_images}
            
            print(f"  Found {len(fundus_dict)} unique fundus IDs")
            print(f"  Found {len(oct_dict)} unique OCT IDs")
            
            # Find common image IDs (case-insensitive)
            common_ids = set(fundus_dict.keys()) & set(oct_dict.keys())
            
            if not common_ids:
                print("\nWarning: No matching fundus/OCT pairs found for class", class_name)
                print("This could be due to different naming conventions between fundus and OCT images.")
                
                # Print sample IDs for debugging
                fundus_ids = list(fundus_dict.keys())
                oct_ids = list(oct_dict.keys())
                
                print(f"\nSample fundus IDs ({len(fundus_ids)} total):")
                for fid in fundus_ids[:5]:
                    print(f"  {fid} -> {fundus_dict[fid].name}")
                if len(fundus_ids) > 5:
                    print(f"  ... and {len(fundus_ids) - 5} more")
                
                print(f"\nSample OCT IDs ({len(oct_ids)} total):")
                for oid in oct_ids[:5]:
                    print(f"  {oid} -> {oct_dict[oid].name}")
                if len(oct_ids) > 5:
                    print(f"  ... and {len(oct_ids) - 5} more")
                
                print("\nTrying alternative matching strategy...")
                
                # Try more flexible matching
                matched_pairs = []
                fundus_stems = {f.stem.upper(): f for f in fundus_images}
                oct_stems = {o.stem.upper(): o for o in oct_images}
                
                # Try to find partial matches
                for f_stem, f_path in fundus_stems.items():
                    # Try exact match first
                    if f_stem in oct_stems:
                        matched_pairs.append((f_path, oct_stems[f_stem]))
                        continue
                    
                    # Try removing common suffixes
                    for suffix in ['_FUNDUS', '_OCT', '_LEFT', '_RIGHT', '_L', '_R']:
                        if f_stem.endswith(suffix):
                            base = f_stem[:-len(suffix)]
                            if base in oct_stems:
                                matched_pairs.append((f_path, oct_stems[base]))
                                break
                
                if matched_pairs:
                    print(f"Found {len(matched_pairs)} pairs using flexible matching")
                    for f_path, o_path in matched_pairs[:3]:
                        print(f"  {f_path.name} <-> {o_path.name}")
                    if len(matched_pairs) > 3:
                        print(f"  ... and {len(matched_pairs) - 3} more")
                    
                    # Add the matched pairs
                    for f_path, o_path in matched_pairs:
                        samples.append({
                            'fundus': str(f_path),
                            'oct': str(o_path),
                            'label': class_idx,
                            'class_name': class_name
                        })
                    total_pairs += len(matched_pairs)
                    continue
                
                print("No matches found with any strategy")
                continue
            
            # Add samples for the common IDs
            print(f"\nFound {len(common_ids)} matching pairs for class {class_name}")
            print("Sample matches:")
            for img_id in list(common_ids)[:3]:
                print(f"  {img_id}:")
                print(f"    Fundus: {fundus_dict[img_id].name}")
                print(f"    OCT:    {oct_dict[img_id].name}")
            if len(common_ids) > 3:
                print(f"  ... and {len(common_ids) - 3} more")
            
            # Add samples
            class_samples = []
            for img_id in common_ids:
                class_samples.append({
                    'fundus': str(fundus_dict[img_id]),
                    'oct': str(oct_dict[img_id]),
                    'label': class_idx,
                    'class_name': class_name
                })
            
            samples.extend(class_samples)
            total_pairs += len(class_samples)
        
        print("\n" + "="*80)
        print(f"Finished loading dataset")
        print(f"Total classes processed: {len(self.classes)}")
        print(f"Total image pairs loaded: {total_pairs}")
        print("="*80 + "\n")
        
        if total_pairs == 0:
            print("ERROR: No valid image pairs were found. Please check:")
            print("1. The directory structure matches the expected format")
            print("2. Image files have valid extensions (.jpg, .png, etc.)")
            print("3. Corresponding fundus and OCT images have matching names")
            print("4. File permissions allow reading the image files")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load fundus image
        fundus_img = Image.open(sample['fundus']).convert('RGB')
        
        # Load OCT image
        oct_img = Image.open(sample['oct']).convert('RGB')
        
        # Apply transforms if specified
        if self.transform is not None:
            fundus_img = self.transform(fundus_img)
            oct_img = self.transform(oct_img)  # Same transform for both modalities
        
        return (fundus_img, oct_img), (sample['label'], sample['class_name']), sample['fundus']
    
    def label_statistics(self):
        """Count number of samples per class."""
        cls_count = {}
        for sample in self.samples:
            label = sample['label']
            if label not in cls_count:
                cls_count[label] = 0
            cls_count[label] += 1
        return cls_count
            
    def label_weights_for_balance(self):
        """Calculate weights for class balancing."""
        cls_count = self.label_statistics()
        if not cls_count:
            return []
            
        max_count = max(cls_count.values())
        weights = {}
        for cls, count in cls_count.items():
            weights[cls] = max_count / count if count > 0 else 0.0
            
        # Create weights for each sample
        sample_weights = [weights[sample['label']] for sample in self.samples]
        return sample_weights


def build_dataset_single(mode, args, transform=None, mod='fundus', test_file=None):
    if transform is None:
        transform = build_transform(mode, args)
    root = os.path.join(args.data_path, mode)
    dataset = SingleLabelImageFolder(root, args.n_classes, transform=transform, modality=mod, test_file=test_file)
    return dataset



def build_dataset_multimodal_single(mode, args, transform=None):
    if transform is None:
        transform = build_transform(mode, args)
        # transform_oct = build_transform('test', args)
        transform_oct = build_transform(mode, args)
    f_root = os.path.join(args.data_path, mode)
    o_root = os.path.join(args.data_path_oct, mode)
    dataset = MultiModalSingleImageFolder(f_root, o_root, args.n_classes, mode, transform=transform, transform_oct=transform_oct)
    return dataset


def build_transform(mode, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if mode == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size == 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)