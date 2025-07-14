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
            transform: Optional transform to be applied to each image
        """
        self.split = split
        self.transform = transform
        
        # Get paths for the specified split
        self.fundus_dir = config.processed_data_dir / split / 'fundus'
        self.oct_dir = config.processed_data_dir / split / 'oct'
        
        # Verify directories exist
        if not self.fundus_dir.exists() or not self.oct_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found. Please run prepare_dataset.py first.\n"
                f"Expected directories:\n"
                f"- {self.fundus_dir}\n"
                f"- {self.oct_dir}"
            )
        
        # Get list of classes
        self.classes = sorted([d.name for d in self.fundus_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples with {len(self.classes)} classes")
    
    def _load_samples(self):
        """Load samples from the dataset directory."""
        samples = []
        
        # Iterate through each class directory
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]

        for i, label in self.f_img2labels.items():
            cls_count[label] += 1
        return cls_count
            
    def label_weights_for_balance(self):
        cls_count = self.label_statistics()
        labels_weight_list = []
        for i, label in self.f_img2labels.items():
            weight = 1 / cls_count[label]
            labels_weight_list.append(weight)
        return labels_weight_list


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