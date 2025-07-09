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
        self.modality = modality
        if test_file is None:
            file = 'large9cls.txt'
            path_file = os.path.join(root, file)
        else:
            path_file = test_file
        self.root = '/'.join(os.path.split(root)[:-1])
        self.transform = transform
        self.target_transform = target_transform
        
        self.img2labels = {}
        self.imgs = []
        
        with open(path_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if test_file is not None:
                    img_name = line.strip().split('\t')[0]
                    label = int(line.strip().split('\t')[1])
                else:
                    img_name = line.strip().split(' ')[0]
                    label = int(line.strip().split(' ')[1])
          
                self.imgs.append(img_name)
                self.img2labels[img_name] = label

    def __getitem__(self, index):
        if self.modality == 'fundus':
            img_path = os.path.join(self.root, 'train', 'ImageData', 'cfp-clahe-224x224', self.imgs[index] + '.png')
        if self.modality == 'oct':
            img_path = os.path.join(self.root, 'train', 'ImageData', 'oct-filter-448x448', self.imgs[index] + '.png')
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.split(img_path)[-1][:-4]
        # Get the labels for the corresponding image
        label = self.img2labels[self.imgs[index]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, img_name
    
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