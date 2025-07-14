# MultiEYE: Multi-Modal Eye Disease Classification

Deep learning framework for eye disease classification using fundus and OCT images with Vision Transformers.

## Dataset Structure

Organize your dataset in the following structure:

```
multieye_data/
└── assemble/
    └── train/
        ├── ImageData/
        │   └── images/        # All fundus images are stored here
        └── large9cls.txt      # Label file (format: image_name label)
```

For OCT images (if available), they should be in a parallel structure:

```
multieye_data/
└── assemble_oct/
    └── train/
        ├── ImageData/
        │   └── images/        # All OCT images
        └── large9cls.txt      # Corresponding labels
```

### Class Labels

- 0: Normal
- 1: Dry AMD (dAMD)
- 2: Central Serous Chorioretinopathy (CSC)
- 3: Diabetic Retinopathy (DR)
- 4: Glaucoma (GLC)
- 5: Macular Epiretinal Membrane (MEM)
- 6: Retinal Vein Occlusion (RVO)
- 7: Wet AMD (wAMD)

## Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/HALE.git
cd HALE
```

### 2. Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
1. Organize your dataset in the structure shown above
2. Make sure `large9cls.txt` contains the correct image paths and labels

### 4. Run the Data Processing Script
```bash
python prepare_dataset.py --raw_dir ./multieye_data --output_dir ./processed_data
```

### 5. Test Data Loading
Verify your dataset loads correctly:
```bash
python create_dataset.py --data_dir ./processed_data
```

### 6. Train the Model
```bash
python train.py \
  --data_dir ./processed_data \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --output_dir ./outputs
```

## Advanced Configuration

### Training Options
- `--data_dir`: Path to processed dataset (default: './processed_data')
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--image_size`: Input size (default: 224)
- `--num_workers`: Data loading workers (default: 4)
- `--output_dir`: Checkpoint/log directory
- `--resume`: Path to checkpoint to resume training

### Monitoring
- Training progress is logged to TensorBoard:
  ```bash
  tensorboard --logdir=./outputs
  ```
- Model checkpoints are saved in the output directory

## Model

- Vision Transformer (ViT) based architecture
- Cross-modal attention between fundus and OCT

## License

MIT License - See [LICENSE](LICENSE) for details.

