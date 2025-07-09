# MultiEYE: Multi-Modal Eye Disease Classification

Deep learning framework for eye disease classification using fundus and OCT images with Vision Transformers.

## Dataset Structure

Place your dataset in the following structure:

```
raw_dataset/
├── assemble/
│   ├── train/
│   │   ├── ImageData/     # Fundus images
│   │   └── large9cls.txt  # Labels
│   ├── dev/
│   └── test/
└── assemble_oct/          # OCT images (same structure as assemble)
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

## Quick Start

1. **Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   ```bash
   # Process raw dataset into required format
   python prepare_dataset.py --raw_dir ./raw_dataset --output_dir ./dataset
   ```

3. **Train the Model**
   ```bash
   # Basic training with default parameters
   python train.py \
     --data_dir ./dataset \
     --batch_size 32 \
     --epochs 100 \
     --learning_rate 1e-4 \
     --output_dir ./outputs
   ```

   ### Training Options
   - `--data_dir`: Path to processed dataset directory (default: './dataset')
   - `--batch_size`: Batch size for training (default: 32)
   - `--epochs`: Number of training epochs (default: 100)
   - `--learning_rate`: Initial learning rate (default: 1e-4)
   - `--image_size`: Input image size (default: 224)
   - `--num_workers`: Number of data loading workers (default: 4)
   - `--output_dir`: Directory to save checkpoints and logs
   - `--resume`: Path to checkpoint to resume training

4. **Monitor Training**
   - Training progress is logged to TensorBoard
   - Run `tensorboard --logdir=./outputs` to monitor metrics
   - Model checkpoints are saved in the output directory

## Model

- Vision Transformer (ViT) based architecture
- Cross-modal attention between fundus and OCT
- Grad-CAM visualization support

## License

MIT License - See [LICENSE](LICENSE) for details.

