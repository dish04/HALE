# MultiEYE: Multi-Modal Eye Disease Classification

MultiEye is a deep learning framework for multi-modal eye disease classification using both fundus and OCT (Optical Coherence Tomography) images. The model employs a Vision Transformer (ViT) architecture with cross-modal attention mechanisms to effectively combine information from both imaging modalities for improved disease classification.

## Features

- **Multi-Modal Fusion**: Combines information from fundus and OCT images using cross-attention mechanisms
- **Grad-CAM Visualization**: Includes integrated Grad-CAM for model interpretability
- **Data Augmentation**: Built-in data augmentation for robust training
- **Flexible Architecture**: Easy to modify for different numbers of disease classes
- **Pre-trained Support**: Can be initialized with pre-trained weights for transfer learning

## Dataset Structure

```
assemble/
|-- train/
|   |-- ImageData/
|   `-- large9cls.txt
|-- dev/
|   `-- large9cls.txt
`-- test/
    `-- large9cls.txt
assemble_oct/
|-- train/
|   |-- ImageData/
|   `-- large9cls.txt
|-- dev/
|   `-- large9cls.txt
`-- test/
    `-- large9cls.txt
dataset_source/
|-- assemble/
|   |-- train/
|   |-- dev/
|   `-- test/
`-- assemble_oct/
    |-- train/
    `-- test/
private_paired_data/
```

### Data Organization and Labeling

- `assemble` contains fundus data
- `assemble_oct` contains OCT data
- Images are stored in the `ImageData` folder for each modality
- Labels are defined in `large9cls.txt` files

### Class Labels

The dataset includes the following eye conditions (0-7):

- 0: Normal
- 1: Dry Age-Related Macular Degeneration (dAMD)
- 2: Central Serous Chorioretinopathy (CSC)
- 3: Diabetic Retinopathy (DR)
- 4: Glaucoma (GLC)
- 5: Macular Epiretinal Membrane (MEM)
- 6: Retinal Vein Occlusion (RVO)
- 7: Wet Age-Related Macular Degeneration (wAMD)

### Paired Data

The `private_paired_data` folder contains paired multi-modal data, where each line follows the format:
```
[patient_id] [image_name]
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multieye.git
   cd multieye
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the model, run:

```bash
python train.py --data_dir /path/to/your/data --num_epochs 50 --batch_size 16 --learning_rate 1e-4
```

### Training Arguments

- `--data_dir`: Path to the root directory containing the dataset
- `--num_epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--output_dir`: Directory to save model checkpoints and logs (default: './outputs')
- `--resume`: Path to a checkpoint to resume training from (optional)

## Model Architecture

The model uses a Vision Transformer (ViT) based architecture with the following key components:

1. **Patch Embedding**: Both fundus and OCT images are split into patches and linearly embedded
2. **Cross-Modal Attention**: Bi-directional attention between fundus and OCT features
3. **Classification Head**: Final classification layer for disease prediction
4. **Grad-CAM**: Integrated visualization of important image regions

## Evaluation

To evaluate the model on the test set:

```bash
python evaluate.py --data_dir /path/to/your/data --checkpoint /path/to/checkpoint.pth
```

## Visualization

To generate Grad-CAM visualizations for model interpretability:

```bash
python visualize_gradcam.py --image1 /path/to/fundus.jpg --image2 /path/to/oct.jpg --checkpoint /path/to/checkpoint.pth
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multieye2023,
  author = {Your Name},
  title = {MultiEye: Multi-Modal Eye Disease Classification},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/multieye}}
}
```

## Acknowledgments

- This work was inspired by recent advances in vision transformers and multi-modal learning.
- We thank the open-source community for their valuable contributions.

