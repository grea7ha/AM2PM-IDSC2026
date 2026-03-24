# Glaucoma Detection Bootcamp Project

This repository contains the source code for building, training, and evaluating various deep learning models (DenseNet121, EfficientNetB0, ResNet50) and hybrid architectures for glaucoma detection. It also includes steps for dataset preparation, quality analysis, and Grad-CAM visualizations.

## Project Structure
- `script_*`: Numbered pipeline scripts handling the entire workflow from environment testing to final model comparison.
- `Labels.csv`: Dataset annotations.
- `IDSC2026_Technical_Report.md`: Detailed technical report on the experiments and results.

## Requirements
Ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage
1. Provide the dataset images in the respective directories as mapped in the script variables.
2. Provide the labels in `Labels.csv`.
3. Run the scripts sequentially depending on the components and models you wish to evaluate.

