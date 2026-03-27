# 🧠 Quality-Aware Glaucoma Triage — Team AM2PM
### IDSC 2026 | Mathematics for Hope in Healthcare

> *"What if a nurse in a remote clinic, armed with only a portable camera and a laptop, could flag a high-risk glaucoma patient before any specialist arrived? We built the mathematics to make that possible."*

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-HYGD%20%40%20PhysioNet-green)](https://physionet.org/content/hygd/)
[![License](https://img.shields.io/badge/License-Open%20Source-brightgreen)]()

---

## 📖 Project Overview

Glaucoma is the **silent thief of sight** — it causes irreversible blindness with no warning symptoms, and disproportionately affects populations with limited access to specialized ophthalmologists. This project tackles that problem head-on using deep learning and mathematical modeling.

We built a **Quality-Aware dual-input neural network** that does not simply classify a retinal image as "glaucoma or not." It simultaneously processes the retinal image **and** its associated quality score, teaching the model to mathematically weigh its own diagnostic confidence based on how clear or degraded the scan actually is.

**This is especially critical for rural and mobile screening clinics**, where field cameras frequently produce noisy, suboptimal images — and where the cost of a missed diagnosis is irreversible blindness.

---

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline DenseNet121 | 0.909 | 0.899 | 0.989 | 0.942 | 0.986 |
| **Hybrid DenseNet121 (Proposed ⭐)** | **0.916** | **0.907** | **0.989** | **0.946** | **0.981** |
| Hybrid ResNet50 | 0.931 | 0.959 | 0.949 | 0.954 | 0.975 |
| Hybrid EfficientNetB0 | 0.750 | 0.750 | 1.000 | 0.857 | 0.539 |

> **Recall** is the critical metric in medical screening — a missed glaucoma case means irreversible vision loss. Our champion model achieves **98.9% Recall**.

---

## 📦 Dataset

**Hillel Yaffe Glaucoma Dataset (HYGD)** — accessed via PhysioNet.
- 747 retinal fundus images (JPG format)
- Labels: GON+ (Glaucoma) / GON- (Normal)
- Each image includes an ophthalmologist-assigned Quality Score
- No external labeled data was used

**Access:** https://physionet.org/content/hygd/

---

## 🗂️ Repository Structure

```
AM2PM-IDSC2026/
│
├── 📄 README.md
├── 📄 IDSC2026_Technical_Report.md      # Full 5-page technical report
├── 📄 requirements.txt                  # Pinned Python dependencies
│
├── 📁 scripts/                          # All 38 numbered pipeline scripts
│   ├── script_01_test_environment.py
│   ├── script_02_check_libraries.py
│   ├── ...  (03–30: data exploration, preprocessing, baseline models)
│   ├── script_31_prepare_full_dataset.py
│   ├── script_32_train_hybrid_densenet.py   ← Champion model
│   ├── script_33_evaluate_hybrid_densenet.py
│   ├── ...  (34–37: EfficientNetB0 & ResNet50 hybrids)
│   └── script_38_compare_hybrid_models.py   ← Final comparison
│
├── 📁 data/                             # All CSV datasets & predictions
│   ├── Labels.csv                       # Raw PhysioNet annotations
│   ├── glaucoma_clean_dataset.csv       # Quality-filtered dataset
│   ├── train_dataset.csv                # Baseline training split
│   ├── test_dataset.csv                 # Baseline testing split
│   ├── train_full_dataset.csv           # Full training split (hybrid)
│   ├── test_full_dataset.csv            # Full testing split (hybrid)
│   ├── hybrid_densenet_predictions.csv  # Champion model predictions
│   └── *_predictions.csv               # All other model predictions
│
└── 📁 results/                          # Metrics, histories & plots
    ├── all_model_metrics.json           # Baseline model comparison
    ├── hybrid_all_metrics.json          # Hybrid model comparison
    ├── *_metrics.json                   # Per-model evaluation metrics
    └── *_history.json                   # Per-model training history (20 epochs)
```

---

## 📜 Script Reference — Full Pipeline

### 🔵 Phase 1: Environment & Data Exploration (Scripts 01–07)

| Script | Purpose |
|---|---|
| `script_01_test_environment.py` | Verifies the Python runtime is correctly set up and prints the version. **Start here.** |
| `script_02_check_libraries.py` | Imports and validates all core dependencies (Pandas, NumPy, Matplotlib, PIL) with version reporting. |
| `script_03_load_dataset.py` | Loads `Labels.csv` and prints the first 5 rows, dataset shape, and column names for initial inspection. |
| `script_04_label_distribution.py` | Reports the class balance between GON+ (Glaucoma) and GON- (Normal) — critical for understanding dataset skew. |
| `script_05_patient_analysis.py` | Counts total unique patients and images-per-patient. Reveals that patients have multiple eye scans — key motivation for patient-level splitting. |
| `script_06_quality_analysis.py` | Computes descriptive statistics on the Quality Score column — mean, std, min, max — to understand scan quality distribution. |
| `script_07_plot_labels.py` | Renders a bar chart of the label distribution using Matplotlib. Requires a display environment. |

---

### 🟡 Phase 2: Image Preprocessing Pipeline (Scripts 08–20)

| Script | Purpose |
|---|---|
| `script_08_create_image_paths.py` | Maps each image filename in the CSV to its full file path under the `images/` directory. |
| `script_09_check_images_exist.py` | Validates that every image file referenced in the CSV physically exists on disk. Reports any missing files. |
| `script_10_load_image.py` | Loads the first image in the dataset using PIL and prints its pixel dimensions and color mode. Sanity-check before bulk processing. |
| `script_11_display_images.py` | Displays the first 5 retinal fundus images with their labels and quality scores in a matplotlib grid. |
| `script_12_check_image_sizes.py` | Audits the first 20 images for dimensional consistency. Exposes that raw images have variable sizes — confirming the need for resizing. |
| `script_13_encode_labels.py` | Converts string labels (GON+/GON-) to binary integers (1/0) for use as training targets. |
| `script_14_resize_images.py` | Resizes all 747 images to a uniform `224×224` resolution and saves them to `images_resized/`. |
| `script_15_prepare_image_arrays.py` | Loads all resized images into NumPy arrays and normalizes pixel values to `[0, 1]` range. Outputs the final `(N, 224, 224, 3)` tensor. |
| `script_16_filter_quality.py` | Filters out images with a Quality Score below 5, reducing the dataset from 747 to 618 for the baseline pipeline. |
| `script_17_save_clean_dataset.py` | Applies quality filtering, encodes labels numerically, and saves the clean metadata to `glaucoma_clean_dataset.csv`. |
| `script_18_train_test_split.py` | Performs the critical **patient-level train/test split** — splitting *patients*, not images — ensuring zero data leakage. Saves `train_dataset.csv` and `test_dataset.csv`. |
| `script_19_check_split_distribution.py` | Validates that the class balance is reasonably preserved across the training and testing splits. |
| `script_20_verify_paths.py` | Final pipeline integrity check — verifies all training images exist on disk. Generates a `missing_images_report.csv` if any are absent. |

---

### 🟠 Phase 3: Baseline Model Training & Evaluation (Scripts 21–30)

| Script | Purpose |
|---|---|
| `script_21_train_cnn_model.py` | Trains a custom 3-layer **Vanilla CNN** from scratch as the initial baseline. Establishes a floor for performance comparison. |
| `script_22_evaluate_model.py` | Evaluates the baseline CNN on the test set. Generates a confusion matrix heatmap and classification report. Saves predictions to `test_predictions.csv`. |
| `script_23_visualize_predictions.py` | For each test image: displays the original scan alongside the model's prediction, confidence probability, and a human-readable interpretation (e.g., "Very confident glaucoma"). Also isolates and displays incorrect predictions for failure analysis. |
| `script_24_train_efficientnet.py` | Builds and trains an **EfficientNetB0** transfer learning model using two-phase training: Phase 1 freezes the base (10 epochs, lr=1e-3), Phase 2 fine-tunes the top 20 layers (10 epochs, lr=1e-4). Saves model and training history. |
| `script_25_evaluate_efficientnet.py` | Full evaluation of EfficientNetB0: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and **5 Grad-CAM heatmap overlays** saved to `gradcam_efficientnet/`. |
| `script_26_train_densenet.py` | Builds and trains a **DenseNet121** transfer learning model using identical two-phase training strategy. DenseNet's dense connections provide superior gradient flow for medical imaging tasks. |
| `script_27_evaluate_densenet.py` | Full evaluation of DenseNet121 with all metrics, confusion matrix, ROC curve, and Grad-CAM heatmaps saved to `gradcam_densenet/`. |
| `script_28_compare_models.py` | **Model Shootout:** Loads all three baseline models and runs them against the same test set. Produces a side-by-side metric comparison table, grouped bar chart (`model_comparison.png`), and combined ROC curve (`model_comparison_roc.png`). |
| `script_29_train_resnet50.py` | Builds and trains a **ResNet50** model. Uses ResNet-specific `preprocess_input` normalization (mean-centered, not [0,1]) — an important preprocessing distinction from the other architectures. |
| `script_30_evaluate_resnet50.py` | Full evaluation of ResNet50. Includes a critical fix: loads raw (unnormalized) images separately for Grad-CAM visualization to ensure heatmaps overlay correctly on visible images. |

---

### 🟢 Phase 4: Hybrid Quality-Aware Models (Scripts 31–38)

| Script | Purpose |
|---|---|
| `script_31_prepare_full_dataset.py` | Re-runs the patient-level data split **without quality filtering** — preserving all 747 images including low-quality scans. Also computes `quality_normalized` (Quality Score ÷ 10) as a model-ready feature. Saves `train_full_dataset.csv` and `test_full_dataset.csv`. |
| `script_32_train_hybrid_densenet.py` | **The Champion.** Builds the Quality-Aware Hybrid DenseNet121 using the **Keras Functional API** with two parallel input branches: Branch A (DenseNet121 image features) and Branch B (quality scalar through a Dense layer). Features are concatenated before the final sigmoid classification head. |
| `script_33_evaluate_hybrid_densenet.py` | Full evaluation of the Hybrid DenseNet121: all 5 metrics, confusion matrix, ROC curve, and Grad-CAM heatmaps (quality score displayed as overlay label). Results saved to `hybrid_densenet_predictions.csv`. |
| `script_34_train_hybrid_efficientnet.py` | Applies the identical Quality-as-Feature dual-input architecture to **EfficientNetB0**. Same training strategy and hyperparameter configuration as the Hybrid DenseNet. |
| `script_35_evaluate_hybrid_efficientnet.py` | Full evaluation of Hybrid EfficientNetB0. Grad-CAM analysis reveals the model's attention collapsed to image background corners — identifying it as the weakest architecture despite high raw recall. |
| `script_36_train_hybrid_resnet50.py` | Applies the Quality-as-Feature architecture to **ResNet50**. Applies ResNet-specific `preprocess_input` to the image branch, with the quality scalar branch remaining unaffected. |
| `script_37_evaluate_hybrid_resnet50.py` | Full evaluation of Hybrid ResNet50. Loads unnormalized images for visual Grad-CAM overlays while using preprocessed arrays for inference — maintaining correctness across both paths. |
| `script_38_compare_hybrid_models.py` | **Grand Final:** Loads all three hybrid models and benchmarks them head-to-head. Declares the best model by F1 + ROC-AUC, produces grouped bar chart (`hybrid_model_comparison.png`) and combined ROC curve (`hybrid_model_comparison_roc.png`). All metrics saved to `hybrid_all_metrics.json`. |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- GPU recommended for scripts 21–38 (CPU training is possible but slow)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset Setup
1. Visit https://physionet.org/content/hygd/ and download the HYGD dataset
2. Create an `images/` folder in the project root
3. Place all `.jpg` retinal images inside `images/`
4. Ensure `Labels.csv` is in the project root

---

## 🚀 How to Run

Run scripts **sequentially** from the **project root directory**:

```bash
# Phase 1: Verify environment
python3 scripts/script_01_test_environment.py
python3 scripts/script_02_check_libraries.py

# Phase 2: Explore the dataset
python3 scripts/script_03_load_dataset.py
python3 scripts/script_04_label_distribution.py
python3 scripts/script_05_patient_analysis.py
python3 scripts/script_06_quality_analysis.py

# Phase 3: Preprocess
python3 scripts/script_14_resize_images.py    # ← Run this first
python3 scripts/script_17_save_clean_dataset.py
python3 scripts/script_18_train_test_split.py

# Phase 4: Train baseline models (GPU recommended)
python3 scripts/script_24_train_efficientnet.py
python3 scripts/script_26_train_densenet.py
python3 scripts/script_29_train_resnet50.py

# Phase 5: Train hybrid models
python3 scripts/script_31_prepare_full_dataset.py
python3 scripts/script_32_train_hybrid_densenet.py
python3 scripts/script_34_train_hybrid_efficientnet.py
python3 scripts/script_36_train_hybrid_resnet50.py

# Phase 6: Compare everything
python3 scripts/script_28_compare_models.py
python3 scripts/script_38_compare_hybrid_models.py
```

---

## 📂 Files to Show Judges

### Must Show ✅
| File | Why |
|---|---|
| `IDSC2026_Technical_Report.md` | Core submission — methodology, results, XAI analysis |
| `script_32_train_hybrid_densenet.py` | The champion model — dual-input architecture |
| `script_18_train_test_split.py` | Patient-level split — proves zero data leakage |
| `script_33_evaluate_hybrid_densenet.py` | Grad-CAM + full metrics evaluation |
| `script_38_compare_hybrid_models.py` | Final model comparison logic |
| `hybrid_all_metrics.json` | Machine-readable proof of final results |
| `hybrid_densenet_predictions.csv` | Every test prediction with probability |

### Good to Show 👍
| File | Why |
|---|---|
| `script_31_prepare_full_dataset.py` | Shows the quality normalization innovation |
| `script_28_compare_models.py` | Baseline vs SOTA comparison |
| `glaucoma_clean_dataset.csv` | Shows preprocessing output |
| `hybrid_densenet_history.json` | Training curve data (20 epochs) |

---

## 🧬 Architecture Diagram

```
            ┌─────────────────────────────┐
            │     Retinal Image (224×224) │
            └──────────────┬──────────────┘
                           │
                    DenseNet121 Base
                  (Pre-trained ImageNet)
                           │
                   GlobalAvgPooling2D
                           │
                   BatchNormalization
                           │
                    Dense(256, ReLU)      ← Branch A
                           │
                      Dropout(0.5)
                           │
            ┌──────────────┘
            │
            │       ┌──────────────────────┐
            │       │  Quality Score (1D)  │
            │       └──────────┬───────────┘
            │                  │
            │           Dense(16, ReLU)    ← Branch B
            │                  │
            └──────┬───────────┘
                   │
              Concatenate
                   │
            Dense(64, ReLU)
                   │
              Dropout(0.3)
                   │
            Dense(1, Sigmoid)
                   │
            Glaucoma / Normal
```

---

## 📚 References & Citations

1. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation*, 101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

2. Hillel Yaffe Glaucoma Dataset (HYGD). PhysioNet. https://physionet.org/content/hygd/ *(Cite per the official citation text on the dataset page.)*

---

## 🌐 Competition

**International Data Science Challenge 2026 (IDSC 2026)**
Hosted by Universiti Putra Malaysia (UPM) in collaboration with UNAIR, UNMUL, and UB (Indonesia).

Theme: *Mathematics for Hope in Healthcare* | https://idsc2026.github.io
