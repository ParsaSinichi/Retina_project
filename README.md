# Comparison of handcrafted feature and deep features extracted from RETFound and RETFound green

## Introduction

Age-related macular degeneration (AMD) is a leading cause of blindness, particularly in individuals over 50. Detecting AMD early is crucial for slowing its progression. This report summarizes a study that compares classical feature extraction techniques and deep learning models for AMD detection using fundus images.

## Methodology

The study compares the performance of classical feature extraction methods and deep learning models for AMD detection.

### Classical Feature Extraction Methods

1. **Histogram of Oriented Gradients (HOG):** Captures local intensity gradients and edge directions.
2. **Local Binary Patterns (LBP):** Analyzes texture by comparing each pixel with its neighbors.
3. **Gray Level Co-occurrence Matrix (GLCM):** Measures the spatial relationship of gray levels in an image.
4. **Discrete Wavelet Transform (DWT):** Decomposes images into multiple resolution levels to capture features at different scales.

### Deep Learning Models

1. **RETFound:** A deep learning model specialized in retinal image analysis, fine-tuned for the AMD detection task.
2. **RETFound Green:** A more efficient variant of RETFound, optimized for higher-resolution images with reduced computational requirements.

### Classifiers

- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**

These classifiers were applied to both the features extracted by classical methods and the features extracted by the deep learning models.

## Results and Discussion

| Method                       | Classifier | Accuracy | Sensitivity | Specificity | AUC-ROC |
|------------------------------|------------|----------|-------------|-------------|---------|
| RETFound (fine-tuned)        | SVM        | 0.91     | 0.8539      | 0.9260      | 0.9649  |
| RETFound (fine-tuned)        | LR         | 0.8825   | 0.8988      | 0.8778      | 0.9614  |
| Combined Classical Features  | LR         | 0.8625   | 0.8764      | 0.8585      | 0.9160  |
| Local Binary Patterns (LBP)  | SVM        | 0.825    | 0.8314      | 0.8231      | 0.9030  |

### Key Findings

- **Deep Learning Performance:** The fine-tuned RETFound model outperformed all classical methods, achieving the highest accuracy (0.91) and AUC-ROC (0.9649).
- **Classical Methods:** Despite being outperformed by deep learning, classical methods like LBP still achieved notable results, particularly in lower-resource scenarios.
- **Classifier Comparison:** Logistic Regression generally performed better than SVM across most feature sets.

