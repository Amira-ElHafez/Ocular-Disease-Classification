# Ophthalmology Disease Classification Pipeline

## Project Overview 

This project is an end-to-end, memory-efficient Machine Learning pipeline designed to classify ophthalmology diseases from medical images. Instead of using heavy deep learning models, this pipeline focuses on **robust, illumination-invariant handcrafted feature engineering** combined with a **Soft-Voting Classifier Ensemble**. 

The code automatically handles dataset downloading, class discovery, and smart image sizing, making it highly adaptable to different medical image datasets.

---

##  Technical Methodology

The pipeline follows a strict, step-by-step flow to transform raw clinical images into highly accurate predictions.

### 1. Smart Data Ingestion & Sizing
* **Dynamic Class Discovery:** The code walks through the dataset directories to automatically detect disease classes, ensuring the pipeline is completely class-agnostic and doesn't rely on hardcoded labels.
* **Smart Image Sizing:** Rather than using a fixed, arbitrary image size, the pipeline calculates the median width and height of the dataset. It clamps this value between 128 and 256 pixels and rounds it to the nearest multiple of 16. This guarantees that Histogram of Oriented Gradients (HOG) cells tile perfectly without wasting memory.

### 2. Advanced Image Preprocessing
Medical images often suffer from varying lighting conditions and camera properties. The pipeline applies a series of enhancements to standardize the visual data:
* **Gray-World White Balance:** Removes unnatural camera color casts by normalizing the color channels so that the average scene color becomes a neutral gray.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied to individual color channels to enhance local contrast, making fine blood vessels and optic disc boundaries much clearer without amplifying background noise.

### 3. Feature Engineering (Illumination-Invariant)
To capture the exact pathology of the eye diseases, the pipeline extracts specialized features, primarily using the **Green Channel**, which provides the highest contrast for retinal structures:
* **HOG (Histogram of Oriented Gradients):** Captures the general shape and macro-texture of the eye.
* **LBP (Local Binary Patterns):** Extracts uniform micro-textures and local variations.
* **Color-Ratio Histograms:** Computes Red/Green (R/G) and Blue/Green (B/G) ratios to capture pathological color changes (like hemorrhages) regardless of overall image brightness.

### 4. Dimensionality Reduction & Class Balancing
Before passing the extracted features to the models, the data is optimized to prevent overfitting and bias:
* **Standardization:** `StandardScaler` ensures all features have a mean of 0 and a variance of 1.
* **PCA (Principal Component Analysis):** Reduces the high-dimensional feature space (up to 650 components) while preserving the maximum variance, reducing noise and training time.
* **SMOTE (Synthetic Minority Over-sampling Technique):** Implemented inside a pipeline to synthetically generate samples for underrepresented disease classes in the PCA space, ensuring models do not bias toward majority classes.

### 5. Soft-Voting Machine Learning Ensemble
The system trains five distinct, powerful classifiers using `StratifiedKFold` cross-validation:
1. **Support Vector Machine (SVM):** Optimized via `GridSearchCV` (Linear vs. RBF kernels).
2. **Random Forest:** An ensemble of 200 decision trees using feature subsampling.
3. **K-Nearest Neighbors (KNN):** Uses Euclidean distance in the PCA space.
4. **XGBoost:** Gradient-boosted decision trees for highly accurate, non-linear classification.
5. **Logistic Regression:** Serves as a robust linear baseline.

**The Soft-Voting Classifier:** Finally, the pipeline wraps all five trained models into a Soft-Voting Ensemble. It aggregates the predicted probabilities from every model and predicts the class with the highest combined confidence, outperforming any individual model.

---

## 📊 Evaluation & Visualizations

The code provides rich, automated visual reporting to help researchers understand model performance deeply:
* **t-SNE 2D Projection:** Visualizes the effectiveness of the handcrafted features in separating disease classes before any training occurs.
* **Preprocessing Walkthrough:** Generates a 10-panel visual breakdown of exactly how a single image transforms through White Balance, CLAHE, HOG, LBP, and Color Ratios.
* **Comprehensive Metrics:** Outputs Train/Test Accuracy, Macro F1-Scores, and full Classification Reports.
* **Performance Charts:** 
  * A grouped bar chart comparing Test Accuracy vs. Macro F1 across all models.
  * A line chart highlighting the "Overfitting Gap" (Train vs. Test accuracy).
  * A heatmap displaying the per-class F1 score for every individual classifier and the final ensemble.

##  Memory Management
A key feature of this code is its RAM safety. During feature extraction, instead of dynamically appending vectors to Python lists (which causes massive memory spikes), the script pre-allocates a `NumPy` array based on the total image count and feature dimension. This reduces peak RAM usage by approximately 80%, allowing it to run smoothly on standard machines or Google Colab.

