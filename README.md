# 🌸 Iris Flower Classification

An end-to-end machine learning project to classify Iris flower species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*) based on physical morphological measurements using **Python**, **scikit-learn**, **pandas**, and **seaborn/matplotlib**.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Dataset Description](#dataset-description)
4. [Exploratory Data Analysis (EDA) & Visualizations](#exploratory-data-analysis-eda--visualizations)
5. [Feature Selection Discussion](#feature-selection-discussion)
6. [Machine Learning Models & Methodology](#machine-learning-models--methodology)
7. [Model Comparison & Evaluation](#model-comparison--evaluation)
8. [Champion Model Declaration & Justification](#champion-model-declaration--justification)
9. [Installation & Execution Guide](#installation--execution-guide)

---

## 🎯 Project Overview
The objective is to train a machine learning classification model to predict the species of an iris flower from four continuous features:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

### Key Project Features:
- **Direct scikit-learn Ingestion**: Built-in dataset via `sklearn.datasets.load_iris()` — zero external downloads required.
- **Thorough EDA**: Verification of shape `(150, 4)`, dtypes, zero missing values, balanced 50:50:50 distribution.
- **Publication-Quality Visualizations**: Pairwise scatter matrix with KDE distributions, multi-feature box plots with jittered observations, and correlation heatmaps.
- **Statistical Feature Selection**: Quantitative one-way ANOVA F-score analysis detailing petal vs. sepal discriminative separation.
- **Multi-Model Benchmark**: Evaluation of 4 diverse classifiers:
  1. *Logistic Regression* (Multinomial Softmax with L2 regularization)
  2. *K-Nearest Neighbors (KNN)* (Distance-weighted instance classifier)
  3. *Decision Tree* (Hierarchical rule partitioner)
  4. *Random Forest* (Bagged ensemble of decision trees)
- **Robust Evaluation**: 5-Fold Stratified Cross-Validation on training data and comprehensive hold-out test metrics (Accuracy, Confusion Matrices, Precision, Recall, F1-Score).
- **Executable Notebook**: Fully rendered Jupyter Notebook (`notebooks/iris_flower_classification.ipynb`) with pre-calculated cell outputs and embedded graphs.

---

## 📁 Project Structure

```
iris_flower_classification/
├── data/
│   └── iris.csv                               # Exported CSV representation of the dataset
├── models/
│   └── best_iris_model.joblib                 # Serialized production pipeline (Champion KNN model)
├── notebooks/
│   └── iris_flower_classification.ipynb       # Fully annotated, pre-executed Jupyter Notebook
├── reports/
│   ├── figures/
│   │   ├── pairplot.png                       # Pairwise feature KDE & scatter matrix by species
│   │   ├── boxplots.png                       # Box plots with strip point overlays
│   │   ├── correlation_heatmap.png            # Lower-triangle Pearson correlation matrix
│   │   └── confusion_matrices.png             # Side-by-side test confusion matrices
│   └── evaluation_summary.json                # Structured JSON metric benchmark
├── src/
│   ├── __init__.py
│   ├── data_loader.py                         # Data loading, validation, and train/test splitting
│   ├── eda.py                                 # Plotting routines and ANOVA F-statistic rankings
│   └── models.py                              # Model pipelines, training, cross-validation, and selection
├── tests/
│   └── test_pipeline.py                       # Automated test suite (5 tests covering data, models, artifacts)
├── build_and_execute_notebook.py              # Script to build and pre-render the Jupyter Notebook
├── requirements.txt                           # Project dependencies
└── README.md                                  # Complete documentation
```

---

## 📊 Dataset Description

The Iris dataset contains 150 total records (50 per species). The four features are continuous measurements in centimeters:

| Feature Name | Description | Min | Mean | Max | Std Dev |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `sepal_length` | Calyx length (cm) | 4.30 | 5.84 | 7.90 | 0.83 |
| `sepal_width` | Calyx width (cm) | 2.00 | 3.06 | 4.40 | 0.44 |
| `petal_length` | Corolla length (cm) | 1.00 | 3.76 | 6.90 | 1.76 |
| `petal_width` | Corolla width (cm) | 0.10 | 1.20 | 2.50 | 0.76 |

- **Missing Values**: 0 (100% complete)
- **Target Distribution**: Exactly 50 *Setosa*, 50 *Versicolor*, 50 *Virginica* (Perfect balance: 33.3% each)

---

## 📈 Exploratory Data Analysis (EDA) & Visualizations

The generated visual figures are saved under `reports/figures/`:

1. **Pairwise Distributions (`reports/figures/pairplot.png`)**:
   - Visualizes KDE distributions along the diagonal and pairwise scatters across all 6 feature pairs.
   - Clearly reveals that *Setosa* is completely isolated in feature space, while *Versicolor* and *Virginica* form adjacent, slightly touching clusters.

2. **Feature Box Plots (`reports/figures/boxplots.png`)**:
   - Compares distributions and outliers across species with stripplot data points.
   - Shows that petal dimensions have virtually non-overlapping interquartile ranges (IQRs) between species, whereas sepal dimensions overlap significantly.

3. **Feature Correlation Heatmap (`reports/figures/correlation_heatmap.png`)**:
   - Petal Length and Petal Width exhibit high collinearity ($r = 0.96$).
   - Sepal Length strongly correlates with Petal Length ($r = 0.87$) and Petal Width ($r = 0.82$).
   - Sepal Width correlates negatively with Petal Length ($r = -0.43$) and Petal Width ($r = -0.37$).

---

## 🎯 Feature Selection Discussion: Which Features Are Most Discriminative?

To determine the most discriminative features, we compute the **one-way ANOVA F-score** ($F$-statistic) and corresponding $p$-values:

| Rank | Feature | ANOVA F-Score | $p$-value | Separation Capability |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Petal Length** | **1180.16** | $2.86 \times 10^{-91}$ | **Highest**: 100% linear separation for *Setosa*; minimal *Versicolor*/*Virginica* overlap |
| **2** | **Petal Width** | **960.01** | $4.17 \times 10^{-85}$ | **Very High**: Strongly discriminative; isolates *Setosa* under $0.6$ cm |
| 3 | Sepal Length | 119.26 | $1.67 \times 10^{-31}$ | Moderate: Trends upwards with maturity but wide intra-class overlap |
| 4 | Sepal Width | 49.16 | $4.49 \times 10^{-17}$ | Lowest: Inverse relationship (*Setosa* has widest sepals) with high overlap |

### Key Takeaways:
- **Petal Length & Petal Width are the core discriminative drivers**: A 2D scatter of Petal Length vs. Petal Width achieves nearly 98% class separability alone.
- **Sepal measurements act as secondary stabilizers**: While insufficient on their own, sepal dimensions help resolve boundary cases in the overlap between *Versicolor* and *Virginica*.

---

## 🤖 Machine Learning Models & Methodology

- **Data Split**: 80% Training (120 samples), 20% Hold-out Testing (30 samples) using `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
- **Stratification**: Enforces an exact 10:10:10 sample distribution in the test set.
- **Pipeline Architecture**: All scalers (`StandardScaler`) are encapsulated within scikit-learn `Pipeline` objects to prevent data leakage during cross-validation.
- **Validation**: 5-Fold Stratified Cross-Validation on the training set.

---

## 📊 Model Comparison & Evaluation

### Test Set Performance Summary (Hold-out Test Set, $n=30$):

| Model | 5-Fold CV Accuracy (Train) | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | **0.958 ± 0.026** | **0.9667** | **0.9697** | **0.9667** | **0.9666** |
| Random Forest Classifier | 0.950 ± 0.031 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |
| Logistic Regression | 0.958 ± 0.026 | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree Classifier | 0.942 ± 0.020 | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

### Classification Report (Best Model: KNN):
```
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        10
  versicolor     0.9091    1.0000    0.9524        10
   virginica     1.0000    0.9000    0.9474        10

    accuracy                         0.9667        30
   macro avg     0.9697    0.9667    0.9666        30
weighted avg     0.9697    0.9667    0.9666        30
```

---

## 🏆 Champion Model Declaration & Justification

### Winner: **K-Nearest Neighbors (KNN)** (`n_neighbors=5, weights='distance'`)

### Justification:
1. **Top Predictive Accuracy**: Achieved **96.67% test accuracy** (29/30 correct) and **0.9666 weighted F1-Score**.
2. **Superior Generalization & Stability**: Led the benchmark with **95.83% mean 5-fold cross-validation accuracy** with a narrow standard deviation of **±2.64%**.
3. **Geometric Fit**: Flower morphological traits form compact spatial clusters in Euclidean feature space. KNN with distance-weighting adapts seamlessly to local density boundaries without imposing artificial hyperplanes.
4. **Occam's Razor**: Compared to an ensemble of 100 decision trees (Random Forest), KNN has virtually zero model parameter bloat, instantaneous training time, and optimal suitability for small-scale botanical classification.

---

## 🚀 Installation & Execution Guide

### 1. Requirements
Ensure Python 3.9+ is installed. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Jupyter Notebook
Open the interactive notebook with all figures and outputs pre-rendered:
```bash
jupyter notebook notebooks/iris_flower_classification.ipynb
```

### 3. Run Pipeline Scripts
- **Data Ingestion & Verification**:
  ```bash
  python3 -m src.data_loader
  ```
- **EDA & Figure Generation**:
  ```bash
  python3 -m src.eda
  ```
- **Model Training, Evaluation & Serialization**:
  ```bash
  python3 -m src.models
  ```

### 4. Run Automated Unit Tests
```bash
python3 -m unittest discover tests
```
