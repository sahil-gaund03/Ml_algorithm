<div align="center">

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Syne&weight=800&size=34&pause=1000&color=8B5CF6&center=true&vCenter=true&width=700&lines=🧠+ML+Algorithm+Handbook;Clustering+%7C+PCA+%7C+Regression+%7C+Trees;Learn+by+Building+%26+Exploring" alt="ML Algorithm Handbook" />
</h1>

<p><b>A hands-on Jupyter notebook covering the essential Machine Learning algorithms —<br>
from unsupervised clustering to supervised classification, with real datasets and visual walkthroughs.</b></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3b82f6?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Plotly-3D4DB7?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Algorithms Covered-6-8b5cf6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Datasets Used-6-06b6d4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Challenges %26 Solutions-4-f59e0b?style=flat-square"/>
  <img src="https://img.shields.io/badge/Visualizations-15%2B-ef4444?style=flat-square"/>
</p>

---

</div>

## 📌 Table of Contents

- [✨ Overview](#-overview)
- [🗂️ What's Inside](#️-whats-inside)
- [🧠 Algorithms Covered](#-algorithms-covered)
- [📊 Datasets Used](#-datasets-used)
- [📂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📈 Key Concepts Explained](#-key-concepts-explained)
- [🧰 Tech Stack](#-tech-stack)
- [🌟 Roadmap](#-roadmap)
- [👨‍💻 Author](#-author)
- [📜 License](#-license)

---

## ✨ Overview

This repository is a practical, code-first guide to the core Machine Learning algorithms. Each section follows a structured approach:

```
Concept Explanation  →  Math Intuition  →  Manual Implementation  →  sklearn Example  →  Real Dataset  →  Challenge
```

Whether you're a first-year CS student or a self-taught practitioner, this notebook walks you through the **why** and **how** of each algorithm — not just the API calls.

> 📍 **Author:** Sahil — First-year student at Elphinstone College, Mumbai  
> 🏅 **IBM Certified** in Python for Data Science & AI

---

## 🗂️ What's Inside

The notebook is organized into **5 major chapters**, each with theory, worked examples, real-world applications, and a hands-on challenge:

| # | Chapter | Type | Highlights |
|:--|:--------|:-----|:-----------|
| 1 | **K-Means Clustering** | Unsupervised | Elbow method, Silhouette analysis, Convergence animation |
| 2 | **PCA** | Dimensionality Reduction | Scree plot, Feature weights, 3D scatter via Plotly |
| 3 | **Linear Regression** | Supervised / Regression | Anscombe's Quartet, Manual slope/intercept, Aircraft data |
| 4 | **Logistic Regression** | Supervised / Classification | Sigmoid curve, Eye-movements dataset, Titanic survival |
| 5 | **Decision Trees + Ensembles** | Supervised | Depth sweep, Overfitting viz, Random Forest & XGBoost |

---

## 🧠 Algorithms Covered

### 🔵 K-Means Clustering *(Unsupervised)*

Groups unlabeled data by minimizing distance to cluster centroids. The notebook covers:
- Iterative centroid convergence (animated over 10 iterations)
- **Elbow Method** — finding optimal K via inertia
- **Silhouette Analysis** — measuring cluster cohesion & separation
- Applied to: Iris dataset + Australian Electricity dataset + Titanic (challenge)

```
Choose K  →  Assign points to nearest centroid  →  Recompute centroids  →  Repeat until convergence
```

---

### 🟣 PCA — Principal Component Analysis *(Dimensionality Reduction)*

Finds linear combinations of features that capture the most variance. Covers:
- `explained_variance_ratio_` and **Scree Plot**
- Component weights (how much each feature contributes per PC)
- Manual PCA via `numpy.linalg.eig` — no sklearn!
- 3D scatter plots of the first 3 principal components using **Plotly**
- Also covers: **t-SNE** and **UMAP** as alternative techniques

---

### 🟡 Linear Regression *(Supervised · Regression)*

Fits a line by minimizing sum of squared errors. Covers:
- Manual calculation of slope (β₁) and intercept (β₀) from scratch
- Anscombe's Quartet — same stats, very different data shapes
- **Aircraft Elevators** real-world dataset (R² scoring, feature importance)
- Standardization with `StandardScaler` before fitting
- Comparison with XGBoost regressor

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$$

---

### 🟠 Logistic Regression *(Supervised · Classification)*

Outputs class probabilities via the sigmoid function. Covers:
- Sigmoid curve visualization
- Manual threshold annotation on decision boundary
- **Eye Movements** dataset (multi-feature classification)
- Titanic survival prediction (challenge + solution)
- Head-to-head with XGBoost classifier

$$y = \frac{1}{1 + e^{-z}}$$

---

### 🔴 Decision Trees + Ensembles *(Supervised · Both)*

Splits data by minimizing entropy/MSE at each node. Covers:
- Decision stumps (depth=1) → deep trees → unlimited depth
- **Depth sweep** — plotting train vs test score to find the sweet spot
- Overfitting detection and annotation
- `plot_tree` visualization with filled nodes
- **Random Forest** — bagging ensemble comparison
- **XGBoost** — gradient boosting benchmark

```
Overfitting zone identified:  Train score ↑↑  |  Test score ↓  →  Prune the tree!
```

---

## 📊 Datasets Used

| Dataset | Size | Task | Source |
|:--------|:-----|:-----|:-------|
| **Iris** | 150 rows · 4 features | Clustering / PCA | `sklearn.datasets` |
| **Titanic** | 1,309 rows · 11 features | Classification / Regression | GitHub (mattharrison) |
| **Australian Electricity** | 45,312 rows · 8 features | Clustering | OpenML via HuggingFace |
| **Aircraft Elevators** | 16,599 rows · 18 features | Regression / Trees | OpenML via HuggingFace |
| **Eye Movements** | 10,936 rows · 23 features | Classification | OpenML via HuggingFace |
| **Anscombe's Quartet** | 11 rows · 4 series | Regression demo | Manual |

---

## 📂 Project Structure

```bash
ML-Algorithm/
│
├── 📓 ML algorithm.ipynb        # Main notebook — all algorithms, examples & challenges
└── 📖 README.md                 # You are here
```

> **Note:** The notebook is self-contained. All datasets are loaded inline via `sklearn.datasets` or the HuggingFace `datasets` library. No manual download required.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### 1 · Clone the Repository

```bash
git clone https://github.com/sahil-gaund03/ML-Algorithm.git
cd ML-Algorithm
```

### 2 · Create & Activate a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3 · Install Dependencies

```bash
pip install scikit-learn pandas matplotlib plotly xgboost yellowbrick datasets umap-learn notebook
```

### 4 · Launch the Notebook

```bash
jupyter notebook "ML algorithm.ipynb"
```

---

## 📈 Key Concepts Explained

### How to Find the Best K (K-Means)

```
Elbow Method:        Plot inertia vs K  →  Find the "elbow" where improvement slows
Silhouette Method:   Scores range -1 to 1  →  Higher = better separated clusters
```

### How to Avoid Overfitting (Decision Trees)

```
Depth 1–5:   Underfitting  →  Model too simple to capture patterns
Depth ~10:   Sweet spot    →  Generalizes well to unseen data
Depth 15+:   Overfitting   →  Memorizes training data, fails on test data
```

### When to Use Each Algorithm

| Situation | Recommended Algorithm |
|:----------|:----------------------|
| No labels, find natural groups | K-Means Clustering |
| Too many features, need compression | PCA / t-SNE / UMAP |
| Predict a continuous number | Linear Regression |
| Predict yes/no, true/false | Logistic Regression |
| Interpretable rules from data | Decision Tree |
| Best accuracy, less interpretability | Random Forest / XGBoost |

---

## 🧰 Tech Stack

<div align="center">

| Library | Purpose |
|:--------|:--------|
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) | Core ML algorithms and preprocessing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data manipulation and grouping |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Manual math, eigenvectors for PCA |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c) | Static charts and tree visualization |
| ![Plotly](https://img.shields.io/badge/Plotly-3D4DB7?logo=plotly&logoColor=white) | Interactive 3D scatter plots |
| ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4) | Gradient boosting benchmarks |
| ![Yellowbrick](https://img.shields.io/badge/Yellowbrick-ffcc00) | Silhouette visualizations |
| ![UMAP](https://img.shields.io/badge/UMAP--learn-8b5cf6) | Non-linear dimensionality reduction |
| ![HuggingFace](https://img.shields.io/badge/🤗_datasets-FFD21E) | Real-world dataset loading |

</div>

---

## 🌟 Roadmap

- [x] K-Means Clustering — Iris & Electricity datasets
- [x] Elbow Method & Silhouette Analysis
- [x] PCA — manual numpy implementation + sklearn
- [x] t-SNE and UMAP dimensionality reduction
- [x] Linear Regression — manual + sklearn + XGBoost
- [x] Logistic Regression — sigmoid, boundaries, real data
- [x] Decision Trees — depth sweep & overfitting analysis
- [x] Random Forest & XGBoost benchmarks
- [x] Challenge + Solution for every chapter
- [ ] Support Vector Machines (SVM)
- [ ] Neural Networks (MLPClassifier)
- [ ] Cross-validation & hyperparameter tuning
- [ ] SHAP explainability plots
- [ ] Streamlit interactive dashboard

---

## 👨‍💻 Author

<div align="center">

**Sahil Gaund**  
First-year · Elphinstone College, Mumbai  
🏅 IBM Certified in Python for Data Science & AI

[![GitHub](https://img.shields.io/badge/GitHub-@sahil--gaund03-181717?style=for-the-badge&logo=github)](https://github.com/sahil-gaund03)

*If this helped you learn, a ⭐ on GitHub goes a long way!*

</div>

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<sub>Built with curiosity · Python · scikit-learn · and a lot of Iris data 🌸</sub>
</div>
