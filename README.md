# 🦸 Superhero Attributes and Power Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A comprehensive data mining project exploring classification and clustering techniques on superhero data.

**Course**: DSCI 4411 - Fundamentals of Data Mining  
**Institution**: The American University in Cairo  
**Semester**: Fall 2025

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Key Results](#-key-results)
- [Figures & Visualizations](#-figures--visualizations)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Methodology](#-methodology)
- [Conclusions](#-conclusions)

---

## 🎯 Project Overview

### Objectives
1. **Classification**: Predict whether a superhero is a Hero or Villain based on their attributes
2. **Clustering**: Discover natural groupings/archetypes among characters

### Approach
- Tested **19 different machine learning models**
- Created **7 engineered features** to improve predictions
- Applied **hyperparameter tuning** with GridSearchCV
- Explored **3 clustering algorithms** (K-Means, DBSCAN, Hierarchical)

---

## 📊 Dataset

**Source**: [Kaggle Super-Heros Dataset](https://www.kaggle.com/datasets/kenil1719/super-heros)

| Metric | Value |
|--------|-------|
| Total Records | 1,200 |
| Features | 17 original + 7 engineered |
| Target | `is_good` (65% Hero / 35% Villain) |
| Missing Values | None |

### Feature Categories

| Category | Features |
|----------|----------|
| **Physical** | height_cm, weight_kg, age, years_active |
| **Behavioral** | power_level, public_approval_rating, training_hours_per_week, civilian_casualties_past_year |
| **Powers** | super_strength, flight, energy_projection, telepathy, healing_factor, shape_shifting, invisibility, telekinesis |

### Engineered Features
```
total_powers         = sum of all 8 power flags
power_efficiency     = power_level / years_active
training_intensity   = training_hours / age
casualty_rate        = casualties / years_active
approval_power_ratio = approval / power_level
bmi                  = weight / height²
experience_score     = years_active × training_hours
```

---

## 🏆 Key Results

### Classification Performance (Top 5 of 19 Models)

| Model | CV Accuracy | Test Accuracy | F1 Score |
|-------|-------------|---------------|----------|
| 🥇 **LDA** | 63.9% | **65.0%** | 0.778 |
| 🥈 **SVM (Linear)** | 65.0% | **65.0%** | 0.788 |
| 🥉 Logistic Regression | 63.8% | 64.6% | 0.776 |
| AdaBoost | 63.5% | 64.6% | 0.768 |
| Random Forest | 62.6% | 64.2% | 0.768 |

### Best Tuned Model
```
Gradient Boosting (Tuned)
├── Parameters: learning_rate=0.01, max_depth=3, n_estimators=100
├── CV Accuracy: 65.0%
└── Test Accuracy: 65.0%
```

### Top Predictive Features
1. `power_level`
2. `training_intensity` *(engineered)*
3. `training_hours_per_week`

### Clustering Results
| Algorithm | Best Config | Silhouette Score |
|-----------|-------------|------------------|
| **K-Means** | k=2 | **0.167** |
| Hierarchical | n=2, ward | 0.154 |
| DBSCAN | eps=1.5 | Poor |

---

## 📈 Figures & Visualizations

All figures are saved in the `figures/` directory:

### Exploratory Data Analysis

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| ![](figures/target_distribution.png) | **Target Distribution**: 65% Heroes vs 35% Villains | Slight class imbalance |
| ![](figures/power_comparison.png) | **Power Comparison**: Heroes vs Villains power distribution | Powers are equally distributed - no predictive power |
| ![](figures/correlation_heatmap.png) | **Correlation Heatmap**: Feature relationships | No strong correlations with target |

### Classification Results

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| ![](figures/model_comparison_all.png) | **Model Comparison**: All 19 models ranked | Linear models perform as well as complex ones |
| ![](figures/feature_importance_tuned.png) | **Feature Importance**: Random Forest (tuned) | power_level and training_intensity are most important |
| ![](figures/confusion_matrix_best.png) | **Confusion Matrix**: Best model performance | Model is slightly biased toward predicting "Hero" |

### Clustering Results

| Figure | Description | Key Insight |
|--------|-------------|-------------|
| ![](figures/elbow_silhouette.png) | **Elbow + Silhouette**: Optimal k selection | k=2 has best silhouette score |
| ![](figures/clustering_pca_comparison.png) | **PCA Visualization**: Clusters vs ground truth | Clusters are power-based, not morality-based |

---

## 📁 Project Structure

```
superhero_project/
│
├── 📓 superhero_analysis.ipynb          # Main analysis notebook (source code)
├── 📓 superhero_analysis_executed.ipynb # Executed notebook with outputs
│
├── 📄 report.md                         # Comprehensive project report
├── 📄 README.md                         # This file
├── 📄 presentation.md                   # Presentation slides
│
├── 📊 superhero dataset.csv             # Original dataset
├── 📊 superhero_enhanced_clusters.csv   # Dataset with engineered features + clusters
├── 📊 model_comparison_results.csv      # All model metrics
│
├── 📁 figures/                          # All generated visualizations
│   ├── target_distribution.png
│   ├── power_comparison.png
│   ├── correlation_heatmap.png
│   ├── model_comparison_all.png
│   ├── feature_importance_tuned.png
│   ├── confusion_matrix_best.png
│   ├── elbow_silhouette.png
│   ├── clustering_pca_comparison.png
│   └── ... (22 figures total)
│
├── 📁 venv/                             # Python virtual environment
├── 📄 requirements.txt                  # Python dependencies
└── 📄 .gitignore                        # Git ignore rules
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/elbarbary/superhero-classification.git
cd superhero-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the analysis
jupyter notebook superhero_analysis.ipynb
```

### Running the Complete Analysis
```bash
# Execute all cells and save outputs
jupyter nbconvert --to notebook --execute superhero_analysis.ipynb --output superhero_analysis_executed.ipynb
```

---

## 🔬 Methodology

### Classification Pipeline

```
1. Data Loading
       ↓
2. Feature Engineering (7 new features)
       ↓
3. Train/Test Split (80/20, stratified)
       ↓
4. Feature Scaling (StandardScaler)
       ↓
5. Model Training (19 algorithms)
       ↓
6. Hyperparameter Tuning (GridSearchCV)
       ↓
7. Ensemble Methods (Voting, Stacking)
       ↓
8. Evaluation (Accuracy, F1, Confusion Matrix)
```

### Clustering Pipeline

```
1. Feature Selection (behavioral + engineered)
       ↓
2. Scaling (StandardScaler)
       ↓
3. Optimal k Selection (Elbow + Silhouette)
       ↓
4. K-Means / DBSCAN / Hierarchical
       ↓
5. PCA Visualization
       ↓
6. Cluster Profiling
```

### Models Tested

| Category | Count | Models |
|----------|-------|--------|
| Linear | 3 | Logistic Regression, LDA, QDA |
| Tree-based | 7 | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGB, AdaBoost, XGBoost |
| SVM | 3 | Linear, RBF, Polynomial |
| Instance-based | 2 | KNN (k=5, k=10) |
| Probabilistic | 1 | Naive Bayes |
| Neural Network | 3 | MLP (small/medium/large) |
| **Total** | **19** | |

---

## 💡 Conclusions

### Key Findings

1. **🎯 Accuracy Ceiling at ~65%**
   - All 19 models plateau around 65% accuracy
   - Simple linear models perform as well as complex ensembles
   - The dataset lacks signal to distinguish heroes from villains

2. **⚡ Powers Don't Define Morality**
   - All 8 superpowers are equally distributed between heroes and villains
   - A character's abilities don't predict their moral alignment

3. **📊 Top Predictors**
   - Behavioral metrics (power_level, training hours) matter most
   - Engineered features like `training_intensity` ranked highly

4. **🔍 Natural Clusters are Power-Based**
   - K-Means finds high-power vs low-power groups (k=2)
   - Clustering doesn't align with hero/villain labels

### Limitations
- Dataset appears synthetically generated
- Missing narrative features (origin stories, affiliations)
- Binary hero/villain labels oversimplify character complexity

---

## 📝 License

This project is for educational purposes (AUC DSCI 4411 course project).

---

## 👥 Team

**The American University in Cairo - Fall 2025**

---

**⭐ If you found this project useful, please star the repository!**
