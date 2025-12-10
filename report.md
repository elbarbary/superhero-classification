# Superhero Attributes and Power Classification

## Comprehensive Project Report

**DSCI 4411 - Fundamentals of Data Mining**  
**The American University in Cairo - Fall 2025**  
**Project 8: Superhero Attributes and Power Classification**

---

## 📋 Table of Contents
1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Methodology](#3-methodology)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Classification Results](#5-classification-results)
6. [Clustering Analysis](#6-clustering-analysis)
7. [Discussion](#7-discussion)
8. [Conclusions](#8-conclusions)
9. [Code & Figures Reference](#9-code--figures-reference)

---

## 1. Introduction

### Problem Statement
The goal of this project is to analyze a dataset of 1,200 superheroes and villains to:
1. **Classification**: Build machine learning models to predict whether a character is a hero or villain based on their attributes (powers, physical traits, behavioral metrics)
2. **Clustering**: Identify natural groupings or "archetypes" among characters to understand patterns in superhero universes

### Motivation
Understanding what distinguishes heroes from villains and identifying character archetypes has applications in:
- Content recommendation systems for comics/movies
- Character design and storytelling
- Understanding narrative patterns across fictional universes

---

## 2. Dataset Description

### Source
**Kaggle Super-Heros Dataset**: https://www.kaggle.com/datasets/kenil1719/super-heros

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Records | 1,200 |
| Total Features | 17 |
| Target Variable | `is_good` (binary) |
| Class Balance | 65% Heroes / 35% Villains |
| Missing Values | None |

### Feature Categories

#### Physical Attributes (4 features)
| Feature | Description | Range |
|---------|-------------|-------|
| `height_cm` | Height in centimeters | 150 - 250 |
| `weight_kg` | Weight in kilograms | 45 - 128 |
| `age` | Character age | 18 - 100+ |
| `years_active` | Years as a superhero/villain | 1 - 50 |

#### Behavioral Metrics (4 features)
| Feature | Description | Range |
|---------|-------------|-------|
| `training_hours_per_week` | Training intensity | 0 - 60 |
| `civilian_casualties_past_year` | Collateral damage | 0 - 10 |
| `power_level` | Overall power rating | 0 - 100 |
| `public_approval_rating` | Public perception | 0 - 100 |

#### Power Flags (8 binary features)
| Power | % of Characters |
|-------|-----------------|
| `super_strength` | 28.8% |
| `flight` | 31.4% |
| `energy_projection` | 30.1% |
| `telepathy` | 30.4% |
| `healing_factor` | 30.8% |
| `shape_shifting` | 31.7% |
| `invisibility` | 31.5% |
| `telekinesis` | 31.8% |

---

## 3. Methodology

### 3.1 Data Preprocessing

#### Feature Engineering (7 New Features Created)
To improve model performance, we engineered 7 new features:

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `total_powers` | Sum of all 8 power flags | Total number of abilities |
| `power_efficiency` | power_level / (years_active + 1) | Power gained per year |
| `training_intensity` | training_hours / (age + 1) | Relative training effort |
| `casualty_rate` | casualties / (years_active + 1) | Damage per year active |
| `approval_power_ratio` | approval / (power_level + 1) | Public perception vs power |
| `bmi` | weight / (height/100)² | Body mass index |
| `experience_score` | years_active × training_hours | Combined experience metric |

#### Data Scaling
- **StandardScaler** applied for algorithms sensitive to feature scale (SVM, KNN, Neural Networks)
- Tree-based models (Random Forest, Gradient Boosting) used unscaled data

### 3.2 Classification Approach

We tested **19 different classification algorithms** across 6 categories:

| Category | Models Tested |
|----------|---------------|
| **Linear** | Logistic Regression, LDA, QDA |
| **Tree-based** | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, XGBoost |
| **SVM** | Linear kernel, RBF kernel, Polynomial kernel |
| **Instance-based** | KNN (k=5), KNN (k=10) |
| **Probabilistic** | Gaussian Naive Bayes |
| **Neural Network** | MLP (50), MLP (100,50), MLP (100,100,50) |
| **Ensemble** | Voting Classifier, Stacking Classifier |

#### Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation for top models:

**Random Forest Parameters:**
- n_estimators: [100, 200, 300]
- max_depth: [5, 10, 15, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

**Gradient Boosting Parameters:**
- n_estimators: [100, 200]
- learning_rate: [0.01, 0.1, 0.2]
- max_depth: [3, 5, 7]

**SVM Parameters:**
- C: [0.1, 1, 10, 100]
- gamma: ['scale', 'auto', 0.1, 0.01]
- kernel: ['rbf', 'poly']

### 3.3 Clustering Approach

Three clustering algorithms tested:

| Algorithm | Parameters Tested |
|-----------|-------------------|
| **K-Means** | k = 2 to 9 |
| **DBSCAN** | eps = [0.5, 1.0, 1.5, 2.0], min_samples = [3, 5, 10] |
| **Agglomerative** | n_clusters = [2, 3, 4, 5], linkage = ['ward', 'complete', 'average'] |

**Clustering Features Selected:**
- power_level
- civilian_casualties_past_year
- training_hours_per_week
- years_active
- public_approval_rating
- total_powers (engineered)
- power_efficiency (engineered)

---

## 4. Exploratory Data Analysis

### Figure 1: Target Distribution
![Target Distribution](figures/target_distribution.png)

**What it shows:** The class balance of heroes vs villains in the dataset.
- **Heroes (1)**: 780 characters (65%)
- **Villains (0)**: 420 characters (35%)
- **Implication**: Slight class imbalance, but not severe enough to require SMOTE or undersampling.

---

### Figure 2: Power Comparison (Heroes vs Villains)
![Power Comparison](figures/power_comparison.png)

**What it shows:** Bar chart comparing the percentage of heroes vs villains with each superpower.
- **Key Finding**: Powers are distributed almost equally between heroes and villains
- **Implication**: Individual powers alone cannot predict morality - a character's powers don't determine if they're good or evil

---

### Figure 3: Correlation Heatmap
![Correlation Heatmap](figures/correlation_heatmap.png)

**What it shows:** Pearson correlation between all numerical features.
- **Notable Correlations:**
  - Height and weight are positively correlated (expected)
  - Power flags show no correlation with `is_good` (near zero)
  - No strong multicollinearity issues
- **Implication**: Features are relatively independent; no need to remove highly correlated features

---

### Figure 4: Model Comparison (All 19 Models)
![Model Comparison All](figures/model_comparison_all.png)

**What it shows:** Horizontal bar chart comparing test accuracy and cross-validation accuracy for all 19 classification models.
- **Top Performers**: LDA, SVM (Linear), Logistic Regression all achieve ~65%
- **Worst Performers**: XGBoost, some MLP configurations
- **Implication**: Simple linear models perform as well as complex ones - the problem is linearly separable but with limited signal

---

### Figure 5: Feature Importance (Tuned Random Forest)
![Feature Importance Tuned](figures/feature_importance_tuned.png)

**What it shows:** Feature importance scores from the hyperparameter-tuned Random Forest model.
- **Top 3 Features:**
  1. `power_level`
  2. `training_intensity` (engineered)
  3. `training_hours_per_week`
- **Implication**: Behavioral metrics matter more than specific powers; feature engineering was valuable

---

### Figure 6: Confusion Matrix (Best Model)
![Confusion Matrix Best](figures/confusion_matrix_best.png)

**What it shows:** Confusion matrix for the best-performing model (Gradient Boosting Tuned).
- **True Positives (Heroes correctly classified)**: High
- **True Negatives (Villains correctly classified)**: Moderate
- **Implication**: Model is slightly biased toward predicting "Hero" due to class imbalance

---

### Figure 7: Elbow Method & Silhouette Analysis
![Elbow Silhouette](figures/elbow_silhouette.png)

**What it shows:** Combined plot of K-Means inertia (elbow method) and silhouette scores for different k values.
- **Best k**: k=2 has highest silhouette score (0.167)
- **Implication**: The data naturally forms 2 main clusters (likely corresponding to high-power vs low-power characters)

---

### Figure 8: Clustering PCA Visualization
![Clustering PCA Comparison](figures/clustering_pca_comparison.png)

**What it shows:** Side-by-side comparison of K-Means cluster assignments vs actual hero/villain labels in PCA-reduced space.
- **Left**: K-Means clusters based on behavioral features
- **Right**: Actual ground truth labels
- **Implication**: Clusters don't perfectly align with hero/villain labels - unsupervised clustering finds different patterns than the supervised target

---

## 5. Classification Results

### 5.1 All Models Comparison

| Rank | Model | CV Accuracy | Test Accuracy | F1 Score |
|------|-------|-------------|---------------|----------|
| 1 | **LDA** | 63.9% | **65.0%** | 0.778 |
| 2 | **SVM (Linear)** | 65.0% | **65.0%** | 0.788 |
| 3 | Logistic Regression | 63.8% | 64.6% | 0.776 |
| 4 | AdaBoost | 63.5% | 64.6% | 0.768 |
| 5 | Random Forest | 62.6% | 64.2% | 0.768 |
| 6 | Gradient Boosting | 62.9% | 63.3% | 0.766 |
| 7 | KNN (k=5) | 57.7% | 60.8% | 0.737 |
| ... | ... | ... | ... | ... |
| 19 | XGBoost | 58.5% | 58.3% | 0.708 |

### 5.2 Hyperparameter Tuning Results

| Model | Best Parameters | CV Score | Test Score |
|-------|-----------------|----------|------------|
| Random Forest | max_depth=15, n_estimators=200, min_samples_leaf=4 | 65.7% | 63.3% |
| Gradient Boosting | learning_rate=0.01, max_depth=3, n_estimators=100 | 65.0% | **65.0%** |
| SVM | C=1, gamma='scale', kernel='poly' | 65.0% | 62.1% |

### 5.3 Ensemble Methods

| Ensemble | CV Accuracy | Test Accuracy |
|----------|-------------|---------------|
| Voting (RF + GB + LR) | 64.1% | 63.8% |
| Stacking (RF + GB + KNN → LR) | 62.3% | 63.3% |

**Key Finding**: Ensemble methods did NOT outperform individual tuned models, indicating we've reached the performance ceiling for this dataset.

---

## 6. Clustering Analysis

### 6.1 K-Means Results

| k | Silhouette Score | Interpretation |
|---|------------------|----------------|
| 2 | **0.167** (Best) | High-power vs Low-power characters |
| 3 | 0.142 | Adds mid-tier power level |
| 4 | 0.128 | Further fragmentation |
| 5+ | <0.12 | Diminishing returns |

### 6.2 DBSCAN Results
- Most configurations produced either 1 giant cluster or excessive noise points
- Best: eps=1.5, min_samples=5 → 3 clusters, many noise points
- **Conclusion**: DBSCAN not ideal for this uniformly distributed data

### 6.3 Hierarchical Clustering Results

| n_clusters | Linkage | Silhouette |
|------------|---------|------------|
| 2 | ward | 0.154 |
| 2 | complete | 0.143 |
| 3 | ward | 0.121 |

### 6.4 Final Cluster Archetypes (k=2)

| Cluster | Size | Power Level | Casualties | Training | Character Type |
|---------|------|-------------|------------|----------|----------------|
| **0** | ~600 | High (60+) | Higher | Moderate | High-Power Characters |
| **1** | ~600 | Low-Mid (<60) | Lower | Varies | Regular Characters |

---

## 7. Discussion

### 7.1 Why Classification Accuracy Plateaus at ~65%

Several factors limit model performance:

1. **Weak Feature-Target Correlation**: The correlation between features and `is_good` is near zero for most features
2. **Balanced Power Distribution**: Heroes and villains have nearly identical power distributions
3. **Missing Narrative Features**: The hero/villain distinction in comics often depends on:
   - Origin story
   - Motivations and intentions
   - Specific narrative events
   - Affiliation (Avengers vs. Hydra)
   
   None of these are captured in the numerical dataset.

4. **Synthetic Data Limitations**: The dataset appears to be synthetically generated with random assignments, explaining the weak signal

### 7.2 Feature Engineering Impact

The engineered features provided marginal improvement:
- `training_intensity` ranked #2 in feature importance
- `power_efficiency` ranked in top 10
- However, overall accuracy gain was only ~1-2%

### 7.3 Clustering Insights

The clustering analysis revealed that:
- The data naturally splits into **power tiers** rather than hero/villain groups
- Character power level is a stronger organizing principle than moral alignment
- This aligns with comic book lore where heroes and villains span all power levels

---

## 8. Conclusions

### Key Findings

1. **Linear models match complex models**: Logistic Regression performs as well as Random Forest, Gradient Boosting, and Neural Networks, indicating the problem is simple but lacks signal

2. **Accuracy ceiling of ~65%**: Despite testing 19 models, hyperparameter tuning, feature engineering, and ensemble methods, accuracy cannot exceed ~65%

3. **Powers don't determine morality**: All 8 superpowers are equally distributed between heroes and villains

4. **Natural clusters are power-based**: Unsupervised clustering finds high-power vs low-power groups, not hero vs villain

5. **Top predictive features**: power_level, training_intensity, training_hours_per_week

### Recommendations for Future Work

1. **Acquire richer data**: Text descriptions, origin stories, team affiliations
2. **Multi-class classification**: Predict specific alignments (Lawful Good, Chaotic Evil, etc.)
3. **Graph-based analysis**: Model relationships between characters
4. **Deep learning on text**: Use NLP on character bios if available

---

## 9. Code & Figures Reference

### 📁 Final Code Location
The complete analysis is in:
```
/home/barbary/data_mining/superhero_project/superhero_analysis.ipynb
```

The executed notebook with all outputs is:
```
/home/barbary/data_mining/superhero_project/superhero_analysis_executed.ipynb
```

### 📊 Figures Generated

| Figure File | Description | Code Section |
|-------------|-------------|--------------|
| `target_distribution.png` | Hero/Villain class distribution | Section 2: EDA |
| `power_comparison.png` | Powers by hero vs villain | Section 2: EDA |
| `correlation_heatmap.png` | Feature correlation matrix | Section 2: EDA |
| `model_comparison_all.png` | All 19 models comparison | Section 4: Classification |
| `feature_importance_tuned.png` | RF feature importance | Section 7: Best Model |
| `confusion_matrix_best.png` | Best model confusion matrix | Section 7: Best Model |
| `elbow_silhouette.png` | Optimal k selection | Section 8: Clustering |
| `clustering_pca_comparison.png` | PCA cluster visualization | Section 8: Clustering |

### 📄 Output Files

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Metrics for all 19 models |
| `superhero_enhanced_clusters.csv` | Dataset with engineered features + cluster labels |

### 🔧 Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
xgboost>=1.7.0 (optional)
```

---

**Report prepared for DSCI 4411 - Fundamentals of Data Mining**  
**The American University in Cairo - Fall 2025**
