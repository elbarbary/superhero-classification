# Superhero Attributes and Power Classification

## Project Report

**DSCI 4411 - Fundamentals of Data Mining**  
**The American University in Cairo - Fall 2025**

---

## 1. Introduction

This project explores classification and clustering techniques using a dataset of 1,200 superheroes/villains. The goal is to:
1. Build models to classify heroes vs villains based on attributes
2. Identify meaningful archetypes through clustering

### Dataset: 1200 records with 17 features
- **Target**: `is_good` (65% Hero, 35% Villain)
- **Features**: Physical (height, weight, age), Behavioral (power_level, casualties, training), Powers (8 binary flags)

---

## 2. Methods

### 2.1 Feature Engineering (7 New Features)
- `total_powers`: Count of all powers
- `power_efficiency`: power_level / years_active
- `training_intensity`: training_hours / age
- `casualty_rate`: casualties / years_active
- `approval_power_ratio`: approval / power_level
- `bmi`: weight / height²
- `experience_score`: years_active × training_hours

### 2.2 Classification Models Tested (19 Total)
| Category | Models |
|----------|--------|
| Linear | Logistic Regression, LDA, QDA |
| Tree-based | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost |
| Instance-based | KNN (k=5, k=10) |
| SVM | Linear, RBF, Polynomial kernels |
| Probabilistic | Naive Bayes |
| Neural Network | MLP (small/medium/large) |
| Ensemble | Voting Classifier, Stacking Classifier |

### 2.3 Hyperparameter Tuning
GridSearchCV with 5-fold CV for RF, GB, and SVM.

### 2.4 Clustering Methods
- K-Means (k=2-9)
- DBSCAN (various eps/min_samples)
- Agglomerative Hierarchical (ward/complete/average linkage)

---

## 3. Results

### 3.1 Classification Performance
| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| **Gradient Boosting (Tuned)** | **65.0%** | **65.0%** |
| SVM (Linear) | 65.0% | 65.0% |
| LDA | 63.9% | 65.0% |
| Logistic Regression | 63.8% | 64.6% |
| Random Forest | 62.6% | 64.2% |

**Key Insight**: All models plateau around 65% accuracy. The dataset lacks strong signal to differentiate hero/villain beyond this ceiling.

### 3.2 Top Predictive Features
1. `power_level`
2. `training_intensity` (engineered)
3. `training_hours_per_week`

### 3.3 Clustering Results
| Method | Best Config | Silhouette |
|--------|-------------|------------|
| **K-Means** | **k=2** | **0.167** |
| Hierarchical | n=2, ward | ~0.15 |
| DBSCAN | Various | Poor separation |

---

## 4. Conclusions

1. **Feature engineering helped**: Engineered features like `training_intensity` ranked highly
2. **Model ceiling ~65%**: The hero/villain label likely depends on unobserved narrative factors
3. **Clustering finds 2 natural groups**: High power vs low power characters
4. **Powers alone don't predict morality**: Behavioral metrics matter more

---

## Appendix: Files
- `superhero_analysis.ipynb`: Complete analysis code
- `superhero_enhanced_clusters.csv`: Dataset with clusters
- `model_comparison_results.csv`: All model metrics
