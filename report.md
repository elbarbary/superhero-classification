# Superhero Attributes and Power Classification

## Project Report

**DSCI 4411 - Fundamentals of Data Mining**  
**The American University in Cairo - Fall 2025**  
**Due: December 10, 2025**

---

## 1. Introduction

This project explores classification and clustering techniques using a dataset of 1,200 superheroes with various attributes including powers, physical traits, and behavioral data. The primary objectives are:

1. **Classification**: Build models to predict whether a character is a hero or villain based on their attributes
2. **Clustering**: Identify superhero archetypes by grouping similar characters together

### Dataset Overview

The dataset contains 17 features:
- **Physical Attributes**: height (cm), weight (kg), age
- **Experience Metrics**: years_active, training_hours_per_week
- **Behavioral Data**: civilian_casualties_past_year, power_level, public_approval_rating
- **Power Flags** (binary): super_strength, flight, energy_projection, telepathy, healing_factor, shape_shifting, invisibility, telekinesis
- **Target Variable**: is_good (1=Hero, 0=Villain)

The dataset is balanced with 65% heroes (780) and 35% villains (420), with no missing values.

---

## 2. Methods

### 2.1 Data Exploration and Preprocessing

**Exploratory Data Analysis (EDA):**
- Distribution analysis of numerical features by hero/villain status
- Power frequency analysis across all characters
- Correlation heatmap to identify feature relationships
- Box plots comparing heroes vs villains on key metrics

**Data Preprocessing:**
- Train/test split (80/20) with stratification
- StandardScaler normalization for SVM and Logistic Regression

### 2.2 Classification Models

Three classifier models were implemented and compared:

1. **Logistic Regression**: Linear baseline model with L2 regularization
2. **Random Forest Classifier**: Ensemble method with 100 trees, max_depth=10
3. **Support Vector Machine (SVM)**: RBF kernel with C=1.0

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, 5-fold Cross-Validation

### 2.3 Clustering Analysis

**K-Means Clustering:**
- Elbow method and Silhouette Score to determine optimal k
- K=4 clusters selected for interpretable superhero archetypes
- PCA (2 components) for visualization
- Cluster profiling to identify archetype characteristics

---

## 3. Results

### 3.1 Classification Results

| Model | Test Accuracy | CV Accuracy |
|-------|---------------|-------------|
| Logistic Regression | ~72% | ~71% (±3%) |
| Random Forest | ~68% | ~67% (±4%) |
| SVM | ~70% | ~69% (±3%) |

**Key Finding**: Logistic Regression achieved the best performance, suggesting a relatively linear decision boundary between heroes and villains.

**Most Important Features for Classification:**
1. public_approval_rating - Strongest predictor
2. civilian_casualties_past_year - Villains have higher casualties
3. power_level - Higher power levels associated with heroism
4. training_hours_per_week - Similar across both groups

### 3.2 Clustering Results

Four distinct superhero archetypes were discovered:

1. **Public Heroes**: High power, high approval, low casualties
2. **Street-Level Heroes**: Moderate power, community focus
3. **Vigilante Heroes**: High power but lower public approval
4. **Supervillains**: High power combined with high casualty rates

**Silhouette Score**: ~0.12-0.15 (moderate cluster separation due to feature overlap)

### 3.3 Key Visualizations

The analysis generated the following visualizations:
- Target distribution pie/bar charts
- Numerical feature distributions by class
- Power distribution comparisons (heroes vs villains)
- Correlation heatmap
- Feature importance charts (LR, RF)
- Model comparison bar charts
- Confusion matrices for all three models
- Elbow method and silhouette score plots
- PCA cluster visualization
- Cluster characteristic analysis

---

## 4. Conclusions

### Key Insights

1. **Behavioral features matter more than powers**: Public approval rating and civilian casualties are stronger predictors of hero/villain status than specific superpowers.

2. **Powers are evenly distributed**: All 8 superpowers appear in similar proportions among heroes and villains, indicating powers don't determine morality.

3. **Four distinct archetypes exist**: K-Means clustering revealed meaningful groupings that align with comic book tropes (public heroes, vigilantes, street-level heroes, villains).

4. **Linear models perform well**: The success of Logistic Regression suggests the hero/villain distinction follows relatively straightforward patterns.

### Limitations

- Dataset is synthetic and may not reflect real comic book character distributions
- Limited feature set (no origin story, affiliations, or universe data)
- Binary classification doesn't capture anti-heroes or morally complex characters

### Future Work

- Include character names and universe affiliations for richer analysis
- Implement additional classifiers (XGBoost, Neural Networks)
- Explore hierarchical clustering for archetype refinement
- Time-series analysis if temporal data becomes available

---

## Appendix: Files

- `superhero_analysis.ipynb` - Jupyter notebook with all code and analysis
- `superhero dataset.csv` - Original dataset
- `superhero_with_clusters.csv` - Dataset with cluster assignments
- `figures/` - All generated visualizations
- `README.md` - Project documentation
