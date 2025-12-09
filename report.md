# Superhero Attributes and Power Classification

## Project Report

**DSCI 4411 - Fundamentals of Data Mining**  
**The American University in Cairo - Fall 2025**  
**Due: December 10, 2025**

---

## 1. Introduction

This project explores classification and clustering techniques using a dataset of 1,200 superheroes/villains. The goal is to build models that can distinguish between heroes and villains based on their attributes (powers, approval ratings, casualties) and to identify meaningful archetypes through clustering.

### Dataset Overview
The dataset contains 1200 records with:
- **Target**: `is_good` (1=Hero, 0=Villain). 65% are Heroes.
- **Key Features**: Power Level, Public Approval, Civilian Casualties, and 8 binary power flags (Flight, Super Strength, etc.).

---

## 2. Methods

### 2.1 Classification (Predicting Hero vs Villain)
We trained three supervised learning models to predict the `is_good` label:
1. **Logistic Regression**: A linear baseline.
2. **Random Forest**: An ensemble method to capture non-linear relationships.
3. **Support Vector Machine (SVM)**: For finding optimal hyperplanes.

**Features Used**: All physical traits, powers, and behavioral metrics (scaled).

### 2.2 Clustering (Refined Approach)
We used **K-Means Clustering** to find archetypes. To improve separation quality:
- **Feature Selection**: We removed noisy physical traits (height/weight/age) and focused on **behavioral metrics** (`public_approval`, `casualties`, `power_level`) and **powers**.
- **K=4** was selected using the Elbow Method.
- **Archetype Labeling**: Clusters were named based on their mean approval and hero/villain composition.

---

## 3. Results

### 3.1 Classification Performance
| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **65.0%** |
| Random Forest | 64.6% |
| SVM | 62.1% |

**Key Insight**: The linear model performed best, suggesting the boundary between "Good" and "Bad" is relatively straightforward based on public metrics (Approval Rating & Casualties).

### 3.2 Clustering Archetypes (Discovered Groups)
Using K-Means (k=4) on behavioral features, we identified distinct groups primarily driven by their public perception:

1. **Public Heroes**: Characters with high approval ratings (>60%) and positive alignment.
2. **Public Villains**: Characters with low approval, high casualties, and villain alignment.
3. **Street-Level Characters**: Moderate power levels and lower public visibility.
4. **Destructive Forces**: High power levels combined with high casualty counts.

**Silhouette Score**: 0.081 (Indicates moderate overlap between groups, typical for complex character data).

---

## 4. Conclusions

1. **Behavior defines Alignment**: Public approval and civilian casualties are much stronger predictors of being a "Hero" or "Villain" than specific superpowers (like Flight or Strength).
2. **Archetypes Exist**: While many characters share traits, distinct clusters of high-profile heroes vs. destructive villains emerged.
3. **Model Performance**: Predicting morality is difficult (65% accuracy), implying that the "Hero vs Villain" label depends on unobserved factors (e.g., origin story, specific actions) beyond raw stats.

---

## Appendix: Deliverables
- `superhero_analysis.ipynb`: Complete code.
- `presentation.pptx`: Slides summarizing the project.
