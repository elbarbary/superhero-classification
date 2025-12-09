# Superhero Classification Project
## DSCI 4411 - Fall 2025

---

# 1. Project Goal

> Analyze patterns across superheroes to classify them and identify archetypes.

### Objectives
1. **Classification**: Predict Hero vs Villain (`is_good`).
2. **Clustering**: Discover character archetypes (e.g., "Vigilante", "God-tier").

---

# 2. Dataset

**1200 Characters**
*   **Target**: 65% Heroes, 35% Villains
*   **Features**:
    *   **Behavior**: Public Approval, Casualties, Power Level
    *   **Powers**: Flight, Strength, Telepathy, etc.
    *   **Physical**: Height, Weight, Age

---

# 3. Approach

### Classification Models
*   **Logistic Regression** (Baseline)
*   **Random Forest** (Tree-based)
*   **SVM** (Kernel-based)

### Clustering Strategy (Improved)
*   **Feature Selection**: Removed noise (Height, Weight). Focused on Behavior + Powers.
*   **Algorithm**: K-Means (k=4).
*   **Naming**: Automated logic based on Approval & Alignment.

---

# 4. Results: Classification

| Model | Accuracy |
| :--- | :--- |
| **Logistic Regression** | **65.0%** 🏆 |
| Random Forest | 64.6% |
| SVM | 62.1% |

**Insight**: Public metrics (Approval, Casualties) are the best predictors.

---

# 5. Results: Clustering

We discovered **4 Main Archetypes**:

1.  **Public Heroes** (High Approval, Good)
2.  **Public Villains** (High Casualties, Bad)
3.  **Street-Level Heroes** (Lower Power, Local impact)
4.  **Destructive Forces** (High Power, High Casualties)

---

# 6. Conclusion

*   **Behavior > Powers**: Morality is defined by actions (casualties), not abilities.
*   **Linear Separation**: Simple models work best for this data.
*   **Archetypes**: Distinct groups exist based on public perception.

### Thank You!
