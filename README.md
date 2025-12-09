# Superhero Attributes and Power Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A data mining project exploring classification and clustering techniques on a dataset of 1,200 superheroes.

## 📖 Project Overview

This project analyzes superhero data to:
1. **Classify** characters as heroes or villains using machine learning
2. **Cluster** superheroes into archetypes based on their attributes

## 🗂️ Dataset

**Source**: [Kaggle Super-Heros Dataset](https://www.kaggle.com/datasets/kenil1719/super-heros)

| Feature Type | Attributes |
|--------------|------------|
| Physical | height_cm, weight_kg, age |
| Experience | years_active, training_hours_per_week |
| Behavioral | civilian_casualties_past_year, power_level, public_approval_rating |
| Powers (binary) | super_strength, flight, energy_projection, telepathy, healing_factor, shape_shifting, invisibility, telekinesis |
| Target | is_good (1=Hero, 0=Villain) |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/superhero-classification.git
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

## 📊 Results Summary

### Classification Performance
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~72% |
| Random Forest | ~68% |
| SVM | ~70% |

### Discovered Archetypes
- **Public Heroes**: High power + high approval
- **Street-Level Heroes**: Moderate power, community focus
- **Vigilantes**: High power, mixed approval
- **Supervillains**: High power + high casualties

## 📁 Project Structure

```
superhero-classification/
├── superhero_analysis.ipynb     # Main analysis notebook
├── superhero dataset.csv        # Original data
├── superhero_with_clusters.csv  # Data with cluster labels
├── report.md                    # Full project report
├── figures/                     # Generated visualizations
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 👥 Team

DSCI 4411 - Fundamentals of Data Mining  
The American University in Cairo - Fall 2025

## 📜 License

MIT License
