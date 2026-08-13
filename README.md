# HybridCXR

Hybrid machine learning models for Chest X-Ray (CXR) disease classification using multiple algorithms.

## Overview

This project implements and compares three machine learning models (GBM, Random Forest, SVM) for classifying Chest X-Ray images. Features are extracted and visualized using PCA, t-SNE, and zone importance analysis.

## Models

- **Gradient Boosting Machine (GBM)**
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**

## Project Structure

`
HybridCXR/
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
├── results.json        # Model comparison results
├── models/             # Trained model files (.pkl)
├── assets/             # Visualization images
│   ├── comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── tsne_pca.png
│   └── zone_importance.png
└── .streamlit/
    └── config.toml
`

## Installation

`ash
pip install -r requirements.txt
streamlit run app.py
`

## Results

Model performance comparison stored in esults.json. Visualizations include confusion matrices, ROC curves, feature importance, and dimensionality reduction plots.

## Tech Stack

- Python
- Streamlit
- Scikit-learn (GBM, Random Forest, SVM)
- Matplotlib / Seaborn
- Pandas / NumPy

## License

MIT
