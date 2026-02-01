# Fraud & Default Prediction with Machine Learning

![](https://ci3.googleusercontent.com/meips/ADKq_NaZCZa8TRZab7adCDhWVBh0C8pJc0Jr2xzt3QqQGJ1e5pxO0rGYrvkSuoCbbTQQLvfK3tfAQ2tHb8Imnrm_yb2UasiNxtGa_Voi2mzqYeeORjqrqPUchRknoY-TSiMVQWOm9deSAOQUgdbprPxmWBrRt0mEtvAQBOA8mImdAnJ5xrIk7xQN_ht9IHpSgEsQbSq2lhotAiXlAXhopw1qCOt4cWUlJQ=s0-d-e1-ft#https://quickin-media-production.s3.sa-east-1.amazonaws.com/pzOWjW04KkvJQuz8sCb7JeZTzd4BrDRjoTzTOpj7K8L8ByBnzellwfi7YPZozUYo/1733919539795.jpg)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-blue)

## Project Overview

This project aims to build a **fraud and default prediction model** using Machine Learning techniques to classify customers as:

- Adimplente (Non-default)
- Inadimplente (Default/Fraud risk)

The project follows a complete data science pipeline:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Class balancing with SMOTE
- Model training and comparison
- Hyperparameter tuning with K-Fold
- Threshold optimization
- Model deployment with Streamlit

## Business Problem

Financial institutions need to reduce credit risk by identifying customers with higher probability of default.

**Objective:**
> Predict the probability of customer default using demographic, behavioral and transactional features.

## Dataset Features

Main variables used:
- `idade_cliente`
- `renda`
- `valor_compra`
- `score_pessoa`
- `score_email`
- `tipo_de_cliente`
- `produto_1, produto_2, produto_3`
- `uf`
- `dia_semana`
- `hora_da_compra`

Target:
- `status_fraude` (0 = Adimplente, 1 = Inadimplente)

## Exploratory Data Analysis (EDA)

Key insights:
- Higher default rate among low-income groups
- Some Brazilian states show higher default rates
- Fraudulent clients have lower credit scores
- Score_person and score_email show strong separation between classes

Visualizations include:
- Boxplots (scores vs fraud)
- Default rate by income range
- Default rate by state (UF)
- Feature distributions
- Correlation analysis

## Data Preprocessing

Steps performed:
1. Missing values handling
2. Outlier detection using Z-score
3. Feature engineering for categorical variables
4. Encoding categorical variables
5. Feature scaling
6. Train/Test split

## Class Imbalance Treatment (SMOTE)

Before SMOTE: 0 = 22949 1 = 3324
After SMOTE: 0 = 22949 1 = 22949


Balanced dataset improved model recall significantly.

## Models Evaluated

| Model | Accuracy | Recall | F1-score |
|------|---------|--------|----------|
| Naive Bayes | 0.52 | 0.73 | 0.28 |
| Logistic Regression | 0.61 | 0.64 | 0.29 |
| Decision Tree | 0.73 | 0.31 | 0.23 |
| KNN | 0.69 | 0.29 | 0.19 |
| Gradient Boosting | 0.79 | 0.24 | 0.23 |
| Random Forest | 0.83 | 0.17 | 0.20 |
| XGBoost | 0.83 | 0.15 | 0.19 |
| CatBoost | 0.83 | 0.15 | 0.18 |
| LightGBM | 0.85 | 0.10 | 0.14 |

## Best Model: LightGBM + Hyperparameter Tuning + K-Fold

After RandomizedSearchCV + Stratified K-Fold:

**Final Metrics (Threshold = 0.30):**

- Accuracy: **0.933**
- Recall: **0.812**
- F1-score: **0.754**
- ROC AUC: **0.965**

## Final Evaluation

### Confusion Matrix
- High True Positive detection
- Balanced trade-off between false positives and false negatives

### ROC Curve
- AUC = 0.965 (excellent discriminative power)

### Threshold Optimization
Different thresholds tested to balance recall vs precision for business use case.

## Deployment (Streamlit App)

The trained model was deployed using Streamlit to predict default probability in real-time.

Features:
- User input form
- Probability of default
- Risk classification
- Interactive UI

Run app:
```bash
streamlit run app.py

Case_tecnico_Koin
 ┣  app
 ┃ ┗ app.py
 ┣  models
 ┃ ┗ modelo_tuned_lightgbm_kfold.pkl
 ┣  notebooks
 ┃ ┗ analysis.ipynb
 ┣  data
 ┃ ┗ dataset.csv
 ┣ README.md
 ┗ requirements.txt
```

### Technologies Used

Python
Pandas / NumPy
Scikit-learn
LightGBM
XGBoost
CatBoost
TensorFlow / Keras
Matplotlib / Seaborn
Streamlit
Joblib

### Installation
pip install -r requirements.txt

How to Run

Clone repository
git clone https://github.com/yourusername/fraud-default-prediction.git

Run notebook
jupyter notebook

Run Streamlit app
streamlit run app/app.py

### Conclusion

The LightGBM model with SMOTE, K-Fold and threshold optimization achieved strong predictive performance and is suitable for real-world fraud detection scenarios.

This solution can support:

Credit approval decisions

Risk analysis

Fraud prevention strategies

### Future Improvements

SHAP explainability

Real-time API deployment

Deep Learning optimization

Cost-sensitive learning

Feature selection automation

### Author

Rafael Gallo
Data Scientist
MBA in Data Science & AI

LinkedIn: https://linkedin.com/in/rafaelgallo

GitHub: https://github.com/rafaelgallo
