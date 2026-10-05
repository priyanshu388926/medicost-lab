# MediCost Lab

An interactive web app that predicts yearly medical insurance charges and flags
high-cost customers. Built for Unit 2 (Linear Models) of the ML course.

Team: Priyanshu, Ashish Kumar, Salil Pratap Singh, Vedang Pratap Singh

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Needs Python 3.10 or newer.

## Files

| File | What it holds |
|---|---|
| `app.py` | The Streamlit interface (overview + 7 pages) |
| `models.py` | Data loading, all models, and a from-scratch Bayesian logistic regression with the Laplace approximation |
| `data/insurance.csv` | The dataset |
| `MediCost_Lab_Infographic.png` | Hour 18 infographic: Discover and Analyze ML Algorithms |

## Dataset

Medical Cost Personal dataset: 1,338 US insurance customers with age, sex, BMI,
number of children, smoker, region and yearly charges. One duplicate row is
removed, leaving 1,337.

- Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance
- GitHub mirror used here: https://github.com/stedy/Machine-Learning-with-R-datasets

Two targets:
- Regression: `charges` (US dollars per year).
- Classification: `high_cost` = 1 when charges are above the median ($9,386).

## Syllabus coverage (Unit 2)

| Hour | Topic | Page |
|---|---|---|
| 10 | Maximum likelihood estimation, least squares | 2 |
| 11 | Robust regression, ridge regression, Bayesian linear regression | 3 |
| 12 | Discriminant functions (LDA) | 4 |
| 13 | Generative (Naive Bayes) vs discriminative (logistic) models | 4 |
| 14 | Laplace approximation, Bayesian logistic regression | 5 |
| 15 | Kernel functions, kernels in a GLM, kernel trick | 6 |
| 16 | SVMs | 4, 6 |
| 17 | Practice 1: linear regression for prediction | 2, 3, 7 |
| 17 | Practice 2: Bayesian logistic regression and SVM for classification | 4, 5, 7 |
| 18 | Discover and Analyze ML Algorithms infographic | PNG file |

## Results (25% held-out test set)

Regression of yearly charges:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Least squares (MLE) | $5,940 | $4,069 | 0.796 |
| Robust (Huber) | $6,189 | $3,278 | 0.778 |
| Ridge (λ = 10) | $6,084 | $4,227 | 0.786 |
| Bayesian linear | $5,947 | $4,077 | 0.795 |
| Kernel ridge (RBF) | $4,774 | $2,862 | 0.868 |

Classification of high-cost customers (baseline 50%):

| Model | Accuracy | Recall | ROC AUC |
|---|---|---|---|
| LDA | 0.901 | 0.910 | 0.940 |
| Gaussian Naive Bayes | 0.743 | 0.485 | 0.941 |
| Logistic regression | 0.907 | 0.910 | 0.943 |
| Bayesian logistic (Laplace) | 0.907 | 0.910 | 0.943 |
| Kernel GLM (RBF features) | 0.913 | 0.910 | 0.950 |
| SVM linear | 0.907 | 0.910 | 0.945 |
| SVM polynomial | 0.919 | 0.892 | 0.937 |
| SVM RBF | 0.925 | 0.910 | 0.965 |

## Main findings

- Smoking is the dominant cost driver: it adds about $23,000 a year in the linear model.
- Linear models miss the smoker x BMI interaction; the RBF kernel captures it and lifts R² from 0.80 to 0.87.
- Huber regression is barely affected when training bills are corrupted; least squares degrades steadily.
- Naive Bayes ranks customers well (AUC 0.94) but its probabilities are miscalibrated, so its accuracy is only 74%.

## Limits

- US data from a textbook dataset; not representative of Indian insurance pricing.
- The high-cost label is derived from the regression target, so the two tasks are related.
- Hyperparameters are set by hand through the sliders, not tuned by cross-validation.
