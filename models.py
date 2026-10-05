"""Data and models for MediCost Lab (ML course, Unit 2: linear models).

Hour 10  maximum likelihood / least squares
Hour 11  robust, ridge and Bayesian linear regression
Hour 12  discriminant functions (LDA)
Hour 13  generative vs discriminative models
Hour 14  Laplace approximation, Bayesian logistic regression
Hour 15  kernel functions, kernels in a GLM, kernel trick
Hour 16  support vector machines
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, HuberRegressor,
                                  LinearRegression, LogisticRegression, Ridge)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).parent / "data" / "insurance.csv"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
FEATURES = NUMERIC + CATEGORICAL
REG_TARGET = "charges"
CLF_TARGET = "high_cost"
SEED = 42

DESCRIPTIONS = {
    "age": "Age of the policy holder (years)",
    "sex": "female / male",
    "bmi": "Body mass index (kg/m²)",
    "children": "Number of dependants covered",
    "smoker": "Smokes: yes / no",
    "region": "US region: northeast, northwest, southeast, southwest",
    "charges": "Medical costs billed in one year (US dollars)",
    "high_cost": "1 if charges are above the median, else 0",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    df[CLF_TARGET] = (df[REG_TARGET] > df[REG_TARGET].median()).astype(int)
    return df


def preprocessor(numeric=NUMERIC, categorical=CATEGORICAL):
    # Standardise numbers; one-hot encode categories, dropping one level so
    # that linear weights are identifiable.
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical),
    ])


def feature_names(pipe):
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def split(df, target, test_size=0.25):
    strat = df[target] if target == CLF_TARGET else None
    return train_test_split(df[FEATURES], df[target], test_size=test_size,
                            random_state=SEED, stratify=strat)


# --------------------------------------------------------------------------
# Regression (hours 10, 11, 15)
# --------------------------------------------------------------------------
def gaussian_log_likelihood(y, yhat):
    """Log-likelihood of the data under y = f(x) + N(0, sigma²), with sigma² at its MLE."""
    resid = np.asarray(y) - np.asarray(yhat)
    n, sigma2 = len(resid), float(np.mean(resid ** 2))
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1), sigma2


def kernel_ridge(kernel="rbf", alpha=0.1, gamma=0.1, degree=2):
    """Ridge regression in a kernel feature space (the kernel trick in a GLM)."""
    return TransformedTargetRegressor(
        regressor=KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma, degree=degree, coef0=1),
        transformer=StandardScaler())


def regressor_zoo(ridge_alpha=10.0, kr_alpha=0.1, kr_gamma=0.1):
    return {
        "Least squares (MLE)": LinearRegression(),
        "Robust (Huber loss)": HuberRegressor(max_iter=2000),
        "Ridge regression": Ridge(alpha=ridge_alpha),
        "Bayesian linear regression": BayesianRidge(),
        "Kernel ridge (RBF)": kernel_ridge("rbf", kr_alpha, kr_gamma),
    }


def fit_regressors(df, **kw):
    Xtr, Xte, ytr, yte = split(df, REG_TARGET)
    fitted = {n: Pipeline([("prep", preprocessor()), ("model", m)]).fit(Xtr, ytr)
              for n, m in regressor_zoo(**kw).items()}
    return fitted, (Xtr, Xte, ytr, yte)


def corrupt(y, fraction, factor=5.0, seed=0):
    """Multiply a random fraction of targets by `factor` (simulated billing errors)."""
    y = np.asarray(y, dtype=float).copy()
    k = int(round(fraction * len(y)))
    if k:
        idx = np.random.default_rng(seed).choice(len(y), k, replace=False)
        y[idx] *= factor
    return y


# --------------------------------------------------------------------------
# Bayesian logistic regression with the Laplace approximation (hour 14)
# --------------------------------------------------------------------------
class LaplaceLogisticRegression(BaseEstimator, ClassifierMixin):
    """Bishop PRML section 4.5.

    Prior        p(w) = N(0, alpha^-1 I)
    MAP          w_map = argmax  log p(t|w) + log p(w)
    Laplace      q(w) = N(w_map, S_N),  S_N^-1 = alpha I + sum y(1-y) phi phi'
    Predictive   p(C1|x) ~= sigmoid(kappa(s2) * mu),
                 mu = w_map' phi,  s2 = phi' S_N phi,  kappa = (1 + pi s2/8)^-1/2
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def _phi(X):
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Phi, t = self._phi(X), np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])

        def neg_log_post(w):
            a = Phi @ w
            return np.sum(np.logaddexp(0, a) - t * a) + 0.5 * self.alpha * w @ w

        def grad(w):
            return Phi.T @ (expit(Phi @ w) - t) + self.alpha * w

        self.w_map_ = minimize(neg_log_post, np.zeros(Phi.shape[1]), jac=grad, method="L-BFGS-B").x
        p = expit(Phi @ self.w_map_)
        H = self.alpha * np.eye(Phi.shape[1]) + (Phi * (p * (1 - p))[:, None]).T @ Phi
        self.S_N_ = np.linalg.inv(H)
        return self

    def activation(self, X):
        Phi = self._phi(X)
        return Phi @ self.w_map_, np.einsum("ij,jk,ik->i", Phi, self.S_N_, Phi)

    def predict_proba_map(self, X):
        return expit(self.activation(X)[0])

    def predict_proba(self, X):
        mu, s2 = self.activation(X)
        p = expit(mu / np.sqrt(1 + np.pi * s2 / 8))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------
# Classification (hours 12, 13, 15, 16)
# --------------------------------------------------------------------------
def svm_with_proba(**kw):
    """An SVM outputs a signed distance. Platt scaling turns it into a probability."""
    return CalibratedClassifierCV(SVC(**kw), method="sigmoid", ensemble=False)


def classifier_zoo(alpha=1.0, C=1.0):
    return {
        "LDA (discriminant function)": LinearDiscriminantAnalysis(),
        "Gaussian Naive Bayes (generative)": GaussianNB(),
        "Logistic regression (discriminative)": LogisticRegression(max_iter=2000),
        "Bayesian logistic (Laplace)": LaplaceLogisticRegression(alpha=alpha),
        "Kernel GLM (RBF features + logistic)": make_pipeline(
            Nystroem(kernel="rbf", gamma=0.1, n_components=150, random_state=0),
            LogisticRegression(max_iter=2000)),
        "SVM - linear kernel": svm_with_proba(kernel="linear", C=C),
        "SVM - polynomial kernel": svm_with_proba(kernel="poly", degree=3, C=C),
        "SVM - RBF kernel": svm_with_proba(kernel="rbf", C=C),
    }


def fit_classifiers(df, **kw):
    Xtr, Xte, ytr, yte = split(df, CLF_TARGET)
    fitted = {n: Pipeline([("prep", preprocessor()), ("model", m)]).fit(Xtr, ytr)
              for n, m in classifier_zoo(**kw).items()}
    return fitted, (Xtr, Xte, ytr, yte)
