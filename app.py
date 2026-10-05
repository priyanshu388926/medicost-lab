"""MediCost Lab - Unit 2 (linear models) on a medical insurance dataset.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import models as M

st.set_page_config(page_title="MediCost Lab", page_icon="🩺", layout="wide")

# Fixed colour roles: a colour always means the same thing across pages.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
REG_COLORS = [BLUE, ORANGE, AQUA, YELLOW, VIOLET]
CLF_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, RED, VIOLET]
TWO = {0: BLUE, 1: ORANGE}
MUTED, INK = "#898781", "#52514e"
SEQ_BLUE = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]
COST = {0: "Low cost", 1: "High cost"}


def show(fig, height=380, **layout):
    layout.setdefault("legend", dict(orientation="h", y=1.12, x=0))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), **layout)
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line_width=2)
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def money(v):
    return f"${v:,.0f}"


@st.cache_data
def get_data():
    return M.load_data()


@st.cache_resource(show_spinner="Training regression models...")
def get_regressors(ridge_alpha, kr_alpha, kr_gamma):
    return M.fit_regressors(get_data(), ridge_alpha=ridge_alpha, kr_alpha=kr_alpha, kr_gamma=kr_gamma)


@st.cache_resource(show_spinner="Training classifiers...")
def get_classifiers(alpha, C):
    return M.fit_classifiers(get_data(), alpha=alpha, C=C)


@st.cache_data(show_spinner="Simulating billing errors...")
def outlier_curve(fractions):
    df = get_data()
    Xtr, Xte, ytr, yte = M.split(df, M.REG_TARGET)
    rows = []
    for f in fractions:
        y_bad = M.corrupt(ytr, f)
        for name, reg in [("Least squares (MLE)", LinearRegression()),
                          ("Robust (Huber loss)", HuberRegressor(max_iter=2000))]:
            pipe = Pipeline([("prep", M.preprocessor()), ("model", reg)]).fit(Xtr, y_bad)
            pred = pipe.predict(Xte)
            rows.append([f, name, M.rmse(yte, pred), float(np.mean(np.abs(yte - pred)))])
    return pd.DataFrame(rows, columns=["fraction", "model", "rmse", "mae"])


df = get_data()
MEDIAN = df.charges.median()

PAGES = [
    "Overview",
    "1 · Data",
    "2 · Least squares and MLE",
    "3 · Robust, ridge, Bayesian",
    "4 · Classifiers",
    "5 · Laplace approximation",
    "6 · Kernels and SVM",
    "7 · Cost predictor",
]
st.sidebar.title("🩺 MediCost Lab")
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.caption(f"Data: Medical Cost Personal dataset, {len(df):,} insurance customers.")

# ==========================================================================
if page == "Overview":
    st.title("MediCost Lab")
    st.write(
        "An insurer wants to know **what a customer's yearly medical bill will be** "
        "(regression) and **whether the customer will be a high-cost customer** "
        "(classification). This app answers both with the linear models of Unit 2, "
        "and shows where linear models stop working and kernels take over.")
    c = st.columns(4)
    c[0].metric("Customers", f"{len(df):,}")
    c[1].metric("Median yearly charge", money(MEDIAN))
    c[2].metric("Smokers", f"{(df.smoker == 'yes').mean():.1%}")
    c[3].metric("Smoker vs non-smoker mean bill",
                f"{df[df.smoker == 'yes'].charges.mean() / df[df.smoker == 'no'].charges.mean():.1f}x")

    st.subheader("Two tasks")
    st.markdown(
        f"- **Regression target**: `charges`, the yearly medical bill in US dollars.\n"
        f"- **Classification target**: `high_cost`, which is 1 when charges are above the "
        f"median of {money(MEDIAN)}. The classes are balanced 50 / 50 by construction.\n"
        f"- **Inputs**: age, sex, BMI, number of children, smoker, region.")

    st.subheader("Unit 2 syllabus map")
    st.dataframe(pd.DataFrame([
        ["10", "Maximum likelihood estimation, least squares", "2 · Least squares and MLE"],
        ["11", "Robust linear regression, ridge regression, Bayesian linear regression", "3 · Robust, ridge, Bayesian"],
        ["12", "Linear models for classification: discriminant function", "4 · Classifiers (LDA)"],
        ["13", "Probabilistic generative and discriminative models", "4 · Classifiers"],
        ["14", "Laplace approximation, Bayesian logistic regression", "5 · Laplace approximation"],
        ["15", "Kernel functions, kernels in GLM, kernel trick", "6 · Kernels and SVM"],
        ["16", "SVMs", "4 and 6"],
        ["17", "Practice 1: linear regression for prediction", "2, 3 and 7"],
        ["17", "Practice 2: Bayesian logistic regression and SVM for classification", "4, 5 and 7"],
    ], columns=["Hour", "Topic", "Where in this app"]), hide_index=True, width="stretch")

# ==========================================================================
elif page == "1 · Data":
    st.title("The data")
    st.write(f"**{df.shape[0]:,} customers, {len(M.FEATURES)} input features**, no missing values. "
             f"One duplicated row was removed.")
    t1, t2 = st.tabs(["Table and summary", "What drives the bill"])
    with t1:
        st.dataframe(df.head(15), width="stretch")
        st.dataframe(pd.DataFrame({"Column": M.DESCRIPTIONS.keys(), "Meaning": M.DESCRIPTIONS.values()}),
                     hide_index=True, width="stretch")
        st.write("**Summary statistics**")
        st.dataframe(df[M.NUMERIC + [M.REG_TARGET]].describe().T.round(1), width="stretch")
    with t2:
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            for k, (lab, col) in enumerate([("no", BLUE), ("yes", ORANGE)]):
                s = df[df.smoker == lab]
                fig.add_scatter(x=s.age, y=s.charges, mode="markers",
                                name="Smoker" if lab == "yes" else "Non-smoker",
                                marker=dict(color=col, size=6, opacity=0.6))
            show(fig, height=420, xaxis_title="Age", yaxis_title="Yearly charges ($)")
            st.caption("Charges rise with age along three parallel bands. Smokers sit in the upper two.")
        with right:
            fig = go.Figure()
            for lab, col, name in [("no", BLUE, "Non-smoker"), ("yes", ORANGE, "Smoker")]:
                s = df[df.smoker == lab]
                fig.add_scatter(x=s.bmi, y=s.charges, mode="markers", name=name,
                                marker=dict(color=col, size=6, opacity=0.6))
            fig.add_vline(x=30, line_dash="dot", line_color=MUTED, annotation_text="BMI 30")
            show(fig, height=420, xaxis_title="BMI", yaxis_title="Yearly charges ($)")
            st.caption("For smokers, charges jump at BMI 30. For non-smokers BMI barely matters. "
                       "This interaction is what a purely linear model cannot represent.")
        g = df.groupby("smoker").charges.agg(["count", "mean", "median", "std"]).round(0)
        g.index = ["Non-smoker", "Smoker"]
        st.dataframe(g, width="stretch")

# ==========================================================================
elif page == "2 · Least squares and MLE":
    st.title("Least squares is maximum likelihood")
    st.write("Assume each bill is a linear function of the inputs plus Gaussian noise: "
             "**y = wᵀx + ε, ε ~ N(0, σ²)**. Maximising the likelihood of the data over w is then "
             "exactly the same as minimising the sum of squared errors.")
    st.latex(r"\ln p(\mathbf{y}\mid \mathbf{w},\sigma^2) = -\frac{N}{2}\ln(2\pi\sigma^2) "
             r"- \frac{1}{2\sigma^2}\sum_{n=1}^{N}\left(y_n - \mathbf{w}^T\mathbf{x}_n\right)^2")

    st.subheader("See it on one feature")
    ns = df[df.smoker == "no"]
    x, y = ns.age.values.astype(float), ns.charges.values
    w_ls = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    w = st.slider("Slope: dollars added per year of age (non-smokers only)", 0.0, 550.0, 120.0, 5.0)
    b = float(np.mean(y - w * x))
    slopes = np.linspace(0, 550, 111)
    nll = [-M.gaussian_log_likelihood(y, s * x + np.mean(y - s * x))[0] for s in slopes]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=x, y=y, mode="markers", name="Non-smokers",
                        marker=dict(color=BLUE, size=5, opacity=0.45))
        xs = np.array([18, 64])
        fig.add_scatter(x=xs, y=w * xs + b, mode="lines", name=f"Your line, slope {w:.0f}", line_color=ORANGE)
        show(fig, xaxis_title="Age", yaxis_title="Yearly charges ($)")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=slopes, y=nll, mode="lines", name="Negative log-likelihood", line_color=BLUE,
                        showlegend=False)
        fig.add_scatter(x=[w], y=[-M.gaussian_log_likelihood(y, w * x + b)[0]], mode="markers",
                        name="Your line", marker=dict(color=ORANGE, size=12), showlegend=False)
        fig.add_vline(x=w_ls, line_dash="dot", line_color=MUTED,
                      annotation_text=f"least squares: {w_ls:.0f}")
        show(fig, xaxis_title="Slope ($ per year of age)", yaxis_title="Negative log-likelihood")
    cur = M.rmse(y, w * x + b)
    c = st.columns(3)
    c[0].metric("Your RMSE", money(cur))
    c[1].metric("Least squares RMSE", money(M.rmse(y, w_ls * x + np.mean(y - w_ls * x))))
    c[2].metric("Least squares slope", f"${w_ls:,.0f} per year")
    st.caption("The likelihood curve bottoms out exactly at the least squares slope. "
               "Move the slider to check.")

    st.subheader("Full model: all six features")
    regs, (Xtr, Xte, ytr, yte) = get_regressors(10.0, 0.1, 0.1)
    ols = regs["Least squares (MLE)"]
    pred_tr, pred_te = ols.predict(Xtr), ols.predict(Xte)
    ll, sigma2 = M.gaussian_log_likelihood(ytr, pred_tr)
    c = st.columns(4)
    c[0].metric("Test R²", f"{r2_score(yte, pred_te):.3f}")
    c[1].metric("Test RMSE", money(M.rmse(yte, pred_te)))
    c[2].metric("σ (MLE of noise std)", money(np.sqrt(sigma2)))
    c[3].metric("Train log-likelihood", f"{ll:,.0f}")
    left, right = st.columns(2)
    with left:
        coef = pd.Series(ols.named_steps["model"].coef_, index=M.feature_names(ols)).sort_values()
        fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation="h", marker_color=BLUE,
                               text=[money(v) for v in coef.values], textposition="outside",
                               hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
        show(fig, height=380, xaxis_title="Weight in dollars (numeric features per 1 std)",
             xaxis_range=[coef.min() * 1.6, coef.max() * 1.25])
        st.caption("Being a smoker adds about "
                   f"{money(coef['smoker_yes'])} a year, far more than any other feature.")
    with right:
        resid = ytr - pred_tr
        grid = np.linspace(resid.min(), resid.max(), 300)
        fig = go.Figure()
        fig.add_histogram(x=resid, histnorm="probability density", nbinsx=60, name="Residuals",
                          marker_color=BLUE, opacity=0.75)
        fig.add_scatter(x=grid, y=stats.norm.pdf(grid, 0, np.sqrt(sigma2)), mode="lines",
                        name="Assumed N(0, σ²)", line_color=ORANGE)
        show(fig, height=380, xaxis_title="Residual ($)", yaxis_title="density")
        st.caption("The Gaussian noise assumption is visibly wrong here: residuals are far more peaked than "
                   "the bell curve, with separate humps on both sides. That motivates the robust and "
                   "kernel models on the next pages.")

# ==========================================================================
elif page == "3 · Robust, ridge, Bayesian":
    st.title("Robust, ridge and Bayesian linear regression")
    ridge_alpha = st.slider("Ridge penalty λ", 0.1, 500.0, 10.0)
    regs, (Xtr, Xte, ytr, yte) = get_regressors(ridge_alpha, 0.1, 0.1)
    rows = []
    for name, m in regs.items():
        p = m.predict(Xte)
        rows.append([name, M.rmse(yte, p), float(np.mean(np.abs(yte - p))), r2_score(yte, p)])
    st.dataframe(pd.DataFrame(rows, columns=["Model", "Test RMSE ($)", "Test MAE ($)", "Test R²"]).round(3),
                 hide_index=True, width="stretch",
                 column_config={"Test RMSE ($)": st.column_config.NumberColumn(format="%.0f"),
                                "Test MAE ($)": st.column_config.NumberColumn(format="%.0f")})
    st.caption("25% held-out test set. Kernel ridge is included for comparison and explained on page 6.")

    st.subheader("Robust regression: what happens when some bills are wrong")
    st.write("We multiply a random share of the **training** bills by 5, as if they were data-entry "
             "errors, then test on clean data. Squared loss lets each bad bill pull hard on the fit. "
             "Huber loss grows only linearly for large errors.")
    fr = tuple(np.round(np.arange(0, 0.16, 0.02), 2))
    oc = outlier_curve(fr)
    f_sel = st.select_slider("Share of corrupted training bills", fr, 0.06, format_func=lambda v: f"{v:.0%}")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, col in [("Least squares (MLE)", BLUE), ("Robust (Huber loss)", ORANGE)]:
            s = oc[oc.model == name]
            fig.add_scatter(x=s.fraction, y=s.mae, mode="lines+markers", name=name, line_color=col)
        fig.add_vline(x=f_sel, line_dash="dot", line_color=MUTED)
        show(fig, xaxis_title="Share of corrupted training bills", xaxis_tickformat=".0%",
             yaxis_title="Test MAE on clean data ($)", yaxis_rangemode="tozero")
    with right:
        s = oc[np.isclose(oc.fraction, f_sel)].set_index("model")
        c0 = oc[np.isclose(oc.fraction, 0)].set_index("model")
        c = st.columns(2)
        for i, name in enumerate(["Least squares (MLE)", "Robust (Huber loss)"]):
            c[i].metric(name + " MAE", money(s.loc[name, "mae"]),
                        f"{s.loc[name, 'mae'] - c0.loc[name, 'mae']:+,.0f} vs clean", delta_color="inverse")
        st.markdown(
            "- **Least squares** assumes Gaussian noise, so a bill that is 5x too large looks "
            "extremely unlikely and the line bends to explain it.\n"
            "- **Huber** behaves like least squares for small errors and like absolute error for "
            "large ones, so its fit hardly moves.\n"
            "- On clean data Huber has the lower MAE but the higher RMSE: it fits the typical "
            "customer and gives up on the expensive few.")

    left, right = st.columns(2)
    with left:
        st.subheader("Ridge: weights shrink as λ grows")
        lams = np.logspace(-2, 4, 40)
        Z = regs["Ridge regression"].named_steps["prep"].transform(Xtr)
        names = M.feature_names(regs["Ridge regression"])
        path = np.array([Ridge(alpha=l).fit(Z, ytr).coef_ for l in lams])
        top = np.argsort(-np.abs(path[0]))[:4]
        fig = go.Figure()
        for j in range(path.shape[1]):
            if j not in top:
                fig.add_scatter(x=lams, y=path[:, j], mode="lines", line_color="#c3c2b7",
                                showlegend=False, hoverinfo="skip")
        for k, j in enumerate(top):
            fig.add_scatter(x=lams, y=path[:, j], mode="lines", name=names[j], line_color=REG_COLORS[k])
        fig.add_vline(x=ridge_alpha, line_dash="dot", line_color=MUTED)
        show(fig, xaxis_type="log", xaxis_title="λ (log scale), dotted line = your λ", yaxis_title="Weight ($)",
             xaxis_range=[-2, 4])
        st.caption("Ridge minimises squared error + λ‖w‖². It is the MAP estimate under a Gaussian "
                   "prior on w. The four largest weights are coloured; the rest are grey.")
    with right:
        st.subheader("Bayesian: every prediction has an error bar")
        bayes = regs["Bayesian linear regression"]
        mu, sd = bayes.named_steps["model"].predict(bayes.named_steps["prep"].transform(Xte), return_std=True)
        order = np.argsort(mu)[:: max(1, len(mu) // 60)]
        fig = go.Figure()
        fig.add_scatter(x=np.arange(len(order)), y=mu[order], mode="lines", name="Predicted mean",
                        line_color=BLUE, error_y=dict(type="data", array=2 * sd[order], color="#86b6ef",
                                                      thickness=1, width=0))
        fig.add_scatter(x=np.arange(len(order)), y=yte.values[order], mode="markers", name="Actual bill",
                        marker=dict(color=ORANGE, size=7))
        show(fig, xaxis_title="Test customers, sorted by prediction", yaxis_title="Yearly charges ($)")
        inside = np.mean(np.abs(yte.values - mu) <= 2 * sd)
        st.caption(f"Bars are ±2 predictive standard deviations (about ±{money(2 * sd.mean())}). "
                   f"**{inside:.1%}** of test bills fall inside, close to the 95% expected.")

# ==========================================================================
elif page == "4 · Classifiers":
    st.title("Linear models for classification")
    st.write(f"Target: **high cost**, meaning yearly charges above the median of {money(MEDIAN)}. "
             f"75% train, 25% test, stratified.")
    C = st.select_slider("SVM penalty C", [0.1, 0.5, 1.0, 5.0, 10.0], 1.0)
    clfs, (Xtr, Xte, ytr, yte) = get_classifiers(1.0, C)
    rows, probs = [], {}
    for name, m in clfs.items():
        p = m.predict_proba(Xte)[:, 1]
        yh = (p >= 0.5).astype(int)
        probs[name] = p
        rows.append([name, accuracy_score(yte, yh), precision_score(yte, yh), recall_score(yte, yh),
                     f1_score(yte, yh), roc_auc_score(yte, p)])
    st.dataframe(pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]).round(3),
                 hide_index=True, width="stretch")
    st.caption("Baseline: classes are balanced, so always guessing one class scores 50% accuracy.")

    with st.expander("Three ways to build a linear classifier", expanded=True):
        st.markdown(
            "- **Discriminant function (LDA)**: find the direction w that best separates the class "
            "means relative to the within-class spread, then threshold wᵀx.\n"
            "- **Generative (Naive Bayes, LDA)**: model how each class produces its features, "
            "p(x | class), then apply Bayes rule to get p(class | x).\n"
            "- **Discriminative (logistic regression)**: model p(class | x) = σ(wᵀx) directly and "
            "never model the features themselves.")

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curves")
        chosen = st.multiselect("Models", list(clfs), max_selections=4, default=[
            "LDA (discriminant function)", "Gaussian Naive Bayes (generative)",
            "Logistic regression (discriminative)", "SVM - RBF kernel"])
        fig = go.Figure()
        for name in chosen:
            fpr, tpr, _ = roc_curve(yte, probs[name])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name,
                            line_color=CLF_COLORS[list(clfs).index(name)])
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color=MUTED, dash="dot"))
        show(fig, height=460, xaxis_title="False positive rate", yaxis_title="True positive rate",
             legend=dict(orientation="h", y=-0.25, x=0))
    with right:
        st.subheader("Confusion matrix")
        pick = st.selectbox("Model", list(clfs), index=1)
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
        cm = confusion_matrix(yte, (probs[pick] >= thr).astype(int))
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred. low cost", "Pred. high cost"],
                                   y=["Actual low cost", "Actual high cost"], colorscale=SEQ_BLUE,
                                   text=cm, texttemplate="%{text}", textfont_size=18,
                                   showscale=False, xgap=2, ygap=2))
        show(fig, height=320, yaxis_autorange="reversed")
        st.caption(f"At threshold {thr:.2f}: finds {cm[1, 1]} of {cm[1].sum()} high-cost customers, "
                   f"wrongly flags {cm[0, 1]} of {cm[0].sum()} low-cost customers.")

    st.subheader("Why Naive Bayes ranks well but decides badly")
    nb = probs["Gaussian Naive Bayes (generative)"]
    lr = probs["Logistic regression (discriminative)"]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_histogram(x=lr, nbinsx=25, name="Logistic regression", marker_color=AQUA, opacity=0.8)
        fig.add_histogram(x=nb, nbinsx=25, name="Naive Bayes", marker_color=ORANGE, opacity=0.8)
        show(fig, barmode="group", xaxis_title="Predicted P(high cost) on the test set",
             yaxis_title="Customers")
    with right:
        st.markdown(
            "Naive Bayes assumes the features are independent given the class and that each one "
            "is Gaussian. The one-hot columns (smoker, sex, region) are neither. Multiplying many "
            "badly-modelled factors pushes its probabilities toward one side, so the 0.5 threshold "
            "sits in the wrong place.")
        c = st.columns(2)
        c[0].metric("Naive Bayes says high cost", f"{(nb >= 0.5).mean():.0%}")
        c[1].metric("Logistic says high cost", f"{(lr >= 0.5).mean():.0%}")
        st.caption("True share of high-cost customers is 50%. Its ROC AUC stays high because the "
                   "ranking of customers is still good; only the calibration is off.")

# ==========================================================================
elif page == "5 · Laplace approximation":
    st.title("Bayesian logistic regression with the Laplace approximation")
    left, right = st.columns([3, 2])
    with left:
        st.write("Logistic regression has no closed-form posterior over the weights. The Laplace "
                 "approximation fits a Gaussian at the posterior peak (the MAP weights), using the "
                 "curvature there as the inverse covariance.")
        st.latex(r"q(\mathbf{w}) = \mathcal{N}(\mathbf{w}_{MAP},\, \mathbf{S}_N)")
        st.latex(r"\mathbf{S}_N^{-1} = \alpha\mathbf{I} + \sum_n y_n(1-y_n)\boldsymbol\phi_n\boldsymbol\phi_n^T")
    with right:
        st.write("Predictions then average over that Gaussian instead of trusting one weight vector:")
        st.latex(r"p(C_1 \mid \mathbf{x}) \approx \sigma\!\left(\frac{\mu_a}{\sqrt{1+\pi\sigma_a^2/8}}\right)")

    c1, c2 = st.columns(2)
    n = c1.select_slider("Training customers used", [20, 40, 80, 160, 320, 640, 1002], 40)
    alpha = c2.select_slider("Prior precision α", [0.01, 0.1, 1.0, 10.0], 1.0)
    Xtr, Xte, ytr, yte = M.split(df, M.CLF_TARGET)
    prep = M.preprocessor().fit(Xtr)
    blr = M.LaplaceLogisticRegression(alpha=alpha).fit(prep.transform(Xtr.iloc[:n]), ytr.iloc[:n])
    Zte = prep.transform(Xte)
    mu, s2 = blr.activation(Zte)
    p_map, p_bayes = blr.predict_proba_map(Zte), blr.predict_proba(Zte)[:, 1]

    left, right = st.columns(2)
    with left:
        st.subheader("Point estimate vs Bayesian prediction")
        fig = go.Figure(go.Scatter(
            x=p_map, y=p_bayes, mode="markers",
            marker=dict(color=np.sqrt(s2), colorscale=SEQ_BLUE, size=7, cmin=0,
                        colorbar=dict(title="std of a")),
            hovertemplate="MAP %{x:.3f}<br>Bayesian %{y:.3f}<extra></extra>"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip",
                        line=dict(color=MUTED, dash="dot"))
        show(fig, height=420, showlegend=False, xaxis_title="MAP probability",
             yaxis_title="Bayesian predictive probability")
    with right:
        st.subheader("Weight uncertainty")
        names = ["bias"] + [nm.split("__", 1)[1] for nm in prep.get_feature_names_out()]
        sd = np.sqrt(np.diag(blr.S_N_))
        order = np.argsort(-np.abs(blr.w_map_))
        fig = go.Figure(go.Scatter(
            x=blr.w_map_[order], y=[names[i] for i in order], mode="markers",
            marker=dict(color=BLUE, size=9),
            error_x=dict(type="data", array=2 * sd[order], color="#86b6ef", thickness=2, width=0),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
        show(fig, height=420, showlegend=False, xaxis_title="Weight, with ±2 posterior std",
             yaxis_autorange="reversed")
    c = st.columns(4)
    c[0].metric("Largest shift toward 0.5", f"{np.abs(p_map - p_bayes).max():.3f}")
    c[1].metric("Mean std of activation", f"{np.sqrt(s2).mean():.2f}")
    c[2].metric("Test accuracy", f"{accuracy_score(yte, p_bayes >= 0.5):.1%}")
    c[3].metric("Test ROC AUC", f"{roc_auc_score(yte, p_bayes):.3f}")
    st.caption("With few training customers the weight posterior is wide, so Bayesian predictions are "
               "pulled toward 0.5 (points bend away from the diagonal). With all 1,002 the posterior "
               "is narrow and the two agree. The 0.5 decision boundary itself never moves.")

# ==========================================================================
elif page == "6 · Kernels and SVM":
    st.title("Kernels: when a straight line is not enough")
    st.write("A kernel k(x, x′) returns the inner product of two points in a high-dimensional feature "
             "space without ever computing that space. Any model that only needs inner products can "
             "become non-linear by swapping in a kernel. That is the kernel trick.")
    st.subheader("Kernels in a regression GLM")
    sm = df[df.smoker == "yes"]
    xs = StandardScaler().fit(sm[["bmi"]])
    Zs, ys = xs.transform(sm[["bmi"]]), sm.charges.values
    c1, c2 = st.columns(2)
    gamma = c1.select_slider("RBF width γ", [0.1, 0.5, 1.0, 3.0, 10.0, 50.0], 0.5)
    lam = c2.select_slider("Regulariser λ", [0.001, 0.01, 0.1, 1.0, 10.0], 0.1)
    grid = np.linspace(sm.bmi.min(), sm.bmi.max(), 300)
    Zg = xs.transform(pd.DataFrame({"bmi": grid}))
    fits = {
        "Linear kernel: xᵀx′": M.kernel_ridge("linear", lam).fit(Zs, ys),
        "Polynomial kernel: (γxᵀx′ + 1)³": M.kernel_ridge("poly", lam, 1.0, 3).fit(Zs, ys),
        "RBF kernel: exp(−γ‖x − x′‖²)": M.kernel_ridge("rbf", lam, gamma).fit(Zs, ys),
    }
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_scatter(x=sm.bmi, y=ys, mode="markers", name="Smokers",
                        marker=dict(color=MUTED, size=6, opacity=0.5))
        for (name, m), col in zip(fits.items(), [BLUE, AQUA, ORANGE]):
            fig.add_scatter(x=grid, y=m.predict(Zg), mode="lines", name=name.split(":")[0], line_color=col)
        show(fig, height=420, xaxis_title="BMI (smokers only)", yaxis_title="Yearly charges ($)")
    with right:
        st.dataframe(pd.DataFrame([[n, r2_score(ys, m.predict(Zs))] for n, m in fits.items()],
                                  columns=["Kernel", "R² on these points"]).round(3),
                     hide_index=True, width="stretch")
        st.markdown("Smokers' bills jump at BMI 30. A linear kernel can only draw a straight line "
                    "through the step. The RBF kernel follows it. A very large γ makes the curve "
                    "chase single points, which is overfitting.")
        regs, (Xtr, Xte, ytr, yte) = get_regressors(10.0, 0.1, 0.1)
        r_lin = r2_score(yte, regs["Least squares (MLE)"].predict(Xte))
        r_ker = r2_score(yte, regs["Kernel ridge (RBF)"].predict(Xte))
        c = st.columns(2)
        c[0].metric("Linear: test R²", f"{r_lin:.3f}")
        c[1].metric("RBF kernel: test R²", f"{r_ker:.3f}", f"{r_ker - r_lin:+.3f}")
        st.caption("Both trained on all six features and scored on the 25% test set.")

    st.subheader("SVM decision boundary")
    st.write("Two features, so the boundary can be drawn. The SVM picks the boundary with the widest "
             "margin; only the support vectors (circled) determine it.")
    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("Customers", ["Non-smokers", "All customers"])
    kernel = c2.selectbox("Kernel", ["linear", "poly", "rbf"], index=2,
                          format_func={"linear": "Linear", "poly": "Polynomial (degree 3)", "rbf": "RBF"}.get)
    C = c3.select_slider("Penalty C", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
    g2 = c4.select_slider("γ", [0.01, 0.1, 0.5, 1.0, 5.0, 20.0], 0.5, disabled=kernel == "linear")
    s = df if group == "All customers" else df[df.smoker == "no"]
    s = s.sample(min(600, len(s)), random_state=3)
    sc = StandardScaler().fit(s[["age", "bmi"]])
    Z, yv = sc.transform(s[["age", "bmi"]]), s[M.CLF_TARGET].values
    svm = SVC(kernel=kernel, C=C, gamma=g2, degree=3, coef0=1).fit(Z, yv)
    gx = np.linspace(Z[:, 0].min() - 0.3, Z[:, 0].max() + 0.3, 140)
    gy = np.linspace(Z[:, 1].min() - 0.3, Z[:, 1].max() + 0.3, 140)
    D = svm.decision_function(np.c_[np.repeat(gx, len(gy)), np.tile(gy, len(gx))]).reshape(len(gx), len(gy)).T
    ox, oy = gx * sc.scale_[0] + sc.mean_[0], gy * sc.scale_[1] + sc.mean_[1]
    m = float(np.abs(D).max())
    fig = go.Figure()
    fig.add_contour(x=ox, y=oy, z=D, colorscale=[[0, "#cde2fb"], [0.5, "#f0efec"], [1, "#f6c7b3"]],
                    zmin=-m, zmax=m, showscale=False, opacity=0.8, hoverinfo="skip",
                    contours=dict(showlines=False), ncontours=16)
    fig.add_contour(x=ox, y=oy, z=D, showscale=False, hoverinfo="skip",
                    contours=dict(start=0, end=0, size=1, coloring="none"), line=dict(color=INK, width=2),
                    name="Decision boundary", showlegend=True)
    for k in (0, 1):
        fig.add_scatter(x=s.age[yv == k], y=s.bmi[yv == k], mode="markers", name=COST[k],
                        marker=dict(color=TWO[k], size=8, line=dict(color="white", width=1)))
    fig.add_scatter(x=s.age.values[svm.support_], y=s.bmi.values[svm.support_], mode="markers",
                    name="Support vectors", hoverinfo="skip",
                    marker=dict(symbol="circle-open", size=13, color=INK, line_width=1))
    show(fig, height=520, xaxis_title="Age", yaxis_title="BMI")
    c = st.columns(3)
    c[0].metric("Support vectors", f"{len(svm.support_)} of {len(s)}")
    c[1].metric("Training accuracy", f"{svm.score(Z, yv):.1%}")
    c[2].metric("High-cost customers found", f"{(svm.predict(Z)[yv == 1] == 1).mean():.1%}")
    st.caption("Dark line: decision boundary. Blue region predicts low cost, orange predicts high cost. "
               "Among non-smokers the boundary is essentially an age threshold, so the linear kernel "
               "does as well as RBF. With all customers, smokers of every age are high cost and no "
               "boundary in age and BMI alone can separate them.")

# ==========================================================================
elif page == "7 · Cost predictor":
    st.title("Cost predictor")
    st.write("Describe a customer. The regression models estimate the yearly bill and the classifiers "
             "give the probability of being a high-cost customer.")
    with st.form("customer"):
        c = st.columns(3)
        age = c[0].slider("Age", 18, 64, 35)
        bmi = c[1].slider("BMI", 16.0, 53.0, 28.0, 0.5)
        children = c[2].slider("Children", 0, 5, 1)
        c = st.columns(3)
        smoker = c[0].radio("Smoker", ["no", "yes"], horizontal=True)
        sex = c[1].radio("Sex", ["female", "male"], horizontal=True)
        region = c[2].selectbox("Region", sorted(df.region.unique()))
        go_btn = st.form_submit_button("Predict", type="primary")
    if go_btn:
        row = pd.DataFrame([dict(age=age, bmi=bmi, children=children, sex=sex, smoker=smoker, region=region)])
        regs, _ = get_regressors(10.0, 0.1, 0.1)
        clfs, _ = get_classifiers(1.0, 1.0)
        est = pd.Series({n: float(m.predict(row)[0]) for n, m in regs.items()})
        b = regs["Bayesian linear regression"]
        mu, sd = b.named_steps["model"].predict(b.named_steps["prep"].transform(row), return_std=True)
        pr = pd.Series({n: float(m.predict_proba(row)[0, 1]) for n, m in clfs.items()})

        c = st.columns(3)
        c[0].metric("Kernel ridge estimate", money(max(est["Kernel ridge (RBF)"], 0)))
        c[1].metric("Bayesian 95% range (USD)", f"{max(mu[0] - 2 * sd[0], 0) / 1000:.1f}k to {(mu[0] + 2 * sd[0]) / 1000:.1f}k")
        c[2].metric("Average P(high cost)", f"{pr.mean():.0%}",
                    f"{(pr >= 0.5).sum()} of {len(pr)} models say high cost", delta_color="off")

        left, right = st.columns(2)
        with left:
            st.subheader("Estimated yearly bill")
            fig = go.Figure(go.Bar(x=est.values, y=est.index, orientation="h", marker_color=BLUE,
                                   text=[money(v) for v in est.values], textposition="outside",
                                   hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
            fig.add_vline(x=MEDIAN, line_dash="dot", line_color=MUTED)
            show(fig, height=330, xaxis_title="Yearly charges ($), dotted = median bill", yaxis_autorange="reversed",
                 xaxis_range=[min(0, est.min() * 1.2), max(est.max(), MEDIAN) * 1.3])
        with right:
            st.subheader("Probability of high cost")
            fig = go.Figure(go.Bar(x=pr.values, y=pr.index, orientation="h", marker_color=BLUE,
                                   text=[f"{v:.0%}" for v in pr.values], textposition="outside",
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED)
            show(fig, height=330, xaxis_title="P(high cost), dotted = 0.5 threshold", xaxis_range=[0, 1.2],
                 xaxis_tickformat=".0%", yaxis_autorange="reversed")
        st.info("The linear models add a fixed amount for smoking and a fixed amount per BMI point. "
                "The kernel model can also learn that smoking and high BMI together cost more than "
                "the two added separately, so the estimates differ most for smokers.")
        st.caption("Educational project on a public dataset of US insurance customers. Not a pricing tool.")
