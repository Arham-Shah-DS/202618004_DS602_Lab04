import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Charges Dashboard",
    layout="wide",
)

# --------------------------------------------------------------------------
# Data loading (cached so it only reads the CSV once per session)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# --------------------------------------------------------------------------
# Model fitting (cached so OLS only runs once, not on every widget change)
# --------------------------------------------------------------------------
@st.cache_resource
def fit_model(data):
    formula = "charges ~ age + bmi + children + C(smoker) + C(sex) + C(region) + bmi:C(smoker)"
    return smf.ols(formula=formula, data=data).fit()

model = fit_model(df)

st.title("🏥 Medical Insurance Charges — Statistical Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🔬 Hypothesis Testing Lab", "📈 Live Prediction & Diagnostics"]
)

# ==========================================================================
# TAB 1 — DATA EXPLORATION
# ==========================================================================
with tab1:
    st.header("Data Exploration")

    st.sidebar.header("Filters — Data Exploration")

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

    bmi_min, bmi_max = float(df["bmi"].min()), float(df["bmi"].max())
    bmi_range = st.sidebar.slider("BMI range", bmi_min, bmi_max, (bmi_min, bmi_max))

    region_options = sorted(df["region"].unique().tolist())
    region_selected = st.sidebar.multiselect("Region", options=region_options, default=region_options)

    smoker_options = sorted(df["smoker"].unique().tolist())
    smoker_selected = st.sidebar.multiselect("Smoker", options=smoker_options, default=smoker_options)

    filtered_df = df[
        df["age"].between(age_range[0], age_range[1])
        & df["bmi"].between(bmi_range[0], bmi_range[1])
        & df["region"].isin(region_selected)
        & df["smoker"].isin(smoker_selected)
    ]

    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** records after filtering.")

    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            filtered_df, x="charges", nbins=30, marginal="box",
            title="Distribution of Charges",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        fig_scatter = px.scatter(
            filtered_df, x="bmi", y="charges", color="smoker",
            title="BMI vs. Charges (colored by smoker status)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = filtered_df[["age", "bmi", "children", "charges"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ==========================================================================
with tab2:
    st.header("Hypothesis Testing Lab")
    alpha = 0.05

    st.subheader("Test 1: Compare a Numerical Metric Across Groups")
    cat_options = ["smoker", "sex", "region"]
    num_options = ["charges", "bmi", "age", "children"]

    cat_col = st.selectbox("Categorical grouping variable", options=cat_options, key="cat1")
    num_col = st.selectbox("Numerical metric", options=num_options, key="num1")

    groups = sorted(df[cat_col].unique().tolist())
    group_data = [df[df[cat_col] == g][num_col] for g in groups]

    st.write(f"**H0:** Mean {num_col} is equal across {cat_col} groups.")
    st.write(f"**H1:** Mean {num_col} differs across {cat_col} groups.")

    if len(groups) == 2:
        norm_ps = []
        for g, data in zip(groups, group_data):
            stat_g, p_g = stats.shapiro(data)
            norm_ps.append(p_g)
            st.write(f"Shapiro-Wilk ({g}): p = {p_g:.4e}")

        stat_lev, p_lev = stats.levene(*group_data)
        st.write(f"Levene's test (equal variance): p = {p_lev:.4e}")

        normal = all(p > alpha for p in norm_ps)
        equal_var = p_lev > alpha

        if normal:
            test_stat, p_value = stats.ttest_ind(*group_data, equal_var=equal_var)
            test_name = "Two-Sample t-test"
        else:
            test_stat, p_value = stats.mannwhitneyu(*group_data, alternative="two-sided")
            test_name = "Mann-Whitney U test"

        st.write(f"**Test used:** {test_name}")
        st.write(f"Statistic = {test_stat:.4f}, p-value = {p_value:.4e}")

    else:
        test_stat, p_value = stats.f_oneway(*group_data)
        st.write("**Test used:** One-Way ANOVA")
        st.write(f"F-statistic = {test_stat:.4f}, p-value = {p_value:.4e}")

    if p_value < alpha:
        st.success(f"Reject H0 at α = 0.05 — {num_col} differs significantly across {cat_col} groups.")
    else:
        st.error(f"Fail to Reject H0 at α = 0.05 — no significant difference in {num_col} across {cat_col} groups.")

    st.divider()

    st.subheader("Test 2: Chi-Square Test of Association")
    cat_col2_options = [c for c in cat_options if c != cat_col]
    cat_col2 = st.selectbox("Second categorical variable", options=cat_col2_options, key="cat2")

    st.write(f"**H0:** {cat_col} is independent of {cat_col2}.")
    st.write(f"**H1:** {cat_col} is associated with {cat_col2}.")

    contingency = pd.crosstab(df[cat_col], df[cat_col2])
    st.dataframe(contingency, use_container_width=True)

    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency)
    st.write(f"Chi-Square = {chi2_stat:.4f}, dof = {dof}, p-value = {chi2_p:.4e}")

    if chi2_p < alpha:
        st.success(f"Reject H0 at α = 0.05 — {cat_col} is significantly associated with {cat_col2}.")
    else:
        st.error(f"Fail to Reject H0 at α = 0.05 — no significant association between {cat_col} and {cat_col2}.")

# ==========================================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ==========================================================================
with tab3:
    st.header("Live Prediction & Diagnostics")
    st.write("Adjust the inputs below to get a real-time prediction of medical charges.")

    col1, col2, col3 = st.columns(3)
    with col1:
        input_age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), int(df["age"].median()))
        input_bmi = st.number_input(
            "BMI", float(df["bmi"].min()), float(df["bmi"].max()), float(round(df["bmi"].median(), 1))
        )
    with col2:
        input_children = st.slider(
            "Children", int(df["children"].min()), int(df["children"].max()), int(df["children"].median())
        )
        input_smoker = st.selectbox("Smoker", options=sorted(df["smoker"].unique().tolist()))
    with col3:
        input_sex = st.selectbox("Sex", options=sorted(df["sex"].unique().tolist()))
        input_region = st.selectbox("Region", options=sorted(df["region"].unique().tolist()))

    input_df = pd.DataFrame({
        "age": [input_age],
        "bmi": [input_bmi],
        "children": [input_children],
        "smoker": [input_smoker],
        "sex": [input_sex],
        "region": [input_region],
    })

    pred = model.get_prediction(input_df)
    pred_summary = pred.summary_frame(alpha=0.05)

    predicted_charge = pred_summary["mean"].iloc[0]
    pi_lower = pred_summary["obs_ci_lower"].iloc[0]
    pi_upper = pred_summary["obs_ci_upper"].iloc[0]

    st.metric("Predicted Charges", f"${predicted_charge:,.2f}")
    st.write(f"**95% Prediction Interval:** ${pi_lower:,.2f} — ${pi_upper:,.2f}")

    with st.expander("View Full Model Summary"):
        st.text(str(model.summary()))

    st.divider()
    st.subheader("Residual Diagnostics")

    fitted_vals = model.fittedvalues
    residuals = model.resid

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fig_resid = px.scatter(
            x=fitted_vals, y=residuals,
            labels={"x": "Fitted Values", "y": "Residuals"},
            title="Residuals vs. Fitted Values",
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)

    with dcol2:
        qq = sm.ProbPlot(residuals)
        theo_q = qq.theoretical_quantiles
        samp_q = qq.sample_quantiles
        fig_qq = px.scatter(
            x=theo_q, y=samp_q,
            labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
            title="Q-Q Plot of Residuals",
        )
        line_min = min(theo_q.min(), samp_q.min())
        line_max = max(theo_q.max(), samp_q.max())
        fig_qq.add_shape(
            type="line", x0=line_min, y0=line_min, x1=line_max, y1=line_max,
            line=dict(color="red", dash="dash"),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
    st.write(f"**Jarque-Bera Test:** stat = {jb_stat:.4f}, p-value = {jb_p:.4e}")
    st.write(f"Skew = {jb_skew:.4f}, Kurtosis = {jb_kurt:.4f}")

    with st.expander("View Variance Inflation Factors (VIF)"):
        X = sm.add_constant(df[["age", "bmi", "children"]])
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_data, use_container_width=True)
