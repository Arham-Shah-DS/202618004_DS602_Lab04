# Medical Insurance Charges — Statistical Analysis & Dashboard

**Live App:** [Medical Insurance Charges Dashboard](https://arham-shah-ds-202618004-ds602-lab04-app-ybxwmo.streamlit.app/)

M.Sc. Data Science (Sem 1) — Lab 4: Applied Statistical Modeling & Interactive Web Dashboard.
This project applies hypothesis testing and OLS regression to the Medical Insurance Costs dataset, then exposes the full analysis through an interactive Streamlit dashboard.

---

## Dataset Summary

**Source:** Medical Insurance Costs dataset (`insurance.csv`), 1,338 records.

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the primary beneficiary |
| `sex` | Categorical | Insurance contractor gender (male/female) |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Number of dependents covered |
| `smoker` | Categorical | Smoking status (yes/no) |
| `region` | Categorical | Residential area in the US (4 regions) |
| `charges` | Numeric | Individual medical costs billed (target variable) |

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Arham-Shah-DS/202618004_DS602_Lab04.git
cd 202618004_DS602_Lab04
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the dashboard**
```bash
streamlit run app.py
```
This opens the dashboard at `http://localhost:8501`.

> Note: `insurance.csv` must be in the same folder as `app.py` — it's already included in this repo.

---

## Project Structure

```
├── app.py              # Streamlit dashboard (3 tabs: EDA, Hypothesis Testing, Prediction)
├── lab.ipynb            # Full analysis notebook (EDA, hypothesis tests, OLS model, diagnostics)
├── insurance.csv         # Dataset
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Statistical Findings

### Part 1: Exploratory Data Analysis
Descriptive statistics (mean, median, std, IQR, skewness, kurtosis) were computed for `age`, `bmi`, `children`, and `charges`. `charges` shows strong right-skew, consistent with a small subset of high-cost claims (largely driven by smoking status).

### Hypothesis Test 1 — Smoker vs. Non-Smoker Charges
- **H0:** Mean charges is equal across smoker groups.
- **H1:** Mean charges differs across smoker groups.
- **Normality (Shapiro-Wilk):** non-smokers p = 1.4459e-28, smokers p = 3.6250e-09 — both groups fail normality.
- **Equal variance (Levene's test):** p = 1.5593e-66 — variances are not equal.
- **Test used:** Mann-Whitney U test (chosen since neither group is normally distributed)
- **Result:** statistic = 7403.0000, p-value = 5.2702e-130
- **Conclusion:** Reject H0 at α = 0.05 — charges differ significantly between smokers and non-smokers. This is the expected and by far the strongest effect in the dataset: smoking status dominates medical charges.

### Hypothesis Test 2 — Chi-Square Test of Association (Smoker vs. Sex)
- **H0:** Smoker status is independent of sex.
- **H1:** Smoker status is associated with sex.
- **Result:** Chi-Square = 7.3929, dof = 1, p-value = 6.5481e-03
- **Conclusion:** Reject H0 at α = 0.05 — smoking status is significantly associated with sex in this dataset (though this is a weaker, secondary finding compared to Test 1 — a chi-square p-value near 0.006 is a modest association, not a dominant driver of charges).

### OLS Regression Model
**Formula:** `charges ~ age + bmi + children + C(smoker) + C(sex) + C(region) + bmi:C(smoker)`

- **R² / Adjusted R²:** 0.841 / 0.840 — the model explains about 84% of the variance in medical charges, a strong fit.
- **F-statistic:** 780.0 (p < 0.001) — the model as a whole is highly statistically significant.

**Significant predictors (p < 0.05):**
| Predictor | Coefficient | p-value | Interpretation |
|---|---|---|---|
| `age` | +263.62 | 0.000 | Each additional year of age adds ~$264 to predicted charges |
| `children` | +516.40 | 0.000 | Each additional dependent adds ~$516 |
| `bmi:C(smoker)[T.yes]` | +1443.10 | 0.000 | For smokers, each unit of BMI adds an extra ~$1,443 on top of the baseline BMI effect — smoking sharply amplifies the cost impact of higher BMI |
| `C(region)[T.southeast]` | -1210.13 | 0.002 | Southeast region has significantly lower charges than the baseline region |
| `C(region)[T.southwest]` | -1231.11 | 0.001 | Southwest region also has significantly lower charges than the baseline region |
| `C(smoker)[T.yes]` | -20,420 | 0.000 | Significant, but see note below on interpreting this alongside the interaction term |

**Not statistically significant (p ≥ 0.05):** `bmi` alone (p = 0.358), `C(sex)[T.male]` (p = 0.061, borderline), `C(region)[T.northwest]` (p = 0.124).

**Note on the smoker coefficient:** the raw `C(smoker)[T.yes]` coefficient looks counterintuitive (large and negative) because, once an interaction term (`bmi:C(smoker)`) is in the model, that coefficient represents the effect of being a smoker *at BMI = 0* — a value that never occurs in practice. The real story is in the interaction term: smoking's effect on charges scales strongly and positively with BMI, meaning smoking and high BMI compound each other's cost impact rather than adding independently.

**Diagnostic checks:**
- Residuals vs. Fitted: `[pending — paste output once you run this cell]`
- Q-Q Plot / Jarque-Bera: `[pending]`
- VIF (multicollinearity): `[pending]`

---

## Dashboard Overview

| Tab | Description |
|---|---|
| **Data Exploration** | Sidebar filters (age, BMI, region, smoker), reactive Plotly histograms/scatter plots, live summary statistics |
| **Hypothesis Testing Lab** | Dynamic dropdowns to test any categorical vs. numerical pair (auto-selects t-test/Mann-Whitney/ANOVA), plus a Chi-Square test for categorical associations |
| **Live Prediction & Diagnostics** | Interactive inputs for real-time charge prediction with 95% prediction interval, plus residual diagnostic plots |

---

## Tech Stack
Python · pandas · NumPy · SciPy · statsmodels · Plotly · Streamlit
