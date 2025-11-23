# Analysis Plan: Multivariate Hierarchical Regression

## Study Design
- **Sample:** N=255, split between experimental and control groups
- **Hypothesis:** Experimental manipulation has no meaningful effect on trust (equivalence testing with pre-specified bounds)
- **Outcome:** 5 trust subscales (correlated dimensions)
- **Predictors:**
    - Experimental condition (binary: experimental vs control)
    - Demographics: age (continuous), education level (ordinal/categorical)
    - Psychometric scales: 3 continuous scales (technology affinity, healthcare trust, etc.)

## Analytical Approach
Multivariate hierarchical regression with four sequential models, testing both direct effects and moderation effects.

## Analysis Steps

### 1. Data Preparation
- Load and clean data
- Check for missing values
- Create interaction terms (mean-center continuous predictors first)
- Verify grouping variables are coded correctly

### 2. Descriptive Statistics
- Means, SDs, and ranges for all trust subscales by experimental condition
- Correlation matrix for the 5 trust subscales
- Correlation matrix for all predictors
- Sample size per group

### 3. Assumption Checks
- **Univariate normality:** Histograms and Q-Q plots for each trust subscale
- **Multivariate outliers:** Mahalanobis distance (flag values with p < .001)
- **Homogeneity of covariance:** Box's M test (if violated, use Pillai's Trace)
- **Multicollinearity:** VIF for all predictors (especially after adding interactions; VIF < 10)

### 4. Hierarchical Model Building

**Model 1:** Experimental condition only
- Predictors: condition
- DV: 5 trust subscales (multivariate)

**Model 2:** Add demographics
- Predictors: condition + age + education
- DV: 5 trust subscales

**Model 3:** Add psychometric scales
- Predictors: condition + age + education + 3 psychometric scales
- DV: 5 trust subscales

**Model 4:** Add interactions (moderation tests)
- Predictors: all from Model 3 + condition×age + condition×education + condition×(each psychometric scale)
- DV: 5 trust subscales

### 5. Statistical Tests for Each Model

**Multivariate tests:**
- Wilks' Lambda or Pillai's Trace
- F-statistic, df, p-value
- Partial η² (effect size)
- Test ΔR² between successive models

**Univariate follow-up tests (for each trust subscale):**
- Regression coefficients (unstandardized and standardized)
- 95% confidence intervals
- t-statistics and p-values
- Individual R² for each subscale

### 6. Primary Hypothesis Test (Null Effect)
For Model 1 (condition effect):
- Extract 95% CIs for condition effect on each subscale
- Verify CIs fall within equivalence bounds (specify bounds based on your prior non-inferiority margin)
- Create visualization: CI plot with equivalence region shaded

### 7. Moderation Analysis (Model 4)
For each significant interaction:
- Simple slopes analysis at moderator levels: M - 1SD, M, M + 1SD (continuous) or each category (categorical)
- Plot interactions (predicted trust subscale values by condition at different moderator levels)
- Interpret region of significance if applicable

### 8. Effect Size Summary
- Partial η² for multivariate tests
- Standardized β for univariate effects
- R² and ΔR² for each model
- Cohen's f² for specific effects of interest

### 9. Results Tables
**Table 1:** Descriptive statistics (means, SDs by condition)
**Table 2:** Correlation matrix (trust subscales)
**Table 3:** Model comparison (R², ΔR², F, p for each model)
**Table 4:** Multivariate test results (all four models)
**Table 5:** Univariate results for each subscale (final model coefficients)
**Table 6:** Interaction effects (if significant)

### 10. Visualizations
- Correlation heatmap for trust subscales
- CI plot for condition effects with equivalence bounds
- Interaction plots for significant moderators
- Optional: residual plots for assumption checking

## Key Technical Notes
- Mean-center all continuous predictors before creating interactions
- Use Pillai's Trace if Box's M test is significant
- Apply Holm-Bonferroni correction if testing multiple exploratory moderators
- For equivalence testing: report both whether effect is non-significant AND whether CI falls within equivalence bounds
- Document which hypotheses are confirmatory vs exploratory

## Python Implementation Requirements
- Packages: pandas, numpy, statsmodels, scipy, matplotlib/seaborn
- Use `statsmodels.multivariate.manova.MANOVA` for multivariate tests
- Use `statsmodels.api.OLS` for univariate follow-ups
- Calculate VIF using `statsmodels.stats.outliers_influence.variance_inflation_factor`
- Mahalanobis distance using scipy.spatial.distance or custom function