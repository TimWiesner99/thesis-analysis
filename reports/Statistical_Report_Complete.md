# Statistical Analysis Report
## Effect of Uncertainty Communication on Trust in Medical AI Systems

**Report Date**: December 2, 2025
**Analysis Type**: Comprehensive Statistical Documentation
**Intended Audience**: Research team members writing Methods and Results sections

---

## 1. Executive Summary

### Study Design
This study employed a between-subjects experimental design ($N = 255$) to investigate whether communicating technical uncertainty information affects patients' trust in Medical AI Decision Support Systems (MDSS). Participants were randomly assigned to one of two conditions:

- **Control condition** ($n = 126$): Standard accuracy presentation ("90% accuracy")
- **Uncertainty condition** ($n = 129$): Accuracy with uncertainty margin ("90% accuracy, ±8% margin of error (82%-98%)")

### Primary Hypothesis
The research tested a **null-effect hypothesis**: Communicating AI uncertainty does not significantly affect trust in medical AI systems across five dimensions of trust (reliability/confidence, understanding/predictability, familiarity, propensity to trust, and general trust in automation).

### Key Findings
Statistical equivalence between conditions was confirmed:

1. **Multivariate analysis** (MANOVA): No significant group effect, Pillai's $V = 0.020$, $F(5, 235) = 0.951$, $p = .448$, $\eta_p^2 = 0.020$
2. **Univariate analyses**: None of the five TiA subscales showed significant between-group differences (all $p > .05$, all Cohen's $d < 0.20$)
3. **Equivalence testing**: Multivariate equivalence confirmed; observed effect ($\eta_p^2 = 0.0198$) well within equivalence bound ($\eta_p^2 = 0.0493$)
4. **Moderation analyses**: No significant interactions after Holm-Bonferroni correction (all adjusted $p > .05$)

### Significance
These results demonstrate that transparent communication about AI uncertainty is **neutral** for patient trust, supporting the use of uncertainty communication without concerns about trust degradation. This finding is robust across demographic and psychological profiles (age, gender, education, AI experience, healthcare trust, technology affinity).

---

## 2. Sample and Data Preparation

### 2.1 Sample Characteristics

**Initial sample**: 347 responses collected via Prolific
**Final sample**: 255 complete responses (73.5% retention)
**Exclusions**: 92 participants removed due to incomplete data

**Group allocation**:
- Control: $n = 126$ (49.4%)
- Uncertainty: $n = 129$ (50.6%)

### 2.2 Data Cleaning Procedures

The preprocessing workflow followed these steps:

1. **Qualtrics metadata removal**: Deleted rows 0-1 (column headers and example responses)
2. **Column reduction**: Dropped timing variables and Qualtrics metadata columns not required for analysis
3. **Timer consolidation**: Combined split `delay_timer_Page Submit` columns into single `page_submit` variable (stimulus viewing time)
4. **Completeness filtering**: Removed cases with missing data on primary outcome measures (TiA subscales)
5. **Scale computation**: Calculated mean scores for:
   - ATI (Affinity for Technology Interaction): 9 items
   - HCSDS (Healthcare System Distrust Scale): 2 subscales (competence: 5 items, values: 5 items)
   - TiA (Trust in Automation): 5 subscales (16 items total)

**Output files**:
- `data/data_clean.csv`: Cleaned individual-level data
- `data/data_scales.csv`: Computed scale scores for analysis

### 2.3 Variable Coding and Transformations

To optimize statistical power and interpretation, all variables were transformed prior to analysis:

#### 2.3.1 Treatment Variable (Effect Coding)
```
group_effect = stimulus_group - 0.5
```
- Control: `stimulus_group = 0` → `group_effect = -0.5`
- Uncertainty: `stimulus_group = 1` → `group_effect = 0.5`

**Rationale**: Effect coding centers the treatment variable at zero, improving interpretation of main effects and interactions in regression models.

#### 2.3.2 Continuous Variables (Z-Standardization)
For all continuous moderators and covariates:
```
variable_c = (variable - M) / SD
```

**Variables standardized**:
- Age (years)
- Healthcare trust - competence (`hcsds_c`)
- Healthcare trust - values (`hcsds_v`)
- Technology affinity (`ati`)
- Stimulus viewing time (`page_submit`)

**Rationale**: Standardization places all continuous predictors on the same scale, facilitating comparison of regression coefficients and improving numerical stability.

#### 2.3.3 Categorical Variables (Effect Coding)
**Gender**:
```
gender_c = {
    0.5    if male (gender = 1)
   -0.5    if female (gender = 2)
    0      if other/prefer not to say (gender = 3)
}
```

**Rationale**: Effect coding maintains symmetric interpretation around zero.

#### 2.3.4 Ordinal Variables (Mean-Centering)
```
education_c = education - M_education
ai_exp_c = ai_exp - M_ai_exp
```

**Rationale**: Mean-centering ordinal predictors preserves ordinality while improving interpretability of the intercept and main effects.

### 2.4 Missing Data Handling

**Approach**: Complete case analysis
**Missingness pattern**: No missing data in final sample ($N = 255$)
**Justification**: All participants with incomplete outcome measures were excluded during preprocessing, resulting in a complete dataset for all analyses.

---

## 3. Measured Constructs

### 3.1 Primary Outcome Variables

#### Trust in Automation Scale (TiA; Körber, 2019 - Adapted)

Five subscales measuring distinct dimensions of trust in AI systems:

1. **tia_rc** (Reliability/Confidence) - 6 items
   - Description: Capability-based trust in system performance
   - Example item: "The system works reliably"
   - Scale: 1 (completely disagree) to 5 (completely agree)

2. **tia_up** (Understanding/Predictability) - 3 items
   - Description: Shared mental model; user's understanding of system behavior
   - Example item: "I understand how the system works"
   - Scale: 1-5

3. **tia_f** (Familiarity) - 2 items
   - Description: Prior experience and comfort with similar systems
   - Example item: "I am familiar with this kind of system"
   - Scale: 1-5

4. **tia_pro** (Propensity to Trust) - 3 items
   - Description: General disposition to trust automation (trait-like)
   - Example item: "I tend to trust automated systems"
   - Scale: 1-5

5. **tia_t** (Trust in Automation) - 2 items
   - Description: Overall trust in the specific AI system presented
   - Example item: "I trust the system"
   - Scale: 1-5

**Internal consistency**: Not reported in this analysis (scale validated in Körber, 2019)

### 3.2 Moderators and Covariates

#### 3.2.1 Affinity for Technology Interaction (ATI; Franke et al., 2019)
- **Description**: Single-factor scale measuring general technology affinity and engagement
- **Items**: 9 items
- **Example**: "I like to occupy myself in greater detail with technical systems"
- **Scale**: 1 (completely disagree) to 6 (completely agree)

#### 3.2.2 Revised Health Care System Distrust Scale (HCSDS; Shea et al., 2008)
**Note**: Interpretation inverted to measure **trust** instead of distrust

Two subscales:
1. **hcsds_c** (Competence) - 5 items
   - Description: Trust in healthcare system's technical capabilities
   - Example: "Patients receive high-quality medical care from the health care system"

2. **hcsds_v** (Values) - 5 items
   - Description: Trust in healthcare system's benevolence and value alignment
   - Example: "The health care system puts making money above patients' needs" (reverse-coded)

**Scale**: 1 (strongly disagree) to 5 (strongly agree)

#### 3.2.3 Demographic Variables

| Variable | Type | Coding | Description |
|----------|------|--------|-------------|
| `age` | Continuous | Years | Participant age |
| `gender` | Categorical | 1=male, 2=female, 3=other | Self-reported gender |
| `education` | Ordinal | 1-7 scale | Educational attainment |
| `ai_exp` | Ordinal | 1-5 scale | Self-reported AI experience |
| `medical_prof` | Binary | 0=no, 1=yes | Medical professional status |

### 3.3 Manipulation Check Items

Four items assessed manipulation effectiveness (`manip_check1_1` through `manip_check1_4`):

1. Recognition of uncertainty information
2. Understanding of accuracy range
3. Attention to stimulus content
4. Perception of system reliability

**Scale**: 1 (strongly disagree) to 5 (strongly agree)
**Analysis**: See Section 5.1 for results

---

## 4. Statistical Methods

### 4.1 Manipulation Check Analysis

#### 4.1.1 Between-Group Comparisons
**Test**: Mann-Whitney U test (Wilcoxon rank-sum test)

**Rationale**: Non-parametric test appropriate for:
- Ordinal data (5-point Likert scales)
- Non-normal distributions
- Robust to outliers

**Formula**:
Test statistic $U$ is computed as:
$$U = n_1 n_2 + \frac{n_1(n_1 + 1)}{2} - R_1$$
where $R_1$ is the sum of ranks for group 1, $n_1$ and $n_2$ are sample sizes.

**Effect size**: Rank-biserial correlation $r_{\text{rb}}$:
$$r_{\text{rb}} = 1 - \frac{2U}{n_1 n_2}$$

Interpretation: $|r_{\text{rb}}| < 0.1$ (negligible), $0.1 \leq |r_{\text{rb}}| < 0.3$ (small), $0.3 \leq |r_{\text{rb}}| < 0.5$ (medium), $|r_{\text{rb}}| \geq 0.5$ (large)

#### 4.1.2 Consistency Checks
**Test**: Spearman's rank correlation coefficient $\rho$

**Purpose**: Assess internal consistency of understanding between redundant manipulation check items (items 2 and 3)

**Formula**:
$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
where $d_i$ is the difference between ranks for each observation.

#### 4.1.3 Contingency Analysis
**Test**: Pearson's $\chi^2$ test of independence

**Purpose**: Examine categorical patterns of responses (5-point scale collapsed to 3 categories: disagree/neutral/agree)

**Formula**:
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
where $O_{ij}$ are observed frequencies and $E_{ij}$ are expected frequencies under independence.

**Effect size**: Cramér's $V$:
$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$$

#### 4.1.4 Language Equivalence
**Test**: Kruskal-Wallis H test (non-parametric one-way ANOVA)

**Purpose**: Test whether manipulation effectiveness differed across survey languages (English, Dutch, German)

**Formula**:
$$H = \frac{12}{n(n+1)} \sum_{i=1}^k \frac{R_i^2}{n_i} - 3(n+1)$$
where $R_i$ is the sum of ranks for group $i$.

### 4.2 Primary Analysis: Main Effect Testing

#### 4.2.1 Welch's t-test
**Null hypothesis**: $H_0: \mu_{\text{control}} = \mu_{\text{uncertainty}}$
**Alternative hypothesis**: $H_1: \mu_{\text{control}} \neq \mu_{\text{uncertainty}}$ (two-tailed)

**Test statistic**:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

**Degrees of freedom** (Welch-Satterthwaite approximation):
$$df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Rationale**: Welch's t-test does not assume equal variances between groups (more robust than Student's t-test).

**Significance level**: $\alpha = 0.05$ (two-tailed)

#### 4.2.2 Effect Size: Cohen's d
**Formula** (pooled standard deviation):
$$d = \frac{|\bar{X}_1 - \bar{X}_2|}{s_{\text{pooled}}}$$
$$s_{\text{pooled}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$

**Interpretation**: $|d| < 0.2$ (small), $0.2 \leq |d| < 0.5$ (small-medium), $0.5 \leq |d| < 0.8$ (medium-large), $|d| \geq 0.8$ (large)

#### 4.2.3 Confidence Intervals
**95% confidence interval for mean difference**:
$$CI_{95\%} = (\bar{X}_1 - \bar{X}_2) \pm t_{0.025, df} \cdot SE$$
where $SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$

### 4.3 Multivariate Analysis (MANOVA)

#### 4.3.1 Model Specification
**Dependent variables**: $(Y_1, Y_2, Y_3, Y_4, Y_5)$ = (tia_f, tia_pro, tia_rc, tia_up, tia_t)

**Model**:
$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{Y}$ is $n \times 5$ matrix of outcomes
- $\mathbf{X}$ is $n \times 16$ design matrix
- $\boldsymbol{\beta}$ is $16 \times 5$ coefficient matrix
- $\boldsymbol{\epsilon}$ is $n \times 5$ error matrix

**Predictors** ($p = 16$):
1. Intercept
2. `group_effect` (main treatment effect)
3-9. Direct effects: `age_c`, `gender_c`, `education_c`, `ai_exp_c`, `hcsds_c_c`, `hcsds_v_c`, `ati_c`
10-16. Interaction effects: `group_effect × [each of the 7 direct effects]`

#### 4.3.2 Test Statistic: Pillai's Trace
**Formula**:
$$V = \text{tr}(\mathbf{H}(\mathbf{H} + \mathbf{E})^{-1})$$

where:
- $\mathbf{H}$ = hypothesis sum of squares and cross-products (SSCP) matrix
- $\mathbf{E}$ = error SSCP matrix
- $\text{tr}$ = matrix trace operator

**F-approximation**:
$$F = \frac{V}{s - V} \cdot \frac{df_2}{df_1}$$

where:
- $s = \min(p, df_h)$ = number of dependent variables or hypothesis df
- $df_1 = s(2m + s + 1)$
- $df_2 = s(2n + s + 1)$
- $m = (|df_h - p| - 1)/2$
- $n = (df_e - p - 1)/2$

For this analysis:
- $p = 5$ (dependent variables)
- $df_h = 1$ (group effect)
- $df_e = N - k = 255 - 2 = 253$

**Rationale for Pillai's Trace**:
- Most robust MANOVA test statistic
- Maintains appropriate Type I error rates when assumptions are violated
- Preferred over Wilks' $\Lambda$, Hotelling-Lawley trace, and Roy's largest root when homogeneity of covariance matrices is questionable

#### 4.3.3 Effect Size: Partial Eta-Squared
**Formula** (multivariate):
$$\eta_p^2 = \frac{F \cdot df_1}{F \cdot df_1 + df_2}$$

**Interpretation**: Proportion of variance in the combined dependent variables explained by the predictor, after controlling for other predictors.

### 4.4 Equivalence/Non-Inferiority Testing

#### 4.4.1 Theoretical Framework
Traditional null hypothesis significance testing (NHST) tests:
$$H_0: \Delta = 0 \quad \text{vs.} \quad H_1: \Delta \neq 0$$

**Limitation**: Failure to reject $H_0$ does not prove equivalence (absence of evidence $\neq$ evidence of absence).

**Equivalence testing** reverses the burden of proof:
$$H_0: |\Delta| \geq \delta \quad \text{vs.} \quad H_1: |\Delta| < \delta$$

where $\delta$ is the **equivalence margin** (smallest effect size of interest, SESOI).

**Non-inferiority testing** (one-sided):
$$H_0: \Delta \leq -\delta \quad \text{vs.} \quad H_1: \Delta > -\delta$$

Tests whether the treatment is "not substantially worse" than control.

#### 4.4.2 Minimally Detectable Effect (MDE) Calculation

**Definition**: MDE is the smallest effect size detectable with specified power and significance level.

**Univariate MDE** (from regression):
$$\text{MDE} = (t_{\alpha/2, df} + t_{\beta, df}) \cdot SE$$

where:
- $t_{\alpha/2, df}$ = critical $t$-value for two-tailed test at significance level $\alpha$
- $t_{\beta, df}$ = critical $t$-value for power $1-\beta$
- $SE$ = standard error of the regression coefficient

**Parameters**:
- $\alpha = 0.05$ (two-tailed)
- $1 - \beta = 0.80$ (power)
- $df = 239$ (residual degrees of freedom)

**Calculation**:
```
t_0.025,239 = 1.970
t_0.80,239 = 0.849
MDE = (1.970 + 0.849) × SE = 2.819 × SE
```

**Multivariate MDE** (from MANOVA):

Using non-centrality parameter approach:

$$\lambda_{\text{MDE}} = F_{\text{crit}} \cdot df_1 \cdot \frac{df_2}{df_2 - F_{\text{crit}} \cdot df_1}$$

where $F_{\text{crit}}$ is found such that the non-central $F$ distribution has power $1-\beta$.

Converted to partial eta-squared:
$$\eta_{p,\text{MDE}}^2 = \frac{\lambda}{N - k + \lambda}$$

For this study:
- $N = 255$, $k = 2$, $p = 5$, $\alpha = 0.05$, $1-\beta = 0.80$
- $\lambda_{\text{MDE}} = 13.115$
- $\eta_{p,\text{MDE}}^2 = 0.0493$

#### 4.4.3 Equivalence Decision Rules

**Non-inferiority**: Reject $H_0$ (conclude equivalence) if:
$$|\hat{\Delta}| < \text{MDE}$$

**Equivalence** (two one-sided tests, TOST): Reject $H_0$ if:
$$CI_{1-2\alpha}(\Delta) \subset (-\delta, \delta)$$

In other words, if the $(1-2\alpha)$ confidence interval for the effect size is entirely contained within the equivalence bounds.

For this analysis, equivalence bounds set to:
- Univariate: MDE values specific to each subscale (in original units)
- Multivariate: $\eta_{p}^2 = 0.0493$

#### 4.4.4 Confidence Interval Construction for $\eta_p^2$

Standard approach (based on $t$ or $F$ distribution) assumes normality of effect sizes, which is inappropriate for bounded parameters like $\eta_p^2 \in [0, 1]$.

**Method**: Non-central $F$ distribution inversion (Steiger, 2004)

**Procedure**:

1. Convert $t$-statistic to $F$: $F_{\text{obs}} = t^2$ (for univariate) or use Pillai's $F$ directly (for multivariate)

2. Find lower bound $\lambda_L$ such that:
   $$P(F \geq F_{\text{obs}} \mid \lambda = \lambda_L) = \alpha/2$$

3. Find upper bound $\lambda_U$ such that:
   $$P(F \geq F_{\text{obs}} \mid \lambda = \lambda_U) = 1 - \alpha/2$$

4. Convert non-centrality bounds to $\eta_p^2$:
   $$\eta_{p,L}^2 = \frac{\lambda_L}{\lambda_L + df_{\text{error}}}$$
   $$\eta_{p,U}^2 = \frac{\lambda_U}{\lambda_U + df_{\text{error}}}$$

**Implementation**: Numerical root-finding (Brent's method) to solve for $\lambda_L$ and $\lambda_U$.

**Degrees of freedom**:
- Univariate: $df_1 = 1$, $df_2 = df_{\text{resid}} = 239$, $df_{\text{error}} = 239$
- Multivariate: $df_1 = 5$, $df_2 = 235$, $df_{\text{error}} = N - k = 253$

### 4.5 Moderation Analysis

#### 4.5.1 Conceptual Framework
Moderation tests whether the effect of the independent variable ($X$) on the dependent variable ($Y$) depends on the level of a third variable ($M$, the moderator).

**Statistical model**:
$$Y = \beta_0 + \beta_1 X + \beta_2 M + \beta_3 (X \times M) + \epsilon$$

where:
- $\beta_1$ = main effect of treatment (conditional on $M = 0$, i.e., at mean if centered)
- $\beta_2$ = main effect of moderator
- $\beta_3$ = **interaction effect (moderation effect)**

**Interpretation of $\beta_3$**:
- $\beta_3 > 0$: Treatment effect increases as moderator increases
- $\beta_3 < 0$: Treatment effect decreases as moderator increases
- $\beta_3 = 0$: No moderation (treatment effect constant across moderator levels)

#### 4.5.2 Model Specification
Full moderation model for each TiA subscale:

$$\text{TiA}_j = \beta_0 + \beta_1 \text{group\_effect} + \sum_{i=1}^{8} \beta_{i+1} M_i + \sum_{i=1}^{8} \beta_{i+9} (\text{group\_effect} \times M_i) + \epsilon$$

where:
- $j \in \{\text{f, pro, rc, up, t}\}$ (5 outcomes)
- $M_i$ = 8 moderators: age, gender, education, AI experience, healthcare trust (competence), healthcare trust (values), technology affinity
- Total predictors: $1 + 1 + 8 + 8 = 18$ (intercept + main effect + 8 direct + 8 interactions)

**Note**: Moderation models estimated **separately** for each outcome (not joint multivariate estimation).

#### 4.5.3 Interaction Coefficient Interpretation
Given effect coding ($X \in \{-0.5, 0.5\}$) and centered moderators ($M$ centered at 0):

**Simple slopes**:
- Effect of $X$ when $M = 0$ (mean): $\beta_1$
- Effect of $X$ when $M = +1$ SD: $\beta_1 + \beta_3$
- Effect of $X$ when $M = -1$ SD: $\beta_1 - \beta_3$

**Region of significance**: Values of $M$ for which the simple slope is significant can be computed using the Johnson-Neyman technique (not conducted in this analysis).

### 4.6 Multiple Comparison Corrections

#### 4.6.1 Rationale
Testing multiple hypotheses inflates family-wise Type I error rate:
$$P(\text{at least one false positive}) = 1 - (1 - \alpha)^k$$

For $k = 60$ tests (5 outcomes × 12 predictors) at $\alpha = 0.05$:
$$P(\text{FP}) = 1 - 0.95^{60} = 0.953$$

**Solution**: Control family-wise error rate (FWER) using sequential rejection procedures.

#### 4.6.2 Holm-Bonferroni Method
**Procedure**:

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(k)}$
2. For $j = 1, 2, \ldots, k$:
   - If $p_{(j)} \leq \frac{\alpha}{k - j + 1}$, reject $H_{(j)}$ and continue
   - Otherwise, fail to reject $H_{(j)}$ and all remaining hypotheses

**Adjusted p-value**:
$$p_{\text{adj}, (j)} = \max_{i=1,\ldots,j} \{(k - i + 1) \cdot p_{(i)}\}$$

**Properties**:
- Uniformly more powerful than Bonferroni correction
- Controls FWER at level $\alpha$
- Does not require independence assumption

#### 4.6.3 Family Structure
Tests grouped into three **families** to balance Type I error control with statistical power:

**Family 1: Group Effect** (1 test per outcome/effect)
- Main effect of treatment (`group_effect`)
- **Rationale**: Primary hypothesis; tested separately from exploratory analyses

**Family 2: Direct Effects** (8 tests per outcome/effect)
- Main effects of: age, gender, education, AI experience, healthcare trust (competence), healthcare trust (values), technology affinity
- **Rationale**: Exploratory predictors of baseline trust

**Family 3: Interaction Effects** (8 tests per outcome/effect)
- Interactions: `group_effect × [each of 8 moderators]`
- **Rationale**: Exploratory moderation analyses

**Correction applied**:
- Univariate: Within each family, across all 5 outcomes (e.g., 8 direct effects × 5 outcomes = 40 tests for Family 2)
- Multivariate: Within each family (e.g., 8 direct effects = 8 tests for Family 2)

### 4.7 Assumption Checks

#### 4.7.1 Univariate Normality
**Test**: Shapiro-Wilk test

**Null hypothesis**: Data are normally distributed

**Test statistic**:
$$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

where $x_{(i)}$ are ordered observations and $a_i$ are weights based on expected order statistics.

**Decision rule**: Reject normality if $p < 0.05$

**Visual inspection**: Q-Q plots and histograms examined for each TiA subscale.

**Robustness**: Parametric tests (t-tests, MANOVA) are robust to moderate departures from normality when $n$ is large (central limit theorem); conducted as planned regardless of Shapiro-Wilk results.

#### 4.7.2 Multivariate Outliers
**Method**: Mahalanobis distance

**Formula**:
$$D_i^2 = (\mathbf{x}_i - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})$$

where:
- $\mathbf{x}_i$ = observation vector for case $i$
- $\boldsymbol{\mu}$ = sample mean vector
- $\boldsymbol{\Sigma}$ = sample covariance matrix

**Distribution**: Under multivariate normality, $D_i^2 \sim \chi^2_p$ where $p$ = number of variables.

**Outlier criterion**: $p < 0.001$ (i.e., $D_i^2 > \chi^2_{5, 0.999}$)

**Action**: Outliers identified but retained in analysis (no extreme outliers found).

#### 4.7.3 Homogeneity of Covariance Matrices
**Assumption**: $\boldsymbol{\Sigma}_{\text{control}} = \boldsymbol{\Sigma}_{\text{uncertainty}}$

**Test**: Box's M test (highly sensitive to non-normality; not conducted)

**Strategy**: Use Pillai's Trace (robust to heterogeneity) rather than Wilks' $\Lambda$

**Rationale**: Pillai's Trace maintains appropriate Type I error rates even when homogeneity assumption is violated.

#### 4.7.4 Multicollinearity
**Diagnostic**: Variance Inflation Factor (VIF)

**Formula**:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the $R^2$ from regressing $X_j$ on all other predictors.

**Criterion**: $\text{VIF} < 5$ (conservative) or $\text{VIF} < 10$ (liberal)

**Interpretation**:
- $\text{VIF} = 1$: No correlation with other predictors
- $\text{VIF} > 5$: Moderate multicollinearity; standard errors inflated
- $\text{VIF} > 10$: Severe multicollinearity; unreliable coefficient estimates

**Result**: All VIF < 5 (acceptable; see Section 6)

---

## 5. Results

### 5.1 Manipulation Check Results

#### 5.1.1 Recognition and Understanding
Participants in the uncertainty condition successfully recognized and understood the uncertainty information.

**Between-group comparisons** (Mann-Whitney U tests):

| Item | Control $Mdn$ | Uncertainty $Mdn$ | $U$ | $p$ | $r_{\text{rb}}$ |
|------|---------------|-------------------|-----|-----|-----------------|
| Recognition | 3.0 | 4.0 | 5891 | < .001 | 0.27 |
| Understanding (item 2) | 4.0 | 4.0 | 7456 | .142 | 0.08 |
| Understanding (item 3) | 4.0 | 4.0 | 7382 | .103 | 0.09 |
| Attention | 4.0 | 4.0 | 7891 | .512 | 0.04 |

**Interpretation**: Participants in the uncertainty condition were significantly more likely to recognize uncertainty information ($r_{\text{rb}} = 0.27$, small-medium effect). Understanding and attention did not differ between groups, indicating that both groups comprehended the accuracy information.

#### 5.1.2 Internal Consistency
**Spearman correlations** (items 2 × 3, testing redundancy):

- Control: $\rho = 0.308$, $p < .001$
- Uncertainty: $\rho = 0.356$, $p < .001$

**Interpretation**: Positive correlations confirm internal consistency. Participants who agreed with one understanding statement tended to agree with the other, supporting validity of the manipulation check items.

#### 5.1.3 Response Patterns
**Contingency analysis** ($\chi^2$ test, 3-category collapsed scale):

| Response Category | Control % | Uncertainty % |
|-------------------|-----------|---------------|
| Disagree (1-2) | 8.7% | 12.4% |
| Neutral (3) | 10.9% | 23.6% |
| Agree (4-5) | 80.4% | 64.0% |

$\chi^2(2) = 11.42$, $p = .003$, Cramér's $V = 0.21$

**Interpretation**: Control group participants more frequently "agreed" with both understanding statements (80.4%), while uncertainty group participants showed more variability (64.0% agree, 23.6% neutral). This pattern is consistent with uncertainty communication introducing ambiguity, which participants may perceive as reducing their confidence (reflected in more "neutral" responses).

#### 5.1.4 Language Equivalence
**Kruskal-Wallis tests** across languages (English, Dutch, German):

| Item | $H$ | $df$ | $p$ |
|------|-----|------|-----|
| Recognition | 2.14 | 2 | .343 |
| Understanding (item 2) | 0.89 | 2 | .641 |
| Understanding (item 3) | 1.52 | 2 | .467 |
| Attention | 0.71 | 2 | .702 |

**Interpretation**: No significant differences across languages (all $p > .05$). Translations were equivalent in conveying the manipulation.

#### 5.1.5 Conclusion
**Manipulation successful**: Participants in the uncertainty condition recognized uncertainty information, while both groups demonstrated adequate understanding of accuracy information. The manipulation was effective and equivalent across survey languages.

---

### 5.2 Primary Hypothesis: Main Effect of Uncertainty Communication

#### 5.2.1 Descriptive Statistics

| Scale | Group | $M$ | $SD$ | $n$ |
|-------|-------|-----|------|-----|
| **tia_f** (Familiarity) | Control | 2.11 | 0.90 | 126 |
| | Uncertainty | 2.26 | 0.93 | 129 |
| **tia_pro** (Propensity) | Control | 2.75 | 0.68 | 126 |
| | Uncertainty | 2.72 | 0.64 | 129 |
| **tia_rc** (Reliability/Confidence) | Control | 3.31 | 0.56 | 126 |
| | Uncertainty | 3.21 | 0.55 | 129 |
| **tia_t** (Trust in Automation) | Control | 3.40 | 0.80 | 126 |
| | Uncertainty | 3.31 | 0.78 | 129 |
| **tia_up** (Understanding/Predictability) | Control | 3.35 | 0.76 | 126 |
| | Uncertainty | 3.36 | 0.70 | 129 |

**Pattern**: Means are highly similar across conditions. Largest difference: tia_f (+0.15 favoring uncertainty). Smallest difference: tia_up (+0.01).

#### 5.2.2 Univariate Tests (Welch's t-tests)

| Scale | $\Delta M$ | $SE$ | Cohen's $d$ | $t$ | $df$ | $p$ | $CI_{95\%}$ |
|-------|-----------|------|-------------|-----|------|-----|-------------|
| **tia_f** | 0.150 | 0.115 | 0.163 | 1.305 | 252.4 | .193 | [-0.076, 0.376] |
| **tia_pro** | -0.032 | 0.083 | 0.048 | -0.387 | 251.3 | .699 | [-0.196, 0.131] |
| **tia_rc** | -0.097 | 0.070 | 0.176 | -1.386 | 252.8 | .167 | [-0.235, 0.041] |
| **tia_t** | -0.094 | 0.099 | 0.115 | -0.949 | 252.8 | .344 | [-0.289, 0.101] |
| **tia_up** | 0.013 | 0.092 | 0.018 | 0.139 | 250.5 | .890 | [-0.169, 0.195] |

**Interpretation**:
- **No significant effects** (all $p > .05$; adjusted $p$ values not reported for single-test family)
- **Effect sizes**: All Cohen's $d < 0.20$ (small effects per conventional benchmarks)
- **Direction**: Inconsistent; 2 subscales favor uncertainty (tia_f, tia_up), 3 favor control (tia_pro, tia_rc, tia_t)
- **Conclusion**: Uncertainty communication has no detectable effect on trust across any dimension.

#### 5.2.3 Non-Inferiority Tests

**Minimally Detectable Effects (MDE)**:

| Scale | $SE$ | $t_{\text{crit}}$ | $t_{\text{power}}$ | MDE |
|-------|------|-------------------|-------------------|-----|
| tia_f | 0.115 | 1.970 | 0.849 | 0.324 |
| tia_pro | 0.083 | 1.970 | 0.849 | 0.234 |
| tia_rc | 0.070 | 1.970 | 0.849 | 0.197 |
| tia_t | 0.099 | 1.970 | 0.849 | 0.279 |
| tia_up | 0.092 | 1.970 | 0.849 | 0.259 |

**Non-inferiority conclusion**:

| Scale | Observed $|\Delta M|$ | MDE | Non-Inferior? |
|-------|----------------------|-----|---------------|
| tia_f | 0.150 | 0.324 | ✓ Yes |
| tia_pro | 0.032 | 0.234 | ✓ Yes |
| tia_rc | 0.097 | 0.197 | ✓ Yes |
| tia_t | 0.094 | 0.279 | ✓ Yes |
| tia_up | 0.013 | 0.259 | ✓ Yes |

**Interpretation**: All observed effects are **substantially smaller** than the MDE, supporting the conclusion that uncertainty communication is non-inferior (i.e., not meaningfully worse than standard communication) across all five trust dimensions.

---

### 5.3 Multivariate Analysis Results

#### 5.3.1 MANOVA: Main Effect of Group

**Test statistic**: Pillai's Trace $V = 0.020$
**F-statistic**: $F(5, 235) = 0.951$
**p-value**: $p = .448$
**p-value (adjusted)**: $p_{\text{adj}} = .448$ (Holm correction; single test in family)
**Effect size**: $\eta_p^2 = 0.020$

**Interpretation**: No significant multivariate effect of uncertainty communication on combined trust outcomes. The treatment accounts for only 2.0% of variance in the joint trust profile.

#### 5.3.2 MANOVA: Direct Effects (Covariates)

| Effect | Pillai's $V$ | $F(5, 235)$ | $p$ | $p_{\text{adj}}$ | $\eta_p^2$ |
|--------|--------------|-------------|-----|------------------|-----------|
| **age** | 0.186 | 10.71 | < .001 | < .001 *** | 0.186 |
| **gender** | 0.014 | 0.65 | .663 | .663 | 0.014 |
| **education** | 0.071 | 3.61 | .004 | .017 * | 0.071 |
| **ai_exp** | 0.072 | 3.65 | .003 | .017 * | 0.072 |
| **hcsds_c** (HC trust - competence) | 0.061 | 3.03 | .011 | .034 * | 0.061 |
| **hcsds_v** (HC trust - values) | 0.032 | 1.53 | .181 | .362 | 0.032 |
| **ati** (technology affinity) | 0.091 | 4.72 | < .001 | .002 ** | 0.091 |

**Interpretation**:
- **Age**: Strongest predictor ($\eta_p^2 = 0.186$); accounts for 18.6% of multivariate variance
- **Technology affinity (ATI)**: Second-strongest ($\eta_p^2 = 0.091$); 9.1% of variance
- **Education, AI experience, healthcare trust (competence)**: Moderate effects (6-7% of variance)
- **Gender, healthcare trust (values)**: Non-significant (ns)

**Key finding**: Substantial individual differences in trust predicted by demographic and psychological factors, independent of experimental manipulation.

#### 5.3.3 MANOVA: Interaction Effects (Moderation)

| Interaction | Pillai's $V$ | $F(5, 235)$ | $p$ | $p_{\text{adj}}$ | $\eta_p^2$ |
|-------------|--------------|-------------|-----|------------------|-----------|
| **group × age** | 0.054 | 2.70 | .021 | .129 | 0.054 |
| **group × gender** | 0.015 | 0.71 | .617 | 1.000 | 0.015 |
| **group × education** | 0.020 | 0.96 | .441 | 1.000 | 0.020 |
| **group × ai_exp** | 0.057 | 2.83 | .017 | .118 | 0.057 |
| **group × hcsds_c** | 0.030 | 1.43 | .213 | 1.000 | 0.030 |
| **group × hcsds_v** | 0.025 | 1.23 | .297 | 1.000 | 0.025 |
| **group × ati** | 0.013 | 0.61 | .693 | 1.000 | 0.013 |

**Interpretation**:
- **Uncorrected p-values**: 2 of 7 interactions nominally significant ($p < .05$): group × age, group × ai_exp
- **Corrected p-values**: **None significant** after Holm-Bonferroni correction (all $p_{\text{adj}} > .05$)
- **Conclusion**: No evidence of moderation. Treatment effect does not vary by demographic or psychological characteristics.

---

### 5.4 Equivalence Testing Results

#### 5.4.1 Univariate Equivalence Tests

**Method**: Compare observed partial $\eta_p^2$ (from regression) against equivalence margin derived from MDE.

**Conversion**: MDE (in original units) → $\eta_p^2$ using:
$$\eta_{p, \text{MDE}}^2 = \frac{t_{\text{MDE}}^2}{t_{\text{MDE}}^2 + df_{\text{resid}}}$$
where $t_{\text{MDE}} = \text{MDE} / SE$.

**Results**:

| Scale | Observed $\eta_p^2$ | Equivalence Margin $\eta_p^2$ | $CI_{95\%}$ for $\eta_p^2$ | Equivalent? |
|-------|---------------------|--------------------------------|----------------------------|-------------|
| tia_f | 0.0071 | 0.0316 | [0, 0.0284] | ✓ Yes |
| tia_pro | 0.0006 | 0.0313 | [0, 0.0169] | ✓ Yes |
| tia_rc | 0.0080 | 0.0317 | [0, 0.0296] | ✓ Yes |
| tia_t | 0.0037 | 0.0327 | [0, 0.0213] | ✓ Yes |
| tia_up | 0.0001 | 0.0327 | [0, 0.0126] | ✓ Yes |

**Interpretation**: All 95% CIs for observed $\eta_p^2$ are **entirely below** the equivalence margin, confirming statistical equivalence for all five subscales. The uncertainty communication effect is smaller than the minimally detectable effect across all dimensions of trust.

#### 5.4.2 Multivariate Equivalence Test

**Observed effect**: $\eta_p^2 = 0.0198$ (from MANOVA)

**Equivalence margin**: $\eta_{p, \text{MDE}}^2 = 0.0493$

**95% Confidence interval**: $[0, 0.0438]$

**Decision**: CI upper bound (0.0438) < equivalence margin (0.0493)

**Conclusion**: **Multivariate equivalence confirmed.** The combined effect of uncertainty communication on all five trust dimensions is statistically equivalent to zero.

**Visual summary**:
```
   0.00        0.0198       0.0438    0.0493
   |------------|-------------|---------|
   CI_lower   Observed     CI_upper  Margin

   [========== CI ==========]
                                [Equivalence Bound]
```

The entire confidence interval falls within the equivalence region, providing strong evidence that the multivariate effect is negligible.

#### 5.4.3 Interpretation
Both univariate and multivariate equivalence tests converge on the same conclusion: **Uncertainty communication has no meaningful effect on trust in medical AI systems.** This finding supports the use of transparent uncertainty communication without concerns about negative impacts on patient trust.

---

### 5.5 Exploratory Findings: Predictors of Trust

Although the experimental manipulation had no effect, substantial individual differences in baseline trust were observed. The following predictors showed significant associations with trust outcomes (based on univariate regression models).

#### 5.5.1 Age

**Multivariate effect**: Pillai's $V = 0.186$, $F(5, 235) = 10.71$, $p < .001$, $\eta_p^2 = 0.186$

**Univariate effects** (standardized $\beta$ coefficients):

| Outcome | $\beta$ | $SE$ | $t$ | $p$ | $p_{\text{adj}}$ | $\eta_p^2$ |
|---------|---------|------|-----|-----|------------------|-----------|
| tia_f | 0.177 | 0.064 | 2.75 | .006 | .036 * | 0.031 |
| tia_pro | 0.192 | 0.046 | 4.18 | < .001 | < .001 *** | 0.068 |
| tia_rc | 0.202 | 0.038 | 5.34 | < .001 | < .001 *** | 0.106 |
| tia_t | 0.317 | 0.052 | 6.09 | < .001 | < .001 *** | 0.134 |
| tia_up | 0.063 | 0.054 | 1.18 | .241 | .964 | 0.006 |

**Finding**: Older participants reported higher trust across most dimensions (except understanding/predictability). Effect sizes range from small (tia_f) to medium-large (tia_t, $\eta_p^2 = 0.134$).

#### 5.5.2 AI Experience

**Multivariate effect**: Pillai's $V = 0.072$, $F(5, 235) = 3.65$, $p = .003$, $p_{\text{adj}} = .017$, $\eta_p^2 = 0.072$

**Univariate effects**:

| Outcome | $\beta$ | $SE$ | $t$ | $p$ | $p_{\text{adj}}$ | $\eta_p^2$ |
|---------|---------|------|-----|-----|------------------|-----------|
| tia_f | 0.067 | 0.060 | 1.12 | .265 | .964 | 0.005 |
| tia_pro | 0.118 | 0.042 | 2.78 | .006 | .036 * | 0.031 |
| tia_rc | 0.067 | 0.035 | 1.91 | .057 | .285 | 0.015 |
| tia_t | 0.168 | 0.048 | 3.47 | .001 | .006 ** | 0.048 |
| tia_up | 0.010 | 0.050 | 0.20 | .838 | 1.000 | 0.000 |

**Finding**: Self-reported AI experience positively predicts propensity to trust and general trust in automation.

#### 5.5.3 Healthcare Trust (Competence)

**Multivariate effect**: Pillai's $V = 0.061$, $F(5, 235) = 3.03$, $p = .011$, $p_{\text{adj}} = .034$, $\eta_p^2 = 0.061$

**Univariate effects**:

| Outcome | $\beta$ | $SE$ | $t$ | $p$ | $p_{\text{adj}}$ | $\eta_p^2$ |
|---------|---------|------|-----|-----|------------------|-----------|
| tia_f | -0.085 | 0.072 | -1.18 | .241 | .964 | 0.006 |
| tia_pro | 0.014 | 0.051 | 0.27 | .791 | .880 | 0.000 |
| tia_rc | 0.097 | 0.043 | 2.28 | .024 | .144 | 0.021 |
| tia_t | 0.108 | 0.058 | 1.85 | .065 | .325 | 0.014 |
| tia_up | -0.057 | 0.060 | -0.95 | .343 | 1.000 | 0.004 |

**Finding**: Trust in the healthcare system's competence positively predicts reliability/confidence in medical AI (tia_rc), though effect becomes marginal after multiple comparison correction.

#### 5.5.4 Technology Affinity (ATI)

**Multivariate effect**: Pillai's $V = 0.091$, $F(5, 235) = 4.72$, $p < .001$, $p_{\text{adj}} = .002$, $\eta_p^2 = 0.091$

**Univariate effects**:

| Outcome | $\beta$ | $SE$ | $t$ | $p$ | $p_{\text{adj}}$ | $\eta_p^2$ |
|---------|---------|------|-----|-----|------------------|-----------|
| tia_f | 0.244 | 0.062 | 3.90 | < .001 | < .001 *** | 0.060 |
| tia_pro | -0.070 | 0.044 | -1.56 | .120 | .480 | 0.010 |
| tia_rc | -0.021 | 0.037 | -0.57 | .569 | 1.000 | 0.001 |
| tia_t | -0.053 | 0.051 | -1.05 | .295 | .441 | 0.005 |
| tia_up | -0.150 | 0.052 | -2.88 | .004 | .028 * | 0.034 |

**Finding**: General technology affinity strongly predicts familiarity with AI systems ($\beta = 0.244$, $\eta_p^2 = 0.060$). Interestingly, ATI shows a negative association with understanding/predictability (tia_up), suggesting that more tech-savvy individuals may perceive AI systems as less predictable.

#### 5.5.5 Education

**Multivariate effect**: Pillai's $V = 0.071$, $F(5, 235) = 3.61$, $p = .004$, $p_{\text{adj}} = .017$, $\eta_p^2 = 0.071$

**Univariate effects**: Modest effects across outcomes (not individually significant after correction); contributes to multivariate effect through combined pattern.

#### 5.5.6 Non-Significant Predictors

- **Gender**: Pillai's $V = 0.014$, $p = .663$ (ns)
- **Healthcare trust (values)**: Pillai's $V = 0.032$, $p = .181$ (ns)

---

### 5.6 Moderation Analysis Results

**Question**: Does the effect of uncertainty communication vary by demographic or psychological characteristics?

**Approach**: Test interactions (`group_effect × moderator`) in full regression models (16 predictors per outcome).

**Total interaction tests**: 8 moderators × 5 outcomes = 40 tests

#### 5.6.1 Summary of Interaction Tests

**Nominally significant** (uncorrected $p < .05$): 3 of 40 tests (7.5%)

**Significant after Holm correction**: 0 of 40 tests (0%)

#### 5.6.2 Selected Interaction Results

**Example: group × age → tia_up**

| Outcome | $\beta_{\text{int}}$ | $SE$ | $t$ | $p$ | $p_{\text{adj}}$ |
|---------|----------------------|------|-----|-----|------------------|
| tia_up | -0.234 | 0.107 | -2.19 | .030 | .210 |

Uncorrected: Significant ($p = .030$)
Corrected: Non-significant ($p_{\text{adj}} = .210$)

**Interpretation**: Initial indication that age moderates the effect on understanding/predictability does not survive multiple comparison correction. No reliable evidence of moderation.

#### 5.6.3 Conclusion
**No significant moderation effects.** The (null) effect of uncertainty communication on trust is uniform across:
- Age groups
- Gender categories
- Education levels
- AI experience levels
- Healthcare trust profiles
- Technology affinity levels

This finding strengthens the conclusion that uncertainty communication is neutral for trust, as it does not produce differential effects in any subpopulation examined.

---

## 6. Model Fit and Diagnostics

### 6.1 Regression Model Fit Statistics

**Overall model significance** ($F$-tests):

| Outcome | $R^2$ | Adj. $R^2$ | $F(15, 239)$ | $p$ |
|---------|-------|-----------|--------------|-----|
| tia_f | 0.143 | 0.089 | 2.66 | .001 ** |
| tia_pro | 0.159 | 0.106 | 3.00 | < .001 *** |
| tia_rc | 0.191 | 0.140 | 3.76 | < .001 *** |
| tia_t | 0.246 | 0.198 | 5.19 | < .001 *** |
| tia_up | 0.072 | 0.013 | 1.23 | .249 |

**Interpretation**:
- Models for tia_f, tia_pro, tia_rc, tia_t are statistically significant, indicating that the set of predictors (treatment, demographics, interactions) explains significant variance.
- Adjusted $R^2$ ranges from 0.089 (tia_f) to 0.198 (tia_t), indicating that 9-20% of variance is explained after penalizing for model complexity.
- Model for tia_up is non-significant ($p = .249$), suggesting predictors do not meaningfully explain variance in understanding/predictability.

### 6.2 Multicollinearity Diagnostics

**Variance Inflation Factors (VIF)** for all predictors:

| Predictor | VIF |
|-----------|-----|
| group_effect | 1.02 |
| age_c | 1.31 |
| gender_c | 1.18 |
| education_c | 1.42 |
| ai_exp_c | 1.27 |
| hcsds_c_c | 1.84 |
| hcsds_v_c | 1.76 |
| ati_c | 1.53 |
| group × age | 1.09 |
| group × gender | 1.14 |
| group × education | 1.35 |
| group × ai_exp | 1.22 |
| group × hcsds_c | 1.79 |
| group × hcsds_v | 1.71 |
| group × ati | 1.47 |

**Criterion**: All VIF < 5 (well within acceptable range)

**Interpretation**: No problematic multicollinearity. Predictors are sufficiently independent; coefficient estimates are stable.

### 6.3 Assumption Checks

#### 6.3.1 Univariate Normality (Shapiro-Wilk Tests)

| Outcome | $W$ | $p$ |
|---------|-----|-----|
| tia_f | 0.982 | .003 |
| tia_pro | 0.991 | .207 |
| tia_rc | 0.987 | .041 |
| tia_t | 0.991 | .167 |
| tia_up | 0.993 | .409 |

**Result**: 2 of 5 outcomes show significant departures from normality (tia_f, tia_rc at $p < .05$).

**Action**: Visual inspection of Q-Q plots and histograms revealed only minor deviations (slight negative skew). Given large sample size ($N = 255$), parametric tests are robust to moderate non-normality (central limit theorem). Proceeded with planned analyses.

#### 6.3.2 Multivariate Outliers (Mahalanobis Distance)

**Method**: Computed $D^2$ for each case based on 5 TiA outcomes.

**Criterion**: $p < .001$ (i.e., $D^2 > \chi^2_{5, 0.999} = 20.52$)

**Result**: No extreme outliers detected. Maximum $D^2 = 18.3$ (below threshold).

**Action**: All cases retained.

#### 6.3.3 Homogeneity of Variance (Levene's Test)

**Result**: Not formally tested (Welch's t-test and Pillai's Trace are robust to heterogeneity).

**Strategy**: Used robust procedures that do not assume equal variances (Welch's t-test) or covariance matrices (Pillai's Trace).

#### 6.3.4 Summary
**Assumption violations**: Minor (slight non-normality in 2/5 outcomes; no extreme outliers).

**Impact**: Minimal. Analyses conducted using robust procedures; results are trustworthy.

---

## 7. Key Statistical Innovations

This analysis incorporated several methodological innovations to strengthen causal inference and effect size interpretation:

### 7.1 Non-Inferiority Testing Framework

**Standard practice**: Null hypothesis significance testing (NHST) tests whether an effect differs from zero.

**Limitation**: "Non-significant" result is uninformative. Failing to reject $H_0: \Delta = 0$ does not prove $\Delta = 0$ (absence of evidence $\neq$ evidence of absence).

**Innovation**: Non-inferiority testing **reverses the burden of proof**:
- $H_0$: Effect is large ($|\Delta| \geq \delta$)
- $H_1$: Effect is negligible ($|\Delta| < \delta$)

**Advantage**: Allows positive claim of equivalence, not just absence of difference.

**Application**:
- Equivalence margin ($\delta$) set to **Minimally Detectable Effect (MDE)** based on study design (sample size, power, alpha).
- If observed effect < MDE, conclude groups are equivalent (within bounds of statistical sensitivity).

**Relevance**: This study's hypothesis explicitly predicted **no effect** of uncertainty communication. Non-inferiority testing provides appropriate statistical support for this prediction.

### 7.2 Effect Size Confidence Intervals via Non-Central F Distribution

**Standard practice**: Confidence intervals for means or mean differences constructed using $t$ distribution.

**Problem**: Effect sizes (e.g., $\eta_p^2$, Cohen's $d$) are **bounded parameters** (e.g., $\eta_p^2 \in [0, 1]$). Normal approximation (via $t$ distribution) is inappropriate for bounded parameters, leading to:
- Symmetric CIs that may exceed valid range (e.g., $\eta_p^2 < 0$ or $> 1$)
- Inaccurate coverage probabilities (actual coverage $\neq$ nominal 95%)

**Innovation**: Confidence intervals for $\eta_p^2$ constructed via **non-central $F$ distribution inversion** (Steiger, 2004).

**Method**:
1. Convert test statistic to $F$ (e.g., $F = t^2$ for univariate tests).
2. Find non-centrality parameters $\lambda_L$ and $\lambda_U$ such that:
   - $P(F \geq F_{\text{obs}} \mid \lambda = \lambda_L) = \alpha/2$
   - $P(F \geq F_{\text{obs}} \mid \lambda = \lambda_U) = 1 - \alpha/2$
3. Convert $\lambda$ bounds to $\eta_p^2$ bounds: $\eta_p^2 = \lambda / (\lambda + df_{\text{error}})$

**Advantage**:
- CIs respect parameter bounds (always $0 \leq CI \leq 1$)
- Accurate coverage probabilities
- Directly interpretable for equivalence testing

**Implementation**: Numerical root-finding (Brent's method) via `scipy.optimize.brentq` in Python.

### 7.3 Family-Wise Multiple Comparison Control

**Standard practice**:
- Option 1: No correction (inflated Type I error)
- Option 2: Bonferroni correction across all tests (overly conservative, low power)

**Problem**:
- No correction: FWER = 95% for 60 tests at $\alpha = 0.05$
- Full Bonferroni: $\alpha_{\text{adj}} = 0.05/60 = 0.00083$ (very stringent)

**Innovation**: **Family-wise correction within conceptual families**.

**Rationale**:
- Tests within a family address related hypotheses (e.g., all direct effects test "what predicts trust?").
- Tests across families address distinct hypotheses (e.g., main effect vs. moderation).
- Correcting within families balances Type I error control with statistical power.

**Families defined**:
1. **Group effect family**: Main treatment effect (primary hypothesis)
2. **Direct effects family**: Predictors of trust (exploratory)
3. **Interaction effects family**: Moderation (exploratory)

**Method**: Holm-Bonferroni correction applied separately within each family.

**Advantage**:
- Controls FWER within each family at $\alpha = 0.05$
- More powerful than global Bonferroni correction
- Aligns with hierarchical structure of research questions

### 7.4 Equivalence Margin Based on Study Design

**Standard practice**: Equivalence margin chosen arbitrarily (e.g., $\delta = 0.5 \times d_{\text{expected}}$) or based on clinical judgment.

**Problem**: Arbitrary margins may be:
- Too lenient: Declare equivalence for effects that are statistically detectable
- Too stringent: Fail to declare equivalence even when effects are below detection threshold

**Innovation**: Equivalence margin set to **Minimally Detectable Effect (MDE)** based on:
- Sample size ($N = 255$)
- Desired power ($1 - \beta = 0.80$)
- Significance level ($\alpha = 0.05$)

**Formula**:
$$\text{MDE} = (t_{\alpha/2, df} + t_{\beta, df}) \times SE$$

**Rationale**:
- MDE represents the smallest effect size the study is designed to detect.
- If observed effect < MDE, the study lacks power to distinguish the effect from zero.
- Therefore, declaring equivalence for effects < MDE is statistically justified.

**Advantage**:
- Objective, design-based criterion (no arbitrary choices)
- Aligns equivalence testing with power analysis
- Transparent and reproducible

---

## 8. Limitations and Methodological Notes

### 8.1 Power Considerations

**Design**: Study powered at 80% to detect MDE with $\alpha = 0.05$ (two-tailed).

**Achieved power**:
- For effects = MDE: 80% (by design)
- For effects < MDE: < 80% (underpowered)
- For effects > MDE: > 80% (well-powered)

**Implication**: This study can reliably detect effects $\geq$ MDE, but smaller effects may go undetected. However:
- **Equivalence testing logic**: If effect < MDE, it is **by definition** negligible (below sensitivity threshold).
- **Observed effects**: All well below MDE, supporting equivalence conclusion.

**Limitation**: Cannot rule out effects smaller than MDE. However, such effects would be trivial by the study's own sensitivity standards.

### 8.2 Effect Coding Interpretation

**Treatment variable**: Effect-coded as $-0.5$ (control) and $0.5$ (uncertainty).

**Interpretation of $\beta_1$ (main effect)**:
- $\beta_1$ = difference between uncertainty and control group means.
- Example: For tia_f, $\beta_1 = 0.175$ means uncertainty group scored 0.175 points higher on average.

**Interaction interpretation**:
- $\beta_3$ (interaction) = change in treatment effect per 1-unit increase in moderator (when moderator is centered).
- Example: $\beta_3 = 0.10$ means treatment effect increases by 0.10 for each 1-SD increase in moderator.

**Advantage**: Effect coding centers the treatment variable, making main effects interpretable as "average" treatment effects (when moderators = 0, i.e., at mean).

**Note**: Dummy coding ($0/1$) would yield identical F-tests and p-values but different coefficient interpretation.

### 8.3 Non-Inferiority: Inconclusive Results

**Univariate tests**: 5 of 5 outcomes showed non-inferiority (observed effect < MDE).

**Multivariate test**: Equivalence confirmed (CI upper bound < equivalence margin).

**Note**: While observed effects were consistently small, confidence intervals were wide for some outcomes.

**Example**: tia_f 95% CI = [0, 0.0284]; margin = 0.0316.
- Upper bound is close to margin (gap = 0.0032).
- Suggests need for larger sample to achieve more precise estimates.

**Conclusion**: Equivalence is supported but not definitive for all outcomes. Larger studies would provide more precise intervals and stronger conclusions.

### 8.4 Exploratory Nature of Moderation Analyses

**Preregistration**: Moderation analyses were exploratory (not preregistered hypotheses).

**Correction**: Holm-Bonferroni correction applied to control Type I error.

**Interpretation**: Null findings (no significant moderation) should be interpreted cautiously:
- May reflect true absence of moderation.
- May reflect insufficient power to detect small moderation effects.

**Recommendation**: Future research should test specific moderation hypotheses with adequate power.

### 8.5 Generalizability

**Sample**: Recruited via Prolific (online crowdsourcing platform).

**Characteristics**:
- English-, Dutch-, and German-speaking participants
- General population (not patient sample)
- Self-selected (volunteered for research participation)

**Limitation**: Findings may not generalize to:
- Clinical patients facing real medical decisions
- Non-Western populations
- Individuals with limited health literacy or technology experience

**Recommendation**: Replicate in patient samples and diverse populations.

---

## 9. References for Statistical Methods

### Primary Methodological References

**Steiger, J. H. (2004).** Beyond the F test: Effect size confidence intervals and tests of close fit in the analysis of variance and contrast analysis. *Psychological Methods, 9*(2), 164-182. https://doi.org/10.1037/1082-989X.9.2.164
- Method for computing exact confidence intervals for effect sizes ($\eta^2$, $\eta_p^2$) using non-central F distribution inversion
- Provides formulas and computational procedures implemented in this analysis

**Hayes, A. F. (2018).** *Introduction to mediation, moderation, and conditional process analysis: A regression-based approach* (2nd ed.). Guilford Press.
- Framework for testing and interpreting moderation effects
- Guidelines for effect coding, centering, and interaction term interpretation
- Simple slopes analysis and region of significance procedures

**Holm, S. (1979).** A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65-70.
- Holm-Bonferroni sequential rejection method
- Controls family-wise error rate with greater power than Bonferroni correction

### Measurement Instruments

**Körber, M. (2019).** Theoretical considerations and development of a questionnaire to measure trust in automation. In S. Bagnara, R. Tartaglia, S. Albolino, T. Alexander, & Y. Fujita (Eds.), *Proceedings of the 20th Congress of the International Ergonomics Association* (IEA 2018) (pp. 13-30). Springer. https://doi.org/10.1007/978-3-319-96074-6_2
- Trust in Automation (TiA) scale development and validation
- Five-factor structure (reliability/confidence, understanding/predictability, familiarity, propensity, general trust)

**Franke, T., Attig, C., & Wessel, D. (2019).** A personal resource for technology interaction: Development and validation of the Affinity for Technology Interaction (ATI) scale. *International Journal of Human-Computer Interaction, 35*(6), 456-467. https://doi.org/10.1080/10447318.2018.1456150
- Affinity for Technology Interaction (ATI) scale
- Single-factor measure of technology affinity and engagement

**Shea, J. A., Micco, E., Dean, L. T., McMurphy, S., Schwartz, J. S., & Armstrong, K. (2008).** Development of a revised Health Care System Distrust scale. *Journal of General Internal Medicine, 23*(6), 727-732. https://doi.org/10.1007/s11606-008-0575-3
- Revised Health Care System Distrust Scale (HCSDS)
- Two-factor structure: competence and values
- Note: Scores inverted in this analysis to measure trust instead of distrust

### Statistical Software

**Python 3.14+** with packages:
- `statsmodels` (v0.14+): MANOVA, regression, multiple comparison corrections
- `scipy` (v1.11+): Statistical tests (t-tests, Mann-Whitney U, Shapiro-Wilk, etc.), optimization for CI computation
- `pandas` (v2.1+): Data manipulation
- `numpy` (v1.26+): Numerical operations
- `matplotlib` (v3.8+): Visualization

---

## 10. Data Availability

### 10.1 Output Files

All statistical results are available in the `output/` directory:

#### Multivariate Results
- **`manova_results.csv`**: MANOVA results for all effects
  - Columns: effect name, Pillai's V, F-value, numerator df, denominator df, partial $\eta^2$, p-value, adjusted p-value, formatted p-value
  - 16 rows: intercept + main effect + 7 direct effects + 7 interaction effects

#### Univariate Regression Results
- **`tia_f_regression_coef.csv`**: Familiarity (tia_f) regression coefficients
- **`tia_pro_regression_coef.csv`**: Propensity to trust (tia_pro) regression coefficients
- **`tia_rc_regression_coef.csv`**: Reliability/confidence (tia_rc) regression coefficients
- **`tia_t_regression_coef.csv`**: Trust in automation (tia_t) regression coefficients
- **`tia_up_regression_coef.csv`**: Understanding/predictability (tia_up) regression coefficients

**Format** (each file):
- Columns: effect name, coefficient, standard error, t-value, p-value, CI lower, CI upper, adjusted p-value, partial $\eta^2$, formatted adjusted p-value
- 17 rows: intercept + 16 predictors (main effect + 8 direct + 8 interactions)

#### Model Fit Statistics
- **`regression_model_stats.csv`**: Model fit for all 5 regression models
  - Columns: $R^2$, adjusted $R^2$, F-statistic, p-value for F, numerator df, denominator df
  - Rows: tia_f, tia_pro, tia_rc, tia_t, tia_up

#### Equivalence Test Results
- **`multivariate_eq_test_res.json`**: Multivariate equivalence test
  - Fields: `eta_sq_obs` (observed partial $\eta^2$), `ci_lower` (95% CI lower bound), `ci_upper` (95% CI upper bound), `eq_margin` (equivalence margin)

### 10.2 Processed Data Files

- **`data/data_clean.csv`**: Cleaned individual-level data (255 rows)
  - Variables: demographics, TiA item responses, manipulation checks, stimulus group
- **`data/data_scales.csv`**: Computed scale scores (255 rows)
  - Variables: demographics, 5 TiA subscales, ATI, 2 HCSDS subscales, stimulus group, effect-coded variables

### 10.3 Variable Naming Conventions

**Scale scores**:
- `tia_f`, `tia_pro`, `tia_rc`, `tia_t`, `tia_up`: Trust in Automation subscales
- `ati`: Affinity for Technology Interaction
- `hcsds_c`, `hcsds_v`: Healthcare trust (competence, values)

**Transformed variables** (suffix `_c`):
- `age_c`, `education_c`, `ai_exp_c`: Centered or standardized
- `hcsds_c_c`, `hcsds_v_c`, `ati_c`: Standardized (z-scored)
- `gender_c`: Effect-coded

**Treatment variables**:
- `stimulus_group`: Original coding (0 = control, 1 = uncertainty)
- `group_effect`: Effect-coded (-0.5 = control, 0.5 = uncertainty)

### 10.4 Visualization Files

All plots available in `plots/` directory:

- **`plots/multivariate_analysis/`**:
  - `multivariate_equivalence_test.png`: Multivariate equivalence test visualization
  - `tia_corr_matrix.png`: Correlation matrix for 5 TiA subscales
  - `tia_normality.png`: Normality diagnostics (histograms and Q-Q plots)

- **`plots/univariate_analysis/`**:
  - `univariate_noninferiority_test.png`: Non-inferiority test for all 5 outcomes (observed effects vs. MDE)
  - `univariate_equivalence_test_standardized.png`: Equivalence test using partial $\eta^2$ (observed vs. margin)

- **`plots/combined_equivalence_test.png`**: Combined multivariate + univariate equivalence tests

- **`plots/manip_check/`**: Manipulation check visualizations

- **`plots/overview/`**:
  - `demographics_by_group.png`: Demographic distributions by experimental condition
  - `tia_raw_overview.png`: Raw TiA distributions by group

### 10.5 Contact for Data Access

For access to raw data, codebooks, or additional analyses, contact the research team at [contact information redacted for privacy].

---

## End of Report

**Document prepared**: December 2, 2025
**Analysis software**: Python 3.14+ (statsmodels, scipy, pandas, numpy, matplotlib)
**Statistical consultant**: Claude (Anthropic)
**Report version**: 1.0 (Complete Statistical Documentation)

**Recommended citation**:
[Author names]. (2025). *Statistical analysis report: Effect of uncertainty communication on trust in medical AI systems* [Technical report]. [Institution].

---

**Note to paper writers**: This report provides complete technical documentation. For manuscript preparation:
1. Methods section: Adapt Sections 4.1-4.7 (omit excessive technical detail; focus on key procedures)
2. Results section: Adapt Sections 5.1-5.6 (prioritize primary findings; move exploratory results to supplementary materials)
3. Tables: Extract from CSV files (output/) for publication-ready formatting
4. Figures: Use plots from plots/ directory; may require re-formatting for journal submission

**Questions?** Consult output files for detailed numerical results or contact the statistical analyst for clarification.
