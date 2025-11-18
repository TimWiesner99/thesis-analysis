import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings

def test_moderation(df, outcome, moderator, moderator_centered, categorical=False):
    """
    Test moderation effect using regression with interaction term.

    Parameters:
    -----------
    df : DataFrame
        Data containing all variables
    outcome : str
        Name of outcome variable (e.g., 'TiA_t')
    moderator : str
        Name of moderator variable (for display)
    moderator_centered : str
        Name of centered/coded moderator variable to use in model
    categorical : bool
        Whether moderator is categorical (affects interpretation)

    Returns:
    --------
    dict : Dictionary containing key statistics
    model : Fitted statsmodels regression model
    """
    # Create interaction term
    df_temp = df.copy()
    df_temp['interaction'] = df_temp['group_effect'] * df_temp[moderator_centered]

    # Fit regression model: Y ~ X + M + X*M
    formula = f'{outcome} ~ group_effect + {moderator_centered} + interaction'
    model = smf.ols(formula, data=df_temp).fit()

    # Extract key statistics
    results = {
        'outcome': outcome,
        'moderator': moderator,
        'n': int(model.nobs),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        # Main effect of group (conditional effect when M = 0, i.e., at mean)
        'b_group': model.params['group_effect'],
        'se_group': model.bse['group_effect'],
        'p_group': model.pvalues['group_effect'],
        'ci_group_lower': model.conf_int().loc['group_effect', 0],
        'ci_group_upper': model.conf_int().loc['group_effect', 1],
        # Main effect of moderator
        'b_moderator': model.params[moderator_centered],
        'se_moderator': model.bse[moderator_centered],
        'p_moderator': model.pvalues[moderator_centered],
        # Interaction effect (the moderation effect)
        'b_interaction': model.params['interaction'],
        'se_interaction': model.bse['interaction'],
        'p_interaction': model.pvalues['interaction'],
        'ci_interaction_lower': model.conf_int().loc['interaction', 0],
        'ci_interaction_upper': model.conf_int().loc['interaction', 1],
    }

    return results, model

def print_moderation_results(results):
    """
    Print moderation results in a readable format.
    """
    print(f"\n{'='*80}")
    print(f"Outcome: {results['outcome']} | Moderator: {results['moderator']}")
    print(f"{'='*80}")
    print(f"N = {results['n']}, R² = {results['r_squared']:.4f}, Adj. R² = {results['adj_r_squared']:.4f}")
    print(f"\n{'Coefficient':<25} {'β':<10} {'SE':<10} {'95% CI':<25} {'p':<10}")
    print(f"{'-'*80}")

    # Group main effect
    ci_group = f"[{results['ci_group_lower']:>6.3f}, {results['ci_group_upper']:>6.3f}]"
    p_group = f"{results['p_group']:.4f}" if results['p_group'] >= 0.001 else "<.001"
    print(f"{'Group (conditional)':<25} {results['b_group']:<10.4f} {results['se_group']:<10.4f} {ci_group:<25} {p_group:<10}")

    # Moderator main effect
    p_mod = f"{results['p_moderator']:.4f}" if results['p_moderator'] >= 0.001 else "<.001"
    print(f"{results['moderator']:<25} {results['b_moderator']:<10.4f} {results['se_moderator']:<10.4f} {'':25} {p_mod:<10}")

    # Interaction effect
    ci_int = f"[{results['ci_interaction_lower']:>6.3f}, {results['ci_interaction_upper']:>6.3f}]"
    p_int = f"{results['p_interaction']:.4f}" if results['p_interaction'] >= 0.001 else "<.001"
    sig_marker = " ***" if results['p_interaction'] < 0.001 else " **" if results['p_interaction'] < 0.01 else " *" if results['p_interaction'] < 0.05 else ""
    print(f"{'Interaction':<25} {results['b_interaction']:<10.4f} {results['se_interaction']:<10.4f} {ci_int:<25} {p_int:<10}{sig_marker}")

    print(f"\n{'Interpretation:':<25}")
    if results['p_interaction'] < 0.05:
        direction = "increases" if results['b_interaction'] > 0 else "decreases"
        print(f"  Significant moderation: The effect of uncertainty communication {direction}")
        print(f"  as {results['moderator']} increases.")
    else:
        print(f"  No significant moderation effect detected.")
    print(f"{'='*80}\n")
def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Cohen's d measures the standardized difference between two group means.
    It represents the difference in means divided by the pooled standard deviation.

    Interpretation:
        |d| < 0.2: Small effect (trivial difference)
        0.2 ≤ |d| < 0.5: Small to medium effect
        0.5 ≤ |d| < 0.8: Medium to large effect
        |d| ≥ 0.8: Large effect (substantial difference)

    Args:
        group1: First group data (pandas Series)
        group2: Second group data (pandas Series)

    Returns:
        Cohen's d effect size (float)

    Example:
        >>> control = data[data['stimulus_group'] == 0]['ati']
        >>> treatment = data[data['stimulus_group'] == 1]['ati']
        >>> effect_size = cohens_d(control, treatment)
        >>> print(f"Effect size: {effect_size:.3f}")
        Effect size: 0.234
    """
    # Calculate means
    mean_diff = abs(group1.mean() - group2.mean())

    # Calculate pooled standard deviation
    std1 = group1.std()
    std2 = group2.std()
    pooled_sd = np.sqrt((std1**2 + std2**2) / 2)

    # Calculate Cohen's d
    if pooled_sd > 0:
        d = mean_diff / pooled_sd
    else:
        d = 0.0

    return d


def independent_t_test(group1: pd.Series, group2: pd.Series) -> tuple[float, float, float]:
    """
    Perform an independent samples t-test between two groups.

    This performs Welch's t-test (does not assume equal variances) and returns
    the t-statistic, two-tailed p-value, and degrees of freedom. The p-value
    indicates the probability of observing the data (or more extreme) if the
    null hypothesis (no difference between groups) is true.

    Interpretation:
        α < 0.001: Very strong evidence against null hypothesis (highly significant)
        α < 0.01: Strong evidence against null hypothesis (very significant)
        α < 0.05: Moderate evidence against null hypothesis (significant)
        α ≥ 0.05: Insufficient evidence to reject null hypothesis (not significant)

    Args:
        group1: First group data (pandas Series)
        group2: Second group data (pandas Series)

    Returns:
        Tuple of (t_statistic, p_value, df) where:
            - t_statistic: t-value from the test (float)
            - p_value: Two-tailed p-value (float), ranges from 0 to 1
            - df: Degrees of freedom (float, Welch-Satterthwaite approximation)

    Example:
        >>> control = data[data['stimulus_group'] == 0]['ati']
        >>> treatment = data[data['stimulus_group'] == 1]['ati']
        >>> t_stat, p_value, df = independent_t_test(control, treatment)
        >>> print(f"t({df:.1f}) = {t_stat:.2f}, p = {p_value:.3f}")
        t(83.5) = -1.42, p = 0.159
    """
    # Clean data (remove NaN values)
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    # Perform Welch's t-test (does not assume equal variances)
    # This is generally more robust than Student's t-test
    t_statistic, p_value = stats.ttest_ind(group1_clean, group2_clean, equal_var=False)

    # Calculate Welch-Satterthwaite degrees of freedom for Welch's t-test
    # Formula: df = (s1²/n1 + s2²/n2)² / [(s1²/n1)²/(n1-1) + (s2²/n2)²/(n2-1)]
    n1 = len(group1_clean)
    n2 = len(group2_clean)
    var1 = group1_clean.var()
    var2 = group2_clean.var()

    # Calculate degrees of freedom using Welch-Satterthwaite equation
    numerator = (var1/n1 + var2/n2) ** 2
    denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)

    if denominator > 0:
        df = numerator / denominator
    else:
        # Fallback to simpler approximation if calculation fails
        df = min(n1, n2) - 1

    return t_statistic, p_value, df

def mann_whitney_u_test(group1: pd.Series, group2: pd.Series) -> tuple[float, float, int, int]:
    """
    Perform a Mann-Whitney U test (Wilcoxon rank-sum test) between two groups.

    The Mann-Whitney U test is a non-parametric alternative to the independent
    samples t-test. It tests whether two independent samples come from the same
    distribution by comparing the ranks of the data rather than the raw values.
    This makes it robust to outliers and suitable for ordinal data or non-normal
    distributions.

    Best used when:
    - Data is ordinal (e.g., Likert scales)
    - Distributions are skewed or non-normal
    - Sample sizes are small
    - Data contains outliers
    - Equal variance assumption is violated

    Interpretation of p-value:
        α < 0.001: Very strong evidence of difference (highly significant)
        α < 0.01: Strong evidence of difference (very significant)
        α < 0.05: Moderate evidence of difference (significant)
        α ≥ 0.05: Insufficient evidence to reject equality (not significant)

    Args:
        group1: First group data (pandas Series)
        group2: Second group data (pandas Series)

    Returns:
        Tuple of (U_statistic, p_value, n1, n2) where:
            - U_statistic: Mann-Whitney U statistic (float)
            - p_value: Two-tailed p-value (float), ranges from 0 to 1
            - n1: Sample size of group 1 (int)
            - n2: Sample size of group 2 (int)

    Example:
        >>> control = data[data['stimulus_group'] == 0]['ati']
        >>> treatment = data[data['stimulus_group'] == 1]['ati']
        >>> u_stat, p_value, n1, n2 = mann_whitney_u_test(control, treatment)
        >>> print(f"U = {u_stat:.2f}, p = {p_value:.3f}")
        U = 523.50, p = 0.147

    References:
        Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two
        random variables is stochastically larger than the other. Annals of
        Mathematical Statistics, 18(1), 50-60.
    """
    # Clean data (remove NaN values)
    group1_clean = group1.dropna()
    group2_clean = group2.dropna()

    # Get sample sizes
    n1 = len(group1_clean)
    n2 = len(group2_clean)

    # Perform Mann-Whitney U test (two-sided alternative)
    # scipy.stats.mannwhitneyu returns (U_statistic, p_value)
    # alternative='two-sided' tests if distributions differ in any way
    u_statistic, p_value = stats.mannwhitneyu(group1_clean, group2_clean, alternative='two-sided')

    return u_statistic, p_value, n1, n2


def rank_biserial_correlation(u_statistic: float, n1: int, n2: int) -> float:
    """
    Calculate rank-biserial correlation effect size for Mann-Whitney U test.

    The rank-biserial correlation (r) is a non-parametric effect size measure
    for the Mann-Whitney U test, analogous to Cohen's d for parametric tests.
    It represents the difference between the proportion of pairs where group 1
    ranks higher and the proportion where group 2 ranks higher.

    Formula: r = 1 - (2*U) / (n1 * n2)

    The value ranges from -1 to +1:
        r = +1: All observations in group 1 rank higher than all in group 2
        r = 0: The two groups are randomly intermixed
        r = -1: All observations in group 2 rank higher than all in group 1

    Interpretation of |r|:
        |r| < 0.1: Negligible effect
        0.1 ≤ |r| < 0.3: Small effect
        0.3 ≤ |r| < 0.5: Medium effect
        |r| ≥ 0.5: Large effect

    Args:
        u_statistic: Mann-Whitney U statistic (float)
        n1: Sample size of group 1 (int)
        n2: Sample size of group 2 (int)

    Returns:
        Rank-biserial correlation effect size (float), ranges from -1 to 1

    Example:
        >>> control = data[data['stimulus_group'] == 0]['ati']
        >>> treatment = data[data['stimulus_group'] == 1]['ati']
        >>> u_stat, p_value, n1, n2 = mann_whitney_u_test(control, treatment)
        >>> r = rank_biserial_correlation(u_stat, n1, n2)
        >>> print(f"r = {r:.3f}")
        r = 0.123

    References:
        Kerby, D. S. (2014). The simple difference formula: An approach to teaching
        nonparametric correlation. Comprehensive Psychology, 3, 11.IT.3.1.
    """
    # Calculate rank-biserial correlation
    # Formula: r = 1 - (2*U)/(n1*n2)
    if n1 > 0 and n2 > 0:
        r = 1 - (2 * u_statistic) / (n1 * n2)
    else:
        r = 0.0

    return r


def one_way_anova(*groups: pd.Series) -> tuple[float, float, int, int]:
    """
    Perform a one-way ANOVA test for comparing means across multiple groups.

    ANOVA (Analysis of Variance) tests whether there are significant differences
    between the means of three or more independent groups. It returns the F-statistic,
    p-value, and degrees of freedom. For two groups, consider using independent_t_test() instead.

    The F-statistic represents the ratio of between-group variance to within-group
    variance. A larger F indicates greater differences between group means relative
    to variability within groups.

    Interpretation of p-value:
        α < 0.001: Very strong evidence of differences (highly significant)
        α < 0.01: Strong evidence of differences (very significant)
        α < 0.05: Moderate evidence of differences (significant)
        α ≥ 0.05: Insufficient evidence to reject equality (not significant)

    Interpretation of F-statistic:
        F ≈ 1: Group means are similar (no effect)
        F > 1: Between-group variance exceeds within-group variance
        Larger F: Stronger evidence of group differences

    Args:
        *groups: Variable number of pandas Series, one for each group
                 At least 2 groups required, typically 3 or more

    Returns:
        Tuple of (F_statistic, p_value, df_between, df_within) where:
            - F_statistic: F-value from ANOVA (float, typically > 0)
            - p_value: Probability value (float, ranges from 0 to 1)
            - df_between: Between-groups degrees of freedom (int) = k - 1
            - df_within: Within-groups degrees of freedom (int) = N - k

    Example:
        >>> english = data[data['language'] == 'English']['ati']
        >>> dutch = data[data['language'] == 'Dutch']['ati']
        >>> german = data[data['language'] == 'German']['ati']
        >>> f_stat, p_value, df1, df2 = one_way_anova(english, dutch, german)
        >>> print(f"F({df1}, {df2}) = {f_stat:.2f}, p = {p_value:.3f}")
        F(2, 127) = 2.46, p = 0.089
    """
    # Drop NaN values from each group
    clean_groups = [group.dropna() for group in groups]

    # Perform one-way ANOVA
    # scipy.stats.f_oneway returns (F_statistic, p_value)
    f_statistic, p_value = stats.f_oneway(*clean_groups)

    # Calculate degrees of freedom
    k = len(clean_groups)  # Number of groups
    N = sum(len(group) for group in clean_groups)  # Total sample size

    df_between = k - 1  # Between-groups df
    df_within = N - k   # Within-groups df

    return f_statistic, p_value, df_between, df_within


def eta_squared(*groups: pd.Series) -> float:
    """
    Calculate eta-squared (η²) effect size for ANOVA.

    Eta-squared represents the proportion of total variance in the dependent
    variable that is explained by the independent variable (group membership).
    It is calculated as: η² = SS_between / SS_total

    Interpretation:
        η² < 0.01: Negligible effect
        0.01 ≤ η² < 0.06: Small effect
        0.06 ≤ η² < 0.14: Medium effect
        η² ≥ 0.14: Large effect

    Note: Eta-squared tends to overestimate effect size in small samples.
    Consider omega-squared for smaller samples, but eta-squared is more
    commonly reported in the literature.

    Args:
        *groups: Variable number of pandas Series, one for each group
                 At least 2 groups required

    Returns:
        Eta-squared effect size (float), ranges from 0 to 1

    Example:
        >>> english = data[data['language'] == 'English']['ati']
        >>> dutch = data[data['language'] == 'Dutch']['ati']
        >>> german = data[data['language'] == 'German']['ati']
        >>> effect_size = eta_squared(english, dutch, german)
        >>> print(f"η² = {effect_size:.3f}")
        η² = 0.089
    """
    # Drop NaN values from each group
    clean_groups = [group.dropna() for group in groups]

    # Calculate grand mean (mean of all data points across all groups)
    all_data = pd.concat(clean_groups)
    grand_mean = all_data.mean()

    # Calculate sum of squares between groups (SS_between)
    # For each group: n_i * (mean_i - grand_mean)^2
    ss_between = 0
    for group in clean_groups:
        n_i = len(group)
        mean_i = group.mean()
        ss_between += n_i * (mean_i - grand_mean) ** 2

    # Calculate total sum of squares (SS_total)
    # Sum of squared deviations from grand mean
    ss_total = ((all_data - grand_mean) ** 2).sum()

    # Calculate eta-squared
    if ss_total > 0:
        eta_sq = ss_between / ss_total
    else:
        eta_sq = 0.0

    return eta_sq


def kruskal_wallis_test(*groups: pd.Series) -> tuple[float, float, int]:
    """
    Perform a Kruskal-Wallis H test for comparing distributions across multiple groups.

    The Kruskal-Wallis H test is a non-parametric alternative to one-way ANOVA.
    It tests whether samples from multiple independent groups originate from the
    same distribution by comparing the ranks of the data rather than the raw values.
    This makes it robust to outliers and suitable for ordinal data or non-normal
    distributions.

    Best used when:
    - Data is ordinal (e.g., Likert scales)
    - Distributions are skewed or non-normal
    - Sample sizes are small or unequal
    - Data contains outliers
    - Homogeneity of variance assumption is violated

    Interpretation of p-value:
        α < 0.001: Very strong evidence of differences (highly significant)
        α < 0.01: Strong evidence of differences (very significant)
        α < 0.05: Moderate evidence of differences (significant)
        α ≥ 0.05: Insufficient evidence to reject equality (not significant)

    Interpretation of H-statistic:
        H ≈ 0: Groups have similar distributions (no effect)
        Larger H: Stronger evidence of group differences

    Args:
        *groups: Variable number of pandas Series, one for each group
                 At least 2 groups required, typically 3 or more

    Returns:
        Tuple of (H_statistic, p_value, df) where:
            - H_statistic: Kruskal-Wallis H statistic (float, typically ≥ 0)
            - p_value: Probability value (float, ranges from 0 to 1)
            - df: Degrees of freedom (int) = k - 1, where k is number of groups

    Example:
        >>> english = data[data['language'] == 'English']['ati']
        >>> dutch = data[data['language'] == 'Dutch']['ati']
        >>> german = data[data['language'] == 'German']['ati']
        >>> h_stat, p_value, df = kruskal_wallis_test(english, dutch, german)
        >>> print(f"H({df}) = {h_stat:.2f}, p = {p_value:.3f}")
        H(2) = 5.34, p = 0.069

    References:
        Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion
        variance analysis. Journal of the American Statistical Association,
        47(260), 583-621.
    """
    # Drop NaN values from each group
    clean_groups = [group.dropna() for group in groups]

    # Perform Kruskal-Wallis H test
    # scipy.stats.kruskal returns (H_statistic, p_value)
    h_statistic, p_value = stats.kruskal(*clean_groups)

    # Calculate degrees of freedom
    k = len(clean_groups)  # Number of groups
    df = k - 1  # Degrees of freedom

    return h_statistic, p_value, df


def chi_square_test(contingency_table: pd.DataFrame) -> tuple[float, float, int]:
    """
    Perform a chi-square test of independence for categorical data.

    Tests whether there is a significant association between two categorical variables
    by comparing observed frequencies with expected frequencies under independence.

    Interpretation:
        α < 0.001: Very strong evidence of association (highly significant)
        α < 0.01: Strong evidence of association (very significant)
        α < 0.05: Moderate evidence of association (significant)
        α ≥ 0.05: Insufficient evidence to reject independence (not significant)

    Args:
        contingency_table: Crosstab/contingency table (pandas DataFrame)
                          Rows = categories of variable 1
                          Columns = categories of variable 2

    Returns:
        Tuple of (chi2_statistic, p_value, df) where:
            - chi2_statistic: Chi-square test statistic (float)
            - p_value: Two-tailed p-value (float), ranges from 0 to 1
            - df: Degrees of freedom (int) = (rows - 1) * (cols - 1)

    Example:
        >>> counts = pd.crosstab(data['gender'], data['stimulus_group'])
        >>> chi2, p_value, df = chi_square_test(counts)
        >>> print(f"χ²({df}) = {chi2:.2f}, p = {p_value:.3f}")
        χ²(1) = 5.42, p = 0.020
    """

    # Perform chi-square test of independence
    chi2, p_value, df, expected = stats.chi2_contingency(contingency_table)

    return chi2, p_value, df


def fisher_exact_test(contingency_table: pd.DataFrame) -> tuple[float, float]:
    """
    Perform Fisher's exact test for categorical data.

    Fisher's exact test is used to determine if there are nonrandom associations
    between two categorical variables in a contingency table. It is more accurate
    than the chi-square test for small sample sizes (when expected frequencies < 5).
    The test is "exact" because it calculates the exact probability rather than
    approximating with a distribution.

    Best used when:
    - Sample sizes are small (expected cell frequencies < 5)
    - 2×2 contingency tables (required)
    - Exact p-values are needed rather than approximations

    Interpretation:
        p < 0.001: Very strong evidence of association (highly significant)
        p < 0.01: Strong evidence of association (very significant)
        p < 0.05: Moderate evidence of association (significant)
        p ≥ 0.05: Insufficient evidence to reject independence (not significant)

    Args:
        contingency_table: 2×2 crosstab/contingency table (pandas DataFrame)
                          Must be exactly 2 rows and 2 columns

    Returns:
        Tuple of (odds_ratio, p_value) where:
            - odds_ratio: Odds ratio of the association (float)
            - p_value: Two-tailed p-value (float), ranges from 0 to 1

    Raises:
        ValueError: If contingency table is not 2×2

    Example:
        >>> counts = pd.crosstab(data['gender'], data['stimulus_group'])
        >>> odds_ratio, p_value = fisher_exact_test(counts)
        >>> print(f"OR = {odds_ratio:.2f}, p = {p_value:.3f}")
        OR = 2.14, p = 0.032

    References:
        Fisher, R. A. (1922). On the interpretation of χ² from contingency
        tables, and the calculation of P. Journal of the Royal Statistical
        Society, 85(1), 87-94.
    """
    # Check that table is 2x2
    if contingency_table.shape != (2, 2):
        raise ValueError(
            f"Fisher's exact test requires a 2×2 contingency table. "
            f"Got shape {contingency_table.shape}. "
            f"Use chi-square test for larger tables."
        )

    # Perform Fisher's exact test
    # scipy.stats.fisher_exact returns (odds_ratio, p_value)
    # alternative='two-sided' for two-tailed test
    odds_ratio, p_value = stats.fisher_exact(contingency_table.values, alternative='two-sided')

    return odds_ratio, p_value


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Calculate Cramér's V effect size for categorical associations.

    Cramér's V measures the strength of association between two categorical variables.
    It ranges from 0 (no association) to 1 (perfect association) and is based on the
    chi-square statistic but normalized for sample size and table dimensions.

    Interpretation (for df_min = 1, i.e., 2x2 table):
        V < 0.1: Negligible association
        0.1 ≤ V < 0.3: Small association
        0.3 ≤ V < 0.5: Medium association
        V ≥ 0.5: Large association

    Note: Interpretation thresholds decrease for larger tables (more categories).

    Args:
        contingency_table: Crosstab/contingency table (pandas DataFrame)
                          Rows = categories of variable 1
                          Columns = categories of variable 2

    Returns:
        Cramér's V effect size (float), ranges from 0 to 1

    Example:
        >>> counts = pd.crosstab(data['gender'], data['stimulus_group'])
        >>> effect_size = cramers_v(counts)
        >>> print(f"Cramér's V: {effect_size:.3f}")
        Cramér's V: 0.123
    """

    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)

    if n > 0 and min_dim > 0:
        v = np.sqrt(chi2 / (n * min_dim))
    else:
        v = 0.0

    return v


def pearson_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Calculate Pearson's r correlation coefficient between two continuous variables.

    Pearson's r measures the linear relationship between two continuous variables.
    It ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation),
    with 0 indicating no linear relationship.

    Assumptions:
    - Both variables should be continuous
    - Linear relationship between variables
    - Bivariate normal distribution (for significance testing)
    - No significant outliers

    Interpretation of |r|:
        |r| < 0.3: Weak correlation
        0.3 ≤ |r| < 0.7: Moderate correlation
        |r| ≥ 0.7: Strong correlation

    Interpretation of p-value:
        p < 0.001: Very strong evidence of correlation (highly significant)
        p < 0.01: Strong evidence of correlation (very significant)
        p < 0.05: Moderate evidence of correlation (significant)
        p ≥ 0.05: Insufficient evidence of correlation (not significant)

    Args:
        x: First variable data (pandas Series)
        y: Second variable data (pandas Series)

    Returns:
        Tuple of (r_value, p_value) where:
            - r_value: Pearson correlation coefficient (float, -1 to 1)
            - p_value: Two-tailed p-value (float, 0 to 1)

    Example:
        >>> age = data['age']
        >>> ati = data['ati']
        >>> r, p = pearson_correlation(age, ati)
        >>> print(f"r = {r:.3f}, p = {p:.3f}")
        r = 0.234, p = 0.018
    """
    # Create aligned data by dropping NaN values from both series
    # This ensures both series have the same indices
    combined = pd.concat([x, y], axis=1).dropna()
    x_clean = combined.iloc[:, 0]
    y_clean = combined.iloc[:, 1]

    # Calculate Pearson correlation
    # scipy.stats.pearsonr returns (correlation, p_value)
    r_value, p_value = stats.pearsonr(x_clean, y_clean)

    return r_value, p_value


def spearman_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Calculate Spearman's ρ (rho) rank correlation coefficient between two variables.

    Spearman's ρ measures the monotonic relationship between two variables by
    computing the Pearson correlation between the rank values. It is a non-parametric
    measure that does not assume linear relationships or normal distributions.

    Best used when:
    - Variables are ordinal (e.g., Likert scales)
    - Relationship is monotonic but not necessarily linear
    - Data contains outliers
    - Distributions are not normal

    Interpretation of |ρ|:
        |ρ| < 0.3: Weak correlation
        0.3 ≤ |ρ| < 0.7: Moderate correlation
        |ρ| ≥ 0.7: Strong correlation

    Interpretation of p-value:
        p < 0.001: Very strong evidence of correlation (highly significant)
        p < 0.01: Strong evidence of correlation (very significant)
        p < 0.05: Moderate evidence of correlation (significant)
        p ≥ 0.05: Insufficient evidence of correlation (not significant)

    Args:
        x: First variable data (pandas Series)
        y: Second variable data (pandas Series)

    Returns:
        Tuple of (rho_value, p_value) where:
            - rho_value: Spearman correlation coefficient (float, -1 to 1)
            - p_value: Two-tailed p-value (float, 0 to 1)

    Example:
        >>> likert_scale1 = data['TiA_rc']
        >>> likert_scale2 = data['TiA_up']
        >>> rho, p = spearman_correlation(likert_scale1, likert_scale2)
        >>> print(f"ρ = {rho:.3f}, p = {p:.3f}")
        ρ = 0.456, p < 0.001
    """
    # Create aligned data by dropping NaN values from both series
    combined = pd.concat([x, y], axis=1).dropna()
    x_clean = combined.iloc[:, 0]
    y_clean = combined.iloc[:, 1]

    # Calculate Spearman rank correlation
    # scipy.stats.spearmanr returns (correlation, p_value)
    rho_value, p_value = stats.spearmanr(x_clean, y_clean)

    return rho_value, p_value


def moderation_analysis(df: pd.DataFrame, outcome: str, group_var: str,
                       moderator: str, moderator_name: str = None) -> tuple[dict, object]:
    """
    Test moderation effect using multiple regression with interaction term.

    Moderation analysis tests whether the relationship between a predictor (X) and
    outcome (Y) depends on the level of a third variable (M, the moderator). This is
    implemented via regression with an interaction term:

        Y = β₀ + β₁X + β₂M + β₃(X×M) + ε

    The interaction coefficient (β₃) represents the moderation effect. A significant
    β₃ indicates that the effect of X on Y changes as M increases.

    **Best practices implemented:**
    - Continuous moderators should be mean-centered before calling this function
    - Categorical predictors should use effect coding (-0.5, 0.5) for symmetric interpretation
    - The function returns both a results dictionary and the full fitted model for diagnostics

    **Interpretation:**
    - β₁ (group effect): Effect of X when moderator = 0 (i.e., at mean if centered)
    - β₂ (moderator effect): Effect of moderator when X = 0
    - β₃ (interaction): How much the X→Y effect changes per unit increase in M

    Args:
        df: DataFrame containing all variables
        outcome: Name of dependent variable column
        group_var: Name of independent variable column (e.g., treatment group)
                  Should be effect-coded (-0.5, 0.5) for best interpretation
        moderator: Name of moderator variable column
                  Should be mean-centered if continuous
        moderator_name: Optional display name for moderator (defaults to moderator column name)

    Returns:
        Tuple of (results_dict, fitted_model) where:
            - results_dict: Dictionary containing regression statistics including:
                * outcome, moderator: Variable names
                * n: Sample size
                * r_squared, adj_r_squared: Model fit statistics
                * b_group, se_group, p_group: Main effect of group
                * b_moderator, se_moderator, p_moderator: Main effect of moderator
                * b_interaction, se_interaction, p_interaction: Interaction effect
                * ci_group_lower/upper: 95% CI for group effect
                * ci_interaction_lower/upper: 95% CI for interaction
            - fitted_model: statsmodels OLS regression results object for diagnostics

    Example:
        >>> # Prepare variables (effect code group, center moderator)
        >>> df['group_effect'] = df['stimulus_group'].map({'control': -0.5, 'uncertainty': 0.5})
        >>> df['ati_c'] = df['ati'] - df['ati'].mean()
        >>>
        >>> # Test moderation
        >>> results, model = moderation_analysis(df, 'tia_t', 'group_effect', 'ati_c', 'ATI')
        >>> print(f"Interaction: β = {results['b_interaction']:.3f}, p = {results['p_interaction']:.3f}")
        Interaction: β = 0.092, p = 0.460
        >>>
        >>> # Check residuals for diagnostics
        >>> import matplotlib.pyplot as plt
        >>> plt.scatter(model.fittedvalues, model.resid)

    Notes:
        - Requires statsmodels package
        - Assumes ordinary least squares regression is appropriate
        - Does not check model assumptions (linearity, homoscedasticity, etc.)
        - For significant interactions, follow up with simple slopes analysis

    References:
        Hayes, A. F. (2018). Introduction to mediation, moderation, and conditional
        process analysis (2nd ed.). Guilford Press.
    """
    import statsmodels.formula.api as smf

    # Use provided name or default to variable name
    if moderator_name is None:
        moderator_name = moderator

    # Create interaction term
    df_temp = df.copy()
    df_temp['interaction'] = df_temp[group_var] * df_temp[moderator]

    # Fit regression model: Y ~ X + M + X*M
    formula = f'{outcome} ~ {group_var} + {moderator} + interaction'
    model = smf.ols(formula, data=df_temp).fit()

    # Extract key statistics
    results = {
        'outcome': outcome,
        'moderator': moderator_name,
        'n': int(model.nobs),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        # Main effect of group (conditional effect when M = 0, i.e., at mean if centered)
        'b_group': model.params[group_var],
        'se_group': model.bse[group_var],
        'p_group': model.pvalues[group_var],
        'ci_group_lower': model.conf_int().loc[group_var, 0],
        'ci_group_upper': model.conf_int().loc[group_var, 1],
        # Main effect of moderator
        'b_moderator': model.params[moderator],
        'se_moderator': model.bse[moderator],
        'p_moderator': model.pvalues[moderator],
        # Interaction effect (the moderation effect)
        'b_interaction': model.params['interaction'],
        'se_interaction': model.bse['interaction'],
        'p_interaction': model.pvalues['interaction'],
        'ci_interaction_lower': model.conf_int().loc['interaction', 0],
        'ci_interaction_upper': model.conf_int().loc['interaction', 1],
    }

    return results, model


def print_moderation_results(results: dict, show_interpretation: bool = True) -> None:
    """
    Print moderation analysis results in a formatted, readable table.

    Displays regression coefficients, standard errors, confidence intervals, and
    p-values for the group effect, moderator effect, and interaction effect from
    a moderation analysis. Optionally includes an interpretation of the findings.

    Args:
        results: Dictionary of results from moderation_analysis() function
                Must contain keys: outcome, moderator, n, r_squared, adj_r_squared,
                b_group, se_group, p_group, ci_group_lower, ci_group_upper,
                b_moderator, se_moderator, p_moderator,
                b_interaction, se_interaction, p_interaction,
                ci_interaction_lower, ci_interaction_upper
        show_interpretation: If True, prints interpretation of interaction effect
                            (default: True)

    Returns:
        None (prints to stdout)

    Example:
        >>> results, model = moderation_analysis(df, 'tia_t', 'group_effect', 'ati_c', 'ATI')
        >>> print_moderation_results(results)
        ================================================================================
        Outcome: tia_t | Moderator: ATI
        ================================================================================
        N = 255, R² = 0.0072, Adj. R² = -0.0047

        Coefficient               β          SE         95% CI                    p
        --------------------------------------------------------------------------------
        Group (conditional)       -0.0843    0.0996     [-0.280,  0.112]          0.3978
        ATI                       0.0472     0.0621                                0.4479
        Interaction               0.0919     0.1242     [-0.153,  0.337]          0.4600

        Interpretation:
          No significant moderation effect detected.
        ================================================================================

    Notes:
        - Significance markers: *** p<.001, ** p<.01, * p<.05
        - P-values < .001 are displayed as "<.001"
        - Confidence intervals shown only for group and interaction effects
        - Interpretation based on p < .05 threshold for interaction
    """
    print(f"\n{'='*80}")
    print(f"Outcome: {results['outcome']} | Moderator: {results['moderator']}")
    print(f"{'='*80}")
    print(f"N = {results['n']}, R² = {results['r_squared']:.4f}, Adj. R² = {results['adj_r_squared']:.4f}")
    print(f"\n{'Coefficient':<25} {'β':<10} {'SE':<10} {'95% CI':<25} {'p':<10}")
    print(f"{'-'*80}")

    # Group main effect
    ci_group = f"[{results['ci_group_lower']:>6.3f}, {results['ci_group_upper']:>6.3f}]"
    p_group = f"{results['p_group']:.4f}" if results['p_group'] >= 0.001 else "<.001"
    print(f"{'Group (conditional)':<25} {results['b_group']:<10.4f} {results['se_group']:<10.4f} {ci_group:<25} {p_group:<10}")

    # Moderator main effect
    p_mod = f"{results['p_moderator']:.4f}" if results['p_moderator'] >= 0.001 else "<.001"
    print(f"{results['moderator']:<25} {results['b_moderator']:<10.4f} {results['se_moderator']:<10.4f} {'':25} {p_mod:<10}")

    # Interaction effect
    ci_int = f"[{results['ci_interaction_lower']:>6.3f}, {results['ci_interaction_upper']:>6.3f}]"
    p_int = f"{results['p_interaction']:.4f}" if results['p_interaction'] >= 0.001 else "<.001"
    sig_marker = " ***" if results['p_interaction'] < 0.001 else " **" if results['p_interaction'] < 0.01 else " *" if results['p_interaction'] < 0.05 else ""
    print(f"{'Interaction':<25} {results['b_interaction']:<10.4f} {results['se_interaction']:<10.4f} {ci_int:<25} {p_int:<10}{sig_marker}")

    if show_interpretation:
        print(f"\n{'Interpretation:':<25}")
        if results['p_interaction'] < 0.05:
            direction = "increases" if results['b_interaction'] > 0 else "decreases"
            print(f"  Significant moderation: The effect of the intervention {direction}")
            print(f"  as {results['moderator']} increases.")
        else:
            print(f"  No significant moderation effect detected.")
    print(f"{'='*80}\n")


def interpret_moderation(b_interaction: float, p_interaction: float, moderator: str,
                        moderated_var: str = "intervention") -> str:
    """
    Generate interpretation text for a moderation effect.

    Creates human-readable interpretation of an interaction effect that can be
    saved to CSV files or displayed in reports.

    Args:
        b_interaction: Interaction coefficient (β₃)
        p_interaction: P-value for interaction effect
        moderator: Name of the moderating variable
        moderated_var: Name of the variable being moderated (default: "intervention")

    Returns:
        String interpretation of the moderation effect

    Example:
        >>> interpret_moderation(0.092, 0.460, "ATI")
        'No significant moderation detected (p = .460)'

        >>> interpret_moderation(-0.014, 0.048, "Age")
        'Significant moderation (p = .048): The effect of intervention decreases as Age increases'
    """
    if p_interaction < 0.05:
        direction = "increases" if b_interaction > 0 else "decreases"
        sig_level = "p < .001" if p_interaction < 0.001 else f"p = {p_interaction:.3f}"
        return f"Significant moderation ({sig_level}): The effect of {moderated_var} {direction} as {moderator} increases"
    else:
        return f"No significant moderation detected (p = {p_interaction:.3f})"


def interpret_direct_effect(b_moderator: float, p_moderator: float,
                           moderator: str, outcome: str) -> str:
    """
    Generate interpretation text for a direct effect (main effect of moderator).

    Creates human-readable interpretation of a predictor's main effect on an outcome
    that can be saved to CSV files or displayed in reports.

    Args:
        b_moderator: Regression coefficient for moderator (β₂)
        p_moderator: P-value for moderator effect
        moderator: Name of the predictor variable
        outcome: Name of the outcome variable

    Returns:
        String interpretation of the direct effect

    Example:
        >>> interpret_direct_effect(0.304, 0.0001, "ATI", "TiA_f")
        'Significant positive effect (p < .001): Higher ATI predicts higher TiA_f'

        >>> interpret_direct_effect(-0.142, 0.014, "ATI", "TiA_up")
        'Significant negative effect (p = .014): Higher ATI predicts lower TiA_up'

        >>> interpret_direct_effect(0.047, 0.448, "ATI", "TiA_t")
        'No significant effect detected (p = .448)'
    """
    if p_moderator < 0.05:
        direction = "positive" if b_moderator > 0 else "negative"
        pred_direction = "higher" if b_moderator > 0 else "lower"
        sig_level = "p < .001" if p_moderator < 0.001 else f"p = {p_moderator:.3f}"
        return f"Significant {direction} effect ({sig_level}): Higher {moderator} predicts {pred_direction} {outcome}"
    else:
        return f"No significant effect detected (p = {p_moderator:.3f})"


def format_effect_with_stars(beta: float, p_value: float) -> str:
    """
    Format a regression coefficient with APA-style significance stars.

    Follows APA guidelines for significance markers:
    - *** for p < .001
    - ** for p < .01
    - * for p < .05
    - No marker for p >= .05

    Args:
        beta: Regression coefficient
        p_value: P-value for the coefficient

    Returns:
        Formatted string with coefficient and stars (e.g., "0.304***" or "0.047")

    Example:
        >>> format_effect_with_stars(0.304, 0.0001)
        '0.304***'

        >>> format_effect_with_stars(-0.142, 0.014)
        '-0.142*'

        >>> format_effect_with_stars(0.047, 0.448)
        '0.047'
    """
    stars = ""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"

    return f"{beta:.3f}{stars}"
