import numpy as np
import pandas as pd
from scipy import stats

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
