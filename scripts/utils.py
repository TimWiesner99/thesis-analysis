"""Utility functions for data processing."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load labels and questions once when module is imported
LABELS_PATH = "../data/labels.csv"
QUESTIONS_PATH = "../data/questions.csv"
labels = pd.read_csv(LABELS_PATH)
questions = pd.read_csv(QUESTIONS_PATH)


def get_value_for_label(item_name: str, label_text: str) -> int:
    """
    Get the numeric value for a given answer label.

    Args:
        item_name: The question/item name (e.g., 'consent', 'gender')
        label_text: The answer text to look up (e.g., 'I agree to participate in the study')

    Returns:
        The column number as int

    Example:
        >>> get_value_for_label('consent', 'I agree to participate in the study')
        '1'
    """
    item_row = labels[labels['item'] == item_name]
    if item_row.empty:
        raise ValueError(f"Item '{item_name}' not found in labels.csv")

    # Search through all columns to find the matching label
    for col in item_row.columns:
        if col == 'item':
            continue
        value = item_row[col].iloc[0]
        if pd.notna(value) and value == label_text:
            return col

    raise ValueError(f"Label '{label_text}' not found for item '{item_name}'")


def get_label_for_value(item_name: str, value: str | int) -> str:
    """
    Get the answer label for a given numeric value.

    Args:
        item_name: The question/item name (e.g., 'consent', 'gender')
        value: The numeric value to look up (e.g., '1', 1, '2', 2)

    Returns:
        The label text corresponding to that value

    Example:
        >>> get_label_for_value('consent', '1')
        'I agree to participate in the study'
    """
    item_row = labels[labels['item'] == item_name]
    if item_row.empty:
        raise ValueError(f"Item '{item_name}' not found in labels.csv")

    # Convert value to string to match column names
    value_str = str(value)

    if value_str not in item_row.columns:
        raise ValueError(f"Value '{value}' is not a valid column for item '{item_name}'")

    label_text = item_row[value_str].iloc[0]

    if pd.isna(label_text):
        raise ValueError(f"No label found for value '{value}' in item '{item_name}'")

    return label_text


def get_question_statement(item_name: str) -> str:
    """
    Get the question statement for a given item.

    Args:
        item_name: The question/item name (e.g., 'consent', 'gender', 'ATI_1')

    Returns:
        The question statement text

    Example:
        >>> get_question_statement('consent')
        'Consent Statement\\n\\nI confirm that:...'
        >>> get_question_statement('ATI_1')
        'I like to occupy myself in greater detail with technical systems.'
    """
    item_row = questions[questions['item'] == item_name]
    if item_row.empty:
        raise ValueError(f"Item '{item_name}' not found in questions.csv")

    question_text = item_row['question_statement'].iloc[0]

    if pd.isna(question_text):
        raise ValueError(f"No question statement found for item '{item_name}'")

    return question_text


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


def independent_t_test(group1: pd.Series, group2: pd.Series) -> float:
    """
    Perform an independent samples t-test between two groups.

    This performs Welch's t-test (does not assume equal variances) and returns
    the two-tailed p-value (alpha). The p-value indicates the probability of
    observing the data (or more extreme) if the null hypothesis (no difference
    between groups) is true.

    Interpretation:
        α < 0.001: Very strong evidence against null hypothesis (highly significant)
        α < 0.01: Strong evidence against null hypothesis (very significant)
        α < 0.05: Moderate evidence against null hypothesis (significant)
        α ≥ 0.05: Insufficient evidence to reject null hypothesis (not significant)

    Args:
        group1: First group data (pandas Series)
        group2: Second group data (pandas Series)

    Returns:
        Two-tailed p-value (float), ranges from 0 to 1

    Example:
        >>> control = data[data['stimulus_group'] == 0]['ati']
        >>> treatment = data[data['stimulus_group'] == 1]['ati']
        >>> p_value = independent_t_test(control, treatment)
        >>> print(f"p-value: {p_value:.4f}")
        p-value: 0.0234
    """
    from scipy import stats

    # Perform Welch's t-test (does not assume equal variances)
    # This is generally more robust than Student's t-test
    statistic, p_value = stats.ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)

    return p_value


def chi_square_test(contingency_table: pd.DataFrame) -> float:
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
        Two-tailed p-value (float), ranges from 0 to 1

    Example:
        >>> counts = pd.crosstab(data['gender'], data['stimulus_group'])
        >>> p_value = chi_square_test(counts)
        >>> print(f"p-value: {p_value:.4f}")
        p-value: 0.0234
    """
    from scipy import stats

    # Perform chi-square test of independence
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return p_value


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
    from scipy import stats

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
