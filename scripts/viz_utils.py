"""Visualization utility functions for thesis analysis.

This module provides consistent, clean visualization functions for different data types:
- Likert scale items and computed scales (histogram + KDE)
- Categorical variables (bar plots with readable labels)
- Continuous variables (histogram + KDE)
- Boxplots and mirrored histograms for multi-scale comparisons
- Split histogram with central boxplot for two-group comparisons
- Scatterplots with correlation analysis for examining relationships between variables
- Non-inferiority test visualization with overlapping Gaussian distributions

All functions support optional grouping by experimental condition.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List

from numpy.f2py.auxfuncs import throw_error
from scipy import stats
from .utils import get_label_for_value, get_question_statement
from .stats import (cohens_d, independent_t_test, chi_square_test, cramers_v,
                     fisher_exact_test, one_way_anova, eta_squared,
                     mann_whitney_u_test, rank_biserial_correlation,
                     kruskal_wallis_test, pearson_correlation, spearman_correlation)

# Load metadata
DATA_PATH = Path(__file__).parent.parent / "data"
labels = pd.read_csv(DATA_PATH / "labels.csv")
questions = pd.read_csv(DATA_PATH / "questions.csv")

# Custom color palette (placeholder - customize as needed)
COLORS = {
    'primary': '#578ED6',      # Blue
    'secondary': '#D65F57',    # Purple/Pink
    'accent': '#F18F01',       # Orange
    'neutral': '#6B6B6B',      # Gray
    'control': '#578ED6',      # For control group
    'uncertainty': '#D65F57',  # For uncertainty group
}

# Extended color palette for multiple groups (used when no specific color is defined)
DEFAULT_PALETTE = [
    '#578ED6',  # Blue
    '#D65F57',  # Red/Pink
    '#F18F01',  # Orange
    '#66C2A5',  # Teal
    '#8E44AD',  # Purple
    '#E67E22',  # Dark Orange
    '#3498DB',  # Light Blue
    '#E74C3C',  # Red
    '#95A5A6',  # Gray
    '#16A085',  # Dark Teal
]

# Style configuration
STYLE_CONFIG = {
    'font_size': 11,
    'title_size': 13,
    'label_size': 11,
    'tick_size': 10,
    'grid_alpha': 0.3,
    'hist_alpha': 0.7,
    'kde_linewidth': 2,
    'title_pad': 15,  # Space between title and plot
    'max_label_length': 25,  # Maximum characters for axis labels before truncation
}


def get_group_colors(group_labels: List[str]) -> List[str]:
    """
    Get colors for a list of group labels.

    Uses predefined colors from COLORS dict if available, otherwise assigns
    colors from DEFAULT_PALETTE in order.

    Args:
        group_labels: List of group label strings

    Returns:
        List of color strings (hex codes) matching the group_labels

    Example:
        >>> get_group_colors(['control', 'uncertainty'])
        ['#578ED6', '#D65F57']
        >>> get_group_colors(['English', 'Dutch', 'German'])
        ['#578ED6', '#D65F57', '#F18F01']
    """
    colors = []
    for idx, label in enumerate(group_labels):
        # First check if there's a predefined color for this label
        color = COLORS.get(label.lower(), None)
        if color is None:
            # Use color from default palette, cycling if needed
            color = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
        colors.append(color)
    return colors


def truncate_label(label: str, max_length: int = None) -> str:
    """
    Truncate a label if it exceeds maximum length.

    Args:
        label: The label text to truncate
        max_length: Maximum length (uses STYLE_CONFIG default if None)

    Returns:
        Truncated label with '...' if needed

    Example:
        >>> truncate_label('Very long label that needs truncation', 20)
        'Very long label th...'
    """
    if max_length is None:
        max_length = STYLE_CONFIG['max_label_length']

    if len(label) <= max_length:
        return label
    return label[:max_length-3] + '...'


def get_readable_labels(column_name: str, values: List, trunc: int = 20) -> List[str]:
    """
    Convert numeric values to human-readable labels using labels.csv.

    Args:
        column_name: The column/item name (e.g., 'gender', 'education')
        values: List of numeric values to convert
        trunc: Maximum label length before truncation. -1 means no truncation (default: 20)

    Returns:
        List of readable labels. Returns original value as string if label not found.

    Example:
        >>> get_readable_labels('gender', [1, 2])
        ['Male', 'Female']
        >>> get_readable_labels('education', [1, 2], trunc=15)
        ['Primary Educat...', 'Lower Secondary...']
    """
    readable = []
    for val in values:
        try:
            label = get_label_for_value(column_name, val)
            if trunc > 0:
                label = truncate_label(label, max_length=trunc)
            readable.append(label)
        except (ValueError, KeyError):
            # If label not found, use the numeric value
            readable.append(str(val))
    return readable


def _format_p_value(p_value: float) -> str:
    """
    Format p-value for display in plots using APA style.

    Args:
        p_value: The p-value to format (float between 0 and 1)

    Returns:
        Formatted p-value string in APA style (no leading zero)

    Example:
        >>> _format_p_value(0.0234)
        'p = .023'
        >>> _format_p_value(0.0001)
        'p < .001'
        >>> _format_p_value(0.159)
        'p = .159'
    """
    if p_value < 0.001:
        return "p < .001"
    else:
        # Format with 3 decimal places, remove leading zero
        return f"p = {p_value:.3f}".replace("0.", ".")


def _print_statistical_report(test_type: str,
                              variable_name: str,
                              group_stats: dict,
                              test_stats: dict) -> None:
    """
    Print comprehensive statistical test results to console in APA-style format.

    This internal helper function formats and displays statistical test results
    with group descriptive statistics, test statistics, and interpretation hints.
    Output is designed to be copy-paste ready for manuscripts.

    Args:
        test_type: Type of statistical test ('t-test', 'anova', 'chi-square')
        variable_name: Name of the variable/column being tested
        group_stats: Dictionary containing group descriptive statistics
                    Format for t-test/anova: {group_name: {'n': int, 'mean': float, 'sd': float}}
                    Format for chi-square: {group_name: {'n': int}}
        test_stats: Dictionary containing test statistics
                   For t-test: {'t': float, 'p': float, 'df': float, 'd': float}
                   For anova: {'F': float, 'p': float, 'df1': int, 'df2': int}
                   For chi-square: {'chi2': float, 'p': float, 'df': int, 'V': float}

    Example:
        >>> group_stats = {'Control': {'n': 45, 'mean': 3.21, 'sd': 0.89},
        ...                'Uncertainty': {'n': 43, 'mean': 3.45, 'sd': 0.76}}
        >>> test_stats = {'t': -1.42, 'p': 0.159, 'df': 83.45, 'd': 0.29}
        >>> _print_statistical_report('t-test', 'ati', group_stats, test_stats)
    """
    # Format p-value in APA style (no leading zero)
    def format_p(p_value: float) -> str:
        if p_value < 0.001:
            return "< .001"
        else:
            return f"= {p_value:.3f}".replace("0.", ".")

    # Determine significance level for interpretation
    def get_interpretation(p_value: float) -> str:
        if p_value < 0.001:
            return "highly significant (p < .001)"
        elif p_value < 0.01:
            return "very significant (p < .01)"
        elif p_value < 0.05:
            return "significant (p < .05)"
        else:
            return "not significant (p ≥ .05)"

    print(f"\nStatistical Test Results for '{variable_name}':")

    # Print group descriptive statistics
    if test_type in ['t-test', 'anova']:
        print("  Group Descriptive Statistics:")
        for group_name, stats in group_stats.items():
            n = stats['n']
            mean = stats['mean']
            sd = stats['sd']
            print(f"    {group_name}: n = {n}, M = {mean:.2f}, SD = {sd:.2f}")

    elif test_type in ['mann-whitney', 'kruskal-wallis']:
        print("  Group Descriptive Statistics:")
        for group_name, stats in group_stats.items():
            n = stats['n']
            # Non-parametric tests typically report median, but also support mean for consistency
            if 'median' in stats:
                median = stats['median']
                if 'iqr' in stats:
                    iqr = stats['iqr']
                    print(f"    {group_name}: n = {n}, Mdn = {median:.2f}, IQR = {iqr:.2f}")
                else:
                    print(f"    {group_name}: n = {n}, Mdn = {median:.2f}")
            else:
                # Fallback to mean/SD if median not provided
                mean = stats['mean']
                sd = stats['sd']
                print(f"    {group_name}: n = {n}, M = {mean:.2f}, SD = {sd:.2f}")

    elif test_type == 'chi-square':
        print("  Group Sample Sizes:")
        for group_name, stats in group_stats.items():
            n = stats['n']
            print(f"    {group_name}: n = {n}")

    # Print test results in APA format
    print("  Test: ", end="")

    if test_type == 't-test':
        print("Independent samples t-test (Welch's)")
        t = test_stats['t']
        p = test_stats['p']
        df = test_stats['df']
        d = test_stats['d']

        # APA format: t(df) = X.XX, p = .XXX, d = X.XX
        print(f"  Result: t({df:.1f}) = {t:.2f}, p {format_p(p)}, d = {d:.2f}")
        print(f"  Interpretation: {get_interpretation(p)}")

    elif test_type == 'anova':
        print("One-way ANOVA")
        F = test_stats['F']
        p = test_stats['p']
        df1 = test_stats['df1']
        df2 = test_stats['df2']

        # APA format: F(df1, df2) = X.XX, p = .XXX
        print(f"  Result: F({df1}, {df2}) = {F:.2f}, p {format_p(p)}")
        print(f"  Interpretation: {get_interpretation(p)}")

    elif test_type == 'mann-whitney':
        print("Mann-Whitney U test (non-parametric)")
        U = test_stats['U']
        p = test_stats['p']
        r = test_stats['r']

        # APA format: U = X.XX, p = .XXX, r = X.XX
        print(f"  Result: U = {U:.2f}, p {format_p(p)}, r = {r:.2f}")
        print(f"  Interpretation: {get_interpretation(p)}")

    elif test_type == 'kruskal-wallis':
        print("Kruskal-Wallis H test (non-parametric)")
        H = test_stats['H']
        p = test_stats['p']
        df = test_stats['df']

        # APA format: H(df) = X.XX, p = .XXX
        print(f"  Result: H({df}) = {H:.2f}, p {format_p(p)}")
        print(f"  Interpretation: {get_interpretation(p)}")

    elif test_type == 'chi-square':
        print("Chi-square test of independence")
        chi2 = test_stats['chi2']
        p = test_stats['p']
        df = test_stats['df']
        V = test_stats['V']

        # APA format: χ²(df) = X.XX, p = .XXX, V = X.XX
        print(f"  Result: χ²({df}) = {chi2:.2f}, p {format_p(p)}, V = {V:.2f}")
        print(f"  Interpretation: {get_interpretation(p)}")

    elif test_type == 'fisher':
        print("Fisher's exact test")
        OR = test_stats['OR']
        p = test_stats['p']

        # APA format: OR = X.XX, p = .XXX
        print(f"  Result: OR = {OR:.2f}, p {format_p(p)}")
        print(f"  Interpretation: {get_interpretation(p)}")

    print()  # Blank line separator


def _print_correlation_report(x_var: str,
                              y_var: str,
                              correlation_method: str,
                              correlation_stats: dict,
                              group_name: Optional[str] = None) -> None:
    """
    Print correlation test results to console in APA-style format.

    This internal helper function formats and displays correlation results
    with sample size, correlation coefficient, and p-value. Output is designed
    to be copy-paste ready for manuscripts.

    Args:
        x_var: Name of the x-axis variable
        y_var: Name of the y-axis variable
        correlation_method: Type of correlation ('pearson' or 'spearman')
        correlation_stats: Dictionary containing correlation statistics
                          Format: {'r': float, 'p': float, 'n': int}
                          (or {'rho': float, 'p': float, 'n': int} for Spearman)
        group_name: Optional name of the group (for grouped analyses)

    Example:
        >>> corr_stats = {'r': 0.456, 'p': 0.001, 'n': 88}
        >>> _print_correlation_report('age', 'ati', 'pearson', corr_stats)
    """
    # Format p-value in APA style (no leading zero)
    def format_p(p_value: float) -> str:
        if p_value < 0.001:
            return "< .001"
        else:
            return f"= {p_value:.3f}".replace("0.", ".")

    # Determine correlation strength for interpretation
    def get_strength(coef: float) -> str:
        abs_coef = abs(coef)
        if abs_coef < 0.3:
            return "weak"
        elif abs_coef < 0.7:
            return "moderate"
        else:
            return "strong"

    # Determine significance level for interpretation
    def get_significance(p_value: float) -> str:
        if p_value < 0.001:
            return "highly significant (p < .001)"
        elif p_value < 0.01:
            return "very significant (p < .01)"
        elif p_value < 0.05:
            return "significant (p < .05)"
        else:
            return "not significant (p ≥ .05)"

    # Print header
    if group_name:
        print(f"\nCorrelation Analysis for '{group_name}':")
    else:
        print(f"\nCorrelation Analysis:")

    print(f"  Variables: {x_var} vs {y_var}")
    print(f"  Sample size: n = {correlation_stats['n']}")

    # Print correlation results based on method
    if correlation_method == 'pearson':
        r = correlation_stats['r']
        p = correlation_stats['p']
        print(f"  Method: Pearson's r correlation")
        # APA format: r(n-2) = X.XX, p = .XXX
        df = correlation_stats['n'] - 2
        print(f"  Result: r({df}) = {r:.3f}, p {format_p(p)}")
        print(f"  Interpretation: {get_strength(r)} {('positive' if r > 0 else 'negative')} correlation, {get_significance(p)}")
    else:  # spearman
        rho = correlation_stats['rho']
        p = correlation_stats['p']
        print(f"  Method: Spearman's ρ (rho) rank correlation")
        # APA format: ρ(n-2) = X.XX, p = .XXX
        df = correlation_stats['n'] - 2
        print(f"  Result: ρ({df}) = {rho:.3f}, p {format_p(p)}")
        print(f"  Interpretation: {get_strength(rho)} {('positive' if rho > 0 else 'negative')} correlation, {get_significance(p)}")

    print()  # Blank line separator


def apply_consistent_style(ax: plt.Axes,
                          title: Optional[str] = None,
                          xlabel: Optional[str] = None,
                          ylabel: Optional[str] = None,
                          title_pad: Optional[int] = None) -> None:
    """
    Apply consistent styling to a matplotlib axes object.

    Args:
        ax: The matplotlib axes to style
        title: Optional plot title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)
    """
    # Use default title_pad if not specified
    if title_pad is None:
        title_pad = STYLE_CONFIG['title_pad']

    if title:
        ax.set_title(title, fontsize=STYLE_CONFIG['title_size'],
                    fontweight='bold', pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE_CONFIG['label_size'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE_CONFIG['label_size'])

    ax.tick_params(labelsize=STYLE_CONFIG['tick_size'])
    ax.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_likert_distribution(data: pd.DataFrame,
                             column: str,
                             title: Optional[str] = None,
                             group_by: Optional[str] = None,
                             ax: Optional[plt.Axes] = None,
                             show_stats: bool = True,
                             show_bars: bool = True,
                             show_kde: bool = False,
                             show_correlation: bool = True,
                             likert_range: int = 5,
                             show_labels: bool = False,
                             trunc: int = 20,
                             title_pad: Optional[int] = None,
                             test_method: str = 'parametric') -> plt.Axes:
    """
    Plot distribution of Likert scale items or computed scale scores.

    Shows histogram with optional KDE overlay. Optionally splits by experimental group.

    Args:
        data: DataFrame containing the data
        column: Column name to plot
        title: Plot title (auto-generated if None)
        group_by: Optional column to group by (e.g., 'stimulus_group')
        ax: Optional matplotlib axes (creates new figure if None)
        show_stats: Whether to show mean and SD in legend
        show_bars: Whether to show histogram bars (default: True)
        show_kde: Whether to show KDE (kernel density estimate) overlay (default: False)
        show_correlation: Whether to show effect size and p-value from statistical test between groups (default: True)
        likert_range: Max value of Likert scale (default: 5, for 1-5 scale)
        show_labels: Whether to show Likert labels on x-axis (e.g., "Strongly disagree") (default: False)
        trunc: Max characters for labels before truncation. -1 = no truncation (default: 20)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)
        test_method: Statistical test method ('parametric' or 'nonparametric').
                    - 'parametric': Uses t-test (2 groups) or ANOVA (3+ groups) with Cohen's d or η²
                    - 'nonparametric': Uses Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) with rank-biserial r
                    Default: 'parametric'

    Returns:
        The matplotlib axes object

    Example:
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_kde=True)
        >>> plot_likert_distribution(data, 'ATI_1', likert_range=5, show_labels=True, trunc=15)
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_correlation=True)
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_bars=False, show_kde=True)
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', test_method='nonparametric')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Generate title if not provided
    if title is None:
        title = f"Distribution of {column.upper()}"

    # Remove stats from legend if kde plot is chosen
    if show_kde:
        show_stats = False

    # Check if grouping is requested
    if group_by is not None and group_by in data.columns:
        # Get unique groups
        groups = data[group_by].unique()

        # Store group data for correlation calculation
        group_means = {}
        group_info = {}  # Store group data, mean, std, color, label

        # First pass: collect all group data and calculate means
        for group in groups:
            group_data = data[data[group_by] == group][column].dropna()

            # Get readable label for group
            try:
                group_label = get_label_for_value(group_by, group)
            except (ValueError, KeyError):
                group_label = str(group)

            # Store group data for correlation
            group_means[group_label] = group_data

            # Choose color
            color = COLORS.get(group_label.lower(), COLORS['primary'])

            # Store group info
            group_info[group_label] = {
                'data': group_data,
                'mean': group_data.mean(),
                'std': group_data.std(),
                'color': color
            }

        # Determine label positioning based on mean comparison
        # Sort groups by mean to determine which has higher/lower mean
        sorted_groups = sorted(group_info.items(), key=lambda x: x[1]['mean'])

        # Second pass: plot with adjusted label positions
        for group in groups:
            # Get readable label for group
            try:
                group_label = get_label_for_value(group_by, group)
            except (ValueError, KeyError):
                group_label = str(group)

            info = group_info[group_label]
            group_data = info['data']
            mean = info['mean']
            std = info['std']
            color = info['color']

            # Add stats to label if requested
            label = group_label
            if show_stats:
                label += f" (M={mean:.2f}, SD={std:.2f})"

            # Determine label alignment based on whether this is the lower or higher mean group
            is_lower_mean = (group_label == sorted_groups[0][0])
            is_higher_mean = (group_label == sorted_groups[-1][0])

            # Plot histogram (optional)
            if show_bars:
                ax.hist(group_data, bins=20, alpha=STYLE_CONFIG['hist_alpha'],
                       label=label, color=color, edgecolor='white', linewidth=0.5)

            # Plot KDE (optional)
            if show_kde:
                group_data.plot.kde(ax=ax, color=color, linewidth=STYLE_CONFIG['kde_linewidth'],
                                   label=label if not show_bars else None)

                # Add vertical line for mean
                ax.axvline(mean, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

                # Add text label for mean at top of plot with adjusted positioning
                y_top = ax.get_ylim()[1] * 0.95
                # Adjust horizontal alignment and position based on mean ranking
                if is_lower_mean:
                    mean_ha = 'right'
                    mean_x = mean - 0.05  # Slightly to the left
                elif is_higher_mean:
                    mean_ha = 'left'
                    mean_x = mean + 0.05  # Slightly to the right
                else:
                    mean_ha = 'center'
                    mean_x = mean

                ax.text(mean_x, y_top, f'M={mean:.2f}', color=color, fontsize=STYLE_CONFIG['tick_size']-1,
                       ha=mean_ha, va='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

                # Add horizontal line for standard deviation
                y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.hlines(y_pos, mean - std, mean + std, color=color, linewidth=3, alpha=0.7)

                # Add text label for SD with adjusted positioning
                if is_lower_mean:
                    sd_ha = 'right'
                    sd_x = mean - 0.05  # Slightly to the left
                elif is_higher_mean:
                    sd_ha = 'left'
                    sd_x = mean + 0.05  # Slightly to the right
                else:
                    sd_ha = 'center'
                    sd_x = mean

                ax.text(sd_x, y_pos, f'SD={std:.2f}', color=color, fontsize=STYLE_CONFIG['tick_size']-1,
                       ha=sd_ha, va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

        # Calculate and display correlation/effect size between groups if requested
        if show_correlation and len(group_means) >= 2:
            # Get all groups
            group_names = list(group_means.keys())

            if len(group_means) == 2:
                # Two groups: use parametric or non-parametric test
                group1_data = group_means[group_names[0]]
                group2_data = group_means[group_names[1]]

                if test_method == 'nonparametric':
                    # Non-parametric: Mann-Whitney U test and rank-biserial correlation
                    u_stat, p_value, n1, n2 = mann_whitney_u_test(group1_data, group2_data)
                    effect_size = rank_biserial_correlation(u_stat, n1, n2)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        'U': u_stat,
                        'p': p_value,
                        'r': effect_size
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('mann-whitney', column, group_stats, test_stats)

                    # Add text box with rank-biserial correlation (r) and p-value
                    textstr = f"r = {effect_size:.2f}\n{_format_p_value(p_value)}"
                else:
                    # Parametric: t-test and Cohen's d
                    effect_size = cohens_d(group1_data, group2_data)
                    t_stat, p_value, df = independent_t_test(group1_data, group2_data)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        't': t_stat,
                        'p': p_value,
                        'df': df,
                        'd': effect_size
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('t-test', column, group_stats, test_stats)

                    # Add text box with Cohen's d (δ) and p-value
                    textstr = f"δ = {effect_size:.2f}\n{_format_p_value(p_value)}"
            else:
                # Three or more groups: use parametric or non-parametric test
                all_group_data = [group_means[name] for name in group_names]

                if test_method == 'nonparametric':
                    # Non-parametric: Kruskal-Wallis H test
                    h_stat, p_value, df = kruskal_wallis_test(*all_group_data)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        'H': h_stat,
                        'p': p_value,
                        'df': df
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('kruskal-wallis', column, group_stats, test_stats)

                    # Add text box with H-statistic and p-value
                    textstr = f"H = {h_stat:.2f}\n{_format_p_value(p_value)}"
                else:
                    # Parametric: ANOVA and F-statistic
                    f_stat, p_value, df1, df2 = one_way_anova(*all_group_data)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        'F': f_stat,
                        'p': p_value,
                        'df1': df1,
                        'df2': df2
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('anova', column, group_stats, test_stats)

                    # Add text box with F-statistic and p-value
                    textstr = f"F = {f_stat:.2f}\n{_format_p_value(p_value)}"

            # Display statistics in text box at top-right
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', multialignment='left', bbox=props)
    else:
        # Single distribution (no grouping)
        plot_data = data[column].dropna()

        label = f"Overall"
        if show_stats:
            label += f" (M={plot_data.mean():.2f}, SD={plot_data.std():.2f})"

        # Plot histogram (optional)
        if show_bars:
            ax.hist(plot_data, bins=20, alpha=STYLE_CONFIG['hist_alpha'],
                   label=label, color=COLORS['primary'], edgecolor='white', linewidth=0.5)

        # Plot KDE (optional)
        if show_kde:
            plot_data.plot.kde(ax=ax, color=COLORS['primary'],
                              linewidth=STYLE_CONFIG['kde_linewidth'],
                              label=label if not show_bars else None)

            # Add vertical line for mean
            mean = plot_data.mean()
            ax.axvline(mean, color=COLORS['primary'], linestyle='--', linewidth=1.5, alpha=0.6)

            # Add horizontal line for standard deviation
            std = plot_data.std()
            y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.hlines(y_pos, mean - std, mean + std, color=COLORS['primary'], linewidth=3, alpha=0.7)

    # Set x-axis limits based on Likert scale range (1 to likert_range)
    # This ensures consistent scale across all Likert plots
    ax.set_xlim(0.5, likert_range + 0.5)

    # Set x-axis ticks and labels
    if show_labels:
        # Try to get Likert labels from labels.csv
        # Assumes labels are stored for values 1 through likert_range
        try:
            tick_labels = []
            for i in range(1, likert_range + 1):
                try:
                    label = get_label_for_value(column, i)
                    # Truncate long labels if trunc parameter is set
                    if trunc > 0:
                        label = truncate_label(label, max_length=trunc)
                    tick_labels.append(label)
                except (ValueError, KeyError):
                    # If label not found, use numeric value
                    tick_labels.append(str(i))

            ax.set_xticks(range(1, likert_range + 1))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        except Exception:
            # If labels retrieval fails, fall back to numeric labels
            ax.set_xticks(range(1, likert_range + 1))
    else:
        # Just show numeric labels (1, 2, 3, 4, 5)
        ax.set_xticks(range(1, likert_range + 1))

    # Apply styling
    apply_consistent_style(ax, title=title, xlabel=column.upper(),
                          ylabel='Density', title_pad=title_pad)
    ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'])

    return ax


def plot_categorical_bar(data: pd.DataFrame,
                        column: str,
                        title: Optional[str] = None,
                        group_by: Optional[str] = None,
                        ax: Optional[plt.Axes] = None,
                        show_bars: bool = True,
                        show_percentages: bool = True,
                        show_absolute: bool = True,
                        show_stats: bool = True,
                        trunc: int = 20,
                        title_pad: Optional[int] = None,
                        x_label: Optional[str] = None,
                        test_method: str = 'parametric') -> plt.Axes:
    """
    Plot bar chart for categorical variables with readable labels.

    Args:
        data: DataFrame containing the data
        column: Column name to plot
        title: Plot title (auto-generated if None)
        group_by: Optional column to group by (e.g., 'stimulus_group')
        ax: Optional matplotlib axes (creates new figure if None)
        show_bars: Whether to show bars (default: True)
        show_percentages: Whether to show percentages on bars
        show_absolute: Whether to show absolute counts on bars
        show_stats: Whether to show effect size and p-value when comparing groups (default: True)
        trunc: Max characters for labels before truncation. -1 = no truncation (default: 20)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)
        test_method: Statistical test method ('parametric' or 'nonparametric').
                    - 'parametric': Uses chi-square test with Cramér's V (works for any table size)
                    - 'nonparametric': Uses Fisher's exact test with odds ratio (requires 2×2 table)
                    Default: 'parametric'

    Returns:
        The matplotlib axes object

    Example:
        >>> plot_categorical_bar(data, 'gender', group_by='stimulus_group')
        >>> plot_categorical_bar(data, 'education', trunc=20)
        >>> plot_categorical_bar(data, 'gender', group_by='stimulus_group', show_stats=False)
        >>> plot_categorical_bar(data, 'gender', show_bars=False)
        >>> plot_categorical_bar(data, 'gender', group_by='stimulus_group', test_method='nonparametric')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Generate title if not provided
    if title is None:
        title = f"Distribution of {column.capitalize()}"

    # Get value counts
    if group_by is not None and group_by in data.columns:
        # Grouped bar chart
        counts = data.groupby([column, group_by]).size().unstack(fill_value=0)

        # Get readable labels for x-axis (apply truncation)
        x_labels = get_readable_labels(column, counts.index.tolist(), trunc=trunc)

        # Get readable labels for groups (legend) - NO truncation for legend
        group_labels = get_readable_labels(group_by, counts.columns.tolist(), trunc=-1)
        counts.columns = group_labels

        # Plot grouped bars
        counts_plot = counts.copy()
        counts_plot.index = x_labels

        # Create color mapping: map each group label to its color
        colors = [COLORS.get(label.lower(), COLORS['primary']) for label in group_labels]
        color_dict = dict(zip(group_labels, colors))

        # Plot bars (optional)
        if show_bars:
            counts_plot.plot(kind='bar', ax=ax, color=color_dict, alpha=STYLE_CONFIG['hist_alpha'],
                            edgecolor='white', linewidth=1)

        # Add percentage labels if requested
        if show_bars and show_percentages and show_absolute:
            for container in ax.containers:
                labels = [f'{v:.0f}\n({v/counts.values.sum()*100:.1f}%)' if v > 0 else ''
                         for v in container.datavalues]
                ax.bar_label(container, labels=labels, fontsize=STYLE_CONFIG['tick_size'])
        elif show_bars and show_percentages and not show_absolute:
            for container in ax.containers:
                labels = [f'{v / counts.values.sum() * 100:.1f}%' if v > 0 else ''
                          for v in container.datavalues]
                ax.bar_label(container, labels=labels, fontsize=STYLE_CONFIG['tick_size'])
        elif show_bars and not show_percentages and show_absolute:
            for container in ax.containers:
                labels = [f'{v:.0f}' if v > 0 else ''
                          for v in container.datavalues]
                ax.bar_label(container, labels=labels, fontsize=STYLE_CONFIG['tick_size'])


        # Calculate and display statistics if requested and we have exactly 2 groups
        if show_stats and len(group_labels) == 2:
            # Prepare group statistics for console output (sample sizes)
            group_stats = {}
            for group_label in group_labels:
                # Sum counts for this group across all categories
                n = counts[group_label].sum()
                group_stats[group_label] = {'n': int(n)}

            if test_method == 'nonparametric':
                # Non-parametric: Fisher's exact test (requires 2×2 table)
                if counts.shape != (2, 2):
                    # Fall back to chi-square if not 2×2
                    print(f"Warning: Fisher's exact test requires a 2×2 table. "
                          f"Got {counts.shape}. Falling back to chi-square test.")
                    effect_size = cramers_v(counts)
                    chi2, p_value, df = chi_square_test(counts)
                    test_stats = {
                        'chi2': chi2,
                        'p': p_value,
                        'df': df,
                        'V': effect_size
                    }
                    _print_statistical_report('chi-square', column, group_stats, test_stats)
                    textstr = f"V = {effect_size:.3f}\n{_format_p_value(p_value)}"
                else:
                    # Use Fisher's exact test for 2×2 table
                    odds_ratio, p_value = fisher_exact_test(counts)

                    # Prepare test statistics for console output
                    test_stats = {
                        'OR': odds_ratio,
                        'p': p_value
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('fisher', column, group_stats, test_stats)

                    # Add text box with odds ratio and p-value
                    textstr = f"OR = {odds_ratio:.3f}\n{_format_p_value(p_value)}"
            else:
                # Parametric: Chi-square test and Cramér's V
                effect_size = cramers_v(counts)
                chi2, p_value, df = chi_square_test(counts)

                # Prepare test statistics for console output
                test_stats = {
                    'chi2': chi2,
                    'p': p_value,
                    'df': df,
                    'V': effect_size
                }

                # Print comprehensive report to console
                _print_statistical_report('chi-square', column, group_stats, test_stats)

                # Add text box with Cramér's V and p-value
                textstr = f"V = {effect_size:.3f}\n{_format_p_value(p_value)}"

            # Display statistics box
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', multialignment='left', bbox=props)
    else:
        # Single bar chart (no grouping)
        counts = data[column].value_counts().sort_index()

        # Get readable labels
        x_labels = get_readable_labels(column, counts.index.tolist(), trunc=trunc)

        # Plot bars
        bars = ax.bar(range(len(counts)), counts.values, color=COLORS['primary'],
                     alpha=STYLE_CONFIG['hist_alpha'], edgecolor='white', linewidth=1)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Add percentage labels if requested
        if show_percentages:
            for i, (bar, count) in enumerate(zip(bars, counts.values)):
                height = bar.get_height()
                percentage = count / counts.sum() * 100
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:.0f}\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=STYLE_CONFIG['tick_size'])

    # Apply styling
    apply_consistent_style(ax, title=title, xlabel=(x_label if x_label is not None else column.capitalize()),
                          ylabel='Count', title_pad=title_pad)

    if group_by is not None:
        ax.legend(title=group_by.replace('_', ' ').title(), frameon=False,
                 fontsize=STYLE_CONFIG['font_size'])

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return ax


def plot_continuous_distribution(data: pd.DataFrame,
                                 column: str,
                                 title: Optional[str] = None,
                                 group_by: Optional[str] = None,
                                 ax: Optional[plt.Axes] = None,
                                 show_stats: bool = True,
                                 show_bars: bool = True,
                                 show_kde: bool = False,
                                 show_correlation: bool = True,
                                 bins: int = 30,
                                 title_pad: Optional[int] = None,
                                 test_method: str = 'parametric') -> plt.Axes:
    """
    Plot distribution of continuous variables (age, time, etc.).

    Shows histogram with optional KDE overlay. Optionally splits by experimental group.

    Args:
        data: DataFrame containing the data
        column: Column name to plot
        title: Plot title (auto-generated if None)
        group_by: Optional column to group by (e.g., 'stimulus_group')
        ax: Optional matplotlib axes (creates new figure if None)
        show_stats: Whether to show mean and SD in legend
        show_bars: Whether to show histogram bars (default: True)
        show_kde: Whether to show KDE (kernel density estimate) overlay (default: False)
        show_correlation: Whether to show effect size and p-value from statistical test between groups (default: True)
        bins: Number of histogram bins
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)
        test_method: Statistical test method ('parametric' or 'nonparametric').
                    - 'parametric': Uses t-test (2 groups) or ANOVA (3+ groups) with Cohen's d or η²
                    - 'nonparametric': Uses Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) with rank-biserial r
                    Default: 'parametric'

    Returns:
        The matplotlib axes object

    Example:
        >>> plot_continuous_distribution(data, 'age', group_by='stimulus_group', show_kde=True)
        >>> plot_continuous_distribution(data, 'page_submit', group_by='stimulus_group', show_correlation=True)
        >>> plot_continuous_distribution(data, 'page_submit', group_by='stimulus_group', show_correlation=False)
        >>> plot_continuous_distribution(data, 'age', group_by='stimulus_group', test_method='nonparametric')
        >>> plot_continuous_distribution(data, 'age', group_by='stimulus_group', show_bars=False, show_kde=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Generate title if not provided
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()}"

    # Remove stats from legend if kde plot is chosen
    if show_kde:
        show_stats = False

    # Check if grouping is requested
    if group_by is not None and group_by in data.columns:
        # Get unique groups
        groups = data[group_by].unique()

        # Store group data for effect size calculation
        group_means = {}
        group_info = {}  # Store group data, mean, std, color, label

        # First pass: collect all group data and calculate means
        for group in groups:
            group_data = data[data[group_by] == group][column].dropna()

            # Get readable label for group
            try:
                group_label = get_label_for_value(group_by, group)
            except (ValueError, KeyError):
                group_label = str(group)

            # Store group data for effect size
            group_means[group_label] = group_data

            # Choose color
            color = COLORS.get(group_label.lower(), COLORS['primary'])

            # Store group info
            group_info[group_label] = {
                'data': group_data,
                'mean': group_data.mean(),
                'std': group_data.std(),
                'color': color
            }

        # Determine label positioning based on mean comparison
        # Sort groups by mean to determine which has higher/lower mean
        sorted_groups = sorted(group_info.items(), key=lambda x: x[1]['mean'])

        # Second pass: plot with adjusted label positions
        for group in groups:
            # Get readable label for group
            try:
                group_label = get_label_for_value(group_by, group)
            except (ValueError, KeyError):
                group_label = str(group)

            info = group_info[group_label]
            group_data = info['data']
            mean = info['mean']
            std = info['std']
            color = info['color']

            # Add stats to label if requested
            label = group_label
            if show_stats:
                label += f" (M={mean:.1f}, SD={std:.1f})"

            # Determine label alignment based on whether this is the lower or higher mean group
            is_lower_mean = (group_label == sorted_groups[0][0])
            is_higher_mean = (group_label == sorted_groups[-1][0])

            # Calculate offset for label positioning (proportional to data range)
            data_range = data[column].max() - data[column].min()
            offset = data_range * 0.02  # 2% of data range

            # Plot histogram (optional)
            if show_bars:
                ax.hist(group_data, bins=bins, alpha=STYLE_CONFIG['hist_alpha'],
                       label=label, color=color, edgecolor='white', linewidth=0.5)

            # Plot KDE (optional)
            if show_kde:
                group_data.plot.kde(ax=ax, color=color, linewidth=STYLE_CONFIG['kde_linewidth'],
                                   label=label if not show_bars else None)

                # Add vertical line for mean
                ax.axvline(mean, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

                # Add text label for mean at top of plot with adjusted positioning
                y_top = ax.get_ylim()[1] * 0.95
                if is_lower_mean:
                    mean_ha = 'right'
                    mean_x = mean - offset
                elif is_higher_mean:
                    mean_ha = 'left'
                    mean_x = mean + offset
                else:
                    mean_ha = 'center'
                    mean_x = mean

                ax.text(mean_x, y_top, f'M={mean:.1f}', color=color, fontsize=STYLE_CONFIG['tick_size']-1,
                       ha=mean_ha, va='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

                # Add horizontal line for standard deviation
                y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                ax.hlines(y_pos, mean - std, mean + std, color=color, linewidth=3, alpha=0.7)

                # Add text label for SD with adjusted positioning
                if is_lower_mean:
                    sd_ha = 'right'
                    sd_x = mean - offset
                elif is_higher_mean:
                    sd_ha = 'left'
                    sd_x = mean + offset
                else:
                    sd_ha = 'center'
                    sd_x = mean

                ax.text(sd_x, y_pos, f'SD={std:.1f}', color=color, fontsize=STYLE_CONFIG['tick_size']-1,
                       ha=sd_ha, va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

        # Calculate and display effect size between groups if requested
        if show_correlation and len(group_means) >= 2:
            # Get all groups
            group_names = list(group_means.keys())

            if len(group_means) == 2:
                # Two groups: use parametric or non-parametric test
                group1_data = group_means[group_names[0]]
                group2_data = group_means[group_names[1]]

                if test_method == 'nonparametric':
                    # Non-parametric: Mann-Whitney U test and rank-biserial correlation
                    u_stat, p_value, n1, n2 = mann_whitney_u_test(group1_data, group2_data)
                    effect_size = rank_biserial_correlation(u_stat, n1, n2)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        'U': u_stat,
                        'p': p_value,
                        'r': effect_size
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('mann-whitney', column, group_stats, test_stats)

                    # Add text box with rank-biserial correlation (r) and p-value
                    textstr = f"r = {effect_size:.2f}\n{_format_p_value(p_value)}"
                else:
                    # Parametric: t-test and Cohen's d
                    effect_size = cohens_d(group1_data, group2_data)
                    t_stat, p_value, df = independent_t_test(group1_data, group2_data)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        't': t_stat,
                        'p': p_value,
                        'df': df,
                        'd': effect_size
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('t-test', column, group_stats, test_stats)

                    # Add text box with Cohen's d (δ) and p-value
                    textstr = f"δ = {effect_size:.2f}\n{_format_p_value(p_value)}"
            else:
                # Three or more groups: use parametric or non-parametric test
                all_group_data = [group_means[name] for name in group_names]

                if test_method == 'nonparametric':
                    # Non-parametric: Kruskal-Wallis H test
                    h_stat, p_value, df = kruskal_wallis_test(*all_group_data)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        'H': h_stat,
                        'p': p_value,
                        'df': df
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('kruskal-wallis', column, group_stats, test_stats)

                    # Add text box with H-statistic and p-value
                    textstr = f"H = {h_stat:.2f}\n{_format_p_value(p_value)}"
                else:
                    # Parametric: ANOVA and F-statistic
                    f_stat, p_value, df1, df2 = one_way_anova(*all_group_data)

                    # Prepare group statistics for console output
                    group_stats = {}
                    for name in group_names:
                        info = group_info[name]
                        group_stats[name] = {
                            'n': len(info['data']),
                            'mean': info['mean'],
                            'sd': info['std']
                        }

                    # Prepare test statistics for console output
                    test_stats = {
                        'F': f_stat,
                        'p': p_value,
                        'df1': df1,
                        'df2': df2
                    }

                    # Print comprehensive report to console
                    _print_statistical_report('anova', column, group_stats, test_stats)

                    # Add text box with F-statistic and p-value
                    textstr = f"F = {f_stat:.2f}\n{_format_p_value(p_value)}"

            # Display statistics in text box at top-right
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', multialignment='left', bbox=props)
    else:
        # Single distribution (no grouping)
        plot_data = data[column].dropna()

        label = f"Overall"
        if show_stats:
            label += f" (M={plot_data.mean():.1f}, SD={plot_data.std():.1f})"

        # Plot histogram (optional)
        if show_bars:
            ax.hist(plot_data, bins=bins, alpha=STYLE_CONFIG['hist_alpha'],
                   label=label, color=COLORS['primary'], edgecolor='white', linewidth=0.5)

        # Plot KDE (optional)
        if show_kde:
            plot_data.plot.kde(ax=ax, color=COLORS['primary'],
                              linewidth=STYLE_CONFIG['kde_linewidth'],
                              label=label if not show_bars else None)

            # Add vertical line for mean
            mean = plot_data.mean()
            ax.axvline(mean, color=COLORS['primary'], linestyle='--', linewidth=1.5, alpha=0.6)

            # Add horizontal line for standard deviation
            std = plot_data.std()
            y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            ax.hlines(y_pos, mean - std, mean + std, color=COLORS['primary'], linewidth=3, alpha=0.7)

    # Set x-axis limits based on actual data range
    # This ensures the plot shows only the range where data exists
    all_data = data[column].dropna()
    if len(all_data) > 0:
        data_min = all_data.min()
        data_max = all_data.max()
        # Add small padding (2% of range) for visual clarity
        data_range = data_max - data_min
        padding = data_range * 0.02 if data_range > 0 else 0.1
        ax.set_xlim(data_min - padding, data_max + padding)

    # Apply styling
    xlabel = column.replace('_', ' ').title()
    apply_consistent_style(ax, title=title, xlabel=xlabel,
                          ylabel='Density', title_pad=title_pad)
    ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'])

    return ax


def plot_boxplot(data: pd.DataFrame,
                 columns: List[str],
                 title: Optional[str] = None,
                 group_by: Optional[str] = None,
                 ax: Optional[plt.Axes] = None,
                 show_stats: bool = True,
                 use_full_labels: bool = False,
                 short_labels: Optional[List[str]] = None,
                 title_pad: Optional[int] = None,
                 mirror_hist: bool = False,
                 bins: Optional[int] = None,
                 show_mean: bool = False,
                 test_method: str = 'parametric') -> plt.Axes:
    """
    Plot boxplots or mirrored histograms for multiple items, optionally comparing experimental groups.

    When comparing groups, boxes/histograms are grouped by item with group comparisons side-by-side.
    All items share the same y-axis scale for easy comparison (ideal for Likert scales).

    Args:
        data: DataFrame containing the data
        columns: List of column names to plot (e.g., ['tia_rc', 'tia_up', 'tia_f'])
        title: Plot title (auto-generated if None)
        group_by: Optional column to group by (e.g., 'stimulus_group')
        ax: Optional matplotlib axes (creates new figure if None)
        show_stats: Whether to show effect size and p-value when comparing groups
        use_full_labels: If True, use full labels from labels.csv; if False, use short_labels or column names
        short_labels: Optional list of short labels for x-axis (must match length of columns)
        title_pad: Padding between title and plot (auto-increased when showing stats to prevent overlap)
        mirror_hist: If True, plot mirrored histograms instead of boxplots (ideal for Likert data) (default: False)
        bins: Number of bins for mirrored histograms (auto-detected from data if None) (default: None)
        show_mean: If True, mark the mean on mirrored histograms with a diamond marker (default: False)
        test_method: Statistical test method ('parametric' or 'nonparametric').
                    - 'parametric': Uses t-test (2 groups) or ANOVA (3+ groups) with Cohen's d or η²
                    - 'nonparametric': Uses Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) with rank-biserial r
                    Default: 'parametric'

    Returns:
        The matplotlib axes object

    Note:
        When show_stats=True and group_by is specified, title padding is automatically increased
        to prevent statistical annotations from overlapping with the title.

        Mirrored histograms are particularly useful for Likert scale data as they show discrete bins
        clearly (unlike violin plots which use continuous KDE).

    Example:
        >>> # Simple boxplot with short labels
        >>> plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f'])

        >>> # Compare groups with custom short labels
        >>> plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f'],
        ...              group_by='stimulus_group',
        ...              short_labels=['R/C', 'U/P', 'Fam'])

        >>> # Use full labels from labels.csv
        >>> plot_boxplot(data, ['tia_rc', 'tia_up'],
        ...              group_by='stimulus_group',
        ...              use_full_labels=True)

        >>> # Mirrored histograms (ideal for Likert scales)
        >>> plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f'],
        ...              group_by='stimulus_group',
        ...              mirror_hist=True)

        >>> # Mirrored histograms with mean markers
        >>> plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f'],
        ...              group_by='stimulus_group',
        ...              mirror_hist=True, show_mean=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Generate title if not provided
    if title is None:
        if group_by is not None:
            title = "Comparison Across Items"
        else:
            title = "Distribution Across Items"

    # Determine x-axis labels
    if use_full_labels:
        # Try to get full labels from labels.csv
        x_labels = []
        for col in columns:
            try:
                # Try to get the question text or label
                label = questions[questions['item'] == col]['question'].values
                if len(label) > 0:
                    x_labels.append(truncate_label(label[0], max_length=30))
                else:
                    x_labels.append(col.upper())
            except:
                x_labels.append(col.upper())
    elif short_labels is not None:
        if len(short_labels) != len(columns):
            raise ValueError(f"short_labels length ({len(short_labels)}) must match columns length ({len(columns)})")
        x_labels = short_labels
    else:
        # Use column names as default
        x_labels = [col.upper() for col in columns]

    # Prepare data for plotting
    if group_by is not None and group_by in data.columns:
        # Grouped boxplot
        groups = sorted(data[group_by].unique())
        n_groups = len(groups)
        n_items = len(columns)

        # Get readable labels for all groups first
        group_labels = []
        for group in groups:
            try:
                group_labels.append(get_label_for_value(group_by, group))
            except (ValueError, KeyError):
                group_labels.append(str(group))

        # Get colors for all groups dynamically
        group_colors = get_group_colors(group_labels)
        color_map = dict(zip(groups, group_colors))

        # Calculate positions for boxes
        # Each item gets a cluster of boxes (one per group)
        width = 0.35  # Width of each box
        positions = []
        group_positions = {group: [] for group in groups}

        for i, col in enumerate(columns):
            # Center of this item's cluster
            center = i * (n_groups * width + 0.5)
            for j, group in enumerate(groups):
                pos = center + (j - n_groups/2 + 0.5) * width
                positions.append(pos)
                group_positions[group].append(pos)

        # Store data for effect size calculation
        effect_sizes = []
        p_values = []

        # Compute histogram bins once for all groups (if using mirrored histograms)
        if mirror_hist:
            # Compute histogram range across all data for consistent bins
            all_col_data = pd.concat([data[col].dropna() for col in columns])
            hist_range = (all_col_data.min(), all_col_data.max())

            # Determine number of bins
            if bins is None:
                # Auto-detect from unique values (ideal for Likert scales)
                n_bins = len(all_col_data.unique())
            else:
                n_bins = bins

            # Compute bin edges (shared across all histograms and groups)
            bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
            bin_heights = np.diff(bin_edges)  # Height of each bar
            bin_centers = bin_edges[:-1] + bin_heights / 2  # Center y-position of each bar

            # Calculate global maximum count for normalization (shared scale across all histograms)
            global_max_count = 0
            for group in groups:
                for col in columns:
                    col_data = data[data[group_by] == group][col].dropna()
                    counts, _ = np.histogram(col_data, bins=bin_edges, range=hist_range)
                    global_max_count = max(global_max_count, counts.max())

            # Compute scale factor so widest bar fits within allocated width
            scale_factor = (width * 0.4) / global_max_count if global_max_count > 0 else 1.0

        # Plot boxes/histograms for each group
        for group_idx, group in enumerate(groups):
            # Get readable label and color for this group
            group_label = group_labels[group_idx]
            color = color_map[group]

            # Collect data for this group across all items
            group_data = []
            group_means = []
            for col in columns:
                col_data = data[data[group_by] == group][col].dropna()
                group_data.append(col_data)
                group_means.append(col_data.mean())

            if mirror_hist:
                # Plot mirrored histograms for this group
                # Plot histogram for each item in this group
                for col_idx, (col, x_pos) in enumerate(zip(columns, group_positions[group])):
                    col_data = group_data[col_idx]

                    # Compute histogram counts
                    counts, _ = np.histogram(col_data, bins=bin_edges, range=hist_range)

                    # Normalize counts using global scale factor
                    normalized_counts = counts * scale_factor

                    # Plot symmetric horizontal bars
                    lefts = x_pos - 0.5 * normalized_counts
                    ax.barh(bin_centers, normalized_counts, height=bin_heights, left=lefts,
                           color=color, alpha=STYLE_CONFIG['hist_alpha'],
                           edgecolor='black', linewidth=0.5)

                # Add mean markers if requested
                if show_mean:
                    ax.scatter(group_positions[group], group_means,
                             marker='D', color='white', s=50,
                             edgecolors=color, linewidths=2,
                             zorder=3, label=group_label if group_idx == 0 else '')

                # Add invisible line for legend
                if not show_mean:
                    ax.plot([], [], color=color, linewidth=10,
                           alpha=STYLE_CONFIG['hist_alpha'],
                           label=group_label)
            else:
                # Plot boxplots for this group
                bp = ax.boxplot(group_data,
                               positions=group_positions[group],
                               widths=width,
                               patch_artist=True,
                               labels=['' for _ in columns],  # No labels on individual boxes
                               showfliers=True,  # Show outliers
                               flierprops=dict(marker='o', markerfacecolor=color,
                                             markersize=4, alpha=0.5,
                                             markeredgecolor=color),
                               medianprops=dict(color='black', linewidth=1.5),
                               boxprops=dict(facecolor=color, alpha=STYLE_CONFIG['hist_alpha'],
                                           edgecolor='black', linewidth=1),
                               whiskerprops=dict(color='black', linewidth=1),
                               capprops=dict(color='black', linewidth=1))

                # Add to legend (just once per group)
                bp['boxes'][0].set_label(group_label)

                # Add mean markers if requested
                if show_mean:
                    ax.scatter(group_positions[group], group_means,
                             marker='D', color='white', s=50,
                             edgecolors=color, linewidths=2,
                             zorder=3)

        # Calculate effect sizes and p-values if comparing groups
        if show_stats and n_groups >= 2:
            # Print header for multiple item comparison
            print(f"\n{'='*60}")
            print(f"Multiple Item Comparison: {', '.join(columns)}")
            print(f"{'='*60}")

            for col_idx, col in enumerate(columns):
                # Collect data for all groups for this column
                all_group_data = [data[data[group_by] == group][col].dropna() for group in groups]

                # Find max value across all groups for positioning
                max_val = max(gdata.max() for gdata in all_group_data if len(gdata) > 0)

                # Position text above the item cluster
                text_x = col_idx * (n_groups * width + 0.5)
                text_y = max_val * 1.05  # 5% above max value

                if n_groups == 2:
                    # Two groups: use parametric or non-parametric test
                    group1_data = all_group_data[0]
                    group2_data = all_group_data[1]

                    if test_method == 'nonparametric':
                        # Non-parametric: Mann-Whitney U test and rank-biserial correlation
                        u_stat, p, n1, n2 = mann_whitney_u_test(group1_data, group2_data)
                        effect_size = rank_biserial_correlation(u_stat, n1, n2)

                        effect_sizes.append(effect_size)
                        p_values.append(p)

                        # Prepare group statistics for console output
                        group_stats = {}
                        for i, group_label in enumerate(group_labels):
                            group_stats[group_label] = {
                                'n': len(all_group_data[i]),
                                'mean': all_group_data[i].mean(),
                                'sd': all_group_data[i].std()
                            }

                        # Prepare test statistics for console output
                        test_stats = {
                            'U': u_stat,
                            'p': p,
                            'r': effect_size
                        }

                        # Print comprehensive report to console
                        _print_statistical_report('mann-whitney', col, group_stats, test_stats)

                        # Format text with rank-biserial correlation (r) and p-value
                        text_str = f"r = {effect_size:.3f}\n{_format_p_value(p)}"
                    else:
                        # Parametric: t-test and Cohen's d
                        d = cohens_d(group1_data, group2_data)
                        t_stat, p, df = independent_t_test(group1_data, group2_data)

                        effect_sizes.append(d)
                        p_values.append(p)

                        # Prepare group statistics for console output
                        group_stats = {}
                        for i, group_label in enumerate(group_labels):
                            group_stats[group_label] = {
                                'n': len(all_group_data[i]),
                                'mean': all_group_data[i].mean(),
                                'sd': all_group_data[i].std()
                            }

                        # Prepare test statistics for console output
                        test_stats = {
                            't': t_stat,
                            'p': p,
                            'df': df,
                            'd': d
                        }

                        # Print comprehensive report to console
                        _print_statistical_report('t-test', col, group_stats, test_stats)

                        # Format text with Cohen's d (δ) and p-value
                        text_str = f"δ = {d:.3f}\n{_format_p_value(p)}"
                else:
                    # Three or more groups: use parametric or non-parametric test
                    if test_method == 'nonparametric':
                        # Non-parametric: Kruskal-Wallis H test
                        h_stat, p, df = kruskal_wallis_test(*all_group_data)

                        effect_sizes.append(h_stat)  # Store H-statistic
                        p_values.append(p)

                        # Prepare group statistics for console output
                        group_stats = {}
                        for i, group_label in enumerate(group_labels):
                            group_stats[group_label] = {
                                'n': len(all_group_data[i]),
                                'mean': all_group_data[i].mean(),
                                'sd': all_group_data[i].std()
                            }

                        # Prepare test statistics for console output
                        test_stats = {
                            'H': h_stat,
                            'p': p,
                            'df': df
                        }

                        # Print comprehensive report to console
                        _print_statistical_report('kruskal-wallis', col, group_stats, test_stats)

                        # Format text with H-statistic and p-value
                        text_str = f"H = {h_stat:.3f}\n{_format_p_value(p)}"
                    else:
                        # Parametric: ANOVA and F-statistic
                        f_stat, p, df1, df2 = one_way_anova(*all_group_data)

                        effect_sizes.append(f_stat)  # Store F-statistic
                        p_values.append(p)

                        # Prepare group statistics for console output
                        group_stats = {}
                        for i, group_label in enumerate(group_labels):
                            group_stats[group_label] = {
                                'n': len(all_group_data[i]),
                                'mean': all_group_data[i].mean(),
                                'sd': all_group_data[i].std()
                            }

                        # Prepare test statistics for console output
                        test_stats = {
                            'F': f_stat,
                            'p': p,
                            'df1': df1,
                            'df2': df2
                        }

                        # Print comprehensive report to console
                        _print_statistical_report('anova', col, group_stats, test_stats)

                        # Format text with F-statistic and p-value
                        text_str = f"F = {f_stat:.3f}\n{_format_p_value(p)}"

                # Add text annotation above boxes
                ax.text(text_x, text_y, text_str,
                       ha='center', va='bottom',
                       fontsize=STYLE_CONFIG['tick_size'] - 1,
                       multialignment='left',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='wheat', alpha=0.3,
                               edgecolor='none'))

            # Print footer
            print(f"{'='*60}\n")

        # Set x-tick positions at center of each item cluster
        tick_positions = [i * (n_groups * width + 0.5) for i in range(n_items)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Add legend
        ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'])

    else:
        # Single boxplot/mirrored histogram (no grouping)
        plot_data = [data[col].dropna() for col in columns]
        plot_means = [data[col].dropna().mean() for col in columns]

        if mirror_hist:
            # Plot mirrored histograms
            # Compute histogram range across all data for consistent bins
            all_col_data = pd.concat([data[col].dropna() for col in columns])
            hist_range = (all_col_data.min(), all_col_data.max())

            # Determine number of bins
            if bins is None:
                # Auto-detect from unique values (ideal for Likert scales)
                n_bins = len(all_col_data.unique())
            else:
                n_bins = bins

            # Compute bin edges (shared across all histograms)
            bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
            bin_heights = np.diff(bin_edges)  # Height of each bar
            bin_centers = bin_edges[:-1] + bin_heights / 2  # Center y-position of each bar

            # Calculate global maximum count for normalization (shared scale across all histograms)
            global_max_count = 0
            for col_idx, col in enumerate(columns):
                counts_temp, _ = np.histogram(plot_data[col_idx], bins=bin_edges, range=hist_range)
                global_max_count = max(global_max_count, counts_temp.max())

            # Compute scale factor so widest bar fits within allocated width
            # For single plots, use a fixed width value similar to boxplots
            width_single = 0.35
            scale_factor = (width_single * 0.4) / global_max_count if global_max_count > 0 else 1.0

            # Plot histogram for each item
            for col_idx, col in enumerate(columns):
                col_data = plot_data[col_idx]
                x_pos = col_idx + 1  # Position 1, 2, 3, ...

                # Compute histogram counts
                counts, _ = np.histogram(col_data, bins=bin_edges, range=hist_range)

                # Normalize counts using global scale factor
                normalized_counts = counts * scale_factor

                # Plot symmetric horizontal bars
                lefts = x_pos - 0.5 * normalized_counts
                ax.barh(bin_centers, normalized_counts, height=bin_heights, left=lefts,
                       color=COLORS['primary'], alpha=STYLE_CONFIG['hist_alpha'],
                       edgecolor='black', linewidth=0.5)

            # Add mean markers if requested
            if show_mean:
                ax.scatter(range(1, len(columns) + 1), plot_means,
                         marker='D', color='white', s=50,
                         edgecolors=COLORS['primary'], linewidths=2,
                         zorder=3, label='Mean')
                ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'])

            # Set x-tick labels
            ax.set_xticks(range(1, len(columns) + 1))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        else:
            # Plot boxplots
            bp = ax.boxplot(plot_data,
                           labels=x_labels,
                           patch_artist=True,
                           showfliers=True,
                           flierprops=dict(marker='o', markerfacecolor=COLORS['primary'],
                                         markersize=4, alpha=0.5,
                                         markeredgecolor=COLORS['primary']),
                           medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(facecolor=COLORS['primary'],
                                       alpha=STYLE_CONFIG['hist_alpha'],
                                       edgecolor='black', linewidth=1),
                           whiskerprops=dict(color='black', linewidth=1),
                           capprops=dict(color='black', linewidth=1))

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Apply styling
    # Increase title padding if showing statistical annotations to prevent overlap
    if title_pad is None and group_by is not None and show_stats:
        title_pad = 35  # Extra space for statistical annotations above boxes

    apply_consistent_style(ax, title=title, xlabel='Items',
                          ylabel='Score', title_pad=title_pad)

    return ax


def plot_split_histogram_boxplot(data: pd.DataFrame,
                                  column: str,
                                  group_by: str,
                                  title: Optional[str] = None,
                                  ylabel: Optional[str] = None,
                                  ax: Optional[plt.Axes] = None,
                                  bins: Optional[int] = None,
                                  show_mean: bool = True,
                                  show_stats: bool = True,
                                  show_counts: bool = False,
                                  hist_scale: float = 0.4,
                                  title_pad: Optional[int] = None,
                                  test_method: str = 'parametric') -> plt.Axes:
    """
    Plot split histogram with central boxplot for comparing two groups on a single scale.

    Layout:
    - Left: Histogram of group 1 (extending leftward)
    - Center: Boxplots of both groups with optional mean markers
    - Right: Histogram of group 2 (extending rightward)

    All elements share the same y-axis scale, making it easy to compare distributions.
    Ideal for visualizing Likert scale data with group comparisons.

    Args:
        data: DataFrame containing the data
        column: Column name to plot (e.g., 'tia_rc')
        group_by: Column to group by (must have exactly 2 groups, e.g., 'stimulus_group')
        title: Plot title (auto-generated if None)
        ylabel: Optional custom y-axis label (auto-detected from column name if None)
        ax: Optional matplotlib axes (creates new figure if None)
        bins: Number of histogram bins (auto-detected from unique values if None)
        show_mean: If True, mark the mean on boxplots with a diamond marker (default: True)
        show_stats: Whether to show effect size and p-value (default: True)
        show_counts: If True, show "N (X%)" format; if False, show "X%" only (default: False)
        hist_scale: Maximum histogram width as proportion of plot (default: 0.4)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)
        test_method: Statistical test method ('parametric' or 'nonparametric').
                    - 'parametric': Uses t-test with Cohen's d
                    - 'nonparametric': Uses Mann-Whitney U with rank-biserial correlation r
                    Default: 'parametric'

    Returns:
        The matplotlib axes object

    Raises:
        ValueError: If group_by column does not have exactly 2 groups

    Example:
        >>> # Basic usage
        >>> plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group')

        >>> # With custom bins and wider histograms
        >>> plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
        ...                              bins=7, hist_scale=0.6)

        >>> # Without statistics
        >>> plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
        ...                              show_stats=False, show_mean=False)

        >>> # With absolute counts and percentages
        >>> plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
        ...                              show_counts=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Generate title if not provided
    if title is None:
        title = f"Distribution of {column.upper()}"

    # Validate that we have exactly 2 groups
    groups = sorted(data[group_by].dropna().unique())
    if len(groups) != 2:
        raise ValueError(f"group_by column '{group_by}' must have exactly 2 groups. Found {len(groups)}: {groups}")

    # Extract data for each group
    group1_data = data[data[group_by] == groups[0]][column].dropna()
    group2_data = data[data[group_by] == groups[1]][column].dropna()

    # Get readable labels for groups
    try:
        group1_label = get_label_for_value(group_by, groups[0])
    except (ValueError, KeyError):
        group1_label = str(groups[0])

    try:
        group2_label = get_label_for_value(group_by, groups[1])
    except (ValueError, KeyError):
        group2_label = str(groups[1])

    # Choose colors
    color1 = COLORS.get(group1_label.lower(), COLORS['primary'])
    color2 = COLORS.get(group2_label.lower(), COLORS['secondary'])

    # Setup y-axis scale (shared across all elements)
    all_data = pd.concat([group1_data, group2_data])
    data_range = (all_data.min(), all_data.max())

    # Determine number of bins
    if bins is None:
        # Auto-detect from unique values (ideal for Likert scales)
        n_bins = len(all_data.unique())
    else:
        n_bins = bins

    # Compute bin edges (shared y-axis)
    # Extend edges by 0.5 on each side to center bins on integer values
    # For Likert 1-5: bins at [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], centers at [1, 2, 3, 4, 5]
    bin_edges = np.linspace(data_range[0] - 0.5, data_range[1] + 0.5, n_bins + 1)
    bin_heights = np.diff(bin_edges)  # Height of each bar
    bin_centers = bin_edges[:-1] + bin_heights / 2  # Center y-position of each bar

    # Compute histogram counts for both groups
    hist_range = (bin_edges[0], bin_edges[-1])
    counts1, _ = np.histogram(group1_data, bins=bin_edges, range=hist_range)
    counts2, _ = np.histogram(group2_data, bins=bin_edges, range=hist_range)

    # Calculate normalization factor (shared scale for fair comparison)
    global_max_count = max(counts1.max(), counts2.max())
    scale_factor = hist_scale / global_max_count if global_max_count > 0 else 1.0

    # Normalize counts
    normalized_counts1 = counts1 * scale_factor
    normalized_counts2 = counts2 * scale_factor

    # Define boxplot positions (needed for histogram positioning calculations)
    boxplot_width = 0.25
    box_positions = [-0.15, 0.15]

    # Calculate histogram positions to avoid boxplot overlap and touch plot edges
    gap = 0.05  # Gap between histogram and boxplot
    left_hist_inner = box_positions[0] - boxplot_width/2 - gap
    right_hist_inner = box_positions[1] + boxplot_width/2 + gap

    # Anchor points for histograms (outer edges)
    left_anchor = left_hist_inner - hist_scale
    right_anchor = right_hist_inner + hist_scale

    # Plot left histogram (group 1, extending rightward towards center)
    # All bars start from left_anchor and extend right by their normalized count
    lefts1 = left_anchor
    ax.barh(bin_centers, normalized_counts1, height=bin_heights, left=lefts1,
           color=color1, alpha=STYLE_CONFIG['hist_alpha'],
           edgecolor='black', linewidth=0.5)

    # Plot right histogram (group 2, extending leftward towards center)
    # All bars end at right_anchor and extend left by their normalized count
    lefts2 = right_anchor - normalized_counts2
    ax.barh(bin_centers, normalized_counts2, height=bin_heights, left=lefts2,
           color=color2, alpha=STYLE_CONFIG['hist_alpha'],
           edgecolor='black', linewidth=0.5)

    # Add bin labels to histograms
    threshold = 0.15 * hist_scale  # Threshold for "too thin" bars
    label_offset = 0.02  # Offset for labels outside thin bars

    # Add labels to left histogram
    for i, (count, normalized_count) in enumerate(zip(counts1, normalized_counts1)):
        # Calculate percentage
        total_count1 = counts1.sum()
        percentage = (count / total_count1 * 100) if total_count1 > 0 else 0

        # Format label
        if show_counts:
            label = f"{int(count)}\n({percentage:.0f}%)"
        else:
            label = f"{percentage:.0f}%"

        # Determine position based on bar width
        if normalized_count >= threshold:
            # Inside bar, centered
            x_pos = left_anchor + normalized_count / 2
            ha = 'center'
        else:
            # Outside bar, towards center
            x_pos = left_anchor + normalized_count + label_offset
            ha = 'left'

        y_pos = bin_centers[i]

        # Add text
        ax.text(x_pos, y_pos, label, ha=ha, va='center',
               fontsize=STYLE_CONFIG['tick_size'] - 2,
               color='black', fontweight='bold')

    # Add labels to right histogram
    for i, (count, normalized_count) in enumerate(zip(counts2, normalized_counts2)):
        # Calculate percentage
        total_count2 = counts2.sum()
        percentage = (count / total_count2 * 100) if total_count2 > 0 else 0

        # Format label
        if show_counts:
            label = f"{int(count)}\n({percentage:.0f}%)"
        else:
            label = f"{percentage:.0f}%"

        # Determine position based on bar width
        if normalized_count >= threshold:
            # Inside bar, centered
            x_pos = right_anchor - normalized_count / 2
            ha = 'center'
        else:
            # Outside bar, towards center
            x_pos = right_anchor - normalized_count - label_offset
            ha = 'right'

        y_pos = bin_centers[i]

        # Add text
        ax.text(x_pos, y_pos, label, ha=ha, va='center',
               fontsize=STYLE_CONFIG['tick_size'] - 2,
               color='black', fontweight='bold')

    # Plot central boxplots
    bp = ax.boxplot([group1_data, group2_data],
                    positions=box_positions,
                    widths=boxplot_width,
                    patch_artist=True,
                    vert=True,  # Vertical boxplots
                    showfliers=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))

    # Color the boxplots
    bp['boxes'][0].set_facecolor(color1)
    bp['boxes'][0].set_alpha(STYLE_CONFIG['hist_alpha'])
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(1)

    bp['boxes'][1].set_facecolor(color2)
    bp['boxes'][1].set_alpha(STYLE_CONFIG['hist_alpha'])
    bp['boxes'][1].set_edgecolor('black')
    bp['boxes'][1].set_linewidth(1)

    # Color outliers
    bp['fliers'][0].set_markerfacecolor(color1)
    bp['fliers'][0].set_markeredgecolor(color1)
    bp['fliers'][1].set_markerfacecolor(color2)
    bp['fliers'][1].set_markeredgecolor(color2)

    # Add mean markers if requested
    if show_mean:
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()
        ax.scatter([box_positions[0], box_positions[1]], [mean1, mean2],
                 marker='D', color='white', s=50,
                 edgecolors=[color1, color2], linewidths=2,
                 zorder=3)

    # Add statistical annotations if requested
    box_shift = normalized_counts2[-1] *0.6 + 0.1  # amount by which the statistics box is shifted to the left
    if show_stats:
        # Prepare group statistics for console output (used by both parametric and non-parametric)
        group_stats = {
            group1_label: {
                'n': len(group1_data),
                'mean': group1_data.mean(),
                'sd': group1_data.std()
            },
            group2_label: {
                'n': len(group2_data),
                'mean': group2_data.mean(),
                'sd': group2_data.std()
            }
        }

        if test_method == 'nonparametric':
            # Non-parametric: Mann-Whitney U test and rank-biserial correlation
            u_stat, p_value, n1, n2 = mann_whitney_u_test(group1_data, group2_data)
            effect_size = rank_biserial_correlation(u_stat, n1, n2)

            # Prepare test statistics for console output
            test_stats = {
                'U': u_stat,
                'p': p_value,
                'r': effect_size
            }

            # Print comprehensive report to console
            _print_statistical_report('mann-whitney', column, group_stats, test_stats)

            # Add text box with rank-biserial correlation (r) and p-value
            textstr = f"r = {effect_size:.3f}\n{_format_p_value(p_value)}"
        else:
            # Parametric: t-test and Cohen's d
            effect_size = cohens_d(group1_data, group2_data)
            t_stat, p_value, df = independent_t_test(group1_data, group2_data)

            # Prepare test statistics for console output
            test_stats = {
                't': t_stat,
                'p': p_value,
                'df': df,
                'd': effect_size
            }

            # Print comprehensive report to console
            _print_statistical_report('t-test', column, group_stats, test_stats)

            # Add text box with Cohen's d (δ) and p-value
            textstr = f"δ = {effect_size:.3f}\n{_format_p_value(p_value)}"

        # Display statistics box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1-box_shift, 0.96, textstr, transform=ax.transAxes,
               fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
               horizontalalignment='right', multialignment='left', bbox=props)

    # Set axis limits (tight to histogram edges with small margin)
    margin = 0.05
    ax.set_xlim(left_anchor - margin, right_anchor + margin)

    # Set y-axis limits based on bin edges (already includes proper range for centered bins)
    ax.set_ylim(bin_edges[0], bin_edges[-1])

    # Apply styling
    if ylabel is None:
        # Auto-detect y-axis label from column name
        ylabel = column.replace('_', ' ').title()
        try:
            # Try to get better label from questions.csv
            question_label = questions[questions['item'] == column]['question'].values
            if len(question_label) > 0:
                ylabel = truncate_label(question_label[0], max_length=40)
        except:
            pass

    apply_consistent_style(ax, title=title, xlabel='',
                          ylabel=ylabel, title_pad=title_pad)

    # Remove x-axis ticks (not using tick labels)
    ax.set_xticks([])

    # Add group labels below x-axis (positioned at 25% and 75% of x-axis)
    # Calculate x-axis range and label positions
    x_min = left_anchor - margin
    x_max = right_anchor + margin
    x_range = x_max - x_min
    label_x_1 = x_min + 0.25 * x_range  # 25% from left edge
    label_x_2 = x_min + 0.75 * x_range  # 75% from left edge

    # Add vertical offset for spacing between x-axis and labels
    y_range = bin_edges[-1] - bin_edges[0]
    label_y_offset = 0.05 * y_range
    label_y_position = bin_edges[0] - label_y_offset

    ax.text(label_x_1, label_y_position, group1_label,
           ha='center', va='top',
           fontsize=STYLE_CONFIG['label_size'],
           color=color1,
           fontweight='bold')

    ax.text(label_x_2, label_y_position, group2_label,
           ha='center', va='top',
           fontsize=STYLE_CONFIG['label_size'],
           color=color2,
           fontweight='bold')

    return ax


def create_figure_grid(n_plots: int,
                      ncols: int = 2,
                      figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots for multiple visualizations.

    Args:
        n_plots: Number of plots needed
        ncols: Number of columns in the grid
        figsize: Optional figure size (auto-calculated if None)

    Returns:
        Tuple of (figure, axes_array)

    Example:
        >>> fig, axes = create_figure_grid(6, ncols=2)
        >>> plot_likert_distribution(data, 'ati', ax=axes[0, 0])
    """
    nrows = int(np.ceil(n_plots / ncols))

    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes array for easier indexing
    if n_plots == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    return fig, axes


def set_custom_colors(color_dict: dict) -> None:
    """
    Update the global color palette.

    Args:
        color_dict: Dictionary with color keys to update
                   (e.g., {'primary': '#FF5733', 'control': '#3498DB'})

    Example:
        >>> set_custom_colors({'primary': '#2C3E50', 'accent': '#E74C3C'})
    """
    global COLORS
    COLORS.update(color_dict)


def plot_scatterplot(data: pd.DataFrame,
                    x_column: str,
                    y_column: str,
                    title: Optional[str] = None,
                    group_by: Optional[str] = None,
                    ax: Optional[plt.Axes] = None,
                    show_stats: bool = True,
                    show_regression: bool = True,
                    correlation_method: str = 'pearson',
                    point_size: int = 50,
                    point_alpha: float = 0.6,
                    title_pad: Optional[int] = None) -> plt.Axes:
    """
    Plot scatterplot with correlation analysis for examining relationships between two variables.

    Shows scatter points with optional regression lines. Calculates Pearson's r or Spearman's ρ
    correlation. Optionally groups data by experimental condition to examine group-specific correlations.

    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis variable
        y_column: Column name for y-axis variable
        title: Plot title (auto-generated if None)
        group_by: Optional column to group by (e.g., 'stimulus_group')
        ax: Optional matplotlib axes (creates new figure if None)
        show_stats: Whether to show correlation statistics (default: True)
        show_regression: Whether to show regression line(s) (default: True)
        correlation_method: 'pearson' for linear correlation or 'spearman' for rank correlation (default: 'pearson')
        point_size: Size of scatter points (default: 50)
        point_alpha: Transparency of points, 0-1 (default: 0.6, helps show density)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)

    Returns:
        The matplotlib axes object

    Example:
        >>> # Basic scatterplot with Pearson correlation
        >>> plot_scatterplot(data, 'age', 'ati')

        >>> # Grouped by stimulus group with Spearman correlation
        >>> plot_scatterplot(data, 'TiA_rc', 'TiA_up',
        ...                  group_by='stimulus_group',
        ...                  correlation_method='spearman')

        >>> # Without regression lines, larger points
        >>> plot_scatterplot(data, 'age', 'page_submit',
        ...                  show_regression=False,
        ...                  point_size=80)

        >>> # Multiple scatterplots in a grid
        >>> fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        >>> plot_scatterplot(data, 'age', 'ati', ax=axes[0, 0])
        >>> plot_scatterplot(data, 'ati', 'TiA_t', ax=axes[0, 1])
        >>> plot_scatterplot(data, 'TiA_rc', 'TiA_up', ax=axes[1, 0])
        >>> plot_scatterplot(data, 'TiA_f', 'TiA_pro', ax=axes[1, 1])
        >>> plt.tight_layout()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Generate title if not provided
    if title is None:
        title = f"Correlation: {x_column.upper()} vs {y_column.upper()}"

    # Validate correlation method
    if correlation_method not in ['pearson', 'spearman']:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'")

    # Check if grouping is requested
    if group_by is not None and group_by in data.columns:
        # Grouped scatterplot
        groups = sorted(data[group_by].unique())

        # Get readable labels for all groups first
        group_labels = []
        for group in groups:
            try:
                group_labels.append(get_label_for_value(group_by, group))
            except (ValueError, KeyError):
                group_labels.append(str(group))

        # Get colors for all groups dynamically
        group_colors = get_group_colors(group_labels)
        color_map = dict(zip(groups, group_colors))

        # Plot scatter for each group
        for group_idx, group in enumerate(groups):
            group_label = group_labels[group_idx]
            color = color_map[group]

            # Filter data for this group
            group_mask = data[group_by] == group
            x_data = data[group_mask][x_column].dropna()
            y_data = data[group_mask][y_column].dropna()

            # Align x and y data (remove NaN from either variable)
            combined = pd.concat([x_data, y_data], axis=1).dropna()
            x_clean = combined.iloc[:, 0]
            y_clean = combined.iloc[:, 1]

            # Plot scatter points
            ax.scatter(x_clean, y_clean,
                      color=color, alpha=point_alpha, s=point_size,
                      label=group_label, edgecolors='black', linewidth=0.5)

            # Plot regression line if requested
            if show_regression and len(x_clean) > 1:
                # Calculate linear regression
                coefficients = np.polyfit(x_clean, y_clean, 1)
                poly_func = np.poly1d(coefficients)

                # Create smooth line
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = poly_func(x_line)

                # Plot regression line (slightly thicker, semi-transparent)
                ax.plot(x_line, y_line, color=color, alpha=0.8,
                       linewidth=2, linestyle='--')

        # Calculate and display per-group correlations if requested
        if show_stats:
            # Prepare text for all groups
            text_lines = []
            for group_idx, group in enumerate(groups):
                group_label = group_labels[group_idx]

                # Filter and align data for this group
                group_mask = data[group_by] == group
                x_data = data[group_mask][x_column]
                y_data = data[group_mask][y_column]

                # Calculate correlation
                if correlation_method == 'pearson':
                    corr_coef, p_value = pearson_correlation(x_data, y_data)
                    # Prepare stats for console output
                    corr_stats = {
                        'r': corr_coef,
                        'p': p_value,
                        'n': len(pd.concat([x_data, y_data], axis=1).dropna())
                    }
                else:  # spearman
                    corr_coef, p_value = spearman_correlation(x_data, y_data)
                    # Prepare stats for console output
                    corr_stats = {
                        'rho': corr_coef,
                        'p': p_value,
                        'n': len(pd.concat([x_data, y_data], axis=1).dropna())
                    }

                # Print to console
                _print_correlation_report(x_column, y_column, correlation_method,
                                        corr_stats, group_name=group_label)

                # Add to text box
                symbol = 'r' if correlation_method == 'pearson' else 'ρ'
                text_lines.append(f"{group_label}: {symbol} = {corr_coef:.3f}, {_format_p_value(p_value)}")

            # Display all correlations in a single text box at top-right
            textstr = '\n'.join(text_lines)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', multialignment='left', bbox=props)

        # Add legend
        ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'], loc='best')

    else:
        # Single scatterplot (no grouping)
        x_data = data[x_column].dropna()
        y_data = data[y_column].dropna()

        # Align x and y data
        combined = pd.concat([x_data, y_data], axis=1).dropna()
        x_clean = combined.iloc[:, 0]
        y_clean = combined.iloc[:, 1]

        # Plot scatter points
        ax.scatter(x_clean, y_clean,
                  color=COLORS['primary'], alpha=point_alpha, s=point_size,
                  edgecolors='black', linewidth=0.5)

        # Plot regression line if requested
        if show_regression and len(x_clean) > 1:
            # Calculate linear regression
            coefficients = np.polyfit(x_clean, y_clean, 1)
            poly_func = np.poly1d(coefficients)

            # Create smooth line
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_line = poly_func(x_line)

            # Plot regression line
            ax.plot(x_line, y_line, color=COLORS['primary'], alpha=0.8,
                   linewidth=2, linestyle='--', label='Regression line')

        # Calculate and display correlation if requested
        if show_stats:
            # Calculate correlation
            if correlation_method == 'pearson':
                corr_coef, p_value = pearson_correlation(data[x_column], data[y_column])
                symbol = 'r'
                # Prepare stats for console output
                corr_stats = {
                    'r': corr_coef,
                    'p': p_value,
                    'n': len(combined)
                }
            else:  # spearman
                corr_coef, p_value = spearman_correlation(data[x_column], data[y_column])
                symbol = 'ρ'
                # Prepare stats for console output
                corr_stats = {
                    'rho': corr_coef,
                    'p': p_value,
                    'n': len(combined)
                }

            # Print to console
            _print_correlation_report(x_column, y_column, correlation_method, corr_stats)

            # Display correlation in text box at top-right
            textstr = f"{symbol} = {corr_coef:.3f}\n{_format_p_value(p_value)}\nn = {len(combined)}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', multialignment='left', bbox=props)

        # Add legend if regression line is shown
        if show_regression and len(x_clean) > 1:
            ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'], loc='best')

    # Set axis labels (try to get readable labels from metadata)
    try:
        xlabel = get_label_for_value('column_labels', x_column)
    except (ValueError, KeyError):
        xlabel = x_column.replace('_', ' ').title()

    try:
        ylabel = get_label_for_value('column_labels', y_column)
    except (ValueError, KeyError):
        ylabel = y_column.replace('_', ' ').title()

    # Apply consistent styling
    apply_consistent_style(ax, title=title, xlabel=xlabel,
                          ylabel=ylabel, title_pad=title_pad)

    return ax


def _print_noninferiority_report(variable_name: str,
                                  mean_diff: float,
                                  se: float,
                                  ci_lower: float,
                                  ci_upper: float,
                                  sesoi_position: float,
                                  z_score: float,
                                  p_value: float,
                                  alpha: float,
                                  non_inferior: bool,
                                  test_type: str) -> None:
    """
    Print non-inferiority test results to console in APA-style format.

    This internal helper function formats and displays non-inferiority test results
    in a format ready for copy-pasting into manuscripts.

    Args:
        variable_name: Name of the variable being tested
        mean_diff: Observed mean difference
        se: Standard error of the mean difference
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        sesoi_position: SESOI margin position (negative for lower test)
        z_score: Z-score for the test
        p_value: P-value for the test
        alpha: Significance level
        non_inferior: Whether non-inferiority was established
        test_type: Type of test ('lower' or 'upper')
    """
    # Format p-value in APA style (no leading zero)
    def format_p(p_value: float) -> str:
        if p_value < 0.001:
            return "< .001"
        else:
            return f"= {p_value:.3f}".replace("0.", ".")

    print(f"\nNon-Inferiority Test Results for '{variable_name}':")
    if se is not None:
        print(f"  Mean difference: M_diff = {mean_diff:.3f}, SE = {se:.3f}")
    else:
        print(f"  Effect size: {mean_diff:.4f}")
    print(f"  {(1-alpha)*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  SESOI margin: {sesoi_position:.4f}")
    if z_score is not None and p_value is not None:
        print(f"  z = {z_score:.3f}, p {format_p(p_value)}")
    elif p_value is not None:
        print(f"  p {format_p(p_value)}")
    else:
        print(f"  (p-value: N/A)")

    if non_inferior:
        if p_value is not None:
            print(f"  Result: Non-inferiority established (p {format_p(p_value)})")
        else:
            print(f"  Result: Non-inferiority established")
    else:
        if p_value is not None:
            print(f"  Result: Non-inferiority not established (p {format_p(p_value)})")
        else:
            print(f"  Result: Non-inferiority not established")

    print()  # Blank line separator


def plot_noninferiority_test(effect_size,
                             sesoi,
                             se=None,
                             alpha: float = 0.05,
                             test_type: str = 'lower',
                             mode : str = 'non-inferiority',
                             title: Optional[str] = None,
                             xlabel: Optional[str] = None,
                             ax: Optional[plt.Axes] = None,
                             title_pad: Optional[int] = None,
                             show_stats: bool = True,
                             variable_names: Optional[List[str]] = None,
                             variable_labels: Optional[dict] = None,
                             categories: Optional[dict] = None,
                             category_order: Optional[List[str]] = None,
                             ci_lower_bounds=None,
                             ci_upper_bounds=None,
                             p_values=None,
                             estimate_name = 'μ',
                             column_title_pos_shift: float = 0,
                             height_scale: float = 1.0) -> plt.Axes:
    """
    Visualize non-inferiority test as a forest plot (blobbogram).

    Always uses forest plot style with horizontal CI lines and markers, regardless of
    whether single or multiple variables are provided. This ensures uniform visualization.

    Features:
    - Each variable shown as horizontal CI line with marker at point estimate
    - Variable names displayed on the left (inferred from variable_labels or variable_names)
    - Statistics displayed in columns on the right: mean difference, CI bounds, p-value, verdict
    - Statistics printed to console in APA format
    - Optional categorical grouping with bold headers
    - Supports pre-computed CI bounds for non-normal distributions (e.g., F-distribution)

    For a lower non-inferiority test, non-inferiority is established when the lower
    confidence bound (alpha-percentile) is greater than the SESOI margin.

    Args:
        effect_size: Observed effect size(s), e.g. mean difference(s). Float for single, List[float] for multiple
        sesoi: SESOI margin(s) - positive value(s). Float for single, List[float] for multiple
        se: Standard error(s) of mean difference. Float for single, List[float] for multiple.
            Optional if ci_lower_bounds and ci_upper_bounds are provided.
        alpha: Significance level for the test (default: 0.05)
        test_type: Type of non-inferiority test:
                  - 'lower': Test if new is not worse (mean_diff > -sesoi)
                  - 'upper': Test if new is not better (mean_diff < sesoi)
                  Default: 'lower'
        title: Plot title (auto-generated if None)
        xlabel: X-axis label (default: "Effect Size")
        ax: Optional matplotlib axes (creates new figure if None)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)
        show_stats: Whether to print test statistics to console (default: True)
        variable_names: List of variable names for y-axis labels (uses defaults if None)
        variable_labels: Dict mapping variable names to display labels (e.g., scale_titles)
        categories: Dict mapping category names to lists of variable names for grouping.
                   Variables in each category will be bundled with a bold category header.
                   Example: {'Trust': ['tia_t', 'tia_rc'], 'Understanding': ['tia_up']}
        category_order: List of category names specifying the order to display categories.
                       If None, uses dictionary iteration order (Python 3.7+ preserves insertion order).
                       Example: ['Trust', 'Understanding', 'Other']
        ci_lower_bounds: Pre-computed lower CI bound(s). Float for single, List[float] for multiple.
                        If provided with ci_upper_bounds, 'se' is not required.
                        Useful for non-normal distributions (e.g., F-distribution for Pillai's V).
        ci_upper_bounds: Pre-computed upper CI bound(s). Float for single, List[float] for multiple.
                        Must be provided together with ci_lower_bounds.
        p_values: Pre-computed p-value(s). Float for single, List[float] for multiple.
                 If not provided and 'se' is available, p-values are computed assuming normality.
                 If not provided and 'se' is None, p-values are shown as 'N/A'.

    Returns:
        The matplotlib axes object

    Example:
        >>> # Single variable
        >>> plot_noninferiority_test(effect_size=0.15, sesoi=0.3, se=0.12, alpha=0.05)

        >>> # Multiple variables (forest plot)
        >>> plot_noninferiority_test(
        ...     effect_size=[0.15, -0.03, -0.10],
        ...     sesoi=[0.15, 0.11, 0.09],
        ...     se=[0.11, 0.08, 0.07],
        ...     variable_names=['tia_f', 'tia_pro', 'tia_rc'],
        ...     variable_labels={'tia_f': 'Familiarity', 'tia_pro': 'Propensity'},
        ...     title='Non-Inferiority Analysis: Trust in Automation'
        ... )

        >>> # With pre-computed CI bounds (for non-normal distributions)
        >>> plot_noninferiority_test(
        ...     effect_size=0.02,
        ...     sesoi=0.05,
        ...     ci_lower_bounds=0.0,
        ...     ci_upper_bounds=0.06,
        ...     p_values=0.41,
        ...     variable_names=['Pillai V'],
        ...     title='Multivariate Non-Inferiority Test'
        ... )

        >>> # With categorical grouping
        >>> plot_noninferiority_test(
        ...     effect_size=[0.15, -0.03, -0.10, 0.05],
        ...     sesoi=[0.15, 0.11, 0.09, 0.12],
        ...     se=[0.11, 0.08, 0.07, 0.09],
        ...     variable_names=['tia_t', 'tia_rc', 'tia_up', 'tia_f'],
        ...     categories={'Trust': ['tia_t', 'tia_rc'], 'Understanding': ['tia_up', 'tia_f']},
        ...     category_order=['Trust', 'Understanding'],  # Specify order
        ...     title='Non-Inferiority Analysis by Category'
        ... )

    Notes:
        - For 'lower' non-inferiority: H0: μ_diff ≤ -sesoi vs H1: μ_diff > -sesoi
        - For 'upper' non-inferiority: H0: μ_diff ≥ sesoi vs H1: μ_diff < sesoi
        - The SESOI (margin) should always be provided as a positive value
        - Statistics are shown both in the plot and printed to console
        - When using pre-computed CIs, ensure they match the alpha level specified
        :param estimate_name:
    """
    # Detect mode (single vs multiple)
    is_multiple = hasattr(effect_size, '__iter__') and not isinstance(effect_size, str)

    # Check if pre-computed CIs are provided
    has_precomputed_ci = ci_lower_bounds is not None and ci_upper_bounds is not None

    # Convert to lists for unified processing
    if is_multiple:
        mean_diff_list = list(effect_size)
        sesoi_list = list(sesoi) if hasattr(sesoi, '__iter__') else [sesoi] * len(mean_diff_list)

        # Handle SE (optional when pre-computed CIs are provided)
        if se is not None:
            se_list = list(se) if hasattr(se, '__iter__') else [se] * len(mean_diff_list)
        else:
            se_list = [None] * len(mean_diff_list)

        # Handle pre-computed CI bounds
        if has_precomputed_ci:
            ci_lower_list = list(ci_lower_bounds) if hasattr(ci_lower_bounds, '__iter__') else [ci_lower_bounds] * len(mean_diff_list)
            ci_upper_list = list(ci_upper_bounds) if hasattr(ci_upper_bounds, '__iter__') else [ci_upper_bounds] * len(mean_diff_list)
        else:
            ci_lower_list = [None] * len(mean_diff_list)
            ci_upper_list = [None] * len(mean_diff_list)

        # Handle pre-computed p-values
        if p_values is not None:
            p_value_list = list(p_values) if hasattr(p_values, '__iter__') else [p_values] * len(mean_diff_list)
        else:
            p_value_list = [None] * len(mean_diff_list)

        n_vars = len(mean_diff_list)

        # Validate variable_names
        if variable_names is None:
            variable_names = [f"Var{i+1}" for i in range(n_vars)]
        elif len(variable_names) != n_vars:
            raise ValueError(f"variable_names must have length {n_vars}")
    else:
        # Single mode: wrap in lists for unified processing
        mean_diff_list = [effect_size]
        sesoi_list = [sesoi]
        se_list = [se] if se is not None else [None]
        ci_lower_list = [ci_lower_bounds] if ci_lower_bounds is not None else [None]
        ci_upper_list = [ci_upper_bounds] if ci_upper_bounds is not None else [None]
        p_value_list = [p_values] if p_values is not None else [None]
        n_vars = 1

        if variable_names is None:
            variable_names = ["Effect"]

    # Validate inputs
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if test_type not in ['lower', 'upper']:
        raise ValueError("test_type must be 'lower' or 'upper'")

    # Validate that either SE or pre-computed CIs are provided
    if not has_precomputed_ci and se is None:
        raise ValueError("Either 'se' or both 'ci_lower_bounds' and 'ci_upper_bounds' must be provided")

    for i, s in enumerate(sesoi_list):
        if s <= 0:
            raise ValueError(f"SESOI (margin) must be positive (index {i}: {s})")

    # Validate SE if provided
    if se is not None:
        for i, se_val in enumerate(se_list):
            if se_val is not None and se_val <= 0:
                raise ValueError(f"Standard error must be positive (index {i}: {se_val})")

    # Create figure if needed (will adjust height after category processing)
    # Always use forest plot style sizing (uniform visualization)
    if ax is None:
        # Height scales with number of variables, minimum height of 8 for readability
        fig_height = max(8, 2 + 1.5 * n_vars) * height_scale
        fig, ax = plt.subplots(figsize=(10, fig_height))

    # Calculate statistics for all distributions
    stats_data = []
    for i in range(n_vars):
        md = mean_diff_list[i]
        s = sesoi_list[i]
        se_val = se_list[i]
        precomputed_ci_lower = ci_lower_list[i]
        precomputed_ci_upper = ci_upper_list[i]
        precomputed_p = p_value_list[i]

        # Determine SESOI position based on test type
        if test_type == 'lower':
            sesoi_position = -s
            zone_direction = 'right'
        else:
            sesoi_position = s
            zone_direction = 'left'

        # Compute CI bounds: use pre-computed if available, otherwise calculate from SE
        if precomputed_ci_lower is not None and precomputed_ci_upper is not None:
            ci_lower = precomputed_ci_lower
            ci_upper = precomputed_ci_upper
        elif se_val is not None:
            ci_lower = md - stats.norm.ppf(1 - alpha/2) * se_val
            ci_upper = md + stats.norm.ppf(1 - alpha/2) * se_val
        else:
            raise ValueError(f"No CI bounds available for index {i}: provide either 'se' or pre-computed bounds")

        # Calculate p-value: use pre-computed if available, otherwise calculate from SE
        if precomputed_p is not None:
            p_value = precomputed_p
            z_score = None  # Not applicable when using pre-computed p-value
        elif se_val is not None:
            z_score = (md - sesoi_position) / se_val
            p_value = 1 - stats.norm.cdf(z_score) if test_type == 'lower' else stats.norm.cdf(z_score)
        else:
            p_value = None  # Cannot compute without SE
            z_score = None

        # Determine non-inferiority based on CI bounds
        if test_type == 'lower':
            non_inferior = ci_lower >= sesoi_position
        else:
            non_inferior = ci_upper <= sesoi_position


        # Determine three-level verdict
        if non_inferior:
            if mode == 'non-inferiority':
                verdict = 'Non-inferior'
            elif mode == 'equivalence':
                verdict = 'Equal'
            else:
                raise ValueError(f"Unknown mode '{mode}'")
        elif ci_lower > 0 or ci_upper < 0:
            # Zero is outside the CI - effect is significantly different from zero
            verdict = 'Different'
        else:
            # Neither non-inferior nor significantly different
            verdict = 'Inconclusive'

        stats_data.append({
            'mean_diff': md,
            'sesoi': s,
            'se': se_val,
            'sesoi_position': sesoi_position,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_score': z_score,
            'p_value': p_value,
            'non_inferior': non_inferior,
            'verdict': verdict,
            'zone_direction': zone_direction
        })

        # Print statistics to console if requested
        if show_stats:
            var_label = variable_names[i]
            if variable_labels and var_label in variable_labels:
                var_label = variable_labels[var_label]

            _print_noninferiority_report(
                variable_name=var_label,
                mean_diff=md,
                se=se_val,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sesoi_position=sesoi_position,
                z_score=z_score,
                p_value=p_value,
                alpha=alpha,
                non_inferior=non_inferior,
                test_type=test_type
            )

    # Check if all p-values are missing (None)
    all_p_values_missing = all(s['p_value'] is None for s in stats_data)

    # Handle categories: organize variables and create display order
    display_order = []  # List of (var_index, category_name, is_first_in_category)
    var_to_category = {}  # Maps var_idx to category name

    if is_multiple and categories is not None:
        # Validate categories
        all_categorized_vars = []
        for cat_name, var_list in categories.items():
            all_categorized_vars.extend(var_list)

        # Check that all variables are accounted for
        for var_name in variable_names:
            if var_name not in all_categorized_vars:
                raise ValueError(f"Variable '{var_name}' not found in any category. All variables must be categorized when using categories parameter.")

        # Determine category order
        if category_order is not None:
            # Validate category_order
            if set(category_order) != set(categories.keys()):
                missing = set(categories.keys()) - set(category_order)
                extra = set(category_order) - set(categories.keys())
                raise ValueError(f"category_order must contain exactly the same categories as categories dict. Missing: {missing}, Extra: {extra}")
            ordered_categories = category_order
        else:
            # Use dictionary order (Python 3.7+ preserves insertion order)
            ordered_categories = list(categories.keys())

        # Build display order: just variables with category info
        for cat_name in ordered_categories:
            var_list = categories[cat_name]
            for i, var_name in enumerate(var_list):
                var_idx = variable_names.index(var_name)
                is_first = (i == 0)  # First variable in this category
                display_order.append((var_idx, cat_name, is_first))
                var_to_category[var_idx] = cat_name
    else:
        # No categories: simple order
        for i in range(n_vars):
            display_order.append((i, None, False))

    # Calculate global x-axis range using CI bounds (works for any distribution)
    all_sesoi_positions = [s['sesoi_position'] for s in stats_data]
    all_mean_diffs = [s['mean_diff'] for s in stats_data]
    all_ci_lowers = [s['ci_lower'] for s in stats_data]
    all_ci_uppers = [s['ci_upper'] for s in stats_data]

    # Calculate range based on CI bounds (more robust than SE-based calculation)
    data_min = min(min(all_ci_lowers), min(all_sesoi_positions), 0)
    data_max = max(max(all_ci_uppers), max(all_sesoi_positions), 0)
    data_range = data_max - data_min
    x_margin = data_range * 0.15

    x_min = data_min - x_margin
    x_max = data_max + x_margin

    # Forest plot style: horizontal CI lines with markers (always used)
    unit_spacing = 0.7  # Uniform spacing between ALL elements (categories + variables)
    top_margin = 1.0  # Extra space at top for legend
    bottom_margin = 0.3  # Space at bottom

    # ===================================================================
    # PHASE 1: Build element list and calculate even spacing
    # ===================================================================
    # Build complete list of all elements (categories AND variables)
    elements = []  # List of ('type', data) tuples
    element_positions = {}  # Maps element index to y-position

    if categories is not None:
        # Add categories and their variables in order
        for cat_name in ordered_categories:
            # Add category header as an element
            elements.append(('category', cat_name))

            # Add all variables in this category
            for var_name in categories[cat_name]:
                var_idx = variable_names.index(var_name)
                elements.append(('variable', var_idx))
    else:
        # No categories: only variables
        for i in range(n_vars):
            elements.append(('variable', i))

    # Calculate evenly-spaced positions for ALL elements
    # Start from bottom (matplotlib y=0 is at bottom)
    # First element (index 0) should be at TOP, last element at bottom
    n_elements = len(elements)

    # Adjust figure size based on number of elements
    if ax.figure is not None:
        new_height = max(8, 0.5 * n_elements + 2) * height_scale  # Scale with total elements, min 8 for readability
        ax.figure.set_size_inches(18, new_height)

    for i in range(n_elements):
        # Position from bottom: first element at top, last at bottom
        y_pos = bottom_margin + (n_elements - 1 - i) * unit_spacing + top_margin
        element_positions[i] = y_pos

    # ===================================================================
    # PHASE 2: Render all elements
    # ===================================================================

    # Render each element based on its type
    for i, (elem_type, elem_data) in enumerate(elements):
        y_pos = element_positions[i]

        if elem_type == 'category':
            # Render category header only (no plot elements)
            cat_name = elem_data
            ax.text(-0.25, y_pos, cat_name, transform=ax.get_yaxis_transform(),
                   fontsize=STYLE_CONFIG['font_size'] + 1, weight='bold',
                   va='center', ha='left')

        elif elem_type == 'variable':
            # Render everything for this variable
            var_idx = elem_data

            # Get statistics
            md = stats_data[var_idx]['mean_diff']
            ci_lower = stats_data[var_idx]['ci_lower']
            ci_upper = stats_data[var_idx]['ci_upper']

            # Plot horizontal CI line
            ax.plot([ci_lower, ci_upper], [y_pos, y_pos],
                   color='#4A4A4A', linewidth=2, solid_capstyle='butt', zorder=2)

            # Plot marker at point estimate
            ax.plot(md, y_pos, marker='s', markersize=8,
                   color='#4A4A4A', markerfacecolor='#4A4A4A',
                   markeredgewidth=0, zorder=3)

            # Variable label on the left
            var_name = variable_names[var_idx]
            var_label = variable_labels.get(var_name, var_name) if variable_labels else var_name
            ax.text(-0.02, y_pos, var_label, transform=ax.get_yaxis_transform(),
                   fontsize=STYLE_CONFIG['font_size'], va='center', ha='right')

    # Calculate proper x-axis limits for plot area (excluding statistics)
    # Find max CI upper bound and all SESOI positions across all variables
    all_ci_uppers = [stats_data[i]['ci_upper'] for i in range(n_vars)]
    all_ci_lowers = [stats_data[i]['ci_lower'] for i in range(n_vars)]
    all_sesoi_pos = [stats_data[i]['sesoi_position'] for i in range(n_vars)]

    data_x_max = max(all_ci_uppers)
    data_x_min = min(all_ci_lowers)

    # Add some margin for plot area
    plot_margin = (data_x_max - data_x_min) * 0.15
    plot_x_max = data_x_max + plot_margin
    plot_x_min = data_x_min - plot_margin

    # Ensure we include zero and all sesoi positions
    plot_x_min = min(plot_x_min, 0, min(all_sesoi_pos)) - max(abs(s) for s in all_sesoi_pos) * 0.1
    plot_x_max = max(plot_x_max, 0, max(all_sesoi_pos)) + max(abs(s) for s in all_sesoi_pos) * 0.1

    y_min = min(element_positions.values()) - 0.3
    y_max = max(element_positions.values()) + 0.3

    # Mark zero with dotted line
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)

    # Plot individual non-inferiority margins for each VARIABLE element only
    first_margin_plotted = False
    for i, (elem_type, elem_data) in enumerate(elements):
        if elem_type != 'variable':
            continue  # Skip category elements

        y_pos = element_positions[i]
        var_idx = elem_data
        sesoi_pos = stats_data[var_idx]['sesoi_position']

        # Draw short vertical line at this variable's SESOI position
        y_line_height = unit_spacing * 0.3
        label = 'Non-inferiority margin' if not first_margin_plotted else None
        ax.plot([sesoi_pos, sesoi_pos], [y_pos - y_line_height/2, y_pos + y_line_height/2],
               color='#87CEEB', linestyle='-', linewidth=2.5, alpha=0.6, zorder=1, label=label)

        # Add value label to the left of the line (at top of line)
        label_offset = (plot_x_max - plot_x_min) * 0.02  # Small offset to the left
        ax.text(sesoi_pos - label_offset, y_pos + y_line_height/2, f'{abs(sesoi_pos):.4f}',
               fontsize=STYLE_CONFIG['font_size'] - 2, color='#87CEEB',
               va='center', ha='right', alpha=0.8)

        first_margin_plotted = True

    # Set x-axis limits for plot area only (statistics will be outside)
    ax.set_xlim(plot_x_min, plot_x_max)

    # Add statistics columns on the right side (using axis transform coordinates)
    # This positions them outside the plot area
    # Conditionally exclude p-value column if all p-values are missing
    if all_p_values_missing:
        stats_x_positions = [1.05, 1.20, 1.35]  # Relative to axis (0-1 scale), more spread out
        stat_labels = [estimate_name, 'CI', 'Verdict']
    else:
        stats_x_positions = [1.05, 1.20, 1.35, 1.45]  # Relative to axis (0-1 scale), more spread out
        stat_labels = [estimate_name, 'CI', 'p', 'Verdict']

    # Column headers (using axis transform: x is relative 0-1, y is data coordinates)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    # Position headers at top of figure (y_max), with optional shift
    headers_y = y_max + column_title_pos_shift
    for stat_x, label in zip(stats_x_positions, stat_labels):
        ax.text(stat_x, headers_y, label, fontsize=STYLE_CONFIG['font_size'],
               weight='bold', ha='center', va='bottom', transform=trans)

    # Add statistics for each VARIABLE element only (using same transform)
    for i, (elem_type, elem_data) in enumerate(elements):
        if elem_type != 'variable':
            continue  # Skip category elements

        y_pos = element_positions[i]
        var_idx = elem_data

        md = stats_data[var_idx]['mean_diff']
        ci_lower = stats_data[var_idx]['ci_lower']
        ci_upper = stats_data[var_idx]['ci_upper']
        p_value = stats_data[var_idx]['p_value']
        verdict_text = stats_data[var_idx]['verdict']

        # Format p-value (handle None for pre-computed CIs without p-value)
        if p_value is None:
            p_text = "N/A"
        elif p_value < 0.001:
            p_text = "< .001"
        else:
            p_text = f"{p_value:.3f}".replace("0.", ".")

        # Draw statistics (μ, CI, p, Verdict)
        # Conditionally exclude p_text if all p-values are missing
        if all_p_values_missing:
            stat_values = [
                f'{md:.4f}',
                f'[{ci_lower:.4f}, {ci_upper:.4f}]',
                verdict_text
            ]
        else:
            stat_values = [
                f'{md:.4f}',
                f'[{ci_lower:.4f}, {ci_upper:.4f}]',
                p_text,
                verdict_text
            ]

        # Determine verdict column index based on whether p-values are included
        verdict_col_idx = 2 if all_p_values_missing else 3

        for stat_x, value, col_idx in zip(stats_x_positions, stat_values, range(len(stat_values))):
            # Color code verdict column based on three-level verdict
            if col_idx == verdict_col_idx:  # Verdict column
                if verdict_text in ['Non-inferior', 'Equal']:
                    color = 'darkgreen'
                elif verdict_text == 'Different':
                    color = 'darkblue'
                else:  # Inconclusive
                    color = 'darkorange'
            else:
                color = 'black'

            ax.text(stat_x, y_pos, value,
                   fontsize=STYLE_CONFIG['font_size'] - 1, ha='center', va='center',
                   color=color, transform=trans)

    # Set y-limits and remove y-axis
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([])

    # Add legend for non-inferiority margin
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#87CEEB', linestyle='-',
                             linewidth=2.5, alpha=0.6, label='Non-inferiority margin')]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             fontsize=STYLE_CONFIG['font_size'] - 1, fancybox=True, shadow=False)

    # Set labels
    if title is None:
        title = "Non-Inferiority Analysis"

    if xlabel is None:
        xlabel = "Effect Size"

    # Apply styling (always forest plot style)
    ylabel = ""
    apply_consistent_style(ax, title=title, xlabel=xlabel, ylabel=ylabel, title_pad=title_pad)
    # Adjust title position to the left to avoid interfering with statistics columns
    ax.title.set_horizontalalignment('left')
    ax.title.set_position((0, 1))  # Position at left edge, top

    return ax


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
Example usage in a Jupyter notebook:

# Import functions
from scripts.viz_utils import (
    plot_likert_distribution,
    plot_categorical_bar,
    plot_continuous_distribution,
    plot_boxplot,
    create_figure_grid,
    set_custom_colors
)
import pandas as pd

# Load your processed data
data = pd.read_csv('data/data_with_scales.csv', index_col='ResponseId')

# 1. Single plot - Likert scale with grouping (no KDE by default)
# Default: 1-5 scale with numeric labels
plot_likert_distribution(data, 'ati', group_by='stimulus_group')
plt.show()

# 1b. With KDE overlay (optional)
plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_kde=True)
plt.show()

# 1c. With text labels on x-axis (e.g., "Strongly disagree", etc.)
plot_likert_distribution(data, 'ATI_1', show_labels=True, likert_range=5)
plt.show()

# 1d. For a 7-point Likert scale
plot_likert_distribution(data, 'custom_7pt_item', likert_range=7)
plt.show()

# 2. Single plot - Categorical variable
plot_categorical_bar(data, 'gender', group_by='stimulus_group')
plt.show()

# 3. Single plot - Continuous variable (no KDE by default)
plot_continuous_distribution(data, 'age', group_by='stimulus_group')
plt.show()

# 3b. With KDE overlay (optional)
plot_continuous_distribution(data, 'age', group_by='stimulus_group', show_kde=True)
plt.show()

# 4. Multiple plots in a grid
scales = ['ati', 'tia_rc', 'tia_up', 'tia_f', 'tia_pro', 'tia_t']
fig, axes = create_figure_grid(len(scales), ncols=2)

for idx, scale in enumerate(scales):
    plot_likert_distribution(data, scale, group_by='stimulus_group',
                            ax=axes[idx], show_stats=True)

plt.tight_layout()
plt.show()

# 5. Boxplots - Multiple items comparison
# Simple boxplot with default labels (column names in uppercase)
plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f'])
plt.show()

# 5b. Compare groups with custom short labels
plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f', 'tia_pro', 'tia_t'],
             group_by='stimulus_group',
             short_labels=['Reliability', 'Understanding', 'Familiarity', 'Propensity', 'Trust'],
             title='Trust in Automation Subscales by Group')
plt.show()

# 5c. Use full labels from labels.csv (auto-truncated)
plot_boxplot(data, ['tia_rc', 'tia_up'],
             group_by='stimulus_group',
             use_full_labels=True)
plt.show()

# 5d. Mirrored histograms instead of boxplots (ideal for Likert scales)
plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f', 'tia_pro', 'tia_t'],
             group_by='stimulus_group',
             short_labels=['Reliability', 'Understanding', 'Familiarity', 'Propensity', 'Trust'],
             mirror_hist=True)
plt.show()

# 5e. Mirrored histograms with mean markers
plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f'],
             group_by='stimulus_group',
             short_labels=['R/C', 'U/P', 'Fam'],
             mirror_hist=True, show_mean=True,
             title='Trust in Automation Subscales (with means)')
plt.show()

# 5f. Mirrored histograms with custom bin count
plot_boxplot(data, ['tia_rc', 'tia_up'],
             group_by='stimulus_group',
             short_labels=['Reliability', 'Understanding'],
             mirror_hist=True, bins=7,
             title='Custom bin count example')
plt.show()

# 5g. Split histogram with central boxplot (new visualization type!)
# Perfect for comparing two groups on a single scale
# Left histogram + central boxplots + right histogram
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             title='Trust in Automation: Reliability/Confidence')
plt.show()

# 5h. Split histogram without statistics, no mean markers
plot_split_histogram_boxplot(data, 'tia_up', group_by='stimulus_group',
                             show_stats=False, show_mean=False,
                             title='Trust in Automation: Understanding/Predictability')
plt.show()

# 5i. Split histogram with wider histograms
plot_split_histogram_boxplot(data, 'tia_f', group_by='stimulus_group',
                             hist_scale=0.6,  # Wider histograms (60% of plot width)
                             title='Trust in Automation: Familiarity')
plt.show()

# 5j. Split histogram with custom bins
plot_split_histogram_boxplot(data, 'ati', group_by='stimulus_group',
                             bins=7,  # Use 7 bins instead of auto-detected
                             title='Affinity for Technology Interaction')
plt.show()

# 6. Customize colors
set_custom_colors({
    'control': '#3498DB',      # Blue
    'uncertainty': '#E74C3C',   # Red
    'primary': '#2C3E50'        # Dark gray
})

# 7. Demographics overview
fig, axes = create_figure_grid(4, ncols=2, figsize=(12, 8))

plot_categorical_bar(data, 'gender', ax=axes[0])
plot_continuous_distribution(data, 'age', ax=axes[1])
plot_categorical_bar(data, 'education', ax=axes[2])
plot_categorical_bar(data, 'stimulus_group', ax=axes[3])

plt.tight_layout()
plt.show()
"""
