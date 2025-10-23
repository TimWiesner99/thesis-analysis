"""Visualization utility functions for thesis analysis.

This module provides consistent, clean visualization functions for different data types:
- Likert scale items and computed scales (histogram + KDE)
- Categorical variables (bar plots with readable labels)
- Continuous variables (histogram + KDE)

All functions support optional grouping by experimental condition.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
from .utils import get_label_for_value, cohens_d, independent_t_test

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
                             show_kde: bool = False,
                             show_correlation: bool = True,
                             likert_range: int = 5,
                             show_labels: bool = False,
                             trunc: int = 20,
                             title_pad: Optional[int] = None) -> plt.Axes:
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
        show_kde: Whether to show KDE (kernel density estimate) overlay (default: False)
        show_correlation: Whether to show Cohen's d (δ) and p-value (α) from t-test between groups (default: True)
        likert_range: Max value of Likert scale (default: 5, for 1-5 scale)
        show_labels: Whether to show Likert labels on x-axis (e.g., "Strongly disagree") (default: False)
        trunc: Max characters for labels before truncation. -1 = no truncation (default: 20)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)

    Returns:
        The matplotlib axes object

    Example:
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_kde=True)
        >>> plot_likert_distribution(data, 'ATI_1', likert_range=5, show_labels=True, trunc=15)
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_correlation=True)
        >>> plot_likert_distribution(data, 'ati', group_by='stimulus_group', show_correlation=False)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Generate title if not provided
    if title is None:
        title = f"Distribution of {column.upper()}"

    # Check if grouping is requested
    if group_by is not None and group_by in data.columns:
        # Get unique groups
        groups = data[group_by].unique()

        # Store group data for correlation calculation
        group_means = {}

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

            # Add stats to label if requested
            label = group_label
            if show_stats:
                label += f" (M={group_data.mean():.2f}, SD={group_data.std():.2f})"

            # Plot histogram
            ax.hist(group_data, bins=20, alpha=STYLE_CONFIG['hist_alpha'],
                   label=label, color=color, edgecolor='white', linewidth=0.5)

            # Plot KDE (optional)
            if show_kde:
                group_data.plot.kde(ax=ax, color=color, linewidth=STYLE_CONFIG['kde_linewidth'])

        # Calculate and display correlation/effect size between groups if requested
        if show_correlation and len(group_means) == 2:
            # Get the two groups
            group_names = list(group_means.keys())
            group1_data = group_means[group_names[0]]
            group2_data = group_means[group_names[1]]

            # Calculate effect size (Cohen's d) and p-value (t-test)
            effect_size = cohens_d(group1_data, group2_data)
            p_value = independent_t_test(group1_data, group2_data)

            # Add text box with effect size and p-value using Greek letters
            textstr = f"δ = {effect_size:.4f}\nα = {p_value:.4f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', bbox=props)
    else:
        # Single distribution (no grouping)
        plot_data = data[column].dropna()

        label = f"Overall"
        if show_stats:
            label += f" (M={plot_data.mean():.2f}, SD={plot_data.std():.2f})"

        # Plot histogram
        ax.hist(plot_data, bins=20, alpha=STYLE_CONFIG['hist_alpha'],
               label=label, color=COLORS['primary'], edgecolor='white', linewidth=0.5)

        # Plot KDE (optional)
        if show_kde:
            plot_data.plot.kde(ax=ax, color=COLORS['primary'],
                              linewidth=STYLE_CONFIG['kde_linewidth'])

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
                        show_percentages: bool = True,
                        trunc: int = 20,
                        title_pad: Optional[int] = None) -> plt.Axes:
    """
    Plot bar chart for categorical variables with readable labels.

    Args:
        data: DataFrame containing the data
        column: Column name to plot
        title: Plot title (auto-generated if None)
        group_by: Optional column to group by (e.g., 'stimulus_group')
        ax: Optional matplotlib axes (creates new figure if None)
        show_percentages: Whether to show percentages on bars
        trunc: Max characters for labels before truncation. -1 = no truncation (default: 20)
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)

    Returns:
        The matplotlib axes object

    Example:
        >>> plot_categorical_bar(data, 'gender', group_by='stimulus_group')
        >>> plot_categorical_bar(data, 'education', trunc=20)
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

        # Get readable labels for x-axis
        x_labels = get_readable_labels(column, counts.index.tolist(), trunc=trunc)

        # Get readable labels for groups (legend)
        group_labels = get_readable_labels(group_by, counts.columns.tolist(), trunc=trunc)
        counts.columns = group_labels

        # Plot grouped bars
        counts_plot = counts.copy()
        counts_plot.index = x_labels

        colors = [COLORS.get(label.lower(), COLORS['primary']) for label in group_labels]
        counts_plot.plot(kind='bar', ax=ax, color=colors, alpha=STYLE_CONFIG['hist_alpha'],
                        edgecolor='white', linewidth=1)

        # Add percentage labels if requested
        if show_percentages:
            for container in ax.containers:
                labels = [f'{v:.0f}\n({v/counts.values.sum()*100:.1f}%)' if v > 0 else ''
                         for v in container.datavalues]
                ax.bar_label(container, labels=labels, fontsize=STYLE_CONFIG['tick_size'])
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
    apply_consistent_style(ax, title=title, xlabel=column.capitalize(),
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
                                 show_kde: bool = False,
                                 show_correlation: bool = True,
                                 bins: int = 30,
                                 title_pad: Optional[int] = None) -> plt.Axes:
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
        show_kde: Whether to show KDE (kernel density estimate) overlay (default: False)
        show_correlation: Whether to show Cohen's d (δ) and p-value (α) from t-test between groups (default: True)
        bins: Number of histogram bins
        title_pad: Padding between title and plot (uses STYLE_CONFIG default if None)

    Returns:
        The matplotlib axes object

    Example:
        >>> plot_continuous_distribution(data, 'age', group_by='stimulus_group', show_kde=True)
        >>> plot_continuous_distribution(data, 'page_submit', group_by='stimulus_group', show_correlation=True)
        >>> plot_continuous_distribution(data, 'page_submit', group_by='stimulus_group', show_correlation=False)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Generate title if not provided
    if title is None:
        title = f"Distribution of {column.replace('_', ' ').title()}"

    # Check if grouping is requested
    if group_by is not None and group_by in data.columns:
        # Get unique groups
        groups = data[group_by].unique()

        # Store group data for effect size calculation
        group_means = {}

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

            # Add stats to label if requested
            label = group_label
            if show_stats:
                label += f" (M={group_data.mean():.1f}, SD={group_data.std():.1f})"

            # Plot histogram
            ax.hist(group_data, bins=bins, alpha=STYLE_CONFIG['hist_alpha'],
                   label=label, color=color, edgecolor='white', linewidth=0.5)

            # Plot KDE (optional)
            if show_kde:
                group_data.plot.kde(ax=ax, color=color, linewidth=STYLE_CONFIG['kde_linewidth'])

        # Calculate and display effect size between groups if requested
        if show_correlation and len(group_means) == 2:
            # Get the two groups
            group_names = list(group_means.keys())
            group1_data = group_means[group_names[0]]
            group2_data = group_means[group_names[1]]

            # Calculate effect size (Cohen's d) and p-value (t-test)
            effect_size = cohens_d(group1_data, group2_data)
            p_value = independent_t_test(group1_data, group2_data)

            # Add text box with effect size and p-value using Greek letters
            textstr = f"δ = {effect_size:.4f}\nα = {p_value:.4f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                   fontsize=STYLE_CONFIG['font_size'], verticalalignment='top',
                   horizontalalignment='right', bbox=props)
    else:
        # Single distribution (no grouping)
        plot_data = data[column].dropna()

        label = f"Overall"
        if show_stats:
            label += f" (M={plot_data.mean():.1f}, SD={plot_data.std():.1f})"

        # Plot histogram
        ax.hist(plot_data, bins=bins, alpha=STYLE_CONFIG['hist_alpha'],
               label=label, color=COLORS['primary'], edgecolor='white', linewidth=0.5)

        # Plot KDE (optional)
        if show_kde:
            plot_data.plot.kde(ax=ax, color=COLORS['primary'],
                              linewidth=STYLE_CONFIG['kde_linewidth'])

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

# 5. Customize colors
set_custom_colors({
    'control': '#3498DB',      # Blue
    'uncertainty': '#E74C3C',   # Red
    'primary': '#2C3E50'        # Dark gray
})

# 6. Demographics overview
fig, axes = create_figure_grid(4, ncols=2, figsize=(12, 8))

plot_categorical_bar(data, 'gender', ax=axes[0])
plot_continuous_distribution(data, 'age', ax=axes[1])
plot_categorical_bar(data, 'education', ax=axes[2])
plot_categorical_bar(data, 'stimulus_group', ax=axes[3])

plt.tight_layout()
plt.show()
"""
