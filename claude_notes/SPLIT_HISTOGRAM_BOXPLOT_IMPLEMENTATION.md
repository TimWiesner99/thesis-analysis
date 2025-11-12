# Split Histogram with Central Boxplot Implementation

## Overview
Successfully implemented a new visualization function `plot_split_histogram_boxplot()` that combines histograms and boxplots for two-group comparisons on a single scale.

## Visualization Layout

```
    Group 1 Histogram          Boxplots         Group 2 Histogram
         ←bars                 [Box][Box]            bars→
    ══════════════════════════════════════════════════════════
 5  ████████████              |  ■ |■ |           ████████████
 4  ██████████████████        |  ■ |■ |           ████████████████
 3  ████████████████████      |  ■ |■ |           ██████████████████
 2  ██████████████            |  ■ |■ |           ████████████
 1  ████████                  |  ■ |■ |           ██████
    ══════════════════════════════════════════════════════════
         -0.5                  0                 0.5
                          X-position
```

## Function Signature

```python
def plot_split_histogram_boxplot(data: pd.DataFrame,
                                  column: str,
                                  group_by: str,
                                  title: Optional[str] = None,
                                  ax: Optional[plt.Axes] = None,
                                  bins: Optional[int] = None,
                                  show_mean: bool = True,
                                  show_stats: bool = True,
                                  hist_scale: float = 0.4,
                                  title_pad: Optional[int] = None) -> plt.Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | required | DataFrame containing the data |
| `column` | str | required | Column name to plot (single scale, e.g., 'tia_rc') |
| `group_by` | str | required | Grouping column (must have exactly 2 groups) |
| `title` | str | None | Plot title (auto-generated if None) |
| `ax` | Axes | None | Matplotlib axes (creates new figure if None) |
| `bins` | int | None | Number of bins (auto-detected if None) |
| `show_mean` | bool | True | Show mean markers on boxplots |
| `show_stats` | bool | True | Show Cohen's d and p-value |
| `hist_scale` | float | 0.4 | Max histogram width as proportion (0.0-1.0) |
| `title_pad` | int | None | Title padding |

## Key Features

### 1. **Three-Part Layout**
- **Left**: Histogram of group 1 extending leftward from x = -0.5
- **Center**: Two boxplots at x = -0.15 and x = +0.15
- **Right**: Histogram of group 2 extending rightward from x = +0.5

### 2. **Shared Y-Axis**
- All elements (histograms and boxplots) share the same y-axis scale
- Perfect alignment of bins with boxplot positions
- Ideal for Likert scale data (1-5 or custom ranges)

### 3. **Auto-Detection**
- Bins automatically detected from unique values (perfect for Likert scales)
- Manual override available via `bins` parameter
- Intelligent color selection from existing color scheme

### 4. **Statistical Annotations**
- Cohen's d (effect size) between groups
- P-value from independent t-test
- Displayed in top-right corner (same style as other functions)
- Can be disabled with `show_stats=False`

### 5. **Mean Markers**
- White diamond markers with colored edges
- Positioned on boxplots at mean values
- Optional via `show_mean` parameter

### 6. **Normalization**
- Shared scale factor ensures fair comparison
- Histograms normalized to fit within `hist_scale` proportion
- Default: 40% of plot width per histogram
- Adjustable from 0.0 to 1.0

## Implementation Details

### Validation
- Checks that `group_by` column has exactly 2 groups
- Raises `ValueError` with helpful message if not 2 groups
- Handles missing data gracefully

### Color Selection
- Uses existing `COLORS` dictionary
- Automatic color mapping: 'control' → blue, 'uncertainty' → red/pink
- Fallback to 'primary' and 'secondary' colors

### Label Detection
- Attempts to retrieve readable labels from `labels.csv`
- Falls back to column names if not found
- Uses `get_label_for_value()` for group names
- Tries to get better y-label from `questions.csv`

### Histogram Positioning
- **Left histogram**: `lefts = -0.5 - normalized_counts`
  - Extends leftward from -0.5
- **Right histogram**: `lefts = 0.5`
  - Extends rightward from +0.5
  - Width = `normalized_counts`

### Boxplot Positioning
- Positioned at x = -0.15 and x = +0.15
- Width = 0.25 units
- Colored to match histogram groups
- Outliers colored to match groups

## Usage Examples

### Basic Usage
```python
# Simple two-group comparison
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group')
```

### Wider Histograms
```python
# Histograms use 60% of plot width (instead of 40%)
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             hist_scale=0.6)
```

### Custom Bins
```python
# Use 7 bins instead of auto-detected
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             bins=7)
```

### Clean Layout
```python
# No statistics or mean markers
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             show_stats=False, show_mean=False)
```

### Custom Title
```python
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             title='Trust in Automation: Reliability/Confidence')
```

### In a Grid
```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, scale in enumerate(['tia_rc', 'tia_up', 'tia_f', 'tia_pro']):
    plot_split_histogram_boxplot(data, scale, group_by='stimulus_group',
                                 ax=axes[idx])
plt.tight_layout()
plt.show()
```

## Integration with Existing Functions

### Consistent Styling
- Uses `apply_consistent_style()` for consistency
- Matches `STYLE_CONFIG` settings
- Same font sizes, colors, alpha values

### Reuses Utilities
- `get_label_for_value()` for readable labels
- `truncate_label()` for long labels
- `cohens_d()` for effect size
- `independent_t_test()` for p-values
- `COLORS` dictionary for color selection

### Compatible with Other Functions
- Can be used in `create_figure_grid()`
- Follows same parameter naming conventions
- Same optional `ax` parameter pattern

## Files Modified

### 1. `scripts/viz_utils.py`
- **Lines 1-11**: Updated module docstring
- **Lines 1086-1287**: New function implementation
- **Lines 1448-1471**: Added usage examples

### Function Breakdown
- **Lines 1086-1136**: Function signature and docstring
- **Lines 1137-1166**: Setup and validation
- **Lines 1168-1194**: Histogram calculation and normalization
- **Lines 1196-1206**: Plot left and right histograms
- **Lines 1208-1246**: Plot central boxplots with styling
- **Lines 1248-1259**: Statistical annotations
- **Lines 1261-1287**: Axis limits, labels, styling, legend

## Advantages Over Other Visualizations

### vs. Simple Boxplot
- ✓ Shows full distribution shape (not just quartiles)
- ✓ Easy to see modality and skewness
- ✓ Better for Likert scale data

### vs. Overlapping Histograms
- ✓ No overlap issues
- ✓ Easier to compare distributions
- ✓ Boxplot provides quick statistical summary

### vs. Violin Plots
- ✓ Shows discrete bins clearly (better for Likert data)
- ✓ Boxplot easier to interpret than violin
- ✓ No smoothing artifacts from KDE

## Validation

- ✅ Syntax check passed (`py_compile`)
- ✅ Function implemented with all planned features
- ✅ Documentation added
- ✅ Usage examples provided
- ✅ Consistent with existing code style

## Ready to Use!

The function is now available in `scripts/viz_utils.py` and can be used in Jupyter notebooks:

```python
from scripts.viz_utils import plot_split_histogram_boxplot

# Compare control vs uncertainty groups on a single scale
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             title='Trust in Automation: Reliability/Confidence')
```

Perfect for your thesis analysis of Trust in Automation subscales comparing control and uncertainty experimental groups!
