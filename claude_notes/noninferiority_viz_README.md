# Non-Inferiority Test Visualization

## Overview

A new visualization function `plot_noninferiority_test()` has been added to `scripts/viz_utils.py`. This function creates G*Power-style visualizations for non-inferiority tests, showing two overlapping Gaussian distributions with the non-inferiority zone.

## What Was Created

1. **Function**: `plot_noninferiority_test()` in `scripts/viz_utils.py`
2. **Examples**: Comprehensive Jupyter notebook at `claude_notes/noninferiority_examples.ipynb`
3. **Test Scripts**:
   - `claude_notes/test_noninferiority_viz.py` (full test with multiple scenarios)
   - `claude_notes/syntax_check.py` (syntax validation)

## Function Signature

```python
plot_noninferiority_test(
    mean_diff: float,        # Observed mean difference between samples
    sesoi: float,            # Non-inferiority margin (positive value)
    se: float,               # Standard error of mean difference
    alpha: float = 0.05,     # Significance level
    test_type: str = 'lower', # 'lower' or 'upper'
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title_pad: Optional[int] = None,
    show_stats: bool = True
) -> plt.Axes
```

## Key Features

### Visual Elements
- **Two Gaussian curves** (potentially overlapping):
  - **Blue curve**: Observed distribution (centered at `mean_diff`)
  - **Red/Pink curve**: Null distribution (centered at margin)
- **Critical value**: Black dashed line showing the decision boundary
- **Non-inferiority zone**: Green shaded area where non-inferiority is established
- **Statistics box**: Shows test results (mean difference, SESOI, SE, z-score, p-value)
- **Visual verdict**: Green box if non-inferiority established, red if not

### Test Types

#### Lower Non-Inferiority (`test_type='lower'`)
- **Hypothesis**: H₀: μ_diff ≤ -SESOI vs H₁: μ_diff > -SESOI
- **Use case**: Testing if new treatment is **not worse** than standard
- **Example**: Showing that communicating uncertainty doesn't reduce trust by more than 0.3

#### Upper Non-Inferiority (`test_type='upper'`)
- **Hypothesis**: H₀: μ_diff ≥ SESOI vs H₁: μ_diff < SESOI
- **Use case**: Testing if new treatment is **not better** than standard
- **Example**: Safety studies where excessive improvement might be concerning

## Usage Examples

### Basic Usage

```python
from scripts.viz_utils import plot_noninferiority_test
import matplotlib.pyplot as plt

# Lower non-inferiority test
plot_noninferiority_test(
    mean_diff=0.15,    # Observed difference
    sesoi=0.3,         # Margin (positive value)
    se=0.12,           # Standard error
    alpha=0.05
)
plt.show()
```

### With Your Thesis Data

```python
# Example: Testing if uncertainty communication doesn't harm trust
# beyond acceptable margin of 0.3 (Cohen's d)

plot_noninferiority_test(
    mean_diff=-0.11,   # uncertainty - control
    sesoi=0.3,         # Acceptable harm threshold
    se=0.13,           # SE from your analysis
    alpha=0.05,
    test_type='lower',
    title='Trust in Automation: Non-Inferiority Test',
    xlabel="Mean Difference (Cohen's d): Uncertainty - Control"
)
plt.show()
```

### Multiple Scenarios

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Strong non-inferiority
plot_noninferiority_test(0.25, 0.3, 0.10, ax=axes[0, 0],
                        title='Strong Non-Inferiority')

# Marginal non-inferiority
plot_noninferiority_test(-0.05, 0.3, 0.15, ax=axes[0, 1],
                        title='Marginal Non-Inferiority')

# Failed non-inferiority
plot_noninferiority_test(-0.35, 0.3, 0.12, ax=axes[1, 0],
                        title='Failed Non-Inferiority')

# Upper test
plot_noninferiority_test(0.05, 0.3, 0.12, test_type='upper',
                        ax=axes[1, 1], title='Upper Test')

plt.tight_layout()
plt.show()
```

### Saving Figures

```python
fig, ax = plt.subplots(figsize=(10, 6))
plot_noninferiority_test(
    mean_diff=0.15, sesoi=0.3, se=0.12,
    title='Non-Inferiority Analysis',
    ax=ax
)
plt.tight_layout()

# High-resolution PNG for presentations
plt.savefig('plots/noninferiority_analysis.png', dpi=300, bbox_inches='tight')

# PDF for publications
plt.savefig('plots/noninferiority_analysis.pdf', bbox_inches='tight')

plt.show()
```

## Visual Style Consistency

The function follows your existing visualization style from `viz_utils.py`:
- Uses the same color palette (COLORS dict)
- Applies consistent styling (fonts, grid, spines)
- Matches other plotting functions in the module
- Supports optional matplotlib axes for subplot integration

## Parameters Explained

### Required Parameters

- **mean_diff**: The observed mean difference between groups (e.g., new - standard)
  - Can be positive or negative
  - Should be on the same scale as SESOI (e.g., Cohen's d, raw difference)

- **sesoi**: Smallest Effect Size Of Interest (non-inferiority margin)
  - **Always positive** (function handles the sign internally)
  - Represents the maximum acceptable difference
  - Common values: 0.2-0.5 for Cohen's d

- **se**: Standard error of the mean difference
  - Used to construct both Gaussian distributions
  - Computed from your statistical analysis

### Optional Parameters

- **alpha**: Significance level (default: 0.05)
  - Determines the critical value
  - Common values: 0.01, 0.05, 0.10

- **test_type**: 'lower' (default) or 'upper'
  - 'lower': Test if new is not worse
  - 'upper': Test if new is not better

- **show_stats**: Whether to display statistics box (default: True)
  - Set to False for cleaner presentation plots

## Interpretation Guide

### Reading the Visualization

1. **Check the mean difference position**:
   - If in green zone → non-inferiority established
   - If outside green zone → non-inferiority NOT established

2. **Examine the overlap**:
   - Large overlap → high uncertainty
   - Small overlap → clear conclusion

3. **Look at the critical value**:
   - Shows the decision boundary
   - Depends on alpha and standard error

4. **Read the statistics box**:
   - Shows all test statistics
   - Indicates conclusion with ✓ or ✗

### Statistical Interpretation

- **p < alpha**: Reject H₀, conclude non-inferiority
- **p ≥ alpha**: Fail to reject H₀, cannot conclude non-inferiority

The visualization shows both:
- The **observed** effect (blue distribution)
- The **null hypothesis** assumption (red distribution at margin)

## Testing

Run the syntax check to verify the function:

```bash
python3 claude_notes/syntax_check.py
```

## Examples Notebook

Open and run `claude_notes/noninferiority_examples.ipynb` in DataSpell to see:
- 8 comprehensive examples
- Different scenarios (established, failed, upper, strict margin)
- Multiple plots in grids
- Real-world application examples
- Saving figures

## Integration with Your Analysis

This function fits naturally into your existing workflow:

```python
# In your analysis notebook
from scripts.viz_utils import (
    plot_likert_distribution,
    plot_boxplot,
    plot_noninferiority_test  # New function
)

# After running your statistical tests
plot_noninferiority_test(
    mean_diff=your_cohens_d,
    sesoi=0.3,  # Your chosen margin
    se=your_standard_error,
    title='Non-Inferiority: Trust in Automation'
)
```

## References

- **J. Walker**: "Non-inferiority statistics and equivalence studies"
- **G*Power**: Statistical power analysis tool (visual inspiration)
- **SESOI**: Smallest Effect Size Of Interest framework

## Notes

- The function uses scipy.stats for normal distributions
- All visualizations use your existing color scheme
- Function includes comprehensive error checking
- Fully documented with examples in docstring
- Compatible with matplotlib subplots for multi-panel figures
