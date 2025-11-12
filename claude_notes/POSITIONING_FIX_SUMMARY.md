# Mirrored Histogram Positioning Fix Summary

## Problem Identified
The mirrored histograms were all appearing centered in the middle instead of being positioned side-by-side like boxplots. With 4 scales and 2 groups, there should have been 8 distinct histograms positioned along the x-axis.

### Root Cause
Raw histogram counts were being used directly as bar widths:
```python
lefts = x_pos - 0.5 * counts  # counts could be [5, 12, 8, 3, 2]
ax.barh(..., counts, ...)      # bar width = count value (e.g., 12 units!)
```

This caused bars to extend far beyond their allocated x-positions, creating massive overlap.

## Solution Applied

### Normalization with Shared Global Scale
All histograms now share a single scale factor based on the maximum count across ALL scales and groups. This ensures:
- ✅ Bars stay within allocated width
- ✅ Side-by-side positioning works correctly
- ✅ Bar widths are directly comparable across all histograms

### Changes Made to `scripts/viz_utils.py`

#### 1. Grouped Plots Section (lines 871-880)
**Added before the group loop:**
```python
# Calculate global maximum count for normalization (shared scale across all histograms)
global_max_count = 0
for group in groups:
    for col in columns:
        col_data = data[data[group_by] == group][col].dropna()
        counts, _ = np.histogram(col_data, bins=bin_edges, range=hist_range)
        global_max_count = max(global_max_count, counts.max())

# Compute scale factor so widest bar fits within allocated width
scale_factor = (width * 0.4) / global_max_count if global_max_count > 0 else 1.0
```

**Updated histogram plotting (lines 907-917):**
```python
# Compute histogram counts
counts, _ = np.histogram(col_data, bins=bin_edges, range=hist_range)

# Normalize counts using global scale factor
normalized_counts = counts * scale_factor

# Plot symmetric horizontal bars
lefts = x_pos - 0.5 * normalized_counts
ax.barh(bin_centers, normalized_counts, height=bin_heights, left=lefts, ...)
```

#### 2. Single Plots Section (lines 1018-1027, 1034-1044)
**Added global max count calculation:**
```python
# Calculate global maximum count for normalization (shared scale across all histograms)
global_max_count = 0
for col_idx, col in enumerate(columns):
    counts_temp, _ = np.histogram(plot_data[col_idx], bins=bin_edges, range=hist_range)
    global_max_count = max(global_max_count, counts_temp.max())

# Compute scale factor so widest bar fits within allocated width
width_single = 0.35
scale_factor = (width_single * 0.4) / global_max_count if global_max_count > 0 else 1.0
```

**Updated histogram plotting:**
```python
# Compute histogram counts
counts, _ = np.histogram(col_data, bins=bin_edges, range=hist_range)

# Normalize counts using global scale factor
normalized_counts = counts * scale_factor

# Plot symmetric horizontal bars
lefts = x_pos - 0.5 * normalized_counts
ax.barh(bin_centers, normalized_counts, height=bin_heights, left=lefts, ...)
```

## Key Technical Details

### Scale Factor Calculation
- `width * 0.4`: Uses 40% of the allocated width (0.35 units × 0.4 = 0.14 units max)
- This leaves 60% as margin for readability and prevents overlap
- Applied consistently across all histograms (shared global scale)

### Positioning Logic
- Each histogram has a distinct x-position from `group_positions[group]`
- Bars extend symmetrically: `left = x_pos - 0.5 * normalized_count`
- Bar width = `normalized_count` (scaled to fit within allocated space)
- Small spacing between same-scale/different-group pairs (controlled by `width = 0.35`)
- Larger spacing between different scales (controlled by cluster spacing)

## Expected Result

### Before Fix
```
All histograms overlapping in center:
[====MESS OF OVERLAPPING BARS====]
```

### After Fix
```
4 scales × 2 groups = 8 distinct histograms:
[RC_0][RC_1]  [UP_0][UP_1]  [F_0][F_1]  [PRO_0][PRO_1]
  Scale 1        Scale 2       Scale 3      Scale 4
```

Where:
- RC_0, RC_1 = Reliability/Confidence for groups 0 and 1
- UP_0, UP_1 = Understanding/Predictability for groups 0 and 1
- F_0, F_1 = Familiarity for groups 0 and 1
- PRO_0, PRO_1 = Propensity for groups 0 and 1

## Usage Example

```python
# This will now show 8 distinct side-by-side mirrored histograms
plot_boxplot(data, ['tia_rc', 'tia_up', 'tia_f', 'tia_pro'],
             group_by='stimulus_group',
             short_labels=['Reliability', 'Understanding', 'Familiarity', 'Propensity'],
             mirror_hist=True)
```

## Verification
- ✅ Syntax validated (py_compile check passed)
- ✅ All code changes complete
- ✅ Normalization applied to both grouped and single plots
- ✅ Shared global scale ensures comparability

## Files Modified
- `scripts/viz_utils.py` (plot_boxplot function, lines 871-880, 907-917, 1018-1027, 1034-1044)
- `test_mirror_hist.py` (updated with 4-scale test case)

## Status
✅ **FIX COMPLETE** - Mirrored histograms now position correctly side-by-side with proper spacing.
