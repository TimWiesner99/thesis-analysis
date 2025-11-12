# Histogram Edge Fix: Touch Plot Borders, Avoid Overlap

## Problem Solved
Histograms now touch the plot edges with minimal empty space, and no longer overlap with the central boxplots.

## Changes Made

### Before
- Left histogram started at x = -0.5
- Right histogram ended at x = 0.5
- X-axis limits: (-1.0, 1.0)
- **Result**: 0.5 units of empty space on each side
- **Risk**: Potential overlap with boxplots at ±0.15

### After
- Histograms positioned dynamically based on boxplot positions and hist_scale
- Tight x-axis limits with minimal margin (0.05 units)
- Guaranteed gap between histograms and boxplots (0.05 units)

## Implementation Details

### New Position Calculations (lines 1201-1212)

```python
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
```

### Numerical Example
With default parameters:
- boxplot_width = 0.25
- box_positions = [-0.15, 0.15]
- gap = 0.05
- hist_scale = 0.4

**Calculations:**
- left_hist_inner = -0.15 - 0.125 - 0.05 = **-0.325**
- right_hist_inner = 0.15 + 0.125 + 0.05 = **0.325**
- left_anchor = -0.325 - 0.4 = **-0.725**
- right_anchor = 0.325 + 0.4 = **0.725**

**Result:**
- Left histogram spans: **-0.725 to -0.325** (0.4 units)
- Right histogram spans: **0.325 to 0.725** (0.4 units)
- Boxplots span: -0.275 to -0.025 and 0.025 to 0.275
- Gap: **0.05 units** between histogram and boxplot
- X-limits: **(-0.775, 0.775)** with 0.05 margin

### Updated Histogram Plotting (lines 1214-1226)

```python
# Plot left histogram (group 1, extending rightward towards center)
# All bars start from left_anchor and extend right by their normalized count
lefts1 = left_anchor
ax.barh(bin_centers, normalized_counts1, height=bin_heights, left=lefts1, ...)

# Plot right histogram (group 2, extending leftward towards center)
# All bars end at right_anchor and extend left by their normalized count
lefts2 = right_anchor - normalized_counts2
ax.barh(bin_centers, normalized_counts2, height=bin_heights, left=lefts2, ...)
```

### Updated X-Axis Limits (lines 1279-1281)

```python
# Set axis limits (tight to histogram edges with small margin)
margin = 0.05
ax.set_xlim(left_anchor - margin, right_anchor + margin)
```

## Visual Comparison

### Before
```
|<---0.5 empty--->|<---histogram--->|<-boxplot->|<---histogram--->|<---0.5 empty--->|
                  -0.5              0           0                 0.5
  -1.0                                                                              1.0
```

### After
```
|<-0.05->|<--------histogram-------->|<-0.05->|<-boxplot->|<-0.05->|<--------histogram-------->|<-0.05->|
         -0.725                     -0.325    -0.15       0.15     0.325                       0.725
  -0.775                                                                                              0.775
```

## Benefits

✅ **Efficient space usage**: Histograms now occupy maximum available space

✅ **Touch plot edges**: Only 0.05 units margin on each side

✅ **No overlap**: Guaranteed 0.05 units gap between histograms and boxplots

✅ **Dynamic calculation**: Automatically adjusts based on `hist_scale` parameter

✅ **Cleaner appearance**: Better visual balance

## Files Modified

**File:** `scripts/viz_utils.py`
**Function:** `plot_split_histogram_boxplot()` (lines 1086-1315)

### Specific changes:
1. **Lines 1201-1212**: Added position calculations (moved boxplot definitions up, added gap and anchor calculations)
2. **Line 1216**: Changed `lefts1 = -0.5` to `lefts1 = left_anchor`
3. **Line 1223**: Changed `lefts2 = 0.5 - normalized_counts2` to `lefts2 = right_anchor - normalized_counts2`
4. **Lines 1279-1281**: Replaced static x-limits with dynamic calculation

## Validation

✅ Syntax check passed
✅ All calculations verified
✅ No breaking changes to function signature or usage

## Usage

No changes needed to existing code - all improvements are automatic:

```python
# Same usage as before!
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group')

# With wider histograms
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             hist_scale=0.6)  # Automatically adjusts positioning
```

The function now provides optimal spacing regardless of the `hist_scale` parameter value.
