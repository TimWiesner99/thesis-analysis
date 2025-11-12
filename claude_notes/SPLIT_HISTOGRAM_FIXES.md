# Split Histogram Boxplot Fixes

## Changes Made

Three improvements applied to the `plot_split_histogram_boxplot()` function based on user feedback.

---

## 1. ✅ Mirrored Histograms (Point Towards Middle)

### Before
- Left histogram extended **leftward** from x = -0.5 (away from center)
- Right histogram extended **rightward** from x = 0.5 (away from center)

### After
- Left histogram extends **rightward** from x = -0.5 (towards center)
- Right histogram extends **leftward** from x = 0.5 (towards center)
- Both histograms now "point" towards the central boxplots

### Code Change (lines 1201-1211)
```python
# Before:
lefts1 = -0.5 - normalized_counts1  # Extended left
lefts2 = 0.5  # Extended right

# After:
lefts1 = -0.5  # Start at -0.5, extend right
lefts2 = 0.5 - normalized_counts2  # End at 0.5, extend left
```

---

## 2. ✅ Fixed Bin Alignment with Y-Axis

### Problem
- Bin edges were at integer values [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
- Bin centers ended up at [1.5, 2.5, 3.5, 4.5, 5.5]
- **Bins were NOT aligned with y-axis labels!**

### Solution
- Extended bin edges by 0.5 on each side
- For Likert 1-5: bin edges now at [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
- Bin centers now at [1.0, 2.0, 3.0, 4.0, 5.0]
- **Bins perfectly aligned with y-axis integer labels!**

### Code Changes

**Line 1184: Bin edges calculation**
```python
# Before:
bin_edges = np.linspace(data_range[0], data_range[1], n_bins + 1)

# After:
bin_edges = np.linspace(data_range[0] - 0.5, data_range[1] + 0.5, n_bins + 1)
```

**Line 1270: Y-axis limits**
```python
# Before:
y_padding = (data_range[1] - data_range[0]) * 0.05
ax.set_ylim(data_range[0] - y_padding, data_range[1] + y_padding)

# After:
ax.set_ylim(bin_edges[0], bin_edges[-1])
```

**Lines 1189-1191: Histogram range**
```python
# Added to match bin edges:
hist_range = (bin_edges[0], bin_edges[-1])
counts1, _ = np.histogram(group1_data, bins=bin_edges, range=hist_range)
counts2, _ = np.histogram(group2_data, bins=bin_edges, range=hist_range)
```

---

## 3. ✅ Group Labels Under X-Axis (No Legend)

### Before
- Legend displayed in upper left corner
- X-axis had no labels

### After
- Group labels positioned below x-axis, under their respective boxplots
- No legend (cleaner visual appearance)
- Labels positioned at x = -0.15 (group 1) and x = 0.15 (group 2)
- Labels colored to match their groups
- Labels in bold for emphasis

### Code Changes

**Lines 1203-1205, 1209-1211: Removed label parameters from histograms**
```python
# Before:
ax.barh(..., label=group1_label)
ax.barh(..., label=group2_label)

# After:
ax.barh(...)  # No label parameter
ax.barh(...)  # No label parameter
```

**Lines 1288-1300: Added group labels below x-axis**
```python
# Removed:
ax.legend(frameon=False, fontsize=STYLE_CONFIG['font_size'], loc='upper left')

# Added:
ax.text(-0.15, bin_edges[0], group1_label,
       ha='center', va='top',
       fontsize=STYLE_CONFIG['label_size'],
       color=color1,
       fontweight='bold')

ax.text(0.15, bin_edges[0], group2_label,
       ha='center', va='top',
       fontsize=STYLE_CONFIG['label_size'],
       color=color2,
       fontweight='bold')
```

---

## Visual Comparison

### Before
```
      ←bars                [Box][Box]            bars→
  [histograms point away]  [boxplots]  [histograms point away]
  [misaligned bins]                    [misaligned bins]
  [legend in corner]
```

### After
```
       bars→               [Box][Box]              ←bars
  [histograms point in]    [boxplots]    [histograms point in]
  [aligned bins at 1,2,3,4,5]          [aligned bins at 1,2,3,4,5]
          Control            Uncertainty
       (below x-axis)      (below x-axis)
```

---

## Files Modified

**File:** `scripts/viz_utils.py`
**Function:** `plot_split_histogram_boxplot()` (lines 1086-1302)

### Specific line changes:
1. **Line 1184**: Added `-0.5` and `+0.5` to bin edges calculation
2. **Lines 1189-1191**: Updated histogram range to match bin edges
3. **Lines 1202, 1208**: Fixed histogram positioning (mirrored)
4. **Lines 1203-1205, 1209-1211**: Removed label parameters
5. **Line 1270**: Updated y-axis limits to use bin edges
6. **Lines 1290-1300**: Added group labels below x-axis, removed legend

---

## Results

✅ **Mirrored histograms**: Both point towards center for better visual flow

✅ **Perfect alignment**: Bins centered at integer values (1, 2, 3, 4, 5)

✅ **Cleaner layout**: Group labels below x-axis instead of legend

✅ **Extended y-axis**: Shows full bin range (0.5 to 5.5 for 1-5 Likert scale)

✅ **Syntax validated**: All changes compile successfully

---

## Usage

The function works exactly the same way - all changes are internal improvements:

```python
# No changes needed to existing code!
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group')
```

All three improvements are now active automatically.
