"""Test script for the new split histogram boxplot function"""
import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the new function
from scripts.viz_utils import plot_split_histogram_boxplot

# Create sample Likert scale data (1-5)
np.random.seed(42)
n = 100

# Simulate two groups with different distributions
# Group 0 (control): Lower scores, more concentrated around 2-3
# Group 1 (uncertainty): Higher scores, more concentrated around 3-4
data = pd.DataFrame({
    'stimulus_group': [0] * 50 + [1] * 50,
    'tia_rc': np.concatenate([
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.10, 0.25, 0.35, 0.20, 0.10]),
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.05, 0.10, 0.25, 0.35, 0.25])
    ]),
    'tia_up': np.concatenate([
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.12, 0.28, 0.30, 0.22, 0.08]),
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.06, 0.14, 0.25, 0.32, 0.23])
    ]),
    'tia_f': np.concatenate([
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.08, 0.20, 0.38, 0.25, 0.09]),
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.04, 0.11, 0.27, 0.35, 0.23])
    ]),
    'ati': np.concatenate([
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.10, 0.22, 0.36, 0.24, 0.08]),
        np.random.choice([1, 2, 3, 4, 5], 50, p=[0.06, 0.12, 0.28, 0.34, 0.20])
    ])
})

print("="*70)
print("Testing Split Histogram with Central Boxplot Function")
print("="*70)

# Test 1: Basic usage
print("\nTest 1: Basic usage with default settings")
fig, ax = plt.subplots(figsize=(10, 6))
plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',
                             title='Test 1: Basic Split Histogram')
plt.savefig('test_split_hist_1_basic.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: test_split_hist_1_basic.png")
print("  Layout: Left histogram | Central boxplots | Right histogram")

# Test 2: Without statistics and mean markers
print("\nTest 2: Without statistics and mean markers")
fig, ax = plt.subplots(figsize=(10, 6))
plot_split_histogram_boxplot(data, 'tia_up', group_by='stimulus_group',
                             show_stats=False, show_mean=False,
                             title='Test 2: No Stats, No Mean Markers')
plt.savefig('test_split_hist_2_no_stats.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: test_split_hist_2_no_stats.png")
print("  Clean layout without annotations")

# Test 3: Wider histograms
print("\nTest 3: Wider histograms (hist_scale=0.6)")
fig, ax = plt.subplots(figsize=(10, 6))
plot_split_histogram_boxplot(data, 'tia_f', group_by='stimulus_group',
                             hist_scale=0.6,
                             title='Test 3: Wider Histograms')
plt.savefig('test_split_hist_3_wide.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: test_split_hist_3_wide.png")
print("  Histograms use 60% of plot width instead of default 40%")

# Test 4: Custom bin count
print("\nTest 4: Custom bin count (bins=7)")
fig, ax = plt.subplots(figsize=(10, 6))
plot_split_histogram_boxplot(data, 'ati', group_by='stimulus_group',
                             bins=7,
                             title='Test 4: Custom 7 Bins')
plt.savefig('test_split_hist_4_bins.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: test_split_hist_4_bins.png")
print("  Uses 7 bins instead of auto-detected 5")

# Test 5: All scales in a grid
print("\nTest 5: Multiple scales in a grid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

scales = ['tia_rc', 'tia_up', 'tia_f', 'ati']
titles = [
    'Reliability/Confidence',
    'Understanding/Predictability',
    'Familiarity',
    'Affinity for Technology'
]

for idx, (scale, title) in enumerate(zip(scales, titles)):
    plot_split_histogram_boxplot(data, scale, group_by='stimulus_group',
                                 title=title, ax=axes[idx])

plt.tight_layout()
plt.savefig('test_split_hist_5_grid.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: test_split_hist_5_grid.png")
print("  Grid layout with 4 different scales")

print("\n" + "="*70)
print("All tests completed successfully!")
print("="*70)

print("\n📊 New Visualization Type Available:")
print("   plot_split_histogram_boxplot()")
print("\n✓ Features:")
print("   - Left histogram for group 1 (extending leftward)")
print("   - Central boxplots with mean markers")
print("   - Right histogram for group 2 (extending rightward)")
print("   - Aligned y-axis for easy comparison")
print("   - Auto-detected bins (or manual override)")
print("   - Statistical annotations (Cohen's d, p-value)")
print("   - Consistent styling with existing functions")

print("\n📖 Usage Examples:")
print("   # Basic usage")
print("   plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group')")
print()
print("   # Wider histograms")
print("   plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',")
print("                                hist_scale=0.6)")
print()
print("   # Custom bins")
print("   plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',")
print("                                bins=7)")
print()
print("   # No statistics")
print("   plot_split_histogram_boxplot(data, 'tia_rc', group_by='stimulus_group',")
print("                                show_stats=False, show_mean=False)")
