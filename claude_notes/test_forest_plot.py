"""
Test script for the new forest plot visualization of non-inferiority analysis.

This script tests:
1. Basic forest plot without categories
2. Forest plot with categorical grouping
"""

import os
import sys

# Ensure we're in the right directory for relative imports to work
project_root = 'C:\\Users\\tim20\\Software Projects\\thesis-analysis'
if os.getcwd() != project_root:
    os.chdir(project_root)
    print(f"Changed working directory to: {os.getcwd()}")

# Now try importing - at this point scripts/ should be able to find ../data/
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import after changing directory
sys.path.insert(0, project_root)
from scripts.viz_utils import plot_noninferiority_test

# Test data (simulated)
mean_diffs = [0.15, -0.03, -0.10, 0.05, -0.08, 0.12]
sesoi_values = [0.15, 0.11, 0.09, 0.12, 0.10, 0.14]
se_values = [0.11, 0.08, 0.07, 0.09, 0.08, 0.10]
var_names = ['tia_t', 'tia_rc', 'tia_up', 'tia_f', 'tia_pro', 'tia_x']
var_labels = {
    'tia_t': 'Trust in Automation',
    'tia_rc': 'Reliability/Competence',
    'tia_up': 'Understanding/Predictability',
    'tia_f': 'Familiarity',
    'tia_pro': 'Propensity to Trust',
    'tia_x': 'Extra Variable'
}

# Test 1: Basic forest plot without categories
print("Test 1: Creating basic forest plot...")
fig, ax = plt.subplots(figsize=(14, 6))
plot_noninferiority_test(
    mean_diff=mean_diffs,
    sesoi=sesoi_values,
    se=se_values,
    alpha=0.05,
    variable_names=var_names,
    variable_labels=var_labels,
    title='Test 1: Basic Forest Plot (No Categories)',
    xlabel='Mean Difference',
    ax=ax,
    show_stats=False  # Suppress console output for cleaner test
)
plt.tight_layout()
plt.savefig('claude_notes/test_forest_plot_basic.png', dpi=150)
print("✓ Basic forest plot saved to claude_notes/test_forest_plot_basic.png")
plt.close()

# Test 2: Forest plot with categories
print("\nTest 2: Creating forest plot with categories...")
categories = {
    'Trust Dimensions': ['tia_t', 'tia_rc'],
    'Understanding & Familiarity': ['tia_up', 'tia_f'],
    'Other Factors': ['tia_pro', 'tia_x']
}

fig, ax = plt.subplots(figsize=(14, 8))
plot_noninferiority_test(
    mean_diff=mean_diffs,
    sesoi=sesoi_values,
    se=se_values,
    alpha=0.05,
    variable_names=var_names,
    variable_labels=var_labels,
    categories=categories,
    title='Test 2: Forest Plot with Categorical Grouping',
    xlabel='Mean Difference',
    ax=ax,
    show_stats=False
)
plt.tight_layout()
plt.savefig('claude_notes/test_forest_plot_categories.png', dpi=150)
print("✓ Categorical forest plot saved to claude_notes/test_forest_plot_categories.png")
plt.close()

# Test 3: Single variable (backward compatibility)
print("\nTest 3: Testing backward compatibility with single variable...")
fig, ax = plt.subplots(figsize=(10, 6))
plot_noninferiority_test(
    mean_diff=0.15,
    sesoi=0.20,
    se=0.10,
    alpha=0.05,
    title='Test 3: Single Variable (Backward Compatible)',
    xlabel='Mean Difference',
    ax=ax,
    show_stats=False
)
plt.tight_layout()
plt.savefig('claude_notes/test_forest_plot_single.png', dpi=150)
print("✓ Single variable plot saved to claude_notes/test_forest_plot_single.png")
plt.close()

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)
print("\nPlease review the generated plots to verify:")
print("  1. CI lines are horizontal with square markers at point estimates")
print("  2. Variables are stacked vertically with labels on the left")
print("  3. Statistics columns (μ, CI, p, Verdict) appear on the right")
print("  4. Categories have bold headers with proper spacing")
print("  5. Non-inferiority margin is shown as light blue vertical line")
print("  6. Zero reference line is shown as gray dotted line")
