"""
Test script for the updated non-inferiority visualization function.

This script tests the corrected implementation with:
- One Gaussian centered at mean_diff (black)
- Vertical line at SESOI
- Non-inferiority region extending right from SESOI
- Alpha-level tail shading
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path to import scripts as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.viz_utils import plot_noninferiority_test

# Test case 1: Lower non-inferiority test (positive mean difference)
# Mean difference is in the non-inferiority zone
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Test 1: Non-inferiority established (mean_diff > -sesoi)
plot_noninferiority_test(
    mean_diff=0.15,
    sesoi=0.3,
    se=0.12,
    alpha=0.05,
    test_type='lower',
    title='Test 1: Non-inferiority Established',
    ax=axes[0, 0]
)

# Test 2: Non-inferiority NOT established (mean_diff < -sesoi)
plot_noninferiority_test(
    mean_diff=-0.35,
    sesoi=0.3,
    se=0.12,
    alpha=0.05,
    test_type='lower',
    title='Test 2: Non-inferiority NOT Established',
    ax=axes[0, 1]
)

# Test 3: Mean difference right at the SESOI boundary
plot_noninferiority_test(
    mean_diff=-0.3,
    sesoi=0.3,
    se=0.12,
    alpha=0.05,
    test_type='lower',
    title='Test 3: At SESOI Boundary',
    ax=axes[1, 0]
)

# Test 4: Different alpha level (0.01)
plot_noninferiority_test(
    mean_diff=0.15,
    sesoi=0.3,
    se=0.12,
    alpha=0.01,
    test_type='lower',
    title='Test 4: Alpha = 0.01',
    ax=axes[1, 1]
)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'test_noninferiority_fixed.png')
plt.savefig(output_path, dpi=150)
print(f"Test plot saved to {output_path}")
# Don't show plot interactively (would hang on some systems)
# plt.show()
