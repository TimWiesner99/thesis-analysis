"""
Test script for the plot_noninferiority_test function.

This script demonstrates various use cases of the non-inferiority visualization.
"""

import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from scripts.viz_utils import plot_noninferiority_test

# Example 1: Non-inferiority established (observed difference > critical value)
# Scenario: New treatment shows mean difference of 0.15, margin is 0.3
print("Example 1: Non-inferiority established (lower test)")
fig, ax = plt.subplots(figsize=(10, 6))
plot_noninferiority_test(
    effect_size=0.15,
    sesoi=0.3,
    se=0.12,
    alpha=0.05,
    test_type='lower',
    title='Example 1: Non-Inferiority Established',
    xlabel='Mean Difference (Cohen\'s d)',
    ax=ax
)
plt.tight_layout()
plt.savefig('claude_notes/noninferiority_example1.png', dpi=300, bbox_inches='tight')
print("✓ Saved to claude_notes/noninferiority_example1.png\n")
plt.close()

# Example 2: Non-inferiority NOT established (observed difference < critical value)
# Scenario: New treatment shows mean difference of -0.25, margin is 0.3
print("Example 2: Non-inferiority NOT established (lower test)")
fig, ax = plt.subplots(figsize=(10, 6))
plot_noninferiority_test(
    effect_size=-0.25,
    sesoi=0.3,
    se=0.15,
    alpha=0.05,
    test_type='lower',
    title='Example 2: Non-Inferiority NOT Established',
    xlabel='Mean Difference (Cohen\'s d)',
    ax=ax
)
plt.tight_layout()
plt.savefig('claude_notes/noninferiority_example2.png', dpi=300, bbox_inches='tight')
print("✓ Saved to claude_notes/noninferiority_example2.png\n")
plt.close()

# Example 3: Upper non-inferiority test
# Scenario: Testing if new treatment is not better than standard
print("Example 3: Upper non-inferiority test")
fig, ax = plt.subplots(figsize=(10, 6))
plot_noninferiority_test(
    effect_size=-0.10,
    sesoi=0.3,
    se=0.12,
    alpha=0.05,
    test_type='upper',
    title='Example 3: Upper Non-Inferiority Test',
    xlabel='Mean Difference (Cohen\'s d)',
    ax=ax
)
plt.tight_layout()
plt.savefig('claude_notes/noninferiority_example3.png', dpi=300, bbox_inches='tight')
print("✓ Saved to claude_notes/noninferiority_example3.png\n")
plt.close()

# Example 4: Small margin (strict non-inferiority)
# Scenario: Very strict margin of 0.15
print("Example 4: Strict non-inferiority margin")
fig, ax = plt.subplots(figsize=(10, 6))
plot_noninferiority_test(
    effect_size=0.08,
    sesoi=0.15,
    se=0.08,
    alpha=0.05,
    test_type='lower',
    title='Example 4: Strict Non-Inferiority Margin (SESOI = 0.15)',
    xlabel='Mean Difference (Cohen\'s d)',
    ax=ax
)
plt.tight_layout()
plt.savefig('claude_notes/noninferiority_example4.png', dpi=300, bbox_inches='tight')
print("✓ Saved to claude_notes/noninferiority_example4.png\n")
plt.close()

# Example 5: Multiple plots in a grid
print("Example 5: Multiple non-inferiority tests in a grid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Scenario A: Strong non-inferiority
plot_noninferiority_test(
    effect_size=0.25, sesoi=0.3, se=0.10, alpha=0.05,
    test_type='lower',
    title='A: Strong Non-Inferiority',
    xlabel='Effect Size',
    ax=axes[0, 0]
)

# Scenario B: Marginal non-inferiority
plot_noninferiority_test(
    effect_size=-0.05, sesoi=0.3, se=0.15, alpha=0.05,
    test_type='lower',
    title='B: Marginal Non-Inferiority',
    xlabel='Effect Size',
    ax=axes[0, 1]
)

# Scenario C: Failed non-inferiority
plot_noninferiority_test(
    effect_size=-0.35, sesoi=0.3, se=0.12, alpha=0.05,
    test_type='lower',
    title='C: Failed Non-Inferiority',
    xlabel='Effect Size',
    ax=axes[1, 0]
)

# Scenario D: Upper non-inferiority
plot_noninferiority_test(
    effect_size=0.05, sesoi=0.3, se=0.12, alpha=0.05,
    test_type='upper',
    title='D: Upper Non-Inferiority',
    xlabel='Effect Size',
    ax=axes[1, 1]
)

plt.tight_layout()
plt.savefig('claude_notes/noninferiority_grid.png', dpi=300, bbox_inches='tight')
print("✓ Saved to claude_notes/noninferiority_grid.png\n")
plt.close()

# Example 6: Different alpha levels
print("Example 6: Comparing different alpha levels")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

alphas = [0.01, 0.05, 0.10]
for idx, alpha in enumerate(alphas):
    plot_noninferiority_test(
        effect_size=0.10,
        sesoi=0.3,
        se=0.12,
        alpha=alpha,
        test_type='lower',
        title=f'α = {alpha}',
        xlabel='Effect Size',
        ax=axes[idx]
    )

plt.tight_layout()
plt.savefig('claude_notes/noninferiority_alpha_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved to claude_notes/noninferiority_alpha_comparison.png\n")
plt.close()

print("\n" + "="*60)
print("All test visualizations created successfully!")
print("="*60)
print("\nFiles created:")
print("  - noninferiority_example1.png (non-inferiority established)")
print("  - noninferiority_example2.png (non-inferiority NOT established)")
print("  - noninferiority_example3.png (upper test)")
print("  - noninferiority_example4.png (strict margin)")
print("  - noninferiority_grid.png (multiple scenarios)")
print("  - noninferiority_alpha_comparison.png (different alpha levels)")
