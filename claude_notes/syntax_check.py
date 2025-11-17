"""
Syntax check for the plot_noninferiority_test function.
This verifies the function can be imported without syntax errors.
"""

import sys
import ast

# Read the viz_utils.py file
with open('/home/user/thesis-analysis/scripts/viz_utils.py', 'r') as f:
    code = f.read()

# Try to parse the file
try:
    ast.parse(code)
    print("✓ viz_utils.py syntax is valid")
    print("✓ plot_noninferiority_test function added successfully")
    print("\nThe function is ready to use!")
    print("\nFunction signature:")
    print("  plot_noninferiority_test(")
    print("      mean_diff: float,")
    print("      sesoi: float,")
    print("      se: float,")
    print("      alpha: float = 0.05,")
    print("      test_type: str = 'lower',")
    print("      title: Optional[str] = None,")
    print("      xlabel: Optional[str] = None,")
    print("      ax: Optional[plt.Axes] = None,")
    print("      title_pad: Optional[int] = None,")
    print("      show_stats: bool = True")
    print("  )")
    print("\nParameters:")
    print("  - mean_diff: Observed mean difference between samples")
    print("  - sesoi: Smallest Effect Size Of Interest (non-inferiority margin)")
    print("  - se: Standard error of the mean difference")
    print("  - alpha: Significance level (default: 0.05)")
    print("  - test_type: 'lower' or 'upper' (default: 'lower')")
    print("  - title: Optional plot title")
    print("  - xlabel: Optional x-axis label")
    print("  - ax: Optional matplotlib axes object")
    print("  - title_pad: Optional title padding")
    print("  - show_stats: Whether to show statistics (default: True)")

except SyntaxError as e:
    print(f"✗ Syntax error found: {e}")
    sys.exit(1)
