"""Test script for updated regression coefficients overview table with multivariate effects"""
import pandas as pd
import sys
sys.path.append('..')

from scripts.stats import format_effect_with_stars
import processing.scales as scales

# Remove 'TiA - ' prefix from scale titles
for k in scales.scale_titles:
    scales.scale_titles[k] = scales.scale_titles[k].replace('TiA - ', '')

tia_scales = scales.tia_scales
output_path = '../output/'

# Create predictor label mapping
predictor_labels = {
    'group_effect': 'Group Effect (Uncertainty vs Control)',
    'age': 'Age',
    'gender': 'Gender',
    'education': 'Education',
    'ai_exp': 'AI Experience',
    'hcsds_c': 'Healthcare Trust - Competence',
    'hcsds_v': 'Healthcare Trust - Values',
    'ati': 'Affinity for Technology',
    'group_effect:age': 'Group × Age',
    'group_effect:gender': 'Group × Gender',
    'group_effect:education': 'Group × Education',
    'group_effect:ai_exp': 'Group × AI Experience',
    'group_effect:hcsds_c': 'Group × Healthcare Trust - Competence',
    'group_effect:hcsds_v': 'Group × Healthcare Trust - Values',
    'group_effect:ati': 'Group × Affinity for Technology'
}

# Load MANOVA results for multivariate effects
manova_df = pd.read_csv(f'{output_path}manova_results.csv', index_col=0)
# Exclude Intercept
manova_df = manova_df[manova_df.index != 'Intercept']

print("MANOVA predictor names:")
print(manova_df.index.tolist())
print("\nRegression predictor names:")
print(list(predictor_labels.keys()))

# Load all regression coefficient files
regression_data = {}
for subscale in tia_scales:
    df = pd.read_csv(f'{output_path}{subscale}_regression_coef.csv', index_col=0)
    # Exclude Intercept (no adjusted p-value)
    df = df[df.index != 'Intercept']
    regression_data[subscale] = df

# Build formatted table
results_table = pd.DataFrame(index=predictor_labels.keys())

# Add multivariate effect column (partial eta-squared with stars)
multivariate_values = []
for predictor in results_table.index:
    if predictor in manova_df.index:
        eta_sq = manova_df.loc[predictor, 'partial_eta_sq']
        p_adj = manova_df.loc[predictor, 'p_adj']
        multivariate_values.append(format_effect_with_stars(eta_sq, p_adj))
    else:
        multivariate_values.append('—')

results_table['Multivariate Effect (eta_p_sq)'] = multivariate_values

# Add columns for each TiA subscale with (beta) suffix
for subscale in tia_scales:
    df = regression_data[subscale]

    # Create formatted column with coefficient and stars
    formatted_values = []
    for predictor in results_table.index:
        if predictor in df.index:
            coef = df.loc[predictor, 'coef']
            p_adj = df.loc[predictor, 'p_adj']
            formatted_values.append(format_effect_with_stars(coef, p_adj))
        else:
            formatted_values.append('—')

    # Use scale title as column name with (beta) suffix
    col_name = scales.scale_titles[subscale] + ' (beta)'
    results_table[col_name] = formatted_values

# Apply descriptive labels to row index
results_table.index = [predictor_labels[pred] for pred in results_table.index]

# Display table
print("\n" + "="*140)
print("Overview: Regression Coefficients for Trust in Automation Subscales")
print("="*140)
print("Note: Individual subscales show standardized regression coefficients (beta)")
print("      Multivariate effect shows partial eta-squared (eta_p_sq) from MANOVA")
print("Significance levels: *** p < .001, ** p < .01, * p < .05 (Holm-adjusted)")
print("="*140)
print(results_table.to_string())
print("="*140)

# Save to CSV
results_table.to_csv(f'{output_path}regression_coefficients_overview.csv')
print(f"\nTable saved to: {output_path}regression_coefficients_overview.csv")
