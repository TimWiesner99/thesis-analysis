import pandas as pd
from scripts import scales

tia_scales = scales.tia_scales
hcsds_scales = scales.hcsds_scales
ati_scales = scales.ati_scales
manip_check_scales = scales.manip_check_scales

def effect_coding(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Prepare variables for moderation analysis:

    1. Effect code treatment: stimulus_group as -0.5 (control) and 0.5 (uncertainty)
    2. Standardize continuous variables: For better comparison of beta values between variables
    3. Effect code categorical variables: For symmetric interpretation

    :param data: DataFrame to process, typically data_scales.csv
    :return: DataFrame with effect coded, centred variables.
    '''
    # 1. Effect code treatment: control = -0.5, uncertainty = 0.5
    data['group_effect'] = data['stimulus_group'] - 0.5

    # 2. Normalize all continuous variables
    continuous_vars = hcsds_scales + ati_scales + ['age', 'page_submit']

    for var in continuous_vars:
        data[f'{var}_c'] = (data[var] - data[var].mean())/data[var].std()

    # 3. Effect code gender: male (1) = 0.5, female (2) = -0.5, "other/prefer not to say" (3) = 0
    data['gender_c'] = data['gender'].map({1: 0.5, 2: -0.5, 3: 0})

    # 4. Mean-center ordinal variables (education, AI experience)
    data['education_c'] = data['education'] - data['education'].mean()
    data['ai_exp_c'] = data['ai_exp'] - data['ai_exp'].mean()
    data.drop('ai_exp', axis=1)

    return data



