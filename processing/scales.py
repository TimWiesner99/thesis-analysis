# define variable names
tia_scales = ['tia_f', 'tia_pro', 'tia_rc', 'tia_t', 'tia_up']
hcsds_scales = ['hcsds_c', 'hcsds_v']
ati_scales = ['ati']
manip_check_scales = [f'manip_check1_{i+1}' for i in range(4)]

# define scale names
scale_titles = {
    'ati': 'Affinity for Technology Interaction',
    'hcsds_c': 'Healthcare Trust - Competence',
    'hcsds_v': 'Healthcare Trust - Values',
    'tia_rc': 'TiA - Reliability/Competence',
    'tia_up': 'TiA - Understanding/Predictability',
    'tia_f': 'TiA - Familiarity',
    'tia_pro': 'TiA - Propensity to Trust',
    'tia_t': 'TiA - Trust in Automation',
    'tia_comp': 'TiA - Compound scale'
}