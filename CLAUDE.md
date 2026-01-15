# CLAUDE.md

## Project Overview

Master's thesis project (AI, Radboud University) investigating whether communicating technical uncertainties affects trust in Medical Decision Support Systems (MDSS).

**Important**: Claude Code was used only for supporting functions (especially visualization utilities), NOT for the statistical analysis itself. All analysis logic was developed independently.

## Environment

Uses **uv** with Python 3.14+. Run `uv sync` to install dependencies.

## Directory Structure

- `analysis/`: Analysis notebooks (manipulation checks, univariate/multivariate tests, overview)
- `processing/`: Data preprocessing (`preprocessing.ipynb`, `scales.py`)
- `scripts/`: Python modules (`stats.py`, `utils.py`, `viz_utils.py`)
- `data/`: Raw and processed data files
- `output/`: Statistical results (CSV, JSON)
- `plots/`: Generated visualizations

## Experimental Design

**Groups**: `uncertainty` (treatment) vs `control`

**Primary Outcomes** (Trust in Automation scale, Körber 2019):
- `tia_rc`: Reliability/Competence
- `tia_up`: Understanding/Predictability
- `tia_f`: Familiarity
- `tia_pro`: Propensity to Trust
- `tia_t`: Overall Trust

**Covariates/Moderators**:
- ATI (technology affinity), HCSDS (healthcare trust: `hcsds_c`, `hcsds_v`)
- Demographics: age, gender, education, AI experience

## Statistical Analysis

**Univariate**: Multiple linear regression per TiA subscale with main effect (`group_effect`), direct effects (covariates), and interaction effects (moderation).

**Multivariate**: MANOVA with Pillai's Trace, all five TiA subscales as DVs.

**Equivalence Testing**: Non-inferiority tests using partial eta-squared with CIs via non-central F distribution inversion (Steiger, 2004).

**Multiple Comparison Correction**: Holm method applied per family:
1. Group effect (1 test)
2. Direct effects (8 tests)
3. Interaction effects (8 tests)

**Sample**: N=255 (Control: 126, Uncertainty: 129), α=0.05, power=0.80

## Claude-Created Functions (scripts/)

**viz_utils.py** (primary Claude contribution):
- `plot_noninferiority_test()`: Equivalence/non-inferiority visualization with CI and SESOI margins
- Distribution plots: Likert scales, categorical, continuous variables
- `plot_boxplot()`, `plot_mirrored_histogram()`, `plot_split_histogram()`
- Scatterplots with correlation statistics
- All functions support grouping by experimental condition

**utils.py**:
- `get_label_for_value()`, `get_value_for_label()`: Label conversions
- `get_question_statement()`: Question text lookup
- `apa_p()`: APA-style p-value formatting

**stats.py** (partial Claude contribution):
- `eta_confidence_interval()`: CI for partial eta-squared via non-central F inversion
- Interpretation/formatting helpers: `interpret_moderation()`, `format_effect_with_stars()`
