# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project for a Master's thesis in Artificial Intelligence at Radboud University, investigating the impact of communicating technical uncertainties on trust in medical AI systems. The research explores whether transparent communication about AI uncertainties affects patients' trust in Medical Decision Support Systems (MDSS).

## Environment Setup

This project uses **uv** for dependency management with Python 3.14+. The development environment is managed through **DataSpell**, which handles the Jupyter server and virtual environment automatically.

### Installing Dependencies
```bash
# Install/sync dependencies (creates .venv automatically)
uv sync
```

The `uv.lock` file ensures reproducible environments across machines.

## Project Structure

### Notebooks
- `preprocessing.ipynb`: Data cleaning and scale computation (work in progress)
- `overview.ipynb`: Results visualization and analysis

### Data Files
- `data/raw_data_qualtrics.csv`: Raw survey data exported from Qualtrics
- `data/questions.csv`: Survey question metadata
- `data/labels.csv`: Variable labels and descriptions

## Experimental Design

### Stimulus Groups
- `uncertainty`: Treatment condition (AI uncertainty communicated)
- `control`: No uncertainty communication

### Measured Constructs
- **Affinity for Technology Interaction (ATI)** (Franke et al., 2019)
  - Single scale measuring technology affinity

- **Revised Health Care System Distrust Scale** (Shea et al., 2008)
  - Note: Interpretation inverted to measure *trust* instead of distrust
  - Subscales:
    - `c`: competence
    - `v`: values

- **Trust in Automation scale (adapted)** (Körber, 2019)
  - Subscales:
    - `rc`: reliability/confidence (capability-based trust)
    - `up`: understanding/predictability (shared mental model)
    - `f`: familiarity
    - `pro`: propensity to trust (faith in technology)
    - `t`: trust in automation (general trust)

## Data Processing Notes

### Current preprocessing workflow (in progress):
1. Remove Qualtrics metadata rows (indices 0, 1)
2. Drop unnecessary timing and metadata columns
3. Combine split `delay_timer_Page Submit` columns into single `page_submit` variable
4. Remove incomplete samples (TODO)
5. Compute scale scores for ATI, healthcare trust, and Trust in Automation

### Key Variables
- `stimulus_group`: Experimental condition
- `page_submit`: Time spent viewing stimulus (seconds)
- `gender`, `age`, `education`, `medical_prof`: Demographics
- Scale item prefixes: `ATI_*`, `TiA_*` (Trust in Automation)

## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib: Visualization
- scipy: Statistical analysis
- notebook: Jupyter environment
- pandas-stubs: Type hints for pandas
