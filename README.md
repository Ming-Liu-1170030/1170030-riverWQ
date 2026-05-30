# riverWQ — Predicting E. coli Contamination in NZ Rivers

This project applies machine learning to predict E. coli contamination in New Zealand rivers using catchment-level data, in support of the National Policy Statement for Freshwater Management (NPS-FM 2020). It builds both a regression model (predicting E. coli concentration) and a classification model (predicting the NPS-FM attribute band), and compares their reliability.

## Repository structure

- `data/` — input dataset (`riverWQ_dataset.csv`) and data dictionary
- `src/` — Jupyter notebooks
  - `01_eda.ipynb` — exploratory data analysis (milestone)
  - `02_modelling.ipynb` — modelling work (final submission)
- `figures/` — saved visualisations from the notebooks
- `models/` — saved trained models (final submission)
- `requirements.txt` — Python package dependencies

## Setup

1. Create a virtual environment:

2. Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. Install dependencies:

## Running the notebooks

Open the notebooks in `src/` and run all cells in order:
1. `01_eda.ipynb`
2. `02_modelling.ipynb` (final submission)
