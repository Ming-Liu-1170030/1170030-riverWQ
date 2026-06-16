# riverWQ — Predicting E. coli Contamination in NZ Rivers

This project applies machine learning to predict E. coli contamination in New Zealand rivers using catchment-level data, in support of the National Policy Statement for Freshwater Management (NPS-FM 2020). It builds both a regression model (predicting E. coli concentration) and a classification model (predicting the NPS-FM attribute band), and compares their reliability.

## Repository structure

- `data/` — input dataset (`riverWQ_dataset.csv`) and data dictionary
- `src/` — Jupyter notebooks
  - `01_eda.ipynb` — exploratory data analysis (milestone)
  - `02_modelling.ipynb` — modelling work (final submission)
- `figures/` — saved visualisations from the notebooks
- `models/` — 
'regression_mlp.pkl' (final regression model)
'classification_random_forest.pkl' (final classification model)
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

## Key findings

- Catchment-level features explain only about 25% of E. coli variation (regression R² = 0.24). 

- The classifier is more useful for NPS-FM decisions: it identifies band E sites with 0.90 recall and never confuses them with clean bands.

- Both models are reliable on clean catchments but weaker on Pasture and Urban lowland sites — the same profile as Canterbury, which is absent from the dataset.

