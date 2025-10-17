# Predicting Athlete Performance

This repository contains code, data, and a small web demo used to explore and predict athlete performance and injury risk from cleaned soccer dataset. The project combines data preprocessing, traditional machine learning models (regression and classification), optimization and search algorithm implementations to support predictive modelling and decision-making for training load and player management.

This README summarizes the project goals, dataset, how to run training and the demo website, repository layout, and next steps. It was produced from the project's final report and the source tree in this repository.

## Key goals

- Predict player physical performance and fitness metrics (regression models).
- Classify short-term injury risk from recent load and match data (binary classification).
- Compare model performance and evaluate deployment options.
- Provide a small web UI for interactive prediction and visualization.

## Dataset

- Primary dataset: `data/soccer_data_cleaned.csv`
- The CSV contains cleaned match and training load features derived from raw tracking and log data (see Project Report for variable definitions and preprocessing steps).

Notes: the report documents the cleaning steps, feature engineering (rolling windows, per-minute load normalisation), and label construction used to create the training sets.

## What you'll find in the repo

- `predictingModels/` - training scripts for the main predictive models:
	- `train_f_model.py` – train the fitness/performance regression model.
	- `train_p_model.py` – train an alternate performance model (different target/feature set).
	- `train_injury_classifier.py` – train the injury risk classifier.
- `genetic_model/` - Genetic algorithm based modelling/training utilities used in experiments.
- `functions/` and `Predicting-Athlete-Performance-Website/functions/` - implementations of search and optimization helpers used in comparative evaluation and algorithm demos (A*, BFS, DFS, UCS, greedy, CSP, Genetic).
- `Predicting-Athlete-Performance-Website/` - small Flask-based demo site with:
	- `app.py` – web application entrypoint
	- `static/`, `templates/` – front-end assets and pages
	- `predictingModels/` – copies of the training scripts used by the site
- Notebooks: `app.ipynb`, `comparative_evaluation.ipynb` – exploratory analysis and evaluation notebooks.
- `Model_Testing.py`, `Node.py`, `Problem.py` – experiment harnesses and supporting classes used across scripts.

## Requirements

- Install dependencies with pip using the provided `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

If you use a virtual environment, create and activate it before installing.

## Training models (quick start)

The repository includes simple training scripts. Example commands below assume you are in the repository root and have installed dependencies.

Train the fitness/performance model:

```powershell
python predictingModels/train_f_model.py
```

Train the injury classifier:

```powershell
python predictingModels/train_injury_classifier.py
```

Training scripts will read `data/soccer_data_cleaned.csv` by default. Check the top of each script for configurable paths and hyperparameters.

## Run the demo website

The demo is a small Flask app located in `Predicting-Athlete-Performance-Website/`.

From the repository root (Windows PowerShell):

```powershell
cd Predicting-Athlete-Performance-Website
python app.py
```

By default the app will bind to a local port and serve the UI where you can upload or select sample inputs, run predictions, and view visualizations.

## Project structure

Top-level files and folders:

- `data/` - primary dataset CSV(s)
- `predictingModels/` - model training scripts
- `genetic_model/` - genetic algorithm model training (experiment)
- `functions/` - algorithm implementations
- `Predicting-Athlete-Performance-Website/` - web demo (Flask + static assets)
- `app.ipynb`, `comparative_evaluation.ipynb` - interactive notebooks used in the report

## Contributors

The project contributors and their primary roles are listed below:

- Belbakhouche Akram Khaled
	- A* implementation with caching
	- Genetic helper functions (random individual, fitness evaluation, etc.)
	- Tournament selection crossover

- Hamadi Mohammed Abdellah
	- `Node.py` implementation
	- `Problem.py` implementation
	- Genetic logic and operators
	- Alternative model to speed up genetic algorithm

- Blaha Ibrahim
	- Heuristic and cost design and implementation
	- Greedy Best-First implementation
	- Algorithm comparison, evaluation, and testing
	- Website design and documentation page

- Zerraf Bdreddine
	- Uninformed search implementation (BFS, DFS, UCS)
	- Assisted in algorithm comparison, evaluation, and testing
	- Data gathering
	- Website design and API integration assistance

- Kerai Yassine
	- Data preprocessing, cleaning, and visualization
	- Feature engineering
	- Model training and evaluation (performance, risk, fatigue)
	- Back-end implementation

- Kadouci Abdelhack
	- CSP definition and implementation
	- Design of test functions for search algorithms
	- Problem formulation
	- Back-end implementation

## Acknowledgements

See the `Project_Report.pdf` in the repository for full experimental details, figures, and results.

---