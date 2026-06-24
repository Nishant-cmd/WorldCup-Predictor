# FIFA World Cup 2026 Prediction System

## Overview

The FIFA World Cup 2026 Prediction System is a Data Mining / Machine learning project that predicts the outcomes of FIFA World Cup matches and estimates tournament-winning probabilities using Monte Carlo Simulation.

The project combines historical international football match data, FIFA rankings, ELO ratings, team form statistics, head-to-head records, and tournament context to build predictive models capable of forecasting match outcomes.

The ultimate goal is to simulate the entire FIFA World Cup 2026 tournament thousands of times and estimate the probability of each team winning the competition.


## Objectives

- Predict football match outcomes:
  - Home Win
  - Draw
  - Away Win

- Analyze the impact of:
  - FIFA Rankings
  - ELO Ratings
  - Team Form
  - Goal Statistics
  - Head-to-Head Records
  - Tournament Importance

- Compare multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost

- Simulate FIFA World Cup 2026 using Monte Carlo Simulation.

- Build an interactive dashboard for visualizing predictions and tournament simulations.



## Features

### Team Strength Features

- Home Team ELO Rating
- Away Team ELO Rating
- ELO Difference

- Home FIFA Ranking
- Away FIFA Ranking
- FIFA Ranking Difference

- Home FIFA Points
- Away FIFA Points
- FIFA Points Difference


### Team Form Features

Performance over the last 5 matches:

- Home Team Form
- Away Team Form
- Form Difference



### Goal Statistics Features

Last 5 Matches:

- Average Goals Scored
- Average Goals Conceded

For both teams:

- Goals Scored Difference
- Goals Conceded Difference


### Match Context Features

- Tournament Importance
- Home Advantage
- Neutral Venue Information


### Head-to-Head Features

Historical records between teams:

- Home Team H2H Wins
- Away Team H2H Wins
- H2H Difference



## Datasets

### 1. International Football Results

Historical international football matches.

Dataset:

https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Used For:

- Match Results
- Goal Statistics
- Head-to-Head Records
- Team Form



### 2. FIFA Men's Rankings

Historical FIFA rankings dataset.

Dataset:

https://www.kaggle.com/datasets/cashncarry/fifaworldranking

Used For:

- FIFA Rankings
- FIFA Points


### 3. International Football ELO Ratings

Historical ELO ratings dataset.

Dataset:

https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings

Used For:

- Team Strength Evaluation
- ELO Difference Features



## Project Structure

## Project Structure

```text
WORLDCUP-PREDICTOR
│
├── dashboard/
│
├── data/
│   ├── raw/
│   │   ├── results.csv
│   │   ├── fifa_mens_rank.csv
│   │   └── eloratings.csv
│   │
│   └── processed/
│       ├── match_data/
│       │   ├── future_matches.csv
│       │   ├── historical_fifa_matches.csv
│       │   ├── historical_matches.csv
│       │   ├── matches_with_elo.csv
│       │   ├── matches_with_elo_fifa.csv
│       │   └── matches_with_form.csv
│       │
│       ├── team_data/
│       │   ├── elo_ratings_cleaned.csv
│       │   ├── fifa_yearly_rankings.csv
│       │   └── worldcup_teams.csv
│       │
│       └── training_data/
│           └── final_training_dataset.csv (main training dataset)
│
├── models/
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
├── notebooks/
│   ├── data_processing/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_fifa_exploration.ipynb
│   │   └── 03_elo_exploration.ipynb
│   │
│   ├── features_modeling/
│   │   ├── 01_feature_engineering.ipynb
│   │   ├── 02_elo_feature_engineering.ipynb
│   │   ├── 03_fifa_feature_engineering.ipynb
│   │   ├── 04_form_feature_engineering.ipynb
│   │   ├── 05_goal_feature_engineering.ipynb
│   │   ├── 06_tournament_importance_feature_engineering.ipynb
│   │   └── 07_head-to-head_feature_engineering.ipynb
│   │
│   └── model_training/
│
├── outputs/
│   └── simulation/
│       ├── rf_result.png
│       └── xgb_result.png
│
├── src/
│   └── scripts/
│       ├── data_utils.py
│       ├── logistic_regression.py
│       ├── random_forest.py
│       └── xgboost_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Directory Description

* **dashboard/** – Dashboard and visualization components.
* **data/raw/** – Original datasets collected from external sources.
* **data/processed/** – Cleaned datasets and engineered features used for training.
* **models/** – Serialized machine learning models (.pkl files).
* **notebooks/** – Jupyter notebooks used for data exploration, feature engineering, and model development.
* **outputs/** – Generated visualizations, simulation results, and evaluation plots.
* **src/scripts/** – Reusable Python scripts for training and utility functions.
* **README.md** – Project documentation.
* **requirements.txt** – Python dependencies required to run the project.


## Installation

### Clone Repository

```bash
git https://github.com/Nishant-cmd/WorldCup-Predictor.git

cd  WORLDCUP_PREDICTOR
```

### Create Virtual Environment

Linux

```bash
python -m venv venv

source venv/bin/activate
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Machine Learning Pipeline

```text
Raw Match Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Training Dataset
        │
        ▼
Model Training
        │
        ▼
Best Model Selection
        │
        ▼
Monte Carlo Simulation
        │
        ▼
World Cup Winner Probabilities
        │
        ▼
Dashboard Visualization
```

---

### Current Progress

- [x] Data Collection
- [x] Data Cleaning
- [x] FIFA Ranking Integration
- [x] ELO Rating Integration
- [x] Team Form Features
- [x] Goal Statistics Features
- [x] Tournament Importance Features
- [x] Head-to-Head Features
- [x] Logistic Regression Model
- [x] Random Forest Model
- [x] XGBoost Model
- [ ] Monte Carlo Simulation
- [ ] Dashboard Development
- [ ] Final World Cup Prediction


## Future Improvements

- Expected Goals (xG) Features
- Player-Level Statistics
- Injury Information
- Hyperparameter Optimization
- Live Match Prediction Dashboard
