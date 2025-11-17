# Pass Success Prediction

Machine learning models to predict football pass success using event data with contextual features.

## Project Overview

This project uses StatsBomb open data and the Kloppy library to predict whether a pass will be successful based on contextual features such as pass distance, angle, field position, pressure, and pass type.

**Key Achievement:** 80.4% accuracy (10% better than baseline) with LightGBM model.

## Dataset

- Source: StatsBomb Open Data via Kloppy
- Matches: 7 matches (FIFA World Cup 2018 + NWSL)
- Total Passes: 6,723
- Success Rate: 76.5% baseline

## Model Performance

| Model | Test Accuracy | F1-Score | ROC-AUC |
|-------|--------------|----------|---------|
| LightGBM | 80.4% | 0.866 | 0.853 |
| XGBoost | 79.6% | 0.862 | 0.855 |
| Random Forest | 76.1% | 0.846 | 0.837 |

Baseline (always predict success): 70.4%

## Features (48 total)

- Geometric: distance, angle, progression
- Field position: zones, distance to goals
- Difficulty: long/short/diagonal passes
- Context: pressure, counter attack, pass type

## Technologies

- Kloppy (data loading)
- StatsBomb Open Data
- Scikit-learn, XGBoost, LightGBM
- Pandas, NumPy, Matplotlib

## Installation

```bash
pip install kloppy pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

## Usage

```bash
python src/data_collection/collect_statsbomb_data_v2.py
python src/features/engineer_pass_features.py
python src/models/train_pass_models.py
python src/evaluation/create_visualizations.py
```

## Author

Mehmet Yalcin - https://github.com/mehmetyalc
