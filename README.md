# challenge-ds-meli

A machine learning pipeline for predicting item condition (new or used) on MercadoLibre marketplace.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Pipeline Steps](#pipeline-steps)
8. [Results](#results)


## Project Overview
This repository implements a complete ML pipeline to predict whether an item listed on MercadoLibre is "new" or "used".
It includes data loading, preprocessing, feature engineering, feature selection, and model training with Random Forest and XGBoost.
Structured logging is provided by **loguru**, and error handling is integrated across modules.

## Features
- **Data Loading:** Reads JSONLines dataset.
- **Data Processing:** Expansion of JSON columns, null handling, payment method grouping, warranty and title features, date difference calculation.
- **Feature Selection:** Random Forest–based selection of top features.
- **Model Training:** Final training and evaluation with Random Forest and XGBoost classifiers.
- **Logging & Error Handling:** Consistent structured logging and exception management via `loguru`.

## Project Structure
```
├── LICENSE
├── README.md
├── main.py               # Entry point for the pipeline
├── MLA_100k_checked_v3.jsonlines  # Dataset (JSONLines)
├── pyproject.toml        # Poetry configuration
├── config/
│   ├── config.py         # Feature lists and other settings
│   └── logger.py         # loguru logger configuration
├── notebooks/
|   └── EDA.ipynb 
├── data/
│   ├── raw/              # Raw JSONLines data
│   └── processed/        # Processed CSV data
│   └── final/            # Final data
├── src/
│   ├── utils/            # Utility modules:
│   │   ├── aux_colums_functions.py
│   │   ├── convert_datatype_utils.py
│   │   ├── feature_engineering.py
│   │   └── json_adv_utils.py
|   |   └── image_utils.py
│   ├── processing/       # Data processing pipeline:
│   │   └── data_processing.py
│   └── modeling/         # Modeling modules:
│       ├── feature_selection.py
│       └── xgboost_training.py
└── tests/                # Unit tests (if any)
```

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/challenge-ds-meli.git
   cd challenge-ds-meli
   ```
2. **Install dependencies** (using Poetry)
   ```bash
   poetry install
   poetry shell
   ```


## Usage
Run the full pipeline:
```bash
python main.py
```
Logs will be displayed in the console with colorized, timestamped output.

## Configuration
- **config/config.py:** Defines lists of numerical, categorical, and boolean columns for type conversion.
- **config/logger.py:** Sets up `loguru` for structured logging across modules.

## Pipeline Steps
1. **Data Loading:** `build_dataset()` reads and splits data.
2. **Data Processing:** `process_data()` applies JSON expansion, cleaning, feature engineering, and data type conversions.
3. **Feature Selection:** `feature_selection_random_forest()` selects top features using a Random Forest classifier.
4. **Random Forest Training:** `train_and_evaluate()` trains and evaluates the final Random Forest on all features to select top 15.
5. **XGBoost Training:** `train_and_evaluate_xgboost()` trains and evaluates an XGBoost classifier on the select features.


## Results
- **Random Forest:** Train accuracy ~0.89, Test accuracy ~0.87.  The slight difference between these scores suggests that the model is generalizing well and is not significantly overfit. This Random Forest Classifier was also used for feature selection, identifying the top 15 most influential features based on the Gini impurity criterion.

- **XGBoost:** XGBoost model achieved comparable performance, with a test accuracy of around 89%. For this model, we prioritized Precision on the positive class, reaching a score of approximately 92%. This metric was selected because the most costly business error is misclassifying a "used" item as "new". A high precision score ensures we minimize this specific type of error

