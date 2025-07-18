"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import pandas as pd
from src.processing.data_processing import process_data
from src.modeling.feature_selection import feature_selection_random_forest
from src.modeling.xgboost_training import train_and_evaluate_xgboost
from config.logger import logger


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    logger.info("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()
    
    

    # Insert your code below this line:
    logger.info("Processing data...")
    # Convert to DataFrames for processing
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)

    # Process train and test data
    df_train_processed = process_data(df_train)
    
    df_test_processed = process_data(df_test, reference_df=df_train_processed)

    # Convert back to list of dicts for any subsequent steps if needed
    X_train_processed = df_train_processed.drop(columns=['condition']).to_dict('records')
    y_train_processed = df_train_processed['condition'].tolist()
    
    X_test_processed = df_test_processed.to_dict('records')
    # y_test remains the same as it was not modified

    logger.info("Data processing finished.")
    logger.info(f"Features for modeling: {len(X_train_processed[0].keys())}")
    
    # 2. Feature Selection
    logger.info("Starting feature selection...")
    selected_features, importances, model = feature_selection_random_forest(
        X_train=X_train_processed,
        y_train=y_train_processed,
        X_test=X_test_processed,
        y_test=y_test,
        n_features=15
    )

    # Now you can use the 'selected_features' for the final model training
    logger.info("Feature selection completed.")
    
    # 3. XGBoost Model Training and Evaluation
    train_and_evaluate_xgboost(
        X_train=X_train_processed,
        y_train=y_train_processed,
        X_test=X_test_processed,
        y_test=y_test,
        features=selected_features
    )
    

