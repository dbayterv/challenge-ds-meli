import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any
from config.logger import logger

def train_and_evaluate_xgboost(
    X_train: List[Dict[str, Any]],
    y_train: List[str],
    X_test: List[Dict[str, Any]],
    y_test: List[str],
    features: List[str]
) -> None:
    """
    Trains an XGBoost Classifier on the specified features and evaluates it.
    This function handles label encoding for categorical features.

    Args:
        X_train (List[Dict[str, Any]]): Training data.
        y_train (List[str]): Training labels.
        X_test (List[Dict[str, Any]]): Test data.
        y_test (List[str]): Test labels.
        features (List[str]): List of feature names to use for training.
    """
    logger.info("--- Starting XGBoost Model Training ---")

    # Convert to DataFrames
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)

    # Select only the specified features
    df_train = df_train[features]
    df_test = df_test[features]

    # Label encode target variable
    le_y = LabelEncoder()
    y_train_encoded = le_y.fit_transform(y_train)
    y_test_encoded = le_y.transform(y_test)

    # Identify and encode categorical features
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns
    encoders = {}

    if not categorical_cols.empty:
        logger.info(f"Encoding categorical features for XGBoost: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on combined train and test data to handle all possible values
            combined_data = pd.concat([df_train[col], df_test[col]], axis=0).astype(str)
            le.fit(combined_data)
            
            df_train[col] = le.transform(df_train[col].astype(str))
            df_test[col] = le.transform(df_test[col].astype(str))
            encoders[col] = le

    # Align columns - crucial for consistency
    train_cols = df_train.columns
    df_test = df_test[train_cols]

    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(df_train, y_train_encoded)
    logger.info("XGBoost model training completed.")

    # Make predictions
    y_pred_encoded = model.predict(df_test)
    y_pred = le_y.inverse_transform(y_pred_encoded)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info(f"XGBoost Model Accuracy: {accuracy:.4f}")
    logger.info("XGBoost Classification Report:\n" + report)
    logger.info("--- XGBoost Training Finished ---")
