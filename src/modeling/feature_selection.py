import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict

def feature_selection_random_forest(X_train: List[Dict], y_train: List[str],
                                    X_test: List[Dict], y_test: List[str],
                                    n_features: int = 20, random_state: int = 42,
                                    n_estimators: int = 500, max_depth: int = 15
                                    ) -> Tuple[List[str], pd.DataFrame, RandomForestClassifier]:
    """
    Realiza la selección de características usando Random Forest, detectando y
    codificando automáticamente TODAS las columnas categóricas con LabelEncoder.
    """

    # 1. Convertir listas de diccionarios a DataFrames. Usamos .copy() para evitar warnings.
    X_train_df = pd.DataFrame(X_train).copy()
    X_test_df = pd.DataFrame(X_test).copy()

    # 2. Detección automática de columnas categóricas (tipo 'object').
    # Se identifican las columnas que necesitan codificación.
    columnas_a_codificar = X_train_df.select_dtypes(include=['object']).columns

    print(f"\nSe detectaron {len(columnas_a_codificar)} columnas categóricas para codificar.")

    # Diccionario para guardar los codificadores ajustados
    encoders = {}

    for columna in columnas_a_codificar:
        if columna not in X_test_df.columns:
            print(f"Advertencia: La columna '{columna}' no está en el set de test. Se omitirá.")
            continue
            
        le = LabelEncoder()
        
        # Ajustar (fit) el encoder con TODOS los posibles valores (train + test)
        full_column_data = pd.concat(
            [X_train_df[columna], X_test_df[columna]], axis=0
        ).astype(str)
        le.fit(full_column_data)
        
        # Transformar las columnas en los dataframes de train y test
        X_train_df[columna] = le.transform(X_train_df[columna].astype(str))
        X_test_df[columna] = le.transform(X_test_df[columna].astype(str))
        
        encoders[columna] = le
    
    print("\nCodificación completada.")

    # 3. Alinear columnas para asegurar consistencia final antes de entrenar.
    # Esto garantiza que ambos DataFrames tengan las mismas columnas en el mismo orden.
    X_train_final, X_test_final = X_train_df.align(X_test_df, join='inner', axis=1, fill_value=0)

    print(f"El número total de características para el modelo es: {X_train_final.shape[1]}")
    
    # 4. Entrenar el modelo Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    print("Entrenando el modelo Random Forest...")
    rf.fit(X_train_final, y_train)

    # 5. Obtener y mostrar la importancia de las características
    feature_importance = rf.feature_importances_
    feature_names = X_train_final.columns
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # 6. Seleccionar y mostrar las 'n' características más importantes
    selected_features = feature_importance_df.head(n_features)['feature'].tolist()

    train_accuracy = rf.score(X_train_final, y_train)
    test_accuracy = rf.score(X_test_final, y_test)
    
    print(f"\nAccuracy en entrenamiento: {train_accuracy:.4f}")
    print(f"Accuracy en test: {test_accuracy:.4f}")
    
    print(f"\nTop {n_features} características seleccionadas:")
    print(feature_importance_df.head(n_features).to_string())

    return selected_features, feature_importance_df, rf