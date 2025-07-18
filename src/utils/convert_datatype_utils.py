import pandas as pd
import numpy as np
from typing import List, Any, Callable



def set_datatypes(df: pd.DataFrame, numericas: List[str], booleanas: List[str], categoricas_nom: List[str]) -> pd.DataFrame:
    """
    Convierte las columnas de un DataFrame a los tipos de datos especificados.

    Args:
        df (pd.DataFrame): El DataFrame a procesar.
        numericas (list): Lista de columnas que se convertirán a tipo numérico.
        booleanas (list): Lista de columnas que se convertirán a tipo booleano.
        categoricas_nom (list): Lista de columnas que se convertirán a tipo categórico.

    Returns:
        pd.DataFrame: El DataFrame con los tipos de datos actualizados.
    """
    # Trabajar sobre una copia para evitar advertencias
    df_processed = df.copy()

    # 1. Convertir a numéricas (errores se convierten en NaN)
    df_processed[numericas] = df_processed[numericas].apply(pd.to_numeric, errors='coerce')
    print(f"Columnas numéricas convertidas.")

    # 2. Convertir a booleanas
    df_processed[booleanas] = df_processed[booleanas].astype(bool)
    print(f"Columnas booleanas convertidas.")

    # 3. Convertir a categóricas
    for col in categoricas_nom:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
        else:
            print(f"Error: La columna '{col}' no se encontró en el DataFrame.")
    print(f"Columnas categóricas convertidas.")
    
    return df_processed