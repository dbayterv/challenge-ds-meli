from typing import List, Any
import pandas as pd

def drop_column_variations(df: pd.DataFrame, base_column_name: str) -> pd.DataFrame:
    """
    Encuentra y elimina todas las columnas en un DataFrame que comienzan
    con un nombre base seguido de un número.

    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene las columnas a eliminar.
        base_column_name (str): El prefijo común de las columnas que se desean eliminar.
                                (ej. 'non_mercado_pago_payment_methods_id').

    Returns:
        pd.DataFrame: Un nuevo DataFrame sin las columnas eliminadas.
    """
    # Identificar las columnas que comienzan con el nombre base
    cols_to_drop: List[str] = [
        col for col in df.columns if col.startswith(base_column_name)
    ]

    if not cols_to_drop:
        print(f"No se encontraron columnas que comiencen con '{base_column_name}'.")
        return df

    print(f"Eliminando {len(cols_to_drop)} columnas que comienzan con '{base_column_name}'...")
    
    # Eliminar las columnas identificadas
    df_cleaned = df.drop(columns=cols_to_drop)
    
    return df_cleaned


import pandas as pd
import numpy as np
from typing import Any, Callable

def replace_empty_with_nan(
    df: pd.DataFrame,
    inplace: bool = False,
    extra_check: Callable[[Any], bool] | None = None,
) -> pd.DataFrame:
    """
    Reemplaza en todo el DataFrame los «valores vacíos» por `pd.NA`.
    Esta versión es robusta contra celdas que contienen listas o arrays.
    Args:
        df (pd.DataFrame): El DataFrame a procesar.
        inplace (bool): Si es True, modifica el DataFrame original. Si es False,
                       devuelve un nuevo DataFrame.
        extra_check (Callable[[Any], bool], optional): Una función personalizada que recibe un valor
                                                       y devuelve True si el valor debe considerarse vacío.
                                                       Si no se proporciona, se usa un criterio por defecto.
        Returns:
        pd.DataFrame: Un nuevo DataFrame con los valores vacíos reemplazados por `pd.NA`.
    """
    def _is_empty(val: Any) -> bool:
        # 1. Criterio personalizado del usuario
        if extra_check and extra_check(val):
            return True

        # 2. Chequear tipos compuestos PRIMERO para evitar el error con pd.isna()
        if isinstance(val, (list, tuple, set, dict)):
            return len(val) == 0
        if isinstance(val, np.ndarray):
            return val.size == 0
        
        # 3. Chequear strings
        if isinstance(val, str):
            return not val.strip()

        # 4. AHORA es seguro chequear nulos escalares (None, np.nan)
        if pd.isna(val):
            return True
            
        return False

    # Usa .map() que es el método moderno y recomendado en lugar de .applymap()
    df_cleaned = df.map(lambda x: pd.NA if _is_empty(x) else x)

    if inplace:
        df[:] = df_cleaned
        return df
    
    return df_cleaned

def get_unique_values_from_variations(
    df: pd.DataFrame, base_column_name: str
) -> List[Any]:
    """
    Encuentra todas las columnas que comienzan con un nombre base, recopila 
    los valores únicos y no nulos de ellas, y los devuelve en una 
    única lista sin duplicados.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        base_column_name (str): El prefijo común de las columnas a inspeccionar.

    Returns:
        List[Any]: Una lista que contiene todos los valores únicos encontrados 
                   en las columnas correspondientes.
    """
    # 1. Identificar las columnas relevantes que comienzan con el nombre base
    relevant_cols = [col for col in df.columns if col.startswith(base_column_name)]

    if not relevant_cols:
        print(f"Advertencia: No se encontraron columnas que comiencen con '{base_column_name}'.")
        return []

    # 2. Usar un conjunto (set) para recopilar valores únicos y evitar duplicados
    unique_values_set = set()

    # 3. Iterar sobre cada columna relevante
    for col in relevant_cols:
        # Obtener los valores únicos no nulos de la columna
        unique_in_col = df[col].dropna().unique()
        # Agregar estos valores al conjunto principal
        unique_values_set.update(unique_in_col)

    # 4. Convertir el conjunto a una lista para el retorno final
    return list(unique_values_set)

def summarize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza un DataFrame para encontrar valores nulos, generando un reporte.

    Esta función primero elimina las columnas que están completamente vacías.
    Luego, para las columnas restantes, calcula la cantidad y el porcentaje
    de valores nulos, así como el porcentaje de valores no nulos.

    Args:
        df (pd.DataFrame): El DataFrame de entrada a analizar.

    Returns:
        pd.DataFrame: Un DataFrame que resume las métricas de nulos para cada
                      columna, ordenado de mayor a menor por el porcentaje
                      de nulos.
    """
    # 1. Eliminar columnas donde todos los valores son nulos
    df_cleaned = df.dropna(axis='columns', how='all')
    
    # 2. Calcular las métricas
    total_rows = len(df_cleaned)
    null_count = df_cleaned.isnull().sum()
    null_percentage = (null_count / total_rows) * 100
    value_percentage = 100 - null_percentage

    # 3. Crear el DataFrame de resumen
    summary_df = pd.DataFrame({
        'Cantidad de Nulos': null_count,
        '% de Nulos': null_percentage,
        '% de Valores': value_percentage
    })

    # 4. Filtrar y ordenar el reporte para mayor claridad
    # Muestra solo las columnas que tienen al menos un valor nulo
    summary_df = summary_df[summary_df['Cantidad de Nulos'] > 0]
    summary_df = summary_df.sort_values(by='% de Nulos', ascending=False)

    return summary_df