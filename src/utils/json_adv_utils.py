from __future__ import annotations
import json
from collections import Counter
from typing import List, Dict, Any, Tuple
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def _looks_like_json(value: Any) -> bool:
    """
    Heurística: True si es dict / list / JSON string.
    
    Si es string, verifica si parece JSON (empieza y termina con {} o [] y es parseable).
    Args:
        value (Any): Valor a evaluar.
    Returns:
        bool: True si el valor parece JSON, False en caso contrario.
    """
    if isinstance(value, (dict, list)):
        return True
    if isinstance(value, str):
        value = value.strip()
        if (value.startswith("{") and value.endswith("}")) or (
            value.startswith("[") and value.endswith("]")
        ):
            try:
                json.loads(value)
                return True
            except json.JSONDecodeError:
                return False
    return False


def detect_json_columns(df: pd.DataFrame, sample_size: int = 50) -> List[str]:
    """
    Devuelve lista de nombres de columnas cuyo dtype es 'object'
    y cuyo primer valor no nulo 'parece' JSON según la heurística.
    Se toma una muestra para no escanear todo el DF grande.
    Args:
        df (pd.DataFrame): DataFrame a analizar.
        sample_size (int): Cantidad de muestras a tomar para cada columna.
    Returns:
        List[str]: Lista de nombres de columnas que parecen contener JSON.
    """
    json_cols = []
    for col in df.columns:
        if df[col].dtype != "object":
            continue

        sample = df[col].dropna().head(sample_size)
        if sample.empty:
            continue

        if any(_looks_like_json(v) for v in sample):
            json_cols.append(col)
    return json_cols


def summarize_json_column(
    series: pd.Series, top_n: int = 10
) -> Dict[str, Any]:
    """
    Entrega un dict con:
        • total_rows
        • non_null_rows
        • unique_values
        • top_n (lista de tuplas (valor, frecuencia))
    La lista/JSON se convierte a str para poder contarse.
    Args:
        series (pd.Series): Serie a analizar.
        top_n (int): Cantidad de valores más frecuentes a devolver.
    Returns:
        Dict[str, Any]: Resumen de la columna JSON.
    """
    cleaned = series.dropna().apply(lambda x: json.dumps(x, sort_keys=True))
    counts = Counter(cleaned)
    top = counts.most_common(top_n)

    return {
        "total_rows": len(series),
        "non_null_rows": len(cleaned),
        "unique_values": len(counts),
        "top_n": [(json.loads(k), v) for k, v in top],
    }

def simple_expand_json_column(
    df: pd.DataFrame, column: str, prefix: str | None = None
) -> pd.DataFrame:
    """
    Metodo simple que aplana la columna JSON y la concatena al DataFrame, devolviendo
    un nuevo df sin mutar el original.
    - Para dict → json_normalize.
    - Para list → intenta json_normalize del primer elemento; si es
      lista heterogénea, devuelve exclusivamente la longitud.
    - Si es string JSON, lo convierte a dict.
    Args:
        df (pd.DataFrame): DataFrame original.
        column (str): Nombre de la columna a expandir.
        prefix (str | None): Prefijo para las nuevas columnas.
    Returns:
        pd.DataFrame: DataFrame con la columna expandida.
    """
    ser = df[column]

    def _normalize_cell(cell):
        if isinstance(cell, dict):
            return cell
        if isinstance(cell, list):
            # Si la lista contiene dicts homogéneos, usa el primero como plantilla
            if cell and all(isinstance(x, dict) for x in cell):
                # Flatten keys adding an index suffix: key0, key1...
                return {
                    f"{k}{i}": v for i, item in enumerate(cell) for k, v in item.items()
                }
            # Fallback: longitud
            return {"len": len(cell)}
        # Si es string JSON
        if isinstance(cell, str) and _looks_like_json(cell):
            return json.loads(cell)
        # Cualquier otro caso
        return {}

    flattened = ser.apply(_normalize_cell)
    new_cols = pd.json_normalize(flattened)
    new_cols.index = df.index  # mantener alineación

    if prefix is None:
        prefix = column
    new_cols = new_cols.add_prefix(f"{prefix}_")

    return pd.concat([df.drop(columns=[column]), new_cols], axis=1)

def expand_json_column(
    df: pd.DataFrame, column: str, prefix: str | None = None
) -> pd.DataFrame:
    """
    Aplana una columna JSON-like. Detecta el tipo de contenido (dicts, 
    listas de dicts, o listas de primitivos) y aplica la estrategia adecuada.
        - dicts: Se aplanan con pd.json_normalize.
        - listas de dicts: Se aplanan con claves indexadas (key0, key1...).
        - listas de primitivos: Se aplica One-Hot Encoding (MultiLabelBinarizer).
    Args:
        df (pd.DataFrame): DataFrame original.
        column (str): Nombre de la columna a expandir.
        prefix (str | None): Prefijo para las nuevas columnas.
    Returns:
        pd.DataFrame: DataFrame con la columna expandida.


    """
    series_original = df[column]
    series = series_original.dropna()

    if series.empty:
        return df.drop(columns=[column], errors='ignore')

    # Detectar el tipo de dato del primer elemento no nulo para decidir la estrategia
    first_item = series.iloc[0]
    prefix = prefix or column

    # --- RUTA 1: La columna contiene DICCIONARIOS ---
    if isinstance(first_item, dict):
        new_cols = pd.json_normalize(series).add_prefix(f"{prefix}_")
        new_cols.index = series.index # Asegurar alineación
        return pd.concat([df.drop(columns=[column]), new_cols], axis=1)

    # --- RUTA 2: La columna contiene LISTAS ---
    elif isinstance(first_item, list):
        # Determinar si es una lista de dicts o de primitivos
        first_list_element = next((item for sublist in series if sublist for item in sublist), None)

        # --- RUTA 2a: Lista de DICCIONARIOS ---
        if isinstance(first_list_element, dict):
            def _normalize_list_of_dicts(cell):
                if not isinstance(cell, list) or not cell:
                    return None
                return {f"{k}{i}": v for i, item in enumerate(cell) for k, v in item.items()}

            flattened = series.apply(_normalize_list_of_dicts)
            new_cols = pd.json_normalize(flattened).add_prefix(f"{prefix}_")
            new_cols.index = series.index
            return pd.concat([df.drop(columns=[column]), new_cols], axis=1)

        # --- RUTA 2b: Lista de PRIMITIVOS (tags, etc.) -> APLICAR BINARIZACIÓN ---
        else:
            mlb = MultiLabelBinarizer()
            
            # Asegurar que todos los elementos sean listas para el binarizador
            series_of_lists = series_original.apply(lambda x: x if isinstance(x, list) else [])
            
            dummies = pd.DataFrame(
                mlb.fit_transform(series_of_lists),
                columns=mlb.classes_,
                index=df.index
            ).add_prefix(f"{prefix}_")
            
            return pd.concat([df.drop(columns=[column]), dummies], axis=1)

    # --- RUTA 3: La columna es string JSON (se debe parsear primero) ---
    elif isinstance(first_item, str) and _looks_like_json(first_item):
         # Parsear toda la columna y volver a llamar a la función (recursión)
         df_parsed = df.copy()
         df_parsed[column] = df_parsed[column].apply(lambda x: json.loads(x) if isinstance(x, str) and _looks_like_json(x) else None)
         return expand_json_column(df_parsed, column, prefix)

    # Fallback si no es ninguno de los tipos esperados
    return df.drop(columns=[column])