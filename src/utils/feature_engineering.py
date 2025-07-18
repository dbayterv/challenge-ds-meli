
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Any, Dict, Optional

def group_payment_method(payment_method: str) -> str:
    """Clasifica un método de pago en una categoría lógica."""
    if not isinstance(payment_method, str):
        return 'Other'
    credit_cards = ['Visa', 'MasterCard', 'American Express', 'Diners', 'Tarjeta de crédito', 'Visa Electron', 'Mastercard Maestro']
    if payment_method in credit_cards:
        return 'Tarjeta_Credito'
    if payment_method == 'MercadoPago':
        return 'MercadoPago'
    direct_arrangement = ['Acordar con el comprador', 'Efectivo', 'Contra reembolso']
    if payment_method in direct_arrangement:
        return 'Arreglo_Directo'
    manual_transfer = ['Transferencia bancaria', 'Cheque certificado', 'Giro postal']
    if payment_method in manual_transfer:
        return 'Transferencia_Giro'
    return 'Other'

def encode_payment_method_groups(df: pd.DataFrame, base_name: str) -> pd.DataFrame:
    """
    Identifica columnas de descripción de pago, las agrupa en categorías,
    crea columnas binarias para cada categoría y elimina las columnas originales.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        base_name (str): El prefijo de las columnas de descripción a procesar.

    Returns:
        pd.DataFrame: Un DataFrame con las nuevas columnas de grupos de pago 
                      codificadas y las originales eliminadas.
    """
    # 1. Identificar todas las columnas de descripción de pago
    description_cols = [col for col in df.columns if col.startswith(base_name)]

    if not description_cols:
        print(f"Advertencia: No se encontraron columnas que comiencen con '{base_name}'.")
        return df

    # 2. Para cada fila, obtener todos los grupos de pago únicos que ofrece
    def get_groups_for_row(row):
        methods = [row[col] for col in description_cols if pd.notna(row[col])]
        return {group_payment_method(method) for method in methods}

    list_of_groups = df.apply(get_groups_for_row, axis=1)

    # 3. Usar MultiLabelBinarizer para crear columnas binarias (one-hot)
    mlb = MultiLabelBinarizer()
    encoded_groups = pd.DataFrame(
        mlb.fit_transform(list_of_groups),
        columns=[f"payment_group_{c}" for c in mlb.classes_],
        index=df.index
    )

    # 4. Eliminar las columnas de descripción originales
    df_cleaned = df.drop(columns=description_cols)

    # 5. Unir el DataFrame limpio con las nuevas columnas codificadas
    df_final = pd.concat([df_cleaned, encoded_groups], axis=1)
    
    print(f"Proceso completado. Se eliminaron {len(description_cols)} columnas originales y se crearon {len(encoded_groups.columns)} nuevas columnas de grupos de pago.")
    
    return df_final

INPUTATION_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
        "warranty": {"strategy": "constant", "value": "No warranty"},
        "seller_address_city.name": {"strategy": "mode"},
        "seller_address_state.name": {"strategy": "mode"},
        "shipping_free_methods": {"strategy": "binary_presence"}
    }

def impute_data(df: pd.DataFrame, imputation_config: Dict[str, Dict[str, Optional[str]]] = INPUTATION_CONFIG) -> pd.DataFrame:
    """
    Imputa valores nulos en un DataFrame según una configuración específica.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        imputation_config (dict): Un diccionario que mapea nombres de columnas
            a sus estrategias de imputación. Formato esperado:
            {
                'columna_1': {'strategy': 'constant', 'value': 'valor_constante'},
                'columna_2': {'strategy': 'mode'},
                'columna_3': {'strategy': 'binary_presence'}
            }

    Returns:
        pd.DataFrame: Un nuevo DataFrame con los valores imputados.
    """
    df_imputed = df.copy()

    for column, config in imputation_config.items():
        if column not in df_imputed.columns:
            print(f"Advertencia: La columna '{column}' no se encontró en el DataFrame.")
            continue

        strategy = config.get('strategy')
        print(f"Procesando columna '{column}' con estrategia '{strategy}'...")

        if strategy == 'constant':
            fill_value = config.get('value')
            df_imputed[column].fillna(fill_value, inplace=True)
        
        elif strategy == 'mode':
            mode_value = df_imputed[column].mode()[0]
            df_imputed[column].fillna(mode_value, inplace=True)
        
        elif strategy == 'binary_presence':
            # Convierte a True si no es nulo, y a False si es nulo.
            df_imputed[column] = df_imputed[column].notna()
        
        else:
            print(f"Advertencia: Estrategia '{strategy}' no reconocida para la columna '{column}'.")

    return df_imputed

import pandas as pd
import numpy as np
import re
import unicodedata

def _normalize_text(text: str) -> str:
    """Función auxiliar para limpiar y estandarizar texto."""
    if not isinstance(text, str):
        return ""
    # Convertir a minúsculas
    text = text.lower()
    # Quitar acentos
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Quitar caracteres especiales (mantener letras, números y espacios)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

import re
import numpy as np
import pandas as pd
from unidecode import unidecode   # pip install Unidecode

def _clean(txt: str) -> str:
    """Minúsculas, sin acentos ni signos → texto alfanumérico sencillo."""
    if pd.isna(txt):
        return ""
    txt = unidecode(str(txt).lower())
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)          # quita símbolos
    return re.sub(r"\s+", " ", txt).strip()         # colapsa espacios


def add_warranty_features(df: pd.DataFrame,
                          col: str = "warranty",
                          drop_original: bool = True) -> pd.DataFrame:
    """
    Feature‑engineering de la columna `warranty`.

    Crea:
    - warranty_group  : categoría (string) con prioridad definida.
    - con_garantia    : bool
    - sin_garantia    : bool
    - reputacion      : bool
    - _3_6_meses      : bool   (nombres válidos para pandas)

    Args:
    df  : DataFrame de entrada.
    col : nombre de la columna de texto con la garantía.
    drop_original : elimina la columna original si True.

    Returns: 
    DataFrame con nuevas columnas.
    """
    df_ = df.copy()

    # 1) Normalizar texto
    norm = df_[col].apply(_clean)

    # 2) Reglas regexp por grupo (puedes ampliarlas)
    patterns = {
    # 1) Sin garantía explícita
    #    Ej.: “no warranty”, “sin garantia”, “no garantia”
    "sin_garantia": (
        r"\b(?:no|sin)\s+(?:garanti|warranty)\b"          # no garantia / sin garantia
        r"|(?:no\s+warranty\b)"                           # no warranty
    ),

    # 2) Aval de reputación / calificaciones
    #    Ej.: “mi reputacion”, “buenas calificaciones”, “trayectoria”
    "reputacion": (
        r"\b(?:reputacion|calificacion(?:es)?|trayectoria|confianza)\b"
    ),

    # 3) Garantía de 3‑6 meses
    #    Ej.: “3 meses”, “6 meses”, “3 meses de gtia”, “6 meses de garantia”
    "_3_6_meses": (
        r"\b(?:3|6)\s*(?:mes(?:e|o)?s?|dias?)\b"          # 3 meses / 6 meses / 6 dias (por si acaso)
    ),

    # 4) Garantía genérica o mayor (default “con_garantia”)
    #    Incluye: garantía, de fábrica, 12 meses, 1 año, gtia, sellado, por vida
    "con_garantia": (
        r"\b(?:garant[iia]|gtia|garantiz)\w*"             # garantia, garantias, gtia, garantiza
        r"|\b(?:con\s+garant[iia])\b"                     # con garantia
        r"|\b(?:de|por)\s+vida\b"                         # de vida / por vida
        r"|\b\d+\s+(?:mes(?:e|o)?s?|ano?s?)\b"            # 12 meses, 1 ano, 24 meses, etc.
        r"|\b(?:un\s+ano|1\s+ano)\b"                      # un ano, 1 ano
        r"|\b(?:fabrica|de\s+fabrica)\b"                  # fabrica / de fabrica
        r"|\b(?:sellad[oa]s?)\b"                          # sellado / sellada
        r"|\bsi\.?\b"                                     # “Sí” / “Si” (mezclado en muchos textos)
    ),
}

    # 3) Columnas binarias
    for new_col, pat in patterns.items():
        df_[new_col] = norm.str.contains(pat, regex=True, na=False)

    # 4) Columna single‑label con prioridad (np.select es muy rápido)
    priority = ["sin_garantia", "reputacion", "_3_6_meses", "con_garantia"]
    condlist  = [df_[p] for p in priority]
    df_["warranty_group"] = np.select(condlist, priority, default="otros")

    if drop_original:
        df_.drop(columns=[col], inplace=True)

    return df_


import pandas as pd

def calcular_diferencia_meses(df):
    """
    Calcula la diferencia de meses entre las columnas 'start_time' y 'stop_time' de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'start_time' y 'stop_time'.

    Returns:
        pd.DataFrame: DataFrame con la nueva columna 'diferencia_meses'.
    """
    # Convertir las columnas a formato de fecha
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['stop_time'] = pd.to_datetime(df['stop_time'])

    # Calcular la diferencia en meses
    df['diferencia_meses'] = (df['stop_time'].dt.year - df['start_time'].dt.year) * 12 + \
                             (df['stop_time'].dt.month - df['start_time'].dt.month)

    return df

def add_title_flags(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """
    Agrega las columnas:
      - title_cont_usado  (True si el título sugiere artículo usado)
      - title_cont_nuevo  (True si NO parece usado y se menciona 'nuevo')
    
    Parameters
    ----------
    df : DataFrame de entrada que contiene la columna `title`
    title_col : nombre de la columna con los títulos (por defecto 'title')
    
    Returns
    -------
    DataFrame con las nuevas columnas.
    """
    # Copiamos para no modificar el original
    df = df.copy()

    # Normalizamos el texto
    df["_title_norm"] = df[title_col].astype(str).apply(_clean)

    # Patrón para detectar indicios de “usado”
    used_phrases = [
        r"buen estado",
        r"impecable estado",
        r"excelente estado",
        r"increible estado",
        r"perfecto estado",
        r"antiguo",
        r"viejo",
        r"funciona perfecto",
        r"como nuevo",
    ]
    pattern_used = re.compile("|".join(used_phrases))
    df["title_cont_usado"] = df["_title_norm"].str.contains(pattern_used)

    # Patrón para detectar “nuevo” **solo si no se marcó usado**
    df["title_cont_nuevo"] = (
        ~df["title_cont_usado"] & df["_title_norm"].str.contains(r"\bnuevo\b")
    )

    # (Opcional) si no quieres conservar la columna normalizada:
    df.drop(columns=["_title_norm", title_col], inplace=True)

    return df

import pandas as pd
import numpy as np

def set_datatypes(df, numericas, booleanas, categoricas_nom):
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
            print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame.")
    print(f"Columnas categóricas convertidas.")
    
    return df_processed