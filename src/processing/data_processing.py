import pandas as pd
from typing import List, Dict

# Importar las funciones de utilidad que creaste en tu notebook
from src.utils.json_adv_utils import detect_json_columns, expand_json_column
from src.utils.aux_colums_functions import drop_column_variations, replace_empty_with_nan, get_unique_values_from_variations, summarize_nulls
from src.utils.feature_engineering import group_payment_method, impute_data, add_warranty_features, calcular_diferencia_meses, add_title_flags
from src.utils.convert_datatype_utils import set_datatypes
from config.config import configs

def process_data(df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aplica el preprocesamiento completo a un DataFrame.
    Opcionalmente, alinea las columnas con un DataFrame de referencia.
    """
    # 1. Expansión de columnas JSON
    col_to_exclude = ['deal_ids', 'variations', 'attributes', 'coverage_areas', 'descriptions', 'pictures']
    raw_json_cols = detect_json_columns(df)
    json_cols = [col for col in raw_json_cols if col not in col_to_exclude]
    
    df_expanded_parts = []
    for col in json_cols:
        df_flat = expand_json_column(df, col, prefix=col).drop(columns=df.columns, errors='ignore')
        if not df_flat.empty:
            df_expanded_parts.append(df_flat)
    
    if df_expanded_parts:
        df_concatenated = pd.concat(df_expanded_parts, axis=1)
    else:
        df_concatenated = pd.DataFrame()

    # 2. Eliminar columnas expandidas no relevantes
    no_relevant_columns = ['non_mercado_pago_payment_methods_id', 'non_mercado_pago_payment_methods_type', 'seller_address_country.id',  'seller_address_state.id',  'seller_address_city.id']
    df_cleaned_concat = df_concatenated.copy()
    for col in no_relevant_columns:
        df_cleaned_concat = drop_column_variations(df_cleaned_concat, col)

    # 3. Codificar métodos de pago
    df_cleaned_concat_enc = group_payment_method(
        df_cleaned_concat, 
        'non_mercado_pago_payment_methods_description'
    )

    # 4. Unir con el DataFrame original
    df_concat_raw = pd.concat([df.drop(columns=raw_json_cols, errors='ignore'), df_cleaned_concat_enc], axis=1)

    # 5. Limpieza de nulos y columnas irrelevantes
    df_concat_raw = replace_empty_with_nan(df_concat_raw)
    list_col_drop = ['catalog_product_id','shipping_dimensions','shipping_tags', 'original_price', 'official_store_id', 'video_id' , 'parent_item_id', 'thumbnail','secure_thumbnail', 'permalink', 'subtitle', 'seller_address_country.name'
                 ,'differential_pricing','shipping_methods', 'listing_source', 'site_id', 'id']
    for col in list_col_drop:
        df_concat_raw = drop_column_variations(df_concat_raw, col)
    
    df_concat_imputed = impute_data(df_concat_raw)

    # 6. Feature Engineering
    df_procesado = add_warranty_features(df_concat_imputed, col='warranty', drop_original=True)
    df_procesado = calcular_diferencia_meses(df_procesado)
    list_col_drop_dates = ['date_created','stop_time', 'start_time', 'last_updated']
    for col in list_col_drop_dates:
        df_procesado = drop_column_variations(df_procesado, col)
    
    df_procesado_flat = add_title_flags(df_procesado, title_col='title')
    
    # 7. Establecer tipos de datos
    # Primero, obtenemos las columnas que realmente existen en el DF actual
    # antes de intentar convertirlas.
    existing_numericas = [col for col in configs.col_numericas if col in df_procesado_flat.columns]
    existing_categoricas = [col for col in configs.col_categoricas if col in df_procesado_flat.columns]
    existing_booleanas = [col for col in configs.col_booleanas if col in df_procesado_flat.columns]

    df_final = set_datatypes(df_procesado_flat, numericas=existing_numericas, categoricas_nom=existing_categoricas, booleanas=existing_booleanas)

    # 8. Alinear con el DataFrame de referencia si se proporciona
    if reference_df is not None:
        ref_cols = reference_df.columns
        current_cols = df_final.columns
        
        missing_cols = set(ref_cols) - set(current_cols)
        for c in missing_cols:
            # Evitar añadir la columna target al set de test
            if c != 'condition':
                df_final[c] = 0
        
        # Asegurar que el orden de las columnas es el mismo y no hay columnas extra
        df_final = df_final.reindex(columns=ref_cols.drop('condition', errors='ignore'), fill_value=0)

    return df_final
