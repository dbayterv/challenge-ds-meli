
from pydantic_settings import BaseSettings
from typing import Dict, Optional

class Config(BaseSettings):
    """
    Clase centralizada para manejar hardcodeos de configuración y constantes del proyecto.
    Permite una fácil modificación y acceso a valores clave sin necesidad de buscar en todo el código
    """
    # Rutas de archivos
    PATH_DATA: str = "/Users/david.bayter/Documents/Private/challenge-ds-meli/data/raw/MLA_100k_checked_v3.jsonlines"


    # Columnas y su imputacion a reallizar
    INPUTATION_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
        "warranty": {"strategy": "constant", "value": "No warranty"},
        "seller_address_city.name": {"strategy": "mode"},
        "seller_address_state.name": {"strategy": "mode"},
        "shipping_free_methods": {"strategy": "binary_presence"}
    }


    col_numericas: list[str] = [
    "base_price", "price", "initial_quantity", "sold_quantity",
    "available_quantity", "diferencia_meses"
    ]
    col_booleanas: list[str] = [
        "accepts_mercadopago", "automatic_relist", "shipping_local_pick_up",
        "shipping_free_shipping", "shipping_free_methods", "sub_status_deleted",
        "sub_status_expired", "sub_status_suspended", "tags_dragged_bids_and_visits",
        "tags_dragged_visits", "tags_free_relist", "tags_good_quality_thumbnail",
        "tags_poor_quality_thumbnail", "payment_group_debit_card",
        "payment_group_mercado_pago", "payment_group_credit_card","reputacion",
        "payment_group_cash","payment_group_transfer", "payment_group_check_or_money_order", "sin_garantia", 
        "_3_6_meses", "con_garantia", "title_cont_usado", "title_cont_nuevo","payment_group_other"
    ]
    col_categoricas: list[str] = [
        "site_id", "buying_mode", "international_delivery_mode", "currency_id",
        "status", "shipping_mode", "seller_address_state.name",
        "seller_address_city.name", "warranty_group", "listing_type_id",
        "seller_id", "category_id" 
    ]


def get_configs() -> Config:
    """
    Obtiene una instancia de la clase Config con los valores predeterminados.

    returns:
    Config: Instancia de la clase Config con los valores predeterminados.
    """
    return Config()

configs = get_configs()