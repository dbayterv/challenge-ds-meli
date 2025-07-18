# Proyecto challenge-ds-meli
Este documento sirve como una guía rápida de las decisiones tomadas para abordar el proyecto y su justificación. Para resultados del análisis exploratorio se puede consultar el EDA en el folder **notebook/EDA.ipynb**

## 1. Contexto y Objetivo  
En el marketplace de MercadoLibre necesitamos predecir si un ítem listado es **nuevo** o **usado**.  
El objetivo fue montar un pipeline completo de carga, limpieza, expansión de JSON, ingeniería de características, selección de las 15 variables más predictivas y entrenamiento/evaluación de un clasificador XGBoost.
El requisito mínimo de rendimiento fue alcanzar un Accuracy de 0.86 en datos no vistos y proponer una segunda métrica alineada al negocio.

## 2. Análisis de Datos  
- Se cargó el dataset original (`.jsonlines`) y se detectaron columnas con contenido JSON.  
- Se decidió expandir sólo aquellas columnas consideradas relevantes tras un análisis de su estructura y ruido:  
  - **Relevantes**: `seller_address`, `sub_status`, `shipping`, `non_mercado_pago_payment_methods`, `tags`  
  - **Descartadas** por bajo valor predictivo o excesivo ruido: `deal_ids`, `variations`, `attributes`, `coverage_areas`, `descriptions`, `pictures`  
- El dataset inicial contenía datos semiestructurados en columnas con formato JSON. Se desarrolló un módulo de utilidades en Python (json_adv_utils.py) para expandir de forma segura aquellas columnas que se consideraban relevantes, transformando la data anidada en un formato tabular plano y manejable.
- Se imputaron valores faltantes con estrategias por columna (constante, moda, presencia binaria).

## 3. Ingeniería de Características  
A partir del set limpio se generaron nuevas variables:  
1. **Agrupación de métodos de pago**  
   - Se mapearon 7 grupos lógicos (`debit_card`, `credit_card`, `mercado_pago`, `cash`, `transfer`, `check_or_money_order`, `other`)  que del algun modo reflejan el perfil del vendedor.
   - Se binarizaron con `MultiLabelBinarizer`.  
2. **Flags en el título**  
    Se crearon variables booleanas basadas en la presencia de palabras clave ("impecable estado", "como nuevo" vs. "perfecto estado").
   - `title_cont_usado`: detecta indicios de “usado”  
   - `title_cont_nuevo`: detecta “nuevo” si no aparece “usado”  
3. **Diferencia de meses**
    Se calculó la diferencia_meses_listado para medir el tiempo que un producto lleva publicado, una posible señal de su condición.
   - `diferencia_meses` entre fecha de inicio y fin del listado  
4. **Normalización y Limpieza**
    Se aplicaron procesos de limpieza para estandarizar valores nulos, vacíos ('', [], {}) y normalizar texto (minúsculas, sin acentos)

Finalmente se establecieron tipos de dato numérico, booleano y categórico para cada columna.

## 4. Selección de Características
Para mejorar el rendimiento, reducir el ruido y aumentar la interpretabilidad del modelo, se realizó un proceso de selección de características.
- Método: Se utilizó un modelo **RandomForestClassifier**  como herramienta de selección, aprovechando su capacidad para calcular la importancia de cada característica.  
- Criterio: La importancia se midió a través de la reducción de la impureza de Gini. Esta métrica evalúa cuán "mezcladas" están las clases (nuevo/usado) en un nodo del árbol de decisión. Una característica es importante si sus decisiones logran separar eficientemente las clases, reduciendo al máximo esta impureza.  
- Se codificaron automáticamente las variables categóricas con `LabelEncoder` (entrenando sobre train+test). 
- Se obtuvieron las **15 features** más importantes según la importancia Gini para alimentar el modelo final. 
- El modelo tuvo un Accuracy en entrenamiento: 0.8865 y Accuracy en test: 0.8631. La brecha de ≈ 2,3 pp. indica que el modelo generaliza correctamente a datos no vistos y no muestra signos de overfitting


### Top 15 características seleccionadas

| Rank | Feature                           | Importance |
|-----:|-----------------------------------|-----------:|
| 1    | initial_quantity                  | **0.1928** |
| 2    | available_quantity                | **0.1897** |
| 3    | listing_type_id                   | **0.1533** |
| 4    | sold_quantity                     | 0.0774 |
| 5    | base_price                        | 0.0615 |
| 6    | price                             | 0.0606 |
| 7    | category_id                       | 0.0394 |
| 8    | title_cont_nuevo                  | 0.0247 |
| 9    | seller_id                         | 0.0223 |
| 10   | seller_address_city.name          | 0.0217 |
| 11   | warranty_group                    | 0.0190 |
| 12   | title_cont_usado                  | 0.0147 |
| 13   | automatic_relist                  | 0.0147 |
| 14   | payment_group_credit_card         | 0.0140 |
| 15   | shipping_mode                     | 0.0109 |

## 5. Modelado con XGBoost  
- Entrenamiento de `XGBClassifier` usando únicamente las 15 variables seleccionadas  
- Preprocesamiento:  
  - Codificación de categorías con `LabelEncoder` (train+test)  
  - Alineación de columnas entre train y test  
- Resultado: **Accuracy = 0.89**, superior al target mínimo de 0.86

## 6. Métrica Secundaria: Precisión de la Clase “new”  
- Se eligió **Precision** sobre la clase positiva (“new”). Se obtuvo un 0.91  
- Justificación:  
  - Una predicción de **“nuevo”** que en realidad sea **“usado”** (falso positivo) genera expectativas incumplidas, devoluciones y pérdida de confianza. Impactando directamente en la reputación del marketplace. 
  - Maximizar la precisión garantiza que los ítems etiquetados como nuevos cumplan con esa condición la mayor parte del tiempo.

## 7. Coherencia de las Features Seleccionadas  
- Se contrastó el top-15 con análisis univariado, bivariado y matriz de correlaciones del EDA  
- Las variables elegidas (precio, cantidad, método de pago, tipo de envío, flags de título, garantía, categoría, etc.) coinciden con los hallazgos exploratorios, confirmando su relevancia. Mayor información revisar **notebook/EDA.ipynb**

## 8. Limitaciones y Mejoras Futuras  
- Varias columnas de **alta cardinalidad** (p. ej. `category_id`, `seller_id`) se codificaron con `LabelEncoder`.  
  - Si bien no afecta mucho a Random Forest, puede impactar el rendimiento de XGBoost.  
  - Con más tiempo, se sugiere técnicas alternativas (target-encoding, agregados de proporción new/used).  
- Algunas columnas JSON complejas quedaron descartadas (p. ej. `attributes`, `variations`, `pictures`) por el esfuerzo de análisis versus aporte esperado. Podrían revisarse en futuros ciclos.

## 9. Conclusiones  
El pipeline desarrollado es capaz de predecir con alta precisión la condición de un producto en MercadoLibre.  
La estrategia de EDA, expansión selectiva de JSON, ingeniería de características y selección basada en impureza Gini permitió acotar el modelo a 15 variables clave, esto se traduce en rendimiento e interpretabilidad.  
El clasificador XGBoost superó el umbral de desempeño esperado, y la elección de maximizar la precisión en la clase “new” responde a una necesidad crítica de negocio: evitar falsos positivos que dañen la experiencia de usuario y la confianza en la plataforma.

Cabe destacar que para esta fase del proyecto se implementó una versión inicial del clasificador con sus parámetros por defecto. Esto demuestra la robustez del feature engineering realizado. Para futuras iteraciones, se recomienda aplicar un proceso de optimización de hiperparámetros que permita explorar configuraciones más avanzadas y potenciar aún más las capacidades predictivas del modelo. 
