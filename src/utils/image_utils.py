import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
display = pd.options.display


def plot_univariate_numerical(data, column, log_scale=False):
    """
    Crea un histograma y un boxplot para una columna numérica.
    Args:
        data (pd.DataFrame): DataFrame que contiene la columna.
        column (str): Nombre de la columna numérica a analizar.
        log_scale (bool): Si es True, aplica escala logarítmica al eje X.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Análisis Univariado de: {column}', fontsize=16)

    # Histograma con KDE
    sns.histplot(data=data, x=column, kde=True, ax=axes[0])
    axes[0].set_title('Distribución (Histograma)')

    # Boxplot
    sns.boxplot(data=data, x=column, ax=axes[1])
    axes[1].set_title('Diagrama de Caja (Boxplot)')

    # --- INICIO DE LA MODIFICACIÓN ---
    if log_scale:
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        fig.suptitle(f'Análisis Univariado de: {column} (Escala Log)', fontsize=16)
    # --- FIN DE LA MODIFICACIÓN ---

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_bivariate_cat_num(data, cat_col, num_col, log_scale=False):
    """
    Crea un boxplot para comparar una variable numérica a través de las
    categorías de una variable categórica.
    Args:
        data (pd.DataFrame): DataFrame que contiene las columnas.
        cat_col (str): Nombre de la columna categórica.
        num_col (str): Nombre de la columna numérica.
        log_scale (bool): Si es True, aplica escala logarítmica al eje Y.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=cat_col, y=num_col)
    title = f'Relación entre {num_col} y {cat_col}'

    # --- INICIO DE LA MODIFICACIÓN ---
    if log_scale:
        plt.yscale('log')
        plt.ylabel(f'{num_col} (Escala Log)')
        title += ' (Escala Log)'
    # --- FIN DE LA MODIFICACIÓN ---
    
    plt.title(title, fontsize=14)
    plt.show()

def plot_correlation_heatmap(data, columns):
    """
    Calcula y grafica la matriz de correlación para las columnas dadas.
    Args:
        data (pd.DataFrame): DataFrame que contiene las columnas.
        columns (list): Lista de nombres de columnas para calcular la correlación.
    """
    # Incluimos el target 'condition' en la correlación
    corr_cols = columns + ['condition']
    correlation_matrix = data[corr_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matriz de Correlación de Variables Numéricas', fontsize=6)
    plt.show()

def plot_univariate_categorical(data, column, top_n=20):
    """
    Crea un countplot para una columna categórica o booleana.
    Muestra las top_n categorías más frecuentes si hay muchas.
    Args:
        data (pd.DataFrame): DataFrame que contiene la columna.
        column (str): Nombre de la columna categórica o booleana.
        top_n (int): Número máximo de categorías a mostrar.
    """
    plt.figure(figsize=(12, 6))
    
    # Si la columna tiene muchas categorías únicas, mostramos solo las más comunes
    if data[column].nunique() > top_n:
        # Obtenemos las top_n categorías
        top_categories = data[column].value_counts().nlargest(top_n).index
        # Filtramos el dataframe para el gráfico
        data_to_plot = data[data[column].isin(top_categories)]
        title = f'Distribución de: {column} (Top {top_n} Categorías)'
        order = top_categories
    else:
        data_to_plot = data
        title = f'Distribución de: {column}'
        order = data[column].value_counts().index

    ax = sns.countplot(data=data_to_plot, x=column, order=order, palette='viridis')
    ax.set_title(title, fontsize=16)
    
    # Rotar etiquetas si son largas
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_bivariate_cat_cat(data, cat_col1, cat_col2, top_n=15):
    """
    Crea un countplot agrupado para ver la relación entre dos
    variables categóricas.
    Args:
        data (pd.DataFrame): DataFrame que contiene las columnas.
        cat_col1 (str): Nombre de la primera columna categórica.
        cat_col2 (str): Nombre de la segunda columna categórica.
        top_n (int): Número máximo de categorías a mostrar en cat_col1.
    """
    plt.figure(figsize=(14, 7))

    # Lógica para mostrar solo las categorías principales de cat_col1
    if data[cat_col1].nunique() > top_n:
        top_categories = data[cat_col1].value_counts().nlargest(top_n).index
        data_to_plot = data[data[cat_col1].isin(top_categories)]
        title = f'Relación entre {cat_col1} (Top {top_n}) y {cat_col2}'
        order = top_categories
    else:
        data_to_plot = data
        title = f'Relación entre {cat_col1} y {cat_col2}'
        order = data[cat_col1].value_counts().index
        
    sns.countplot(data=data_to_plot, x=cat_col1, hue=cat_col2, order=order, palette='magma')
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()