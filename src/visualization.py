"""
Módulo de visualizaciones
"""
import matplotlib.pyplot as plt
import seaborn as sns

def graficar_productividad_diaria(df, guardar=True, ruta=None):
    """Genera gráfico de productividad diaria"""
    fig, ax = plt.subplots(figsize=(12, 6))
    # ... código del gráfico
    if guardar and ruta:
        plt.savefig(ruta)
    return fig

def graficar_correlaciones(df, guardar=True, ruta=None):
    """Genera mapa de calor de correlaciones"""
    fig, ax = plt.subplots(figsize=(10, 8))
    # ... código del mapa de calor
    if guardar and ruta:
        plt.savefig(ruta)
    return fig
