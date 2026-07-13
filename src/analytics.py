"""
Módulo de análisis para datos de despachos
Versión actualizada con las mejoras de la Fase 1
"""
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calcular_estadisticas_diarias(df):
    """
    Calcula estadísticas agregadas por día
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        
    Returns:
        pd.DataFrame: Estadísticas diarias
    """
    return df.groupby('fecha').agg({
        'cant_productos': ['sum', 'mean', 'std', 'count'],
        'tiempo_preparacion_minutos': ['sum', 'mean', 'std'],
        'velocidad_despacho': ['mean', 'std']
    }).round(2)

def identificar_mejores_peores_dias(df):
    """
    Identifica los mejores y peores días de rendimiento
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        
    Returns:
        dict: Diccionario con los mejores y peores días
    """
    diario = calcular_estadisticas_diarias(df)
    diario.columns = [
        'total_productos', 'promedio_ruta', 'std_ruta', 'num_rutas',
        'tiempo_total_min', 'tiempo_promedio', 'std_tiempo',
        'velocidad_promedio', 'std_velocidad'
    ]
    
    return {
        'mejor_volumen': diario.loc[diario['total_productos'].idxmax()],
        'peor_volumen': diario.loc[diario['total_productos'].idxmin()],
        'mejor_velocidad': diario.loc[diario['velocidad_promedio'].idxmax()],
        'peor_velocidad': diario.loc[diario['velocidad_promedio'].idxmin()]
    }

def analizar_tendencia(df, columna='velocidad_despacho', grupo='semana'):
    """
    Analiza la tendencia del rendimiento por período
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        columna (str): Columna a analizar (default: 'velocidad_despacho')
        grupo (str): Período de agrupación ('semana', 'mes', 'dia')
        
    Returns:
        dict: Resultados del análisis de tendencia
    """
    df = df.copy()
    
    if grupo == 'semana':
        df['periodo'] = df['fecha'].dt.isocalendar().week
    elif grupo == 'mes':
        df['periodo'] = df['fecha'].dt.to_period('M')
    else:
        df['periodo'] = df['fecha'].dt.date
    
    rendimiento = df.groupby('periodo')[columna].mean()
    
    if len(rendimiento) > 1:
        x = np.arange(len(rendimiento))
        y = rendimiento.values
        mask = ~np.isnan(y)
        
        if mask.sum() > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
            return {
                'slope': slope,
                'r2': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'significativo': p_value < 0.05,
                'tendencia': 'creciente' if slope > 0 else 'decreciente'
            }
    
    return None

def analizar_jornada(df):
    """
    Analiza estadísticas de jornada laboral
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        
    Returns:
        dict: Estadísticas de jornada
    """
    df = df.copy()
    
    # Convertir tiempo_total_jornada a horas
    if 'tiempo_total_jornada' in df.columns:
        df['horas_jornada'] = df['tiempo_total_jornada'].dt.total_seconds() / 3600
        
        # Productividad por hora de jornada
        df['productividad_jornada'] = df['cant_productos'] / df['horas_jornada']
        
        return {
            'promedio_horas': df['horas_jornada'].mean(),
            'min_horas': df['horas_jornada'].min(),
            'max_horas': df['horas_jornada'].max(),
            'std_horas': df['horas_jornada'].std(),
            'productividad_promedio': df['productividad_jornada'].mean(),
            'productividad_min': df['productividad_jornada'].min(),
            'productividad_max': df['productividad_jornada'].max()
        }
    
    return None

def analizar_por_dia_semana(df):
    """
    Analiza el rendimiento por día de la semana
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        
    Returns:
        pd.DataFrame: Rendimiento por día de la semana
    """
    df = df.copy()
    df['dia_semana'] = df['fecha'].dt.day_name()
    
    resultado = df.groupby('dia_semana').agg({
        'velocidad_despacho': 'mean',
        'cant_productos': 'sum',
        'id_ruta': 'count'
    }).round(2)
    
    resultado.columns = ['vel_promedio', 'total_productos', 'num_rutas']
    
    # Ordenar por día de la semana
    orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    resultado = resultado.reindex(orden_dias)
    
    return resultado

def calcular_correlaciones(df):
    """
    Calcula matriz de correlación entre métricas clave
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        
    Returns:
        pd.DataFrame: Matriz de correlación
    """
    columnas = ['cant_productos', 'tiempo_preparacion_minutos', 
                'velocidad_despacho', 'valor_ruta']
    
    # Filtrar columnas existentes
    columnas_existentes = [col for col in columnas if col in df.columns]
    
    return df[columnas_existentes].corr()

def identificar_outliers(df, columna='velocidad_despacho', umbral=10000):
    """
    Identifica outliers en una columna específica
    
    Args:
        df (pd.DataFrame): DataFrame con datos de despachos
        columna (str): Columna a analizar
        umbral (float): Umbral para considerar outlier
        
    Returns:
        pd.DataFrame: DataFrame con los outliers
    """
    return df[df[columna] > umbral].copy()
