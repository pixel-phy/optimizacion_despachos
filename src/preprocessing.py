"""
Módulo de preprocesamiento para datos de despachos
Versión actualizada con las mejoras de la Fase 1
"""
import pandas as pd
import numpy as np

def limpiar_datos(df):
    """
    Limpia y prepara los datos para análisis
    
    Args:
        df (pd.DataFrame): DataFrame original con los datos de despachos
        
    Returns:
        pd.DataFrame: DataFrame limpio y procesado
    """
    df = df.copy()
    
    # 1. Convertir fecha
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 2. Procesar valor_ruta (fórmulas de Excel)
    def calcular_valor_ruta(row):
        valor = row['valor_ruta']
        cantidad = row['cant_productos']
        
        if pd.isna(valor) or valor == '':
            return np.nan
        try:
            if isinstance(valor, str) and '=' in valor:
                return float(3672 * cantidad)
            return float(valor)
        except:
            return np.nan
    
    df['valor_ruta'] = df.apply(calcular_valor_ruta, axis=1)
    
    # 3. Función para convertir tiempos (maneja timedelta, strings, etc.)
    def convertir_a_timedelta(tiempo):
        """Convierte cualquier formato de tiempo a Timedelta"""
        if pd.isna(tiempo):
            return pd.NaT
        if isinstance(tiempo, pd.Timedelta):
            return tiempo
        if hasattr(tiempo, 'hour'):
            return pd.Timedelta(hours=tiempo.hour, minutes=tiempo.minute, seconds=tiempo.second)
        if isinstance(tiempo, str):
            tiempo = tiempo.strip()
            if 'day' in tiempo.lower():
                try:
                    tiempo = tiempo.replace('days', 'day').replace('days,', 'day,')
                    if ',' in tiempo:
                        parts = tiempo.split(',')
                        days = int(parts[0].strip().split()[0])
                        time_part = parts[1].strip()
                    else:
                        parts = tiempo.split()
                        days = int(parts[0])
                        time_part = parts[2] if len(parts) > 2 else parts[1]
                    h, m, s = map(int, time_part.split(':'))
                    return pd.Timedelta(days=days, hours=h, minutes=m, seconds=s)
                except:
                    return pd.NaT
            try:
                parts = tiempo.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return pd.Timedelta(hours=h, minutes=m, seconds=s)
                elif len(parts) == 2:
                    h, m = map(int, parts)
                    return pd.Timedelta(hours=h, minutes=m)
            except:
                pass
        try:
            return pd.to_timedelta(tiempo)
        except:
            return pd.NaT
    
    # 4. Convertir tiempos
    df['tiempo_inicio_ruta_delta'] = df['tiempo_inicio_ruta'].apply(convertir_a_timedelta)
    df['tiempo_fin_ruta_delta'] = df['tiempo_fin_ruta'].apply(convertir_a_timedelta)
    
    # 5. Calcular tiempo de preparación en minutos
    def calcular_tiempo_preparacion(row):
        inicio = row['tiempo_inicio_ruta_delta']
        fin = row['tiempo_fin_ruta_delta']
        
        if pd.isna(inicio) or pd.isna(fin):
            return np.nan
        
        # Si fin es menor que inicio, asumir que pasó de medianoche
        if fin < inicio:
            fin += pd.Timedelta(days=1)
        
        return (fin - inicio).total_seconds() / 60
    
    df['tiempo_preparacion_minutos'] = df.apply(calcular_tiempo_preparacion, axis=1)
    
    # 6. Calcular velocidad de despacho (productos por hora)
    df['velocidad_despacho'] = df['cant_productos'] / (df['tiempo_preparacion_minutos'] / 60)
    
    # 7. Limpiar outliers con criterios actualizados
    # Cambios: 0.5 min (en vez de 1), 50000 (en vez de 10000)
    condiciones = (
        (df['tiempo_preparacion_minutos'] >= 0.5) &  # ← Cambio: 0.5 min
        (df['tiempo_preparacion_minutos'] <= 480) &
        (df['velocidad_despacho'] > 0) &
        (df['velocidad_despacho'] < 50000) &          # ← Cambio: 50000
        (df['cant_productos'] > 0)
    )
    
    df_clean = df[condiciones].copy()
    
    # 8. Clasificar dificultad (umbrales fijos - se mantiene para compatibilidad)
    def clasificar_dificultad(velocidad):
        if velocidad >= 100:
            return 'Fácil'
        elif velocidad >= 50:
            return 'Media'
        else:
            return 'Difícil'
    
    df_clean['mi_dificultad'] = df_clean['velocidad_despacho'].apply(clasificar_dificultad)
    
    # 9. Eliminar columnas auxiliares
    df_clean = df_clean.drop(columns=['tiempo_inicio_ruta_delta', 'tiempo_fin_ruta_delta'], errors='ignore')
    
    return df_clean


def agregar_clasificacion_percentiles(df):
    """
    Agrega clasificación por percentiles (balanceada)
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'velocidad_despacho'
        
    Returns:
        pd.DataFrame: DataFrame con columna 'dificultad_percentiles'
    """
    df = df.copy()
    
    percentil_33 = df['velocidad_despacho'].quantile(0.33)
    percentil_66 = df['velocidad_despacho'].quantile(0.66)
    
    def clasificar_por_percentiles(velocidad):
        if velocidad >= percentil_66:
            return 'Alta (top 33%)'
        elif velocidad >= percentil_33:
            return 'Media (33-66%)'
        else:
            return 'Baja (bottom 33%)'
    
    df['dificultad_percentiles'] = df['velocidad_despacho'].apply(clasificar_por_percentiles)
    
    return df
