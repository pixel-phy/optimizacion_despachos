#!/usr/bin/env python3
"""
Script de automatización semanal: reentrena modelos con nuevos datos.

Uso:
    python entrenar_modelos.py

Flujo:
    1. Ejecuta NB1 (limpieza y EDA)
    2. Ejecuta NB2 (reentrenamiento ML)
    3. Verifica que los archivos se hayan generado correctamente
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# CONFIGURACIÓN

PROJECT_ROOT = Path(__file__).parent
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# Archivos que deben existir después de la ejecución
ARCHIVOS_VERIFICAR = [
    DATA_DIR / 'despachos_clean.csv',
    MODELS_DIR / 'random_forest_model.pkl',
    MODELS_DIR / 'metadatos.pkl',
    MODELS_DIR / 'scaler.pkl',
    MODELS_DIR / 'kmeans_model.pkl',
]

# FUNCIONES

def print_header(titulo):
    """Imprime un encabezado formateado"""
    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)


def ejecutar_notebook(nombre):
    """
    Ejecuta un notebook Jupyter de principio a fin.
    
    Args:
        nombre: Nombre del archivo .ipynb
    
    Returns:
        bool: True si se ejecutó correctamente
    """
    notebook_path = NOTEBOOKS_DIR / nombre
    
    if not notebook_path.exists():
        print(f"ERROR: No se encontró {notebook_path}")
        return False
    
    print(f"\nEjecutando {nombre}...")
    print(f"   Esto puede tomar 1-2 minutos...")
    
    try:
        resultado = subprocess.run(
            [
                sys.executable, '-m', 'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',  # Sobreescribe el mismo archivo
                '--ExecutePreprocessor.timeout=300',  # 5 min timeout
                str(notebook_path)
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if resultado.returncode == 0:
            print(f"{nombre} ejecutado correctamente")
            return True
        else:
            print(f"Error ejecutando {nombre}:")
            print(f"   {resultado.stderr[-500:]}")  # Últimas 500 líneas
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout: {nombre} tardó más de 5 minutos")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False


def verificar_archivos():
    """Verifica que los archivos de salida existan"""
    print("\nVerificando archivos generados...")
    todos_ok = True
    
    for archivo in ARCHIVOS_VERIFICAR:
        if archivo.exists():
            tamaño_kb = archivo.stat().st_size / 1024
            print(f"{archivo.name}: {tamaño_kb:.1f} KB")
        else:
            print(f"FALTA: {archivo.name}")
            todos_ok = False
    
    return todos_ok


def mostrar_resumen():
    """Muestra un resumen de lo que se generó"""
    print("\n" + "=" * 60)
    print("  RESUMEN DE ARCHIVOS GENERADOS")
    print("=" * 60)
    
    # Datos
    csv_path = DATA_DIR / 'despachos_clean.csv'
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\nDatos procesados:")
        print(f"   Registros: {len(df):,}")
        print(f"   Columnas: {len(df.columns)}")
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            print(f"   Rango: {df['fecha'].min().strftime('%Y-%m-%d')} → {df['fecha'].max().strftime('%Y-%m-%d')}")
            print(f"   Días: {df['fecha'].nunique()}")
            print(f"   Rutas únicas: {df['id_ruta'].nunique()}")
    
    # Modelo
    import joblib
    meta_path = MODELS_DIR / 'metadatos.pkl'
    if meta_path.exists():
        metadatos = joblib.load(meta_path)
        print(f"\nModelo entrenado:")
        print(f"   Algoritmo: {metadatos.get('mejor_modelo', 'N/A')}")
        print(f"   Fecha: {metadatos.get('fecha_entrenamiento', 'N/A')[:10]}")
        if 'metricas' in metadatos:
            rf = metadatos['metricas'].get('random_forest', {})
            print(f"   R²: {rf.get('R²', 'N/A')}")
            print(f"   MAE: {rf.get('MAE (min)', 'N/A')} min")
        print(f"   Features: {len(metadatos.get('features', []))}")


# PROGRAMA PRINCIPAL

if __name__ == '__main__':
    inicio = datetime.now()
    
    print_header("ENTRENAMIENTO AUTOMÁTICO DE MODELOS")
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio del proyecto: {PROJECT_ROOT}")
    
    # Verificar que el Excel fuente existe
    excel_path = DATA_DIR.parent / 'raw' / 'io_data_bimbo.xlsx'
    if not excel_path.exists():
        print(f"\nERROR: No se encontró {excel_path}")
        print("   Asegúrate de haber agregado los nuevos datos al Excel antes de ejecutar este script.")
        sys.exit(1)
    else:
        tamaño_mb = excel_path.stat().st_size / (1024 * 1024)
        print(f"\nDatos fuente: io_data_bimbo.xlsx ({tamaño_mb:.1f} MB)")
    
    # ============================================================
    # PASO 1: NB1 - Limpieza y EDA
    # ============================================================
    print_header("PASO 1/2: Limpieza y EDA (NB1)")
    
    if ejecutar_notebook('01_eda_cleaning.ipynb'):
        print("Datos limpios generados")
    else:
        print("\nFalló NB1. Abortando.")
        sys.exit(1)
    
    # ============================================================
    # PASO 2: NB2 - Machine Learning
    # ============================================================
    print_header("PASO 2/2: Entrenamiento ML (NB2)")
    
    if ejecutar_notebook('02_machine_learning.ipynb'):
        print("Modelo reentrenado")
    else:
        print("\nFalló NB2. Abortando.")
        sys.exit(1)
    
    # ============================================================
    # VERIFICACIÓN
    # ============================================================
    print_header("VERIFICACIÓN FINAL")
    
    if verificar_archivos():
        print("\nTodos los archivos generados correctamente")
    else:
        print("\nFaltan algunos archivos. Revisa los errores anteriores.")
    
    mostrar_resumen()
    
    # ============================================================
    # FINAL
    # ============================================================
    fin = datetime.now()
    duracion = (fin - inicio).total_seconds()
    
    print_header("ENTRENAMIENTO COMPLETADO")
    print(f"   Duración total: {duracion:.0f} segundos")
    print(f"   Finalizado: {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n   Próximo paso: Reinicia el Dashboard con:")
    print(f"      streamlit run run_dashboard.py")
    print()
