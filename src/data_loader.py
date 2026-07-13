"""
Módulo para carga de datos
"""
import pandas as pd
import sqlite3
import os

def cargar_datos_excel(ruta):
    """Carga datos desde archivo Excel"""
    return pd.read_excel(ruta)

def cargar_datos_csv(ruta):
    """Carga datos desde archivo CSV"""
    return pd.read_csv(ruta)

def cargar_datos_sqlite(ruta, tabla='despachos'):
    """Carga datos desde SQLite"""
    conn = sqlite3.connect(ruta)
    return pd.read_sql_query(f"SELECT * FROM {tabla}", conn)
