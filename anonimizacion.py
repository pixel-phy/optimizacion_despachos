import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv("data/processed/despachos_clean.csv")

# ---- ANONIMIZACIÓN ----

# 1. Anonimizar fechas: desplazar días aleatoriamente pero mantener estructura temporal
np.random.seed(42)
dias_shift = np.random.randint(60, 120, size=len(df))
df['fecha'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(dias_shift, unit='D')

# 2. Recalcular semana y mes basado en nuevas fechas
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
df['mes'] = df['fecha'].dt.month_name()
df['dia_semana'] = df['fecha'].dt.day_name()

# 3. Anonimizar id_ruta: reemplazar por números aleatorios
np.random.seed(123)
rutas_originales = df['id_ruta'].unique()
rutas_anonimas = np.random.choice(range(10000, 10000 + len(rutas_originales)), 
                                   size=len(rutas_originales), replace=False)
mapa_rutas = dict(zip(rutas_originales, rutas_anonimas))
df['id_ruta'] = df['id_ruta'].map(mapa_rutas)

# 4. Anonimizar horas (sumar un offset fijo para preservar diferencias)
offset_minutos = np.random.randint(60, 180)  # 1-3 horas de offset
for col in ['hora_inicio_jornada', 'hora_fin_jornada', 'tiempo_inicio_ruta', 
            'tiempo_fin_ruta', 'tiempo_total_jornada', 'tiempo_inicio_ruta_delta', 
            'tiempo_fin_ruta_delta']:
    df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')
    if df[col].notna().any():
        df[col] = (df[col] + pd.Timedelta(minutes=offset_minutos)).dt.strftime('%H:%M:%S')

# 5. Eliminar columna valor_ruta (información financiera sensible)
df = df.drop(columns=['valor_ruta'])

# ---- VERIFICACIÓN ----
print("Columnas después de anonimizar:")
print(df.columns.tolist())
print(f"\nFilas: {len(df)}")
print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
print("\nPrimeras 3 filas:")
print(df.head(3))

# ---- GUARDAR ----
df.to_csv("kaggle_upload/despachos_clean.csv", index=False)
print("\nDatos anonimizados guardados en kaggle_upload/despachos_clean.csv")
