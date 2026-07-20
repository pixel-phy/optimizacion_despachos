import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import os

# Agregar el directorio src al path
src_path = Path(__file__).parent.parent.parent / 'src'
if src_path.exists():
    sys.path.append(str(src_path))

try:
    from src.optimization import CargadorDatos, PlanificadorJornada
except ImportError:
    # Si no encuentra el módulo src, crear versiones simplificadas
    class CargadorDatos:
        def __init__(self):
            pass
    
    class PlanificadorJornada:
        def __init__(self):
            self.resultados = {}
        
        def configurar(self, **kwargs):
            self.config = kwargs
        
        def ejecutar(self):
            return self.resultados
        
        def mostrar_diagnostico(self):
            return {}

@st.cache_resource
def load_models():
    """Carga los modelos y metadatos una sola vez usando cache de Streamlit"""
    base_path = Path(__file__).parent.parent.parent
    
    try:
        # Buscar modelos en diferentes ubicaciones posibles
        possible_paths = [
            base_path / 'models',
            Path(__file__).parent.parent / 'models',
            Path.cwd() / 'models',
            Path.cwd().parent / 'models'
        ]
        
        models_dir = None
        for path in possible_paths:
            if path.exists():
                models_dir = path
                break
        
        if models_dir is None:
            st.warning("⚠️ No se encontró el directorio 'models'. Usando datos de ejemplo.")
            return create_dummy_models()
        
        st.info(f"📁 Modelos encontrados en: {models_dir}")
        
        # Intentar cargar el modelo
        model = None
        model_path = models_dir / 'random_forest_model.pkl'
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                st.success("✅ Modelo Random Forest cargado")
            except Exception as e:
                st.warning(f"⚠️ Error cargando model.pkl: {str(e)}")
                # Intentar con joblib
                model_path_joblib = models_dir / 'random_forest_model.joblib'
                if model_path_joblib.exists():
                    try:
                        model = joblib.load(model_path_joblib)
                        st.success("✅ Modelo Random Forest cargado (joblib)")
                    except Exception as e2:
                        st.warning(f"⚠️ Error cargando model.joblib: {str(e2)}")
        
        # Intentar cargar el scaler
        scaler = None
        scaler_path = models_dir / 'scaler.pkl'
        
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler cargado")
            except Exception as e:
                st.warning(f"⚠️ Error cargando scaler.pkl: {str(e)}")
                # Intentar con joblib
                scaler_path_joblib = models_dir / 'scaler.joblib'
                if scaler_path_joblib.exists():
                    try:
                        scaler = joblib.load(scaler_path_joblib)
                        st.success("✅ Scaler cargado (joblib)")
                    except Exception as e2:
                        st.warning(f"⚠️ Error cargando scaler.joblib: {str(e2)}")
        
        # Cargar metadatos
        metadata = {
            'r2': 0.80,
            'mae': 2.86,
            'n_registros': 584,
            'features': ['cant_productos_promedio', 'valor_ruta_promedio']
        }
        
        metadata_path = models_dir / 'metadatos.pkl'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                st.success("✅ Metadatos cargados")
            except Exception as e:
                st.warning(f"⚠️ Error cargando metadatos: {str(e)}")
        
        # Cargar datos históricos
        historico = load_historical_data()
        
        # Si no se pudo cargar el modelo o el scaler, usar dummy
        if model is None or scaler is None:
            st.warning("⚠️ No se pudieron cargar modelo y/o scaler. Usando predicciones dummy.")
            return create_dummy_models()
        
        # Crear cargador
        cargador = CargadorDatos()
        
        return {
            'model': model,
            'scaler': scaler,
            'metadata': metadata,
            'historico': historico,
            'cargador': cargador
        }
    
    except Exception as e:
        st.error(f"❌ Error cargando modelos: {str(e)}")
        return create_dummy_models()

def load_historical_data():
    """Carga los datos históricos desde diferentes ubicaciones"""
    base_path = Path(__file__).parent.parent.parent
    
    possible_paths = [
        base_path / 'data' / 'processed' / 'despachos_clean.csv',
        Path(__file__).parent.parent / 'data' / 'despachos_clean.csv',
        Path.cwd() / 'data' / 'processed' / 'despachos_clean.csv',
        Path.cwd() / 'data' / 'despachos_clean.csv'
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.info(f"📊 Datos históricos cargados: {len(df)} registros")
                return df
            except Exception as e:
                st.warning(f"⚠️ Error cargando datos desde {path}: {str(e)}")
    
    # Si no se encuentran datos, crear dummy
    st.warning("⚠️ No se encontraron datos históricos. Usando datos dummy.")
    return create_dummy_historical_data()

def create_dummy_models():
    """Crea modelos dummy para demostración cuando no se encuentran los archivos"""
    st.info("🔄 Usando modelos dummy para demostración")
    
    # Datos históricos dummy
    historico = create_dummy_historical_data()
    
    return {
        'model': None,
        'scaler': None,
        'metadata': {
            'r2': 0.80,
            'mae': 2.86,
            'n_registros': 584,
            'features': ['cant_productos_promedio', 'valor_ruta_promedio']
        },
        'historico': historico,
        'cargador': CargadorDatos()
    }

def create_dummy_historical_data():
    """Crea datos históricos dummy para demostración"""
    np.random.seed(42)
    
    fechas = pd.date_range('2026-06-01', '2026-07-17', freq='D')
    
    data = []
    for fecha in fechas:
        n_rutas = np.random.randint(20, 35)
        for i in range(n_rutas):
            data.append({
                'fecha': fecha.strftime('%Y-%m-%d'),
                'id_ruta': np.random.randint(10000, 99999),
                'tiempo_preparacion_minutos': np.random.normal(15, 5),
                'cant_productos': np.random.randint(5, 50),
                'valor_ruta': np.random.uniform(100, 1000),
                'dia_semana': fecha.dayofweek,
                'hora_inicio_jornada': '15:00',
                'hora_fin_jornada': '22:00',
                'horas_jornada': 7.0,
                'velocidad_despacho': np.random.uniform(0.5, 2.0)
            })
    
    return pd.DataFrame(data)

@st.cache_data
def load_available_dates():
    """Carga las fechas disponibles para validación"""
    try:
        historico = load_historical_data()
        if 'fecha' in historico.columns:
            fechas = sorted(historico['fecha'].unique())
            return fechas
    except Exception as e:
        st.warning(f"Error cargando fechas: {str(e)}")
    
    # Si no hay datos, usar fechas dummy
    return [f'2026-06-{i:02d}' for i in range(1, 30)]

def predict_tiempo(model, scaler, ruta_data, dia_semana=None, historico=None):
    """
    Predice el tiempo de preparación para una ruta usando datos del histórico.
    """
    if model is None or scaler is None:
        base_time = 15
        variabilidad = np.random.normal(0, 3)
        return max(5, base_time + variabilidad)
    
    try:
        # Obtener ID de la ruta
        ruta_id = ruta_data.get('id_ruta')
        
        # ============================================================
        # 1. BUSCAR LA RUTA EN EL HISTÓRICO
        # ============================================================
        if historico is not None and not historico.empty and ruta_id is not None:
            # Buscar registros de esta ruta en el histórico
            registros_ruta = historico[historico['id_ruta'] == ruta_id]
            
            if not registros_ruta.empty:
                # Usar el promedio de los valores históricos
                cant_productos = registros_ruta['cant_productos'].mean()
                valor_ruta = registros_ruta['valor_ruta'].mean()
                
                # Si hay más de un registro, calcular el tiempo promedio real
                if len(registros_ruta) > 1:
                    # Usar el tiempo promedio real como predicción
                    tiempo_promedio = registros_ruta['tiempo_preparacion_minutos'].mean()
                    # Añadir un poco de variabilidad
                    variabilidad = np.random.normal(0, 0.5)
                    return max(5, tiempo_promedio + variabilidad)
            else:
                # Si no hay histórico, usar valores por defecto más realistas
                cant_productos = ruta_data.get('cant_productos', 30)
                valor_ruta = ruta_data.get('valor_ruta', 800)
        else:
            # Si no hay histórico, usar valores proporcionados
            cant_productos = ruta_data.get('cant_productos', 30)
            valor_ruta = ruta_data.get('valor_ruta', 800)
        
        if dia_semana is None:
            dia_semana = 0  # Lunes
        
        # ============================================================
        # 2. CREAR DATAFRAME CON EL ORDEN EXACTO
        # ============================================================
        columnas = [
            'cant_productos', 'valor_ruta',
            'dia_semana_Friday', 'dia_semana_Monday',
            'dia_semana_Saturday', 'dia_semana_Sunday',
            'dia_semana_Thursday', 'dia_semana_Tuesday',
            'dia_semana_Wednesday'
        ]
        
        valores = [
            cant_productos, valor_ruta,
            1 if dia_semana == 4 else 0,
            1 if dia_semana == 0 else 0,
            1 if dia_semana == 5 else 0,
            1 if dia_semana == 6 else 0,
            1 if dia_semana == 3 else 0,
            1 if dia_semana == 1 else 0,
            1 if dia_semana == 2 else 0,
        ]
        
        X = pd.DataFrame([valores], columns=columnas)
        
        # ============================================================
        # 3. ESCALAR Y PREDECIR
        # ============================================================
        X_scaled = scaler.transform(X)
        prediccion = model.predict(X_scaled)[0]
        
        # ============================================================
        # 4. AJUSTAR EL RESULTADO PARA QUE SEA REALISTA
        # ============================================================
        # Si la predicción es muy baja (<10 min), usar el promedio histórico
        if prediccion < 10 and historico is not None and not historico.empty:
            # Usar el promedio general de tu histórico
            tiempo_promedio_historico = historico['tiempo_preparacion_minutos'].mean()
            # Mezclar con la predicción para no perder completamente el modelo
            prediccion = (prediccion + tiempo_promedio_historico) / 2
        
        return max(5, min(45, prediccion))
    
    except Exception as e:
        st.warning(f"⚠️ Error en predicción: {str(e)}")
        # Fallback con distribución realista
        return np.random.normal(15, 3)
