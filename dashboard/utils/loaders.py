"""
Módulo de carga para el dashboard
"""
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

# ============================================================
# IMPORTAR DESDE src (VERSIÓN UNIFICADA)
# ============================================================
try:
    from src.data_loader import CargadorDatos
    from src.optimization import optimizar_asignacion, _asignacion_secuencial, PlanificadorJornada
    from src.models import ModeloPrediccionTiempo, ClusteringRutas
    print("✅ Módulos src importados correctamente")
except ImportError as e:
    print(f"⚠️ Error importando src: {e}")
    # Fallback: clases simplificadas (solo para desarrollo)
    class CargadorDatos:
        def __init__(self, data_path=None, models_path=None):
            self.df_historico = None
            self.feature_order = None
            self.model = None
            self.scaler = None
    
    def optimizar_asignacion(df_pool, n_operadores, **kwargs):
        return {'df_asignacion': pd.DataFrame(), 'cargas_operadores': [], 'makespan': 0}
    
    class PlanificadorJornada:
        def __init__(self):
            self.diagnostico_completo = None

# ============================================================
# FUNCIONES DE CARGA PARA DASHBOARD
# ============================================================

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
        
        # ============================================================
        # 1. CARGAR MODELO - USAR CargadorDatos
        # ============================================================
        cargador = CargadorDatos(
            data_path=str(base_path / 'data' / 'processed' / 'despachos_clean.csv'),
            models_path=str(models_dir)
        )
        
        # ============================================================
        # 2. CARGAR SCALER (opcional)
        # ============================================================
        scaler = None
        scaler_path = models_dir / 'scaler.pkl'
        if scaler_path.exists():
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                st.success("✅ Scaler cargado")
            except Exception as e:
                st.warning(f"⚠️ Error cargando scaler: {str(e)}")
        
        # ============================================================
        # 3. CARGAR METADATOS
        # ============================================================
        metadata = {
            'r2': 0.80,
            'mae': 2.86,
            'n_registros': 584,
            'features': cargador.feature_order if cargador.feature_order else []
        }
        
        metadata_path = models_dir / 'metadatos.pkl'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                st.success("✅ Metadatos cargados")
            except Exception as e:
                st.warning(f"⚠️ Error cargando metadatos: {str(e)}")
        
        # ============================================================
        # 4. VERIFICAR QUE TODO ESTÉ CARGADO
        # ============================================================
        if cargador.model is None:
            st.warning("⚠️ No se pudo cargar el modelo. Usando predicciones dummy.")
            return create_dummy_models()
        
        return {
            'model': cargador.model,
            'scaler': scaler,
            'metadata': metadata,
            'historico': cargador.df_historico,
            'cargador': cargador,
            'feature_order': cargador.feature_order
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
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.warning(f"⚠️ Error cargando datos desde {path}: {str(e)}")
    
    # Si no se encuentran datos, crear dummy
    st.warning("⚠️ No se encontraron datos históricos. Usando datos dummy.")
    return create_dummy_historical_data()

def create_dummy_models():
    """Crea modelos dummy para demostración"""
    st.info("🔄 Usando modelos dummy para demostración")
    
    historico = create_dummy_historical_data()
    
    return {
        'model': None,
        'scaler': None,
        'metadata': {
            'r2': 0.80,
            'mae': 2.86,
            'n_registros': 584,
            'features': []
        },
        'historico': historico,
        'cargador': CargadorDatos(),
        'feature_order': []
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
    
    return [f'2026-06-{i:02d}' for i in range(1, 30)]

def predict_tiempo(model, scaler, ruta_data, dia_semana=None, historico=None):
    """
    Predice el tiempo de preparación para una ruta usando el modelo Random Forest.
    El modelo espera 8 features: cant_productos_log, valor_ruta_log, dia_semana,
    es_jueves, es_lunes_o_viernes, velocidad_historica_ruta, frecuencia_ruta, es_ruta_flash
    """
    if model is None:
        base_time = 15
        variabilidad = np.random.normal(0, 3)
        return max(5, base_time + variabilidad)
    
    try:
        # Obtener ID de la ruta
        ruta_id = ruta_data.get('id_ruta')
        
        # Buscar la ruta en el histórico
        if historico is not None and not historico.empty and ruta_id is not None:
            registros_ruta = historico[historico['id_ruta'] == ruta_id]
            
            if not registros_ruta.empty:
                cant_productos = registros_ruta['cant_productos'].mean()
                valor_ruta = registros_ruta['valor_ruta'].mean()
                
                # Si hay más de un registro, usar el promedio real
                if len(registros_ruta) > 1:
                    tiempo_promedio = registros_ruta['tiempo_preparacion_minutos'].mean()
                    variabilidad = np.random.normal(0, 0.5)
                    return max(5, tiempo_promedio + variabilidad)
            else:
                cant_productos = ruta_data.get('cant_productos', 30)
                valor_ruta = ruta_data.get('valor_ruta', 800)
        else:
            cant_productos = ruta_data.get('cant_productos', 30)
            valor_ruta = ruta_data.get('valor_ruta', 800)
        
        if dia_semana is None:
            dia_semana = 0
        
        # ============================================================
        # CONSTRUIR FEATURES EN EL FORMATO QUE ESPERA EL MODELO
        # ============================================================
        # El modelo espera: cant_productos_log, valor_ruta_log, dia_semana,
        # es_jueves, es_lunes_o_viernes, velocidad_historica_ruta, frecuencia_ruta, es_ruta_flash
        
        # Transformaciones logarítmicas
        cant_productos_log = np.log1p(max(1, cant_productos))
        valor_ruta_log = np.log1p(max(1, valor_ruta))
        
        # Flags
        es_jueves = 1 if dia_semana == 3 else 0
        es_lunes_o_viernes = 1 if dia_semana in [0, 4] else 0
        
        # Valores por defecto
        velocidad_historica = 1500
        frecuencia_ruta = 1
        es_ruta_flash = 0
        
        # Crear DataFrame en el orden correcto
        X = pd.DataFrame([[
            cant_productos_log,
            valor_ruta_log,
            dia_semana,
            es_jueves,
            es_lunes_o_viernes,
            velocidad_historica,
            frecuencia_ruta,
            es_ruta_flash
        ]], columns=[
            'cant_productos_log', 'valor_ruta_log', 'dia_semana',
            'es_jueves', 'es_lunes_o_viernes',
            'velocidad_historica_ruta', 'frecuencia_ruta', 'es_ruta_flash'
        ])
        
        # Si hay scaler, intentar usarlo (pero Random Forest no lo necesita realmente)
        if scaler is not None:
            try:
                # Verificar que el scaler espera las mismas features
                if hasattr(scaler, 'feature_names_in_'):
                    # Reordenar si es necesario
                    X = X[list(scaler.feature_names_in_)]
                X_scaled = scaler.transform(X)
                prediccion = model.predict(X_scaled)[0]
            except Exception as e:
                # Si falla, usar sin escalar
                prediccion = model.predict(X)[0]
        else:
            prediccion = model.predict(X)[0]
        
        return max(5, min(45, prediccion))
    
    except Exception as e:
        st.warning(f"⚠️ Error en predicción: {str(e)}")
        return np.random.normal(15, 3)


def crear_planificador_jornada(cargador=None):
    """Crea una instancia de PlanificadorJornada"""
    from src.optimization import PlanificadorJornada
    return PlanificadorJornada(cargador_instancia=cargador)
