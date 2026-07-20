"""
Módulo para carga de datos y modelos
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from datetime import datetime

# ==================================================
# FUNCIONES BÁSICAS DE CARGA
# ==================================================

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


# ==================================================
# CLASE CARGADOR DE DATOS Y MODELOS
# ==================================================

class CargadorDatos:
    """
    Carga el histórico completo de despachos y los modelos de la Fase 2.
    """
    
    def __init__(self, data_path=None, models_path=None):
        # Rutas por defecto relativas al proyecto
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'despachos_clean.csv')
        if models_path is None:
            models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
            
        self.data_path = data_path
        self.models_path = models_path
        self.model = None
        self.metadatos = None
        self.df_historico = None
        self.feature_order = None
        self.scaler = None
        
        self._cargar_artefactos()
        self._cargar_historico()
        
    def _cargar_artefactos(self):
        """Carga el modelo Random Forest y metadatos de la Fase 2"""
        print("\nCargando modelos de la Fase 2...")
        
        try:
            # Cargar modelo Random Forest
            model_path = os.path.join(self.models_path, 'random_forest_model.pkl')
            self.model = joblib.load(model_path)
            print(f"✓ Modelo Random Forest cargado")
            print(f"   Espera {self.model.n_features_in_} features")
            
            # Obtener nombres de features del modelo
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_order = list(self.model.feature_names_in_)
                print(f"   Features: {self.feature_order}")
            else:
                # Fallback
                self.feature_order = [
                    'cant_productos_log', 'valor_ruta_log', 'dia_semana',
                    'es_jueves', 'es_lunes_o_viernes',
                    'velocidad_historica_ruta', 'frecuencia_ruta', 'es_ruta_flash'
                ]
                print(f"   Features (por defecto): {self.feature_order}")
            
            # Cargar scaler (opcional)
            scaler_path = os.path.join(self.models_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler cargado")
                if hasattr(self.scaler, 'feature_names_in_'):
                    print(f"   Features del scaler: {self.scaler.feature_names_in_}")
            else:
                print(f"⚠ Scaler no encontrado. Random Forest no necesita escalado.")
            
            # Cargar metadatos (opcional)
            meta_path = os.path.join(self.models_path, 'metadatos.pkl')
            if os.path.exists(meta_path):
                self.metadatos = joblib.load(meta_path)
                print(f"✓ Metadatos cargados")
                if 'metricas' in self.metadatos:
                    rf_metrics = self.metadatos['metricas'].get('random_forest', {})
                    print(f"   MAE del modelo: {rf_metrics.get('MAE (min)', 'N/A')} min")
            else:
                print(f"⚠ Metadatos no encontrados.")
            
        except Exception as e:
            print(f"Error al cargar modelos: {e}")
            raise
    
    def _cargar_historico(self):
        """Carga el archivo completo de despachos históricos"""
        try:
            self.df_historico = pd.read_csv(self.data_path)
            print(f"\n✓ Datos históricos cargados: {len(self.df_historico)} registros")
            
            required = ['id_ruta', 'cant_productos', 'valor_ruta', 'dia_semana']
            missing = [c for c in required if c not in self.df_historico.columns]
            if missing:
                print(f"⚠ Columnas faltantes: {missing}")
            else:
                print(f"✓ Columnas requeridas verificadas")
            
        except FileNotFoundError:
            print(f"Archivo no encontrado: {self.data_path}")
            raise
    
    def obtener_rutas_especificas(self, lista_ids_ruta):
        """
        Busca en el histórico los datos de las rutas solicitadas.
        
        Preserva las columnas originales del histórico para uso en optimización.
        """
        ids_buscados = [str(r) for r in lista_ids_ruta]
        
        self.df_historico['id_ruta_str'] = self.df_historico['id_ruta'].astype(str)
        mascara = self.df_historico['id_ruta_str'].isin(ids_buscados)
        df_encontradas = self.df_historico[mascara].copy()
        
        if 'fecha' in df_encontradas.columns:
            df_encontradas = df_encontradas.sort_values('fecha', ascending=False)
        
        df_pool = df_encontradas.groupby('id_ruta_str').first().reset_index()
        df_pool['id_ruta'] = df_pool['id_ruta_str']
        df_pool = df_pool.drop(columns=['id_ruta_str'])
        self.df_historico = self.df_historico.drop(columns=['id_ruta_str'], errors='ignore')
        
        ids_encontrados = set(df_pool['id_ruta'].astype(str))
        rutas_faltantes = set(ids_buscados) - ids_encontrados
        
        if rutas_faltantes:
            faltantes_lista = sorted(list(rutas_faltantes), key=lambda x: int(x))
            print(f"\n⚠ {len(rutas_faltantes)} rutas sin datos históricos:")
            if len(faltantes_lista) <= 10:
                print(f"   IDs: {faltantes_lista}")
            else:
                print(f"   Primeras 10: {faltantes_lista[:10]}")
            print(f"   Se estimarán tiempos con promedios del modelo")
            df_nuevas = self._crear_registros_estimados(list(rutas_faltantes))
            df_pool = pd.concat([df_pool, df_nuevas], ignore_index=True)
        
        df_pool = df_pool.reset_index(drop=True)
        df_base = df_pool.copy()
        
        df_features = self._reconstruir_features(df_pool)
        df_predicho = self._predecir_tiempos(df_features)
        
        df_base['tiempo_estimado'] = df_predicho['tiempo_estimado'].values
        df_base = df_base.sort_values('tiempo_estimado', ascending=False).reset_index(drop=True)
        
        print(f"\n✓ {len(df_base)} rutas preparadas para planificación")
        print(f"   Tiempo total estimado: {df_base['tiempo_estimado'].sum():.1f} min ", end="")
        print(f"({df_base['tiempo_estimado'].sum()/60:.1f} horas)")
        print(f"   Tiempo promedio por ruta: {df_base['tiempo_estimado'].mean():.1f} min")
        print(f"   Columnas disponibles: {list(df_base.columns)}")
        
        return df_base
    
    def _crear_registros_estimados(self, lista_ids):
        """Crea registros sintéticos para rutas sin histórico usando promedios"""
        registros = []
        
        prom_cant = self.df_historico['cant_productos'].mean() if 'cant_productos' in self.df_historico.columns else 8
        prom_valor = self.df_historico['valor_ruta'].mean() if 'valor_ruta' in self.df_historico.columns else 250000
        dia_actual = datetime.now().weekday()
        
        for ruta_id in lista_ids:
            registro = {
                'id_ruta': ruta_id,
                'cant_productos': prom_cant,
                'valor_ruta': prom_valor,
                'dia_semana': dia_actual,
                'velocidad_despacho': 1500,
                'frecuencia_ruta': 1,
                'es_ruta_flash': 0
            }
            registros.append(registro)
        
        return pd.DataFrame(registros)
    
    def _reconstruir_features(self, df):
        """
        Prepara features con los nombres que espera el modelo.
        El modelo espera: cant_productos_log, valor_ruta_log, dia_semana,
        es_jueves, es_lunes_o_viernes, velocidad_historica_ruta, frecuencia_ruta, es_ruta_flash
        
        Resetear índice al inicio para evitar problemas de alineación.
        """
        if self.feature_order is None:
            raise ValueError("feature_order no definido. Verificar carga de modelos.")
        
        print(f"\n   Reconstruyendo {len(self.feature_order)} features del modelo:")
        for f in self.feature_order:
            print(f"      - {f}")
        
        df = df.reset_index(drop=True)
        n_filas = len(df)
        
        # 1. Cantidad productos (conversión a log)
        if 'cant_productos' in df.columns:
            cant_productos_raw = df['cant_productos']
        else:
            cant_productos_raw = pd.Series([500] * n_filas, index=df.index)
        cant_productos_log = np.log1p(cant_productos_raw.clip(lower=0))
        print(f"      ✓ cant_productos_log: {cant_productos_log.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 2. Valor ruta (conversión a log)
        if 'valor_ruta' in df.columns:
            valor_ruta_raw = df['valor_ruta']
        else:
            valor_ruta_raw = pd.Series([1000000] * n_filas, index=df.index)
        valor_ruta_log = np.log1p(valor_ruta_raw.clip(lower=0))
        print(f"      ✓ valor_ruta_log: {valor_ruta_log.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 3. Día de semana
        if 'dia_semana' in df.columns:
            dia_semana = df['dia_semana']
            if dia_semana.dtype == 'object':
                dia_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                           'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                dia_semana = dia_semana.map(dia_map)
            dia_semana = pd.to_numeric(dia_semana, errors='coerce').fillna(0)
        else:
            dia_semana = pd.Series([0] * n_filas, index=df.index)
        print(f"      ✓ dia_semana: {dia_semana.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 4. Es jueves
        es_jueves = (dia_semana == 3).astype(int)
        print(f"      ✓ es_jueves: {es_jueves.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 5. Es lunes o viernes
        es_lunes_o_viernes = (dia_semana.isin([0, 4])).astype(int)
        print(f"      ✓ es_lunes_o_viernes: {es_lunes_o_viernes.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 6. Velocidad histórica
        if 'velocidad_historica_ruta' in df.columns:
            velocidad_historica = df['velocidad_historica_ruta']
        elif 'velocidad_despacho' in df.columns:
            velocidad_historica = df['velocidad_despacho']
        else:
            velocidad_historica = pd.Series([1500] * n_filas, index=df.index)
        print(f"      ✓ velocidad_historica_ruta: {velocidad_historica.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 7. Frecuencia de ruta
        if 'frecuencia_ruta' in df.columns:
            frecuencia_ruta = df['frecuencia_ruta']
        else:
            frecuencia_ruta = pd.Series([1] * n_filas, index=df.index)
        print(f"      ✓ frecuencia_ruta: {frecuencia_ruta.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # 8. Es ruta flash
        if 'es_ruta_flash' in df.columns:
            es_ruta_flash = df['es_ruta_flash']
        else:
            es_ruta_flash = pd.Series([0] * n_filas, index=df.index)
        print(f"      ✓ es_ruta_flash: {es_ruta_flash.iloc[0] if n_filas > 0 else 'N/A'}")
        
        # Construir DataFrame con TODAS las features
        df_resultado = pd.DataFrame({
            'cant_productos_log': cant_productos_log,
            'valor_ruta_log': valor_ruta_log,
            'dia_semana': dia_semana,
            'es_jueves': es_jueves,
            'es_lunes_o_viernes': es_lunes_o_viernes,
            'velocidad_historica_ruta': velocidad_historica,
            'frecuencia_ruta': frecuencia_ruta,
            'es_ruta_flash': es_ruta_flash
        })
        
        # Asegurar el orden correcto
        df_resultado = df_resultado[self.feature_order]
        
        print(f"\n   DataFrame features shape: {df_resultado.shape}")
        print(f"   Columnas features: {list(df_resultado.columns)}")
        
        return df_resultado
    
    def _predecir_tiempos(self, df):
        """
        Predice SIN usar el scaler (Random Forest no necesita escalado).
        """
        # Verificar que tenemos 8 features
        if len(df.columns) != self.model.n_features_in_:
            raise ValueError(
                f"Esperaba {self.model.n_features_in_} features, "
                f"pero recibí {len(df.columns)}: {list(df.columns)}"
            )
        
        # Verificar orden de columnas
        if list(df.columns) != self.feature_order:
            raise ValueError(
                f"Orden de columnas incorrecto.\n"
                f"Esperado: {self.feature_order}\n"
                f"Recibido: {list(df.columns)}"
            )
        
        # PREDECIR DIRECTAMENTE - SIN SCALER
        # Random Forest no necesita escalado de features
        X_input = df.values
        predicciones = self.model.predict(X_input)
        
        # Guardar predicción (mínimo 3 minutos)
        df['tiempo_estimado'] = np.maximum(predicciones, 3.0)
        
        return df


# ==================================================
# FUNCIONES DE VALIDACIÓN RÁPIDA
# ==================================================

def verificar_carga():
    """Verifica que los datos se cargan correctamente"""
    try:
        cargador = CargadorDatos()
        print(f"\n✓ Carga exitosa")
        print(f"   Registros históricos: {len(cargador.df_historico)}")
        print(f"   Features del modelo: {len(cargador.feature_order)}")
        return True
    except Exception as e:
        print(f"✗ Error en la carga: {e}")
        return False


def cargar_modelos():
    """Carga solo los modelos sin datos históricos"""
    import joblib
    import os
    
    models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    model = joblib.load(os.path.join(models_path, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
    metadatos = joblib.load(os.path.join(models_path, 'metadatos.pkl'))
    kmeans = joblib.load(os.path.join(models_path, 'kmeans_model.pkl'))
    
    return {
        'model': model,
        'scaler': scaler,
        'metadatos': metadatos,
        'kmeans': kmeans
    }
