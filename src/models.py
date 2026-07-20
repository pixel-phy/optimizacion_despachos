"""
Módulo de Machine Learning para Analítica de Rendimiento Personal.
Implementa modelos predictivos y clustering para optimización de bodega.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os

class ModeloPrediccionTiempo:
    """
    Modelo de predicción de tiempos de preparación para rutas de despacho.
    
    Decisiones de negocio que habilita:
    - Planificación precisa de jornadas
    - Identificación de cuellos de botella
    - Asignación óptima de personal
    """
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = RobustScaler()  # Cambiado a RobustScaler
        self.feature_cols = None
        self.mejor_modelo = None
        
    def entrenar(self, X_train, y_train, feature_cols):
        """
        Entrena modelos Random Forest y XGBoost.
        """
        self.feature_cols = feature_cols
        
        # Random Forest - Robusto a outliers en datos de bodega
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_train, y_train)
        
        self._seleccionar_mejor_modelo(X_train, y_train)
        
    def _seleccionar_mejor_modelo(self, X_val, y_val):
        """Selecciona el mejor modelo basado en MAE"""
        pred_rf = self.rf_model.predict(X_val)
        pred_xgb = self.xgb_model.predict(X_val)
        
        mae_rf = mean_absolute_error(y_val, pred_rf)
        mae_xgb = mean_absolute_error(y_val, pred_xgb)
        
        self.mejor_modelo = 'Random Forest' if mae_rf < mae_xgb else 'XGBoost'
        
    def predecir(self, X):
        """Predice tiempo de preparación usando el mejor modelo"""
        if self.mejor_modelo == 'Random Forest':
            return self.rf_model.predict(X)
        else:
            return self.xgb_model.predict(X)
    
    def get_feature_importance(self):
        """Retorna importancia de features para análisis operativo"""
        if self.mejor_modelo == 'Random Forest':
            importancia = self.rf_model.feature_importances_
        else:
            importancia = self.xgb_model.feature_importances_
            
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importancia
        }).sort_values('importance', ascending=False)
    
    def guardar_modelo(self, path='../models/'):
        """Persiste modelos entrenados para uso en producción"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.rf_model, f'{path}random_forest_model.pkl')
        joblib.dump(self.xgb_model, f'{path}xgboost_model.pkl')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        
    def cargar_modelo(self, path='../models/'):
        """Carga modelos previamente entrenados"""
        self.rf_model = joblib.load(f'{path}random_forest_model.pkl')
        self.xgb_model = joblib.load(f'{path}xgboost_model.pkl')
        self.scaler = joblib.load(f'{path}scaler.pkl')


class ClusteringRutas:
    """
    Segmentación de rutas usando K-Means para optimización operativa.
    
    Perfiles de negocio:
    - Estrella: Alta velocidad, bajo esfuerzo -> Asignar a juniors
    - Normales: Velocidad media -> Asignación estándar
    - Pesadas: Baja velocidad, alto esfuerzo -> Requieren personal experto
    """
    
    def __init__(self):
        self.kmeans = None
        self.scaler = RobustScaler()
        self.k_optimo = None
        self.cluster_profile = None
        
    def encontrar_k_optimo(self, X, k_range=range(2, 11)):
        """Encuentra K óptimo usando método de silueta."""
        from sklearn.metrics import silhouette_score
        
        silhouettes = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouettes.append(silhouette_score(X, labels))
        
        self.k_optimo = k_range[np.argmax(silhouettes)]
        return self.k_optimo
    
    def entrenar(self, X, k=None):
        """Entrena K-Means y perfila clusters operativamente."""
        if k is None:
            k = self.k_optimo if self.k_optimo else 3
            
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.kmeans.fit(X)
        
        return self.kmeans.labels_
    
    def perfilar_clusters(self, df_cluster, labels):
        """Genera perfil operativo de cada cluster."""
        df_cluster['cluster'] = labels
        
        self.cluster_profile = df_cluster.groupby('cluster').agg(
            num_rutas=('id_ruta', 'count'),
            productos_promedio=('cant_productos_promedio', 'mean'),
            valor_promedio=('valor_ruta_promedio', 'mean'),
            tiempo_promedio=('tiempo_promedio', 'mean'),
            velocidad_promedio=('velocidad_promedio', 'mean')
        ).round(1)
        
        self.cluster_profile['dificultad'] = pd.cut(
            self.cluster_profile['velocidad_promedio'],
            bins=[0, 842, 2279, float('inf')],
            labels=['Pesadas', 'Normales', 'Estrella']
        )
        
        return self.cluster_profile
    
    def predecir_cluster(self, X):
        """Asigna cluster a nuevas rutas"""
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
