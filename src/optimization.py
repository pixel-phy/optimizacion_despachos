"""
Módulo de optimización y simulación para planificación de jornadas de bodega.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class SimuladorJornada:
    """
    Simulador de jornada operativa para toma de decisiones en bodega.
    
    Casos de uso:
    1. Planificación diaria: Estimar tiempo total de jornada
    2. Gestión de personal: Identificar necesidad de apoyo
    3. Balance de carga: Distribuir rutas equitativamente
    """
    
    def __init__(self, modelo_prediccion, modelo_clustering, feature_cols, ruta_data):
        """
        Inicializa simulador con modelos entrenados.
        
        Parámetros:
        - modelo_prediccion: Modelo de regresión entrenado
        - modelo_clustering: Modelo K-Means entrenado
        - feature_cols: Lista de features del modelo
        - ruta_data: DataFrame con datos históricos de rutas
        """
        self.modelo = modelo_prediccion
        self.clustering = modelo_clustering
        self.feature_cols = feature_cols
        self.ruta_data = ruta_data
        
        # Umbrales operativos
        self.JORNADA_ESTANDAR_MIN = 480  # 8 horas
        self.ALERTA_CRITICA_MIN = 120     # Rutas > 2 horas son críticas
        
    def preparar_features(self, ruta_id: int) -> pd.DataFrame:
        """Prepara vector de features para una ruta específica"""
        ruta_features = self.ruta_data[
            self.ruta_data['id_ruta'] == ruta_id
        ][self.feature_cols].iloc[0:1]
        
        if len(ruta_features) == 0:
            raise ValueError(f"Ruta {ruta_id} no encontrada en datos históricos")
            
        return ruta_features
    
    def simular_jornada(self, 
                        lista_rutas: List[int], 
                        con_recomendaciones: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Simula una jornada completa y genera recomendaciones operativas.
        
        Parámetros:
        - lista_rutas: Lista de IDs de rutas a despachar
        - con_recomendaciones: Si genera recomendaciones operativas
        
        Retorna:
        - DataFrame con resultados por ruta
        - Diccionario con métricas y recomendaciones de jornada
        """
        
        resultados = []
        tiempo_total = 0
        alertas_criticas = []
        
        for ruta_id in lista_rutas:
            try:
                # Predicción de tiempo
                X_ruta = self.preparar_features(ruta_id)
                tiempo_pred = self.modelo.predict(X_ruta)[0]
                
                # Clasificación de perfil
                try:
                    perfil = self.clustering.predecir_cluster(X_ruta)
                    dificultad = self.clustering.cluster_profile.loc[
                        perfil[0], 'dificultad'
                    ] if self.clustering.cluster_profile is not None else 'No clasificada'
                except:
                    dificultad = 'No clasificada'
                
                # Registrar resultado
                resultados.append({
                    'ruta_id': ruta_id,
                    'tiempo_estimado_min': round(tiempo_pred, 1),
                    'tiempo_estimado_hrs': round(tiempo_pred/60, 2),
                    'dificultad': dificultad,
                    'critica': tiempo_pred > self.ALERTA_CRITICA_MIN
                })
                
                tiempo_total += tiempo_pred
                
            except Exception as e:
                print(f"Error procesando ruta {ruta_id}: {str(e)}")
                continue
        
        # Métricas de jornada
        df_resultados = pd.DataFrame(resultados)
        
        metricas_jornada = {
            'tiempo_total_min': round(tiempo_total, 1),
            'tiempo_total_hrs': round(tiempo_total/60, 2),
            'rutas_totales': len(resultados),
            'rutas_criticas': df_resultados['critica'].sum() if len(df_resultados) > 0 else 0,
            'excede_jornada': tiempo_total > self.JORNADA_ESTANDAR_MIN,
            'horas_extra': max(0, (tiempo_total - self.JORNADA_ESTANDAR_MIN) / 60)
        }
        
        # Generar recomendaciones
        if con_recomendaciones:
            metricas_jornada['recomendaciones'] = self._generar_recomendaciones(
                df_resultados, metricas_jornada
            )
        
        return df_resultados, metricas_jornada
    
    def _generar_recomendaciones(self, df_resultados: pd.DataFrame, 
                                 metricas: Dict) -> List[str]:
        """
        Genera recomendaciones operativas basadas en resultados.
        
        Decisiones de negocio:
        - Asignación de personal según perfil de ruta
        - Necesidad de horas extra
        - Balance de carga entre operadores
        """
        recomendaciones = []
        
        # Recomendaciones por perfil
        if 'dificultad' in df_resultados.columns:
            estrellas = df_resultados[df_resultados['dificultad'].str.contains('Estrella', na=False)]
            pesadas = df_resultados[df_resultados['dificultad'].str.contains('Pesadas', na=False)]
            
            if len(estrellas) > 0:
                recomendaciones.append(
                    f"{len(estrellas)} rutas 'Estrella': Asignar a personal en entrenamiento"
                )
            
            if len(pesadas) > 0:
                recomendaciones.append(
                    f"{len(pesadas)} rutas 'Pesadas': Requieren personal experto y posible apoyo"
                )
        
        # Recomendaciones por tiempo
        if metricas['excede_jornada']:
            horas_extra = metricas['horas_extra']
            if horas_extra <= 2:
                recomendaciones.append(
                    f"{horas_extra:.1f} hrs extra: Evaluar horas extra o redistribuir rutas ligeras"
                )
            else:
                recomendaciones.append(
                    f"{horas_extra:.1f} hrs extra: IMPRESCINDIBLE redistribuir o añadir personal"
                )
        else:
            holgura = (self.JORNADA_ESTANDAR_MIN - metricas['tiempo_total_min']) / 60
            if holgura > 2:
                recomendaciones.append(
                    f"{holgura:.1f} hrs de holgura: Posibilidad de añadir rutas adicionales"
                )
        
        # Balance de carga
        if metricas['rutas_totales'] > 5:
            recomendaciones.append(
                "Recomendación de balance: Asignar mix de rutas Estrella+Normales+Pesadas a cada operador"
            )
        
        # Rutas críticas
        if metricas['rutas_criticas'] > 0:
            rutas_criticas_list = df_resultados[df_resultados['critica']]['ruta_id'].tolist()
            recomendaciones.append(
                f"Rutas críticas detectadas ({metricas['rutas_criticas']}): Rutas {rutas_criticas_list} - Priorizar y asignar apoyo"
            )
        
        return recomendaciones
    
    def optimizar_asignacion(self, lista_rutas: List[int], 
                            num_operadores: int) -> Dict[int, List[int]]:
        """
        Optimiza asignación de rutas a operadores balanceando carga.
        
        Algoritmo: Greedy con balance de tiempo estimado.
        """
        # Predecir tiempos para todas las rutas
        tiempos = {}
        for ruta in lista_rutas:
            X = self.preparar_features(ruta)
            tiempos[ruta] = self.modelo.predict(X)[0]
        
        # Ordenar rutas por tiempo descendente (las más pesadas primero)
        rutas_ordenadas = sorted(tiempos.items(), key=lambda x: x[1], reverse=True)
        
        # Inicializar operadores
        operadores = {i: {'rutas': [], 'tiempo_total': 0} for i in range(num_operadores)}
        
        # Asignación greedy balanceada
        for ruta, tiempo in rutas_ordenadas:
            # Encontrar operador con menor carga
            operador_min = min(operadores.items(), key=lambda x: x[1]['tiempo_total'])[0]
            operadores[operador_min]['rutas'].append(ruta)
            operadores[operador_min]['tiempo_total'] += tiempo
        
        return {op: datos['rutas'] for op, datos in operadores.items()}
