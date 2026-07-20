"""
Módulo de optimización y simulación para planificación de jornadas de bodega.
"""

import pandas as pd
import numpy as np
import pulp
from typing import List, Dict, Tuple
from scipy import stats
import os
import matplotlib.pyplot as plt

# ============================================================
# OPTIMIZACIÓN CON PuLP
# ============================================================

def optimizar_asignacion(df_pool, n_operadores, verbose=True, tiempo_limite=30):
    """
    Optimiza la asignación de rutas a operadores usando Programación Lineal Entera.
    
    Esta función recibe las rutas con sus tiempos estimados y las distribuye entre los operadores disponibles para
    minimizar el makespan.
    
    Args:
        df_pool: DataFrame con columnas ['id_ruta', 'tiempo_estimado']
        n_operadores: Número de operadores disponibles para la jornada
        verbose: Si True, muestra detalles del proceso de optimización
        tiempo_limite: Segundos máximos para el solver CBC
        
    Returns:
        dict con la asignación óptima y métricas:
            - df_asignacion: DataFrame con columnas [id_ruta, operador, tiempo_estimado]
            - cargas_operadores: Lista con carga total de cada operador
            - makespan: Tiempo del operador que más tarda (min)
            - n_operadores, n_rutas, tiempo_total, status
    """
    
    # Validación de datos de entrada
    required = ['id_ruta', 'tiempo_estimado']
    for col in required:
        if col not in df_pool.columns:
            raise ValueError(f"Columna '{col}' no encontrada en df_pool")
    
    rutas = df_pool['id_ruta'].values
    tiempos = df_pool['tiempo_estimado'].values
    m = len(rutas)   # Número de rutas a asignar
    n = n_operadores # Número de operadores disponibles
    
    if verbose:
        print(f"\nOptimizando asignación de {m} rutas para {n} operadores...")
        print(f"Tiempo total a distribuir: {sum(tiempos):.1f} min ({sum(tiempos)/60:.1f} horas)")
    
    # Crear problema de optimización
    prob = pulp.LpProblem("Planificacion_Diaria_Despachos", pulp.LpMinimize)
    
    # Variables de decisión binarias: x[i][j] = 1 si ruta i va al operador j
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(n)), 
                              cat=pulp.LpBinary)
    
    # Variable continua: makespan (tiempo del operador que más tarda)
    C_max = pulp.LpVariable("C_max", lowBound=0, cat=pulp.LpContinuous)
    
    # Función objetivo: minimizar makespan
    prob += C_max, "Minimizar_makespan"
    
    # Restricción 1: Cada ruta asignada a un solo operador
    for i in range(m):
        prob += pulp.lpSum(x[(i, j)] for j in range(n)) == 1, f"Ruta_{i}_unica"
    
    # Restricción 2: Carga por operador <= makespan
    for j in range(n):
        prob += pulp.lpSum(tiempos[i] * x[(i, j)] for i in range(m)) <= C_max, f"Carga_Op_{j}"
    
    # Restricción 3: Ruptura de simetría (Op_0 >= Op_1 >= Op_2 ...)
    # Esto acelera el solver al reducir el espacio de búsqueda
    if n > 1:
        for j in range(n - 1):
            carga_j = pulp.lpSum(tiempos[i] * x[(i, j)] for i in range(m))
            carga_j_sig = pulp.lpSum(tiempos[i] * x[(i, j + 1)] for i in range(m))
            prob += carga_j >= carga_j_sig, f"Simetria_Op_{j}_vs_{j+1}"
            if verbose:
                print(f"   Restricción de simetría: Operador {j+1} >= Operador {j+2}")
    
    # Resolver el problema
    solver = pulp.PULP_CBC_CMD(
        msg=verbose,              # Mostrar progreso del solver
        timeLimit=tiempo_limite,  # Límite de tiempo en segundos
        gapRel=0.05               # 5% de gap de optimalidad aceptable
    )
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    
    # MEJORA: Validar si existe una solución aprovechable
    # (Optimal o factible post-timeout)
    tiene_solucion = (status == 'Optimal') or (C_max.varValue is not None and C_max.varValue > 0)
    
    if not tiene_solucion:
        if verbose:
            print(f"Estado del solver: {status} (sin solución factible)")
            print("Activando asignación secuencial (plan B)...")
        return _asignacion_secuencial(df_pool, n_operadores)
    
    # Extraer asignación de la solución
    asignaciones = []
    for i in range(m):
        for j in range(n):
            valor_x = pulp.value(x[(i, j)])
            if valor_x is not None and valor_x > 0.5:
                asignaciones.append({
                    'id_ruta': rutas[i],
                    'operador': j + 1,  # Numerar operadores desde 1
                    'tiempo_estimado': tiempos[i]
                })
    
    df_asignacion = pd.DataFrame(asignaciones)
    
    # Calcular métricas de la solución
    cargas = []
    for j in range(n):
        carga_j = df_asignacion[df_asignacion['operador'] == j+1]['tiempo_estimado'].sum()
        cargas.append(carga_j)
    
    makespan = max(cargas) if cargas else 0
    
    # Mostrar resultados
    if verbose:
        print(f"\nAsignación completada (Estado: {status})")
        print(f"Makespan planificado: {makespan:.1f} min ({makespan/60:.1f} horas)")
        print(f"Distribución de cargas:")
        for j, carga in enumerate(cargas):
            n_rutas_op = len(df_asignacion[df_asignacion['operador'] == j+1])
            pct = (carga / sum(cargas)) * 100 if sum(cargas) > 0 else 0
            print(f"      Operador {j+1}: {carga:.1f} min ({carga/60:.1f}h) | {n_rutas_op} rutas | {pct:.1f}%")
        
        # Mostrar balanceo
        if len(cargas) > 1:
            desv = np.std(cargas)
            cv = (desv / np.mean(cargas)) * 100 if np.mean(cargas) > 0 else 0
            print(f"Balanceo: desviación std = {desv:.1f} min (CV = {cv:.1f}%)")
    
    return {
        'df_asignacion': df_asignacion,
        'cargas_operadores': cargas,
        'makespan': makespan,
        'n_operadores': n_operadores,
        'n_rutas': m,
        'tiempo_total': sum(tiempos),
        'status': status
    }


def _asignacion_secuencial(df_pool, n_operadores):
    """
    Plan B: Asignación secuencial cuando el solver no encuentra solución óptima.
    
    Algoritmo:
    1. Ordena rutas de mayor a menor tiempo estimado
    2. Asigna cada ruta al operador con menor carga acumulada
    3. Garantiza que todas las rutas sean asignadas
    """
    print("\n" + "="*50)
    print("EJECUTANDO ASIGNACIÓN SECUENCIAL (PLAN B)")
    print("="*50)
    
    # Ordenar rutas de mayor a menor tiempo para mejor balanceo
    df_ordenado = df_pool.sort_values('tiempo_estimado', ascending=False)
    cargas = [0] * n_operadores
    asignaciones = []
    
    for _, row in df_ordenado.iterrows():
        # Encontrar operador con menor carga actual
        op_idx = np.argmin(cargas)
        operador = op_idx + 1
        
        asignaciones.append({
            'id_ruta': row['id_ruta'],
            'operador': operador,
            'tiempo_estimado': row['tiempo_estimado']
        })
        
        cargas[op_idx] += row['tiempo_estimado']
    
    df_asignacion = pd.DataFrame(asignaciones)
    makespan = max(cargas)
    
    print(f"Asignación secuencial completada")
    print(f"Makespan: {makespan:.1f} min ({makespan/60:.1f} horas)")
    for j, carga in enumerate(cargas):
        n_rutas_op = len(df_asignacion[df_asignacion['operador'] == j+1])
        print(f"      Operador {j+1}: {carga:.1f} min ({carga/60:.1f}h) | {n_rutas_op} rutas")
    
    return {
        'df_asignacion': df_asignacion,
        'cargas_operadores': cargas,
        'makespan': makespan,
        'n_operadores': n_operadores,
        'n_rutas': len(df_pool),
        'tiempo_total': sum(cargas),
        'status': 'Secuencial'
    }


# ============================================================
# SIMULADOR CON 3 CAPAS DE REALISMO
# ============================================================

class SimuladorJornada:
    """
    Simula la jornada con 3 capas de realismo:
    1. Variabilidad natural en tiempos de preparación
    2. Eventos programados (reuniones, capacitaciones)
    3. Interrupciones aleatorias (descargas, imprevistos)
    """
    
    def __init__(self, df_asignacion, mae_modelo=3.03, seed=42):
        self.df_asignacion = df_asignacion.copy()
        self.seed = seed
        self.n_operadores = df_asignacion['operador'].nunique()
        self.mae_modelo = mae_modelo
    
    def simular(self, n_iteraciones=1000, eventos_programados=None, 
                eventos_aleatorios=None, hora_inicio_jornada=7):
        """
        Ejecuta simulación Monte Carlo de la jornada completa.
        """
        if eventos_programados is None:
            eventos_programados = []
        if eventos_aleatorios is None:
            eventos_aleatorios = []
        
        # Convertir eventos programados a minutos de jornada
        interrupciones_fijas = self._procesar_eventos_programados(
            eventos_programados, hora_inicio_jornada
        )
        
        # Inicializar acumuladores
        resultados = {
            'makespans': [],
            'cargas_operadores': {j: [] for j in range(1, self.n_operadores + 1)},
            'interrupciones_ocurridas': [],
            'eventos_programados': eventos_programados,
            'eventos_aleatorios': eventos_aleatorios,
            'n_iteraciones': n_iteraciones
        }
        
        print(f"\nIniciando simulación Monte Carlo...")
        print(f"   Iteraciones: {n_iteraciones}")
        print(f"   Operadores: {self.n_operadores}")
        print(f"   Rutas asignadas: {len(self.df_asignacion)}")
        print(f"   MAE del modelo: {self.mae_modelo} min")
        
        # Ejecutar iteraciones Monte Carlo
        for iteracion in range(n_iteraciones):
            # Generador local para cada iteración
            rng = np.random.default_rng(self.seed + iteracion)
            
            # Generar interrupciones aleatorias para esta iteración
            interrupciones_aleatorias = self._generar_aleatorias(
                eventos_aleatorios, rng
            )
            
            # Combinar todas las interrupciones y ordenarlas cronológicamente
            todas_interrupciones = interrupciones_fijas + interrupciones_aleatorias
            todas_interrupciones.sort(key=lambda e: e['minuto_jornada'])
            
            # Simular procesamiento de rutas por operador
            tiempos_operador = {j: 0.0 for j in range(1, self.n_operadores + 1)}
            
            # Procesar ruta por ruta según la asignación
            for _, row in self.df_asignacion.iterrows():
                operador = int(row['operador'])
                tiempo_base = row['tiempo_estimado']
                
                # Tiempo real de la ruta con variabilidad estocástica (Capa 1)
                tiempo_real = rng.normal(tiempo_base, self.mae_modelo)
                tiempo_real = max(tiempo_real, 1.0)  # Mínimo 1 minuto
                
                inicio_ruta = tiempos_operador[operador]
                fin_estimado_ruta = inicio_ruta + tiempo_real
                
                # Calcular pausas por interrupciones que afectan a esta ruta (Capa 2 y 3)
                pausa_total = 0.0
                for interr in todas_interrupciones:
                    if operador in interr['afecta_operadores']:
                        inicio_int = interr['minuto_jornada']
                        fin_int = inicio_int + interr['duracion_min']
                        
                        # Evaluación de solapamiento de intervalos
                        if max(inicio_ruta, inicio_int) < min(fin_estimado_ruta + pausa_total, fin_int):
                            pausa_total += interr['duracion_min']
                
                # Actualizar tiempo acumulado del operador
                tiempos_operador[operador] += (tiempo_real + pausa_total)
            
            # Registrar resultados de esta iteración
            makespan = max(tiempos_operador.values())
            resultados['makespans'].append(makespan)
            resultados['interrupciones_ocurridas'].append(len(interrupciones_aleatorias))
            
            for op, tiempo in tiempos_operador.items():
                resultados['cargas_operadores'][op].append(tiempo)
        
        # Calcular estadísticas agregadas
        self._calcular_estadisticas(resultados)
        
        # Mostrar resumen
        print(f"\nSimulación completada")
        print(f"Makespan promedio: {resultados['mean_makespan']:.1f} min ({resultados['mean_makespan']/60:.1f}h)")
        print(f"Makespan P95: {resultados['p95_makespan']:.1f} min ({resultados['p95_makespan']/60:.1f}h)")
        print(f"Prob. horas extra (>8h): {resultados['prob_horas_extra']:.1f}%")
        print(f"Balanceo (std entre operadores): {resultados['balanceo_std']:.1f} min")
        print(f"Interrupciones promedio: {resultados['promedio_interrupciones']:.1f} por jornada")
        
        return resultados

    def _procesar_eventos_programados(self, eventos, hora_inicio):
        """Convierte eventos con hora del día a minutos transcurridos de jornada."""
        interrupciones = []
        minuto_inicio_jornada = hora_inicio * 60
        
        for evento in eventos:
            hora_str = evento['hora']
            h, m = map(int, hora_str.split(':'))
            minuto_dia = h * 60 + m
            minuto_jornada = max(0, minuto_dia - minuto_inicio_jornada)
            
            afecta = evento.get('afecta', 'todos')
            if afecta == 'todos':
                operadores = list(range(1, self.n_operadores + 1))
            elif isinstance(afecta, list):
                operadores = afecta
            else:
                operadores = [int(afecta)]
            
            interrupciones.append({
                'minuto_jornada': minuto_jornada,
                'duracion_min': evento.get('duracion', 15),
                'afecta_operadores': operadores,
                'descripcion': evento.get('descripcion', 'Evento programado')
            })
        
        return interrupciones

    def _generar_aleatorias(self, eventos_aleatorios, rng):
        """Genera interrupciones aleatorias usando el generador NumPy estipulado."""
        interrupciones = []
        duracion_max = 600  # Ventana de 10 horas de observación
        
        for evento in eventos_aleatorios:
            prob_hora = evento.get('probabilidad', 0.1)
            duracion_media = evento.get('duracion_media', 15)
            duracion_std = evento.get('duracion_std', 8)
            
            for minuto in range(0, duracion_max, 60):
                if rng.random() < prob_hora:
                    minuto_exacto = minuto + rng.integers(0, 60)
                    duracion = max(1.0, rng.normal(duracion_media, duracion_std))
                    
                    afecta = evento.get('afecta', 'aleatorio')
                    if afecta == 'aleatorio':
                        operadores = [int(rng.integers(1, self.n_operadores + 1))]
                    elif afecta == 'todos':
                        operadores = list(range(1, self.n_operadores + 1))
                    else:
                        operadores = [int(afecta)]
                    
                    interrupciones.append({
                        'minuto_jornada': minuto_exacto,
                        'duracion_min': duracion,
                        'afecta_operadores': operadores,
                        'descripcion': evento.get('descripcion', 'Imprevisto')
                    })
        
        return interrupciones

    def _calcular_estadisticas(self, resultados):
        """Calcula métricas agregadas de todas las iteraciones."""
        makespans = np.array(resultados['makespans'])
        
        resultados['mean_makespan'] = np.mean(makespans)
        resultados['std_makespan'] = np.std(makespans)
        resultados['min_makespan'] = np.min(makespans)
        resultados['max_makespan'] = np.max(makespans)
        resultados['p50_makespan'] = np.percentile(makespans, 50)
        resultados['p95_makespan'] = np.percentile(makespans, 95)
        resultados['p99_makespan'] = np.percentile(makespans, 99)
        
        resultados['prob_horas_extra'] = np.mean(makespans > 480) * 100
        resultados['prob_horas_extra_9h'] = np.mean(makespans > 540) * 100
        
        medias_ops = [np.mean(resultados['cargas_operadores'][op]) 
                      for op in resultados['cargas_operadores']]
        resultados['balanceo_std'] = np.std(medias_ops)
        resultados['balanceo_cv'] = (resultados['balanceo_std'] / np.mean(medias_ops) * 100) if np.mean(medias_ops) > 0 else 0
        resultados['promedio_interrupciones'] = np.mean(resultados['interrupciones_ocurridas'])


# ============================================================
# EVALUADOR DE REBALANCEO
# ============================================================

class EvaluadorRebalanceo:
    """
    Evalúa si durante la jornada será necesario redistribuir las rutas 
    pendientes entre los operadores según los resultados estocásticos.
    """
    
    def __init__(self, df_asignacion, umbral_mejora_min=10):
        self.df_asignacion = df_asignacion.copy()
        self.umbral_mejora = umbral_mejora_min
        self.n_operadores = df_asignacion['operador'].nunique()
    
    def evaluar_necesidad(self, resultado_simulacion, horas_monitoreo=None):
        """
        Evalúa la probabilidad de necesitar rebalanceo durante la jornada.
        
        Args:
            resultado_simulacion: Dict con resultados de SimuladorJornada.simular()
            horas_monitoreo: Lista de horas a evaluar (default: 2, 3, 4, 5)
            
        Returns:
            dict con análisis de rebalanceo
        """
        if horas_monitoreo is None:
            horas_monitoreo = [2, 3, 4, 5]
        
        prob_horas_extra = resultado_simulacion['prob_horas_extra']
        mean_makespan = resultado_simulacion['mean_makespan']
        std_makespan = resultado_simulacion['std_makespan']
        cv_makespan = (std_makespan / mean_makespan) if mean_makespan > 0 else 0
        
        analisis = {
            'umbral_mejora_min': self.umbral_mejora,
            'probabilidad_rebalanceo_global': 0.0,
            'momentos_criticos': [],
            'recomendacion': 'MANTENER'
        }
        
        # Determinar probabilidad global y recomendación
        if prob_horas_extra > 50:
            analisis['probabilidad_rebalanceo_global'] = min(95.0, prob_horas_extra * 1.3)
            analisis['recomendacion'] = 'PREPARAR_REBALANCEO'
        elif prob_horas_extra > 20:
            analisis['probabilidad_rebalanceo_global'] = prob_horas_extra
            analisis['recomendacion'] = 'MONITOREAR'
        elif cv_makespan > 0.15:
            analisis['probabilidad_rebalanceo_global'] = 20.0
            analisis['recomendacion'] = 'MONITOREAR'
        else:
            analisis['probabilidad_rebalanceo_global'] = max(0.0, prob_horas_extra)
            analisis['recomendacion'] = 'MANTENER'
        
        # Identificar momentos críticos durante el avance de la jornada
        for hora in horas_monitoreo:
            minuto = hora * 60
            prob_retraso = self._estimar_prob_retraso(resultado_simulacion, minuto)
            
            if prob_retraso > 15.0:
                analisis['momentos_criticos'].append({
                    'hora_jornada': hora,
                    'minuto': minuto,
                    'probabilidad_retraso': round(prob_retraso, 1),
                    'accion': 'REBALANCEAR' if prob_retraso > 45.0 else 'EVALUAR'
                })
        
        # Mostrar resumen
        print(f"\nAnálisis de rebalanceo:")
        print(f"   Probabilidad de necesitar rebalanceo: {analisis['probabilidad_rebalanceo_global']:.1f}%")
        print(f"   Recomendación general: {analisis['recomendacion']}")
        
        if analisis['momentos_criticos']:
            print(f"   Momentos críticos identificados:")
            for mc in analisis['momentos_criticos']:
                print(f"      Hora {mc['hora_jornada']}h ({mc['minuto']} min): {mc['probabilidad_retraso']:.1f}% prob. retraso -> Accional: {mc['accion']}")
        else:
            print(f"   No se identificaron momentos críticos en los puntos de control.")
        
        return analisis
    
    def _estimar_prob_retraso(self, resultado_sim, minuto_corte):
        """
        Estima la probabilidad de que al minuto_corte la jornada esté desfasada
        con respecto al ritmo teórico ideal (8 horas = 480 min).
        """
        makespans = np.array(resultado_sim['makespans'])
        
        # Ritmo esperado: en el minuto_corte deberíamos llevar acumulado (minuto_corte / 480) del tiempo total
        # Si el makespan simulado es mayor a 480 min, a esta hora ya habría retraso acumulado
        limite_tiempo_esperado = (minuto_corte / 480.0) * 480.0 # = minuto_corte
        
        # Proyección de tiempo ejecutado en el minuto de corte según la distribución del Makespan
        ritmo_simulado = makespans * (minuto_corte / 480.0)
        
        # Un retraso significativo ocurre si proyectamos excede en más del umbral_mejora
        porcentaje_retrasados = np.mean(ritmo_simulado > (limite_tiempo_esperado + self.umbral_mejora)) * 100.0
        
        return float(porcentaje_retrasados)


# ============================================================
# PLANIFICADOR DE JORNADA
# ============================================================

class PlanificadorJornada:
    """
    SISTEMA INTEGRADO DE PLANIFICACIÓN DIARIA (GEMELO DIGITAL OPERATIVO)
    
    El sistema integra:
      1. Carga de datos de rutas requeridas.
      2. Optimización MILP de asignación balanceada por operador.
      3. Simulación Monte Carlo estocástica con eventos intermitentes.
      4. Evaluación de rebalanceo dinámico en caliente.
      5. Diagnóstico de viabilidad y generación de reportes visuales.
    """
    
    def __init__(self, cargador_instancia=None):
        """
        Permite pasar una instancia existente de CargadorDatos o instanciar una nueva.
        """
        if cargador_instancia is not None:
            self.cargador = cargador_instancia
        else:
            self.cargador = None
            
        self.config = None
        self.df_pool = None
        self.resultado_opt = None
        self.resultado_sim = None
        self.analisis_rebalanceo = None
        self.diagnostico_completo = None
    
    def configurar(self, rutas, n_operadores=3, eventos_dia=None,
                   hora_inicio=7, meta_horas=8, n_simulaciones=1000,
                   umbral_rebalanceo=10, seed=42):
        """
        Configura los parámetros operativos de la jornada a planificar.
        """
        if not rutas or len(rutas) == 0:
            raise ValueError("La lista de rutas a despachar no puede estar vacía.")
            
        self.config = {
            'rutas': rutas,
            'n_operadores': n_operadores,
            'eventos_dia': eventos_dia or {},
            'hora_inicio': hora_inicio,
            'meta_minutos': meta_horas * 60,
            'meta_horas': meta_horas,
            'n_simulaciones': n_simulaciones,
            'umbral_rebalanceo': umbral_rebalanceo,
            'seed': seed
        }
        
        print("\n" + "="*50)
        print("CONFIGURACIÓN DE LA JORNADA DE TRABAJO")
        print("="*50)
        print(f"Rutas a despachar: {len(rutas)}")
        print(f"Operadores disponibles: {n_operadores}")
        print(f"Hora de inicio: {hora_inicio:02d}:00 AM")
        print(f"Meta de jornada: {meta_horas} horas ({meta_horas * 60} min)")
        print(f"Eventos configurados: {len(eventos_dia) if eventos_dia else 0}")
        print(f"Iteraciones Monte Carlo: {n_simulaciones}")
        print("="*50)
  
    def ejecutar(self):
        """Ejecuta el flujo end-to-end de planificación"""
        if self.config is None:
            raise ValueError("Debe configurar la jornada primero con .configurar()")
        
        self._paso1_cargar_rutas()
        self._paso2_optimizar()
        self._paso3_simular()
        self._paso4_evaluar_rebalanceo()
        self._paso5_generar_diagnostico()
        
        return self.diagnostico_completo
    
    def _paso1_cargar_rutas(self):
        print("\n[PASO 1/5] Carga e inspección de datos de rutas...")
        if self.cargador is not None and hasattr(self.cargador, 'obtener_rutas_especificas'):
            self.df_pool = self.cargador.obtener_rutas_especificas(self.config['rutas'])
        else:
            print("   Nota: Cargador genérico de rutas activo.")
            self.df_pool = pd.DataFrame({'id_ruta': self.config['rutas']})
    
    def _paso2_optimizar(self):
        print("\n[PASO 2/5] Optimizando asignación de rutas (MILP)...")
        self.resultado_opt = optimizar_asignacion(
            self.df_pool,
            n_operadores=self.config['n_operadores'],
            verbose=False
        )
        print(f"   Makespan óptimo estimado: {self.resultado_opt['makespan']:.1f} min")
    
    def _paso3_simular(self):
        print("\n[PASO 3/5] Ejecutando Gemelo Digital (Simulación Monte Carlo)...")
        
        eventos_programados = []
        eventos_aleatorios = []
        
        for nombre, evento in self.config['eventos_dia'].items():
            evento_con_nombre = evento.copy()
            evento_con_nombre['descripcion'] = nombre.replace('_', ' ').title()
            
            if evento.get('tipo') == 'aleatorio':
                eventos_aleatorios.append(evento_con_nombre)
            else:
                eventos_programados.append(evento_con_nombre)
        
        simulador = SimuladorJornada(
            self.resultado_opt['df_asignacion'],
            seed=self.config['seed']
        )
        
        self.resultado_sim = simulador.simular(
            n_iteraciones=self.config['n_simulaciones'],
            eventos_programados=eventos_programados,
            eventos_aleatorios=eventos_aleatorios,
            hora_inicio_jornada=self.config['hora_inicio']
        )
    
    def _paso4_evaluar_rebalanceo(self):
        print("\n[PASO 4/5] Evaluando necesidad de rebalanceo dinámico...")
        
        evaluador = EvaluadorRebalanceo(
            self.resultado_opt['df_asignacion'],
            umbral_mejora_min=self.config['umbral_rebalanceo']
        )
        
        self.analisis_rebalanceo = evaluador.evaluar_necesidad(self.resultado_sim)
    
    def _paso5_generar_diagnostico(self):
        print("\n[PASO 5/5] Consolidando diagnóstico operacional...")
        opt = self.resultado_opt
        sim = self.resultado_sim
        reb = self.analisis_rebalanceo
        cfg = self.config
        
        makespan_p95 = sim['p95_makespan']
        viable = makespan_p95 <= cfg['meta_minutos']
        holgura = cfg['meta_minutos'] - makespan_p95
        
        if viable:
            estado = "VIABLE"
            nivel = "CÓMODA" if holgura > 120 else "NORMAL" if holgura > 60 else "AJUSTADA"
        else:
            estado = "REQUIERE ATENCIÓN"
            nivel = "CRÍTICA"
        
        self.diagnostico_completo = {
            'config': cfg,
            'optimizacion': opt,
            'simulacion': sim,
            'rebalanceo': reb,
            'viabilidad': {
                'viable': viable,
                'estado': estado,
                'nivel': nivel,
                'holgura_min': holgura,
                'holgura_horas': holgura / 60.0
            },
            'resumen': {
                'rutas_total': opt['n_rutas'],
                'tiempo_total_est': opt['tiempo_total'],
                'makespan_planificado': opt['makespan'],
                'makespan_simulado': sim['mean_makespan'],
                'makespan_p95': makespan_p95,
                'prob_horas_extra': sim['prob_horas_extra'],
                'balanceo_min': sim['balanceo_std']
            }
        }
        
        print(f"✔ Diagnóstico finalizado: Jornada {estado} (Nivel: {nivel})")
    
    def mostrar_diagnostico(self):
        """Muestra el diagnóstico consolidado en formato tabular estructurado"""
        if self.diagnostico_completo is None:
            print("No hay diagnóstico generado. Ejecute .ejecutar() primero.")
            return
        
        d = self.diagnostico_completo
        r = d['resumen']
        v = d['viabilidad']
        opt = d['optimizacion']
        reb = d['rebalanceo']
        
        W = 68
        print("\n" + "╔" + "═"*W + "╗")
        print("║" + "DIAGNÓSTICO DE JORNADA LOGÍSTICA".center(W) + "║")
        print("╠" + "═"*W + "╣")
        print("║ " + "DATOS GENERALES DE ENTRADA".ljust(W-1) + "║")
        print(f"║   • Rutas a despachar: {r['rutas_total']} | Operadores: {opt['n_operadores']}".ljust(W+1) + "║")
        print(f"║   • Tiempo total trabajo estimado: {r['tiempo_total_est']:.0f} min ({r['tiempo_total_est']/60:.1f}h)".ljust(W+1) + "║")
        print("║" + " "*W + "║")
        print("║ " + "PLANIFICACIÓN ÓPTIMA (ASIGNACIÓN)".ljust(W-1) + "║")
        for i, carga in enumerate(opt['cargas_operadores']):
            n_rutas_op = len(opt['df_asignacion'][opt['df_asignacion']['operador'] == i+1])
            print(f"║   • Operador {i+1}: {n_rutas_op} rutas -> {carga:.0f} min ({carga/60:.1f}h)".ljust(W+1) + "║")
        print(f"║   • Makespan teórico ideal: {r['makespan_planificado']:.0f} min ({r['makespan_planificado']/60:.1f}h)".ljust(W+1) + "║")
        print("║" + " "*W + "║")
        print("║ " + "EVALUACIÓN SIMULADA DE RIESGOS (MONTE CARLO)".ljust(W-1) + "║")
        print(f"║   • Makespan promedio proyectado: {r['makespan_simulado']:.0f} min ({r['makespan_simulado']/60:.1f}h)".ljust(W+1) + "║")
        print(f"║   • Makespan P95 (Escenario adverso): {r['makespan_p95']:.0f} min ({r['makespan_p95']/60:.1f}h)".ljust(W+1) + "║")
        print(f"║   • Probabilidad de exceso de jornada: {r['prob_horas_extra']:.1f}%".ljust(W+1) + "║")
        print(f"║   • Desviación de balanceo entre op.: {r['balanceo_min']:.1f} min".ljust(W+1) + "║")
        print("║" + " "*W + "║")
        print("║ " + "MONITOREO DE REBALANCEO".ljust(W-1) + "║")
        print(f"║   • Probabilidad de rebalanceo en caliente: {reb['probabilidad_rebalanceo_global']:.1f}%".ljust(W+1) + "║")
        print(f"║   • Recomendación estratégica: {reb['recomendacion']}".ljust(W+1) + "║")
        print("║" + " "*W + "║")
        print("║ " + f"VEREDICTO OPERATIVO: Jornada {v['estado']}".ljust(W-1) + "║")
        print(f"║   • Estado: {v['nivel']} | Holgura P95: {v['holgura_min']:.0f} min ({v['holgura_horas']:.1f}h)".ljust(W+1) + "║")
        
        if not v['viable']:
            deficit = abs(v['holgura_min'])
            print(f"║   ⚠ Acción sugerida: Agregar +1 operador o descolgar ~{max(1, int(deficit/25))} rutas".ljust(W+1) + "║")
        print("╚" + "═"*W + "╝\n")
    
    def graficar_resultados(self, reports_path=None):
        """Genera y guarda el dashboard de métricas clave"""
        if self.diagnostico_completo is None:
            print("No hay diagnóstico generado. Ejecute .ejecutar() primero.")
            return
        
        if reports_path is None:
            reports_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
        os.makedirs(reports_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        opt = self.diagnostico_completo['optimizacion']
        sim = self.diagnostico_completo['simulacion']
        cfg = self.diagnostico_completo['config']
        
        # Gráfico 1: Balanceo de Cargas Teórico
        ax1 = axes[0]
        ops = [f'Op {i+1}' for i in range(opt['n_operadores'])]
        cargas = opt['cargas_operadores']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#2E8B57', '#C73E1D'][:len(ops)]
        
        bars = ax1.bar(ops, cargas, color=colors, edgecolor='white', alpha=0.9)
        ax1.axhline(y=cfg['meta_minutos'], color='red', linestyle='--', linewidth=2, label=f'Meta {cfg["meta_horas"]}h ({cfg["meta_minutos"]}m)')
        
        for bar, c in zip(bars, cargas):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f'{c:.0f}m', ha='center', fontweight='bold')
            
        ax1.set_title('Asignación Teórica de Carga (min)', fontweight='bold')
        ax1.set_ylabel('Minutos asignados')
        ax1.legend()
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Gráfico 2: Distribución del Makespan Simulado
        ax2 = axes[1]
        ax2.hist(sim['makespans'], bins=30, color='#2E86AB', edgecolor='white', alpha=0.7)
        ax2.axvline(x=sim['mean_makespan'], color='green', linewidth=2, label=f'Promedio: {sim["mean_makespan"]:.0f}m')
        ax2.axvline(x=cfg['meta_minutos'], color='red', linestyle='--', linewidth=2, label=f'Meta: {cfg["meta_horas"]}h')
        ax2.axvline(x=sim['p95_makespan'], color='orange', linestyle=':', linewidth=2, label=f'P95: {sim["p95_makespan"]:.0f}m')
        
        ax2.set_title('Distribución de Duración Real (Monte Carlo)', fontweight='bold')
        ax2.set_xlabel('Minutos de Jornada')
        ax2.set_ylabel('Frecuencia (Simulaciones)')
        ax2.legend()
        ax2.grid(linestyle=':', alpha=0.6)
        
        # Gráfico 3: Indicador de Riesgo de Horas Extra
        ax3 = axes[2]
        prob = sim['prob_horas_extra']
        color_riesgo = 'green' if prob < 15 else 'orange' if prob < 40 else 'red'
        
        ax3.barh(['Horas Extra'], [prob], color=color_riesgo, edgecolor='black', height=0.3)
        ax3.axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='Alerta Moderada (20%)')
        ax3.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Límite Crítico (50%)')
        ax3.set_xlim(0, 100)
        ax3.set_title(f'Riesgo de Exceso de Jornada: {prob:.1f}%', fontweight='bold')
        ax3.set_xlabel('Probabilidad (%)')
        ax3.legend()
        ax3.grid(axis='x', linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        # Guardado del reporte visual
        output_file = os.path.join(reports_path, 'diagnostico_jornada.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✔ Dashboard de diagnósticos guardado con éxito en: {output_file}")


# Verificación
if __name__ == "__main__":
    print("="*50)
    print("MOTOR DE OPTIMIZACIÓN Y SIMULACIÓN LISTO")
    print("="*50)
    print("   Funciones:")
    print("      - optimizar_asignacion()")
    print("      - _asignacion_secuencial() (Plan B)")
    print("   Clases:")
    print("      - SimuladorJornada (3 capas de realismo)")
    print("      - EvaluadorRebalanceo")
    print("      - PlanificadorJornada (Orquestador)")
