# utils/planners.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import sys
from pathlib import Path

# Agregar directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

def ejecutar_planificacion_simple(config, models):
    """
    Ejecuta una planificación simple usando los modelos cargados.
    """
    try:
        rutas_ids = config['rutas_ids']
        n_operadores = config['n_operadores']
        n_simulaciones = config['n_simulaciones']
        hora_inicio = config.get('hora_inicio')
        
        if not rutas_ids:
            return None
        
        from utils.loaders import predict_tiempo
        
        # Obtener histórico para predicciones
        historico = models.get('historico', pd.DataFrame())
        
        if hora_inicio:
            fecha_actual = datetime.now()
            dia_semana = fecha_actual.weekday()
        else:
            dia_semana = 0
        
        # ============================================================
        # 1. PREDECIR TIEMPOS USANDO EL HISTÓRICO
        # ============================================================
        tiempos_estimados = []
        for ruta_id in rutas_ids:
            # Buscar datos reales de esta ruta en el histórico
            if not historico.empty:
                registros_ruta = historico[historico['id_ruta'] == ruta_id]
                if not registros_ruta.empty:
                    # Usar datos reales de la ruta
                    ruta_data = {
                        'id_ruta': ruta_id,
                        'cant_productos': registros_ruta['cant_productos'].mean(),
                        'valor_ruta': registros_ruta['valor_ruta'].mean()
                    }
                else:
                    # Si no hay histórico, usar valores estimados
                    ruta_data = {
                        'id_ruta': ruta_id,
                        'cant_productos': np.random.randint(10, 40),
                        'valor_ruta': np.random.uniform(300, 1200)
                    }
            else:
                ruta_data = {
                    'id_ruta': ruta_id,
                    'cant_productos': np.random.randint(10, 40),
                    'valor_ruta': np.random.uniform(300, 1200)
                }
            
            # Predecir tiempo usando el histórico
            tiempo = predict_tiempo(
                models['model'], 
                models['scaler'], 
                ruta_data,
                dia_semana=dia_semana,
                historico=historico
            )
            tiempos_estimados.append(tiempo)
        
        # ============================================================
        # 2. ASIGNAR RUTAS A OPERADORES
        # ============================================================
        # Ordenar por tiempo descendente para mejor balanceo
        indices_ordenados = np.argsort(tiempos_estimados)[::-1]
        
        # Asignación balanceada (operador con menos carga primero)
        cargas = [0] * n_operadores
        asignacion = []
        
        for idx in indices_ordenados:
            ruta_id = rutas_ids[idx]
            tiempo = tiempos_estimados[idx]
            
            # Asignar al operador con menos carga
            op_min = np.argmin(cargas)
            operador = f"Operador {op_min + 1}"
            
            asignacion.append({
                'id_ruta': ruta_id,
                'operador': operador,
                'tiempo_estimado': tiempo
            })
            cargas[op_min] += tiempo
        
        df_asignacion = pd.DataFrame(asignacion)
        
        # ============================================================
        # 3. CALCULAR MAKESPAN Y SIMULACIÓN
        # ============================================================
        carga_operadores = df_asignacion.groupby('operador')['tiempo_estimado'].sum()
        makespan_planificado = carga_operadores.max()
        
        # Simulación Monte Carlo
        tiempos_simulados = []
        for _ in range(n_simulaciones):
            variabilidad = np.random.normal(1, 0.1, len(tiempos_estimados))
            tiempos_sim = np.array(tiempos_estimados) * np.maximum(0.5, variabilidad)
            
            prob_inter = config.get('prob_interrupcion', 0.1)
            duracion_inter = config.get('duracion_interrupcion', 15)
            
            for i in range(len(tiempos_sim)):
                if np.random.random() < prob_inter:
                    tiempos_sim[i] += np.random.exponential(duracion_inter)
            
            cargas_sim = [0] * n_operadores
            for i, (idx, tiempo) in enumerate(zip(indices_ordenados, tiempos_sim)):
                op_min = np.argmin(cargas_sim)
                cargas_sim[op_min] += tiempo
            
            tiempos_simulados.append(max(cargas_sim))
        
        makespan_simulado = np.mean(tiempos_simulados)
        meta_minutos = config['meta_horas'] * 60
        prob_extra = sum(1 for t in tiempos_simulados if t > meta_minutos) / len(tiempos_simulados)
        
        resultados = {
            'asignacion': df_asignacion,
            'tiempos_individuales': tiempos_estimados,
            'tiempos_simulados': tiempos_simulados,
            'makespan_planificado': makespan_planificado,
            'makespan_simulado': makespan_simulado,
            'prob_extra': prob_extra,
            'carga_operadores': carga_operadores.to_dict()
        }
        
        return resultados
    
    except Exception as e:
        st.error(f"❌ Error en planificación: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def optimizar_asignacion(rutas_tiempos, n_operadores):
    """
    Función de optimización usando PuLP (versión simplificada)
    
    Parámetros:
    -----------
    rutas_tiempos : list
        Lista de tuplas (id_ruta, tiempo_estimado)
    n_operadores : int
        Número de operadores disponibles
    
    Retorna:
    --------
    pd.DataFrame : Asignación optimizada
    """
    try:
        # Verificar que tenemos datos
        if not rutas_tiempos:
            return pd.DataFrame()
        
        # Ordenar por tiempo descendente (más largas primero)
        rutas_ordenadas = sorted(rutas_tiempos, key=lambda x: x[1], reverse=True)
        
        # Inicializar cargas de operadores
        cargas = [0] * n_operadores
        asignacion = []
        
        # Asignación greedy: cada ruta al operador con menos carga actual
        for ruta_id, tiempo in rutas_ordenadas:
            # Encontrar operador con menor carga
            op_min = min(range(n_operadores), key=lambda i: cargas[i])
            
            asignacion.append({
                'id_ruta': ruta_id,
                'operador': f"Operador {op_min + 1}",
                'tiempo_estimado': tiempo
            })
            
            cargas[op_min] += tiempo
        
        return pd.DataFrame(asignacion)
    
    except Exception as e:
        st.warning(f"⚠️ Error en optimización: {str(e)}")
        return None


def calcular_estadisticas_operadores(asignacion):
    """
    Calcula estadísticas detalladas por operador
    
    Parámetros:
    -----------
    asignacion : pd.DataFrame
        DataFrame con columnas: id_ruta, operador, tiempo_estimado
    
    Retorna:
    --------
    pd.DataFrame : Estadísticas por operador
    """
    if asignacion.empty:
        return pd.DataFrame()
    
    if 'operador' not in asignacion.columns or 'tiempo_estimado' not in asignacion.columns:
        return pd.DataFrame()
    
    resumen = asignacion.groupby('operador').agg({
        'id_ruta': 'count',
        'tiempo_estimado': ['sum', 'mean', 'min', 'max', 'std']
    }).round(2)
    
    resumen.columns = [
        'N° Rutas',
        'Tiempo Total (min)',
        'Tiempo Promedio (min)',
        'Tiempo Mínimo (min)',
        'Tiempo Máximo (min)',
        'Desv. Estándar (min)'
    ]
    
    resumen = resumen.reset_index()
    resumen['Tiempo Total (horas)'] = (resumen['Tiempo Total (min)'] / 60).round(2)
    
    return resumen


def calcular_balanceo_carga(asignacion):
    """
    Calcula métricas de balanceo de carga
    
    Parámetros:
    -----------
    asignacion : pd.DataFrame
        DataFrame con columnas: operador, tiempo_estimado
    
    Retorna:
    --------
    dict : Métricas de balanceo
    """
    if asignacion.empty:
        return {}
    
    carga_ops = asignacion.groupby('operador')['tiempo_estimado'].sum()
    
    max_carga = carga_ops.max()
    min_carga = carga_ops.min()
    promedio = carga_ops.mean()
    diferencia = max_carga - min_carga
    diferencia_pct = (diferencia / max_carga) * 100 if max_carga > 0 else 0
    
    return {
        'max_carga': max_carga,
        'min_carga': min_carga,
        'promedio': promedio,
        'diferencia': diferencia,
        'diferencia_pct': diferencia_pct,
        'carga_por_operador': carga_ops.to_dict()
    }


def generar_reporte_resultados(resultados):
    """
    Genera un resumen en texto de los resultados
    
    Parámetros:
    -----------
    resultados : dict
        Diccionario con los resultados de la planificación
    
    Retorna:
    --------
    str : Resumen formateado
    """
    if not resultados:
        return "No hay resultados disponibles"
    
    reporte = []
    reporte.append("=" * 50)
    reporte.append("📊 RESUMEN DE PLANIFICACIÓN")
    reporte.append("=" * 50)
    reporte.append(f"📦 Rutas totales: {resultados.get('n_rutas', 0)}")
    reporte.append(f"👥 Operadores: {resultados.get('n_operadores', 0)}")
    reporte.append("")
    reporte.append("📈 MAKESPAN:")
    reporte.append(f"  Planificado: {resultados.get('makespan_planificado', 0):.1f} min ({resultados.get('makespan_planificado', 0)/60:.2f} h)")
    reporte.append(f"  Simulado: {resultados.get('makespan_simulado', 0):.1f} min ({resultados.get('makespan_simulado', 0)/60:.2f} h)")
    reporte.append(f"  Desv. Estándar: {resultados.get('makespan_std', 0):.1f} min")
    reporte.append(f"  Percentil 5-95: {resultados.get('makespan_p5', 0):.1f} - {resultados.get('makespan_p95', 0):.1f} min")
    reporte.append("")
    
    prob_extra = resultados.get('prob_extra', 0) * 100
    reporte.append(f"⚠️ Probabilidad horas extra: {prob_extra:.1f}%")
    
    if prob_extra < 10:
        reporte.append("  ✅ Bajo riesgo")
    elif prob_extra < 30:
        reporte.append("  ⚠️ Riesgo moderado")
    else:
        reporte.append("  🔴 Alto riesgo")
    
    reporte.append("")
    reporte.append("⚖️ CARGA POR OPERADOR:")
    carga_ops = resultados.get('carga_operadores', {})
    for op, carga in carga_ops.items():
        reporte.append(f"  {op}: {carga:.1f} min ({carga/60:.2f} h)")
    
    balanceo = resultados.get('balanceo', 0)
    balanceo_pct = resultados.get('balanceo_pct', 0)
    reporte.append(f"  Diferencia: {balanceo:.1f} min ({balanceo_pct:.1f}%)")
    reporte.append("=" * 50)
    
    return "\n".join(reporte)


# Función de prueba para verificar que todo funciona
def test_planners():
    """Función de prueba para verificar el módulo planners"""
    print("🧪 Probando utils/planners.py...")
    
    # Configuración de prueba
    config_test = {
        'rutas_ids': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'n_operadores': 3,
        'meta_horas': 8.0,
        'n_simulaciones': 100,
        'hora_inicio': datetime.now().time(),
        'prob_interrupcion': 0.1,
        'duracion_interrupcion': 15,
        'eventos': []
    }
    
    # Models dummy
    models_test = {
        'model': None,
        'scaler': None
    }
    
    # Probar ejecución
    resultados = ejecutar_planificacion_simple(config_test, models_test)
    
    if resultados:
        print("✅ Prueba exitosa!")
        print(f"   Makespan planificado: {resultados['makespan_planificado']:.1f} min")
        print(f"   Makespan simulado: {resultados['makespan_simulado']:.1f} min")
        print(f"   Prob. horas extra: {resultados['prob_extra']*100:.1f}%")
        print(f"   Balanceo: {resultados.get('balanceo', 0):.1f} min")
        
        # Probar estadísticas
        stats = calcular_estadisticas_operadores(resultados['asignacion'])
        if not stats.empty:
            print("\n   Estadísticas por operador:")
            print(stats.to_string(index=False))
        
        # Probar reporte
        reporte = generar_reporte_resultados(resultados)
        print("\n" + reporte)
    else:
        print("❌ Prueba fallida")

if __name__ == "__main__":
    test_planners()
