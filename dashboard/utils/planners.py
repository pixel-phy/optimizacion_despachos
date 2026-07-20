# utils/planners.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Agregar directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.optimization import PlanificadorJornada
from src.data_loader import CargadorDatos


def ejecutar_planificacion_simple(config, models):
    """
    Ejecuta una planificación usando PlanificadorJornada de src.
    
    Args:
        config: Diccionario con configuración de la jornada
        models: Diccionario con modelos cargados (se usa para el cargador)
    
    Returns:
        dict: Resultados de la planificación
    """
    try:
        rutas_ids = config['rutas_ids']
        n_operadores = config['n_operadores']
        n_simulaciones = config['n_simulaciones']
        meta_horas = config['meta_horas']
        hora_inicio = config.get('hora_inicio', datetime.now().time())
        eventos = config.get('eventos', [])
        prob_interrupcion = config.get('prob_interrupcion', 0.1)
        duracion_interrupcion = config.get('duracion_interrupcion', 15)
        
        if not rutas_ids:
            return None
        
        # ============================================================
        # 1. OBTENER CARGADOR DESDE models O CREAR UNO NUEVO
        # ============================================================
        if models and 'cargador' in models and models['cargador'] is not None:
            cargador = models['cargador']
        else:
            # Crear nuevo cargador
            cargador = CargadorDatos()
        
        # ============================================================
        # 2. CONFIGURAR EVENTOS EN EL FORMATO ESPERADO POR PlanificadorJornada
        # ============================================================
        eventos_dia = {}
        
        # Eventos programados
        for i, evento in enumerate(eventos):
            eventos_dia[f'evento_{i+1}'] = {
                'hora': evento['hora'],
                'duracion': evento['duracion'],
                'afecta': 'todos' if len(evento.get('afecta', [])) == n_operadores else 'aleatorio',
                'tipo': 'programado',
                'descripcion': f"Evento {i+1}"
            }
        
        # Eventos aleatorios (interrupciones)
        if prob_interrupcion > 0:
            eventos_dia['interrupciones'] = {
                'probabilidad': prob_interrupcion,
                'duracion_media': duracion_interrupcion,
                'duracion_std': duracion_interrupcion * 0.4,
                'afecta': 'aleatorio',
                'tipo': 'aleatorio',
                'descripcion': 'Interrupciones aleatorias'
            }
        
        # ============================================================
        # 3. CREAR Y CONFIGURAR PlanificadorJornada
        # ============================================================
        planificador = PlanificadorJornada(cargador_instancia=cargador)
        
        # Convertir hora_inicio a formato 24h (entero)
        if hasattr(hora_inicio, 'hour'):
            hora_inicio_int = hora_inicio.hour
        else:
            hora_inicio_int = 15
        
        planificador.configurar(
            rutas=rutas_ids,
            n_operadores=n_operadores,
            eventos_dia=eventos_dia,
            hora_inicio=hora_inicio_int,
            meta_horas=meta_horas,
            n_simulaciones=n_simulaciones,
            seed=42
        )
        
        # ============================================================
        # 4. EJECUTAR PLANIFICACIÓN
        # ============================================================
        planificador.ejecutar()
        diagnostico = planificador.diagnostico_completo
        
        if diagnostico is None:
            return None
        
        # ============================================================
        # 5. EXTRAER RESULTADOS
        # ============================================================
        opt = diagnostico['optimizacion']
        sim = diagnostico['simulacion']
        cfg = diagnostico['config']
        
        df_asignacion = opt['df_asignacion']
        
        # Obtener tiempos individuales simulados (desde la simulación)
        tiempos_simulados = sim['makespans']
        
        # Calcular estadísticas adicionales
        makespan_planificado = opt['makespan']
        makespan_simulado = sim['mean_makespan']
        prob_extra = sim['prob_horas_extra'] / 100  # Convertir a fracción
        
        # Carga por operador
        carga_operadores = {}
        for op in sorted(df_asignacion['operador'].unique()):
            carga = df_asignacion[df_asignacion['operador'] == op]['tiempo_estimado'].sum()
            carga_operadores[f'Operador {op}'] = carga
        
        # Balanceo
        cargas_list = list(carga_operadores.values())
        if cargas_list:
            balanceo = max(cargas_list) - min(cargas_list)
            balanceo_pct = (balanceo / max(cargas_list) * 100) if max(cargas_list) > 0 else 0
        else:
            balanceo = 0
            balanceo_pct = 0
        
        # ============================================================
        # 6. CREAR DICCIONARIO DE RESULTADOS
        # ============================================================
        resultados = {
            'asignacion': df_asignacion,
            'tiempos_individuales': [row['tiempo_estimado'] for _, row in df_asignacion.iterrows()],
            'tiempos_simulados': tiempos_simulados,
            'makespan_planificado': makespan_planificado,
            'makespan_simulado': makespan_simulado,
            'makespan_std': sim['std_makespan'],
            'makespan_p5': np.percentile(tiempos_simulados, 5),
            'makespan_p95': sim['p95_makespan'],
            'prob_extra': prob_extra,
            'carga_operadores': carga_operadores,
            'balanceo': balanceo,
            'balanceo_pct': balanceo_pct,
            'n_rutas': len(rutas_ids),
            'n_operadores': n_operadores,
            'meta_horas': meta_horas,
            'diagnostico': diagnostico,
            'planificador': planificador
        }
        
        return resultados
    
    except Exception as e:
        st.error(f"❌ Error en planificación: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
