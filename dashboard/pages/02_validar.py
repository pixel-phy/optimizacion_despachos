# pages/02_validar.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.loaders import load_models, load_available_dates
from utils.visualizations import crear_grafico_validacion_personal

st.set_page_config(
    page_title="Validación Retrospectiva",
    page_icon="🔍",
    layout="wide"
)

st.title("Validación Retrospectiva de Rutas")
st.markdown("---")

st.markdown("""
### Validación de mis rutas personales

Esta validación compara **ÚNICAMENTE mis rutas personales** (21-30 por día) 
con las predicciones del modelo, no el total de rutas del equipo.

""")

# Cargar modelos y datos
models = load_models()
fechas = load_available_dates()

if not fechas:
    st.warning("No hay fechas disponibles para validación")
    st.stop()

# Selección de fecha
col1, col2 = st.columns([2, 1])

with col1:
    fecha_seleccionada = st.selectbox(
        "Selecciona una fecha histórica (mis jornadas):",
        options=fechas,
        format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'),
        key="fecha_validacion"
    )

with col2:
    ejecutar_validacion = st.button(
        "Validar mis rutas",
        type="primary",
        use_container_width=True,
        key="btn_validar"
    )

if ejecutar_validacion and fecha_seleccionada:
    with st.spinner(f"Validando rutas del {fecha_seleccionada}..."):
        try:
            
            # 1. CARGAR SOLO TUS RUTAS de esa fecha
            
            df_historico = models['historico']
            
            # Filtrar SOLO tus rutas de esa fecha
            df_tus_rutas = df_historico[df_historico['fecha'] == fecha_seleccionada].copy()
            
            if df_tus_rutas.empty:
                st.warning(f"No hay registros de rutas para la fecha {fecha_seleccionada}")
                st.info("""
                    **Posibles causas:**
                    - No trabajaste ese día
                    - Los datos no se cargaron correctamente
                    - La fecha no está en tu histórico personal
                """)
            else:
                
                # 2. MOSTRAR RESUMEN DE TUS RUTAS
                
                st.success(f"Encontradas {len(df_tus_rutas)} rutas para el {fecha_seleccionada}")
                
                # Mostrar estadísticas básicas de tus rutas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Tus Rutas",
                        f"{len(df_tus_rutas)}",
                        delta=f"{(len(df_tus_rutas)/87*100):.1f}% del total",
                        delta_color="off"
                    )
                with col2:
                    tiempo_total_real = df_tus_rutas['tiempo_preparacion_minutos'].sum()
                    st.metric(
                        "Tiempo Total Real",
                        f"{tiempo_total_real:.1f} min",
                        delta=f"{tiempo_total_real/60:.1f} horas",
                        delta_color="off"
                    )
                with col3:
                    tiempo_promedio = df_tus_rutas['tiempo_preparacion_minutos'].mean()
                    st.metric(
                        "Tiempo Promedio/Ruta",
                        f"{tiempo_promedio:.1f} min",
                        delta=f"±{df_tus_rutas['tiempo_preparacion_minutos'].std():.1f} min",
                        delta_color="off"
                    )
                
                
                # 3. PREPARAR DATOS PARA SIMULACIÓN (SOLO TUS RUTAS)
                
                rutas_tuyas = df_tus_rutas['id_ruta'].tolist()
                tiempos_reales = df_tus_rutas['tiempo_preparacion_minutos'].tolist()
                
                # Mostrar tus rutas
                with st.expander("Ver rutas de ese día"):
                    st.dataframe(
                        df_tus_rutas[['id_ruta', 'tiempo_preparacion_minutos', 'cant_productos', 'valor_ruta']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                # 4. EJECUTAR SIMULACIÓN CON RUTAS
                
                from utils.planners import ejecutar_planificacion_simple
                from datetime import time
                
                # Configurar simulación para tus rutas
                config_validacion = {
                    'rutas_ids': rutas_tuyas,
                    'n_operadores': 1,
                    'hora_inicio': time(15, 0),
                    'meta_horas': 8,
                    'n_simulaciones': 500,
                    'eventos': [],
                    'prob_interrupcion': 0.1,
                    'duracion_interrupcion': 15
                }
                
                # Ejecutar simulación
                resultados = ejecutar_planificacion_simple(config_validacion, models)
                
                if resultados:
                    
                    # 5. COMPARAR DATOS REALES VS SIMULADOS
                    
                    st.divider()
                    st.subheader(f"Comparativa de tus rutas - {fecha_seleccionada}")
                    
                    # ============================================================
                    # CORRECCIÓN #7: ALINEAR tiempos simulados con reales por id_ruta
                    # Versión robusta: maneja diferencias de tipo (int vs str) y
                    # rutas nuevas que el CargadorDatos podría no haber agregado.
                    # ============================================================
                    
                    df_asignacion = resultados.get('asignacion', pd.DataFrame())
                    
                    if df_asignacion.empty:
                        st.error("❌ Error: No se recibió la asignación de rutas.")
                        st.stop()
                    
                    # Normalizar tipos: asegurar que ambos sean int para comparar
                    df_asignacion['id_ruta'] = df_asignacion['id_ruta'].astype(int)
                    rutas_tuyas_int = [int(r) for r in rutas_tuyas]
                    
                    # Crear diccionario ruta_id -> tiempo_simulado
                    mapa_simulados = dict(zip(df_asignacion['id_ruta'], df_asignacion['tiempo_estimado']))
                    
                    # Debug: mostrar qué rutas faltan
                    rutas_faltantes = [r for r in rutas_tuyas_int if r not in mapa_simulados]
                    if rutas_faltantes:
                        st.warning(f"⚠️ {len(rutas_faltantes)} rutas no encontradas en la simulación: {rutas_faltantes}")
                        
                        # Solución: para rutas faltantes, usar el promedio de las encontradas
                        if mapa_simulados:
                            tiempo_promedio_sim = sum(mapa_simulados.values()) / len(mapa_simulados)
                            for ruta_id in rutas_faltantes:
                                mapa_simulados[ruta_id] = tiempo_promedio_sim
                            st.info(f"Se usó el tiempo promedio ({tiempo_promedio_sim:.1f} min) para las rutas faltantes")
                        else:
                            st.error("❌ No se pudo recuperar ninguna ruta. Abortando.")
                            st.stop()
                    
                    # Alinear tiempos simulados con el orden de tiempos_reales
                    tiempos_simulados = [mapa_simulados[r] for r in rutas_tuyas_int]
                    
                    # CORRECCIÓN #1: Calcular tiempo simulado como SUMA de predicciones puras
                    tiempo_simulado_total = sum(tiempos_simulados)
                    
                    # Métricas comparativas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Tiempo Real",
                            f"{tiempo_total_real:.1f} min",
                            delta=f"{tiempo_total_real/60:.1f} horas",
                            delta_color="off"
                        )
                    
                    # CORRECCIÓN DEL BUG st.metric: delta solo se pasa UNA vez como keyword
                    with col2:
                        diferencia_sim_real = tiempo_simulado_total - tiempo_total_real
                        st.metric(
                            "Tiempo Simulado",
                            f"{tiempo_simulado_total:.1f} min",
                            delta=f"{diferencia_sim_real:+.1f} min vs real",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        error = abs(tiempo_total_real - tiempo_simulado_total)
                        error_pct = (error / tiempo_total_real) * 100 if tiempo_total_real > 0 else 0
                        st.metric(
                            "Error Total",
                            f"{error:.1f} min",
                            delta=f"{error_pct:.1f}%",
                            delta_color="inverse"
                        )
                    
                    with col4:
                        errores_ruta = [abs(r - s) for r, s in zip(tiempos_reales, tiempos_simulados)]
                        mae = np.mean(errores_ruta) if errores_ruta else 0
                        st.metric(
                            "Error Promedio/Ruta",
                            f"{mae:.2f} min",
                            delta=f"±{np.std(errores_ruta):.2f}" if errores_ruta else "",
                            delta_color="off"
                        )
                    
                    st.divider()
                    
                    # 6. GRÁFICO COMPARATIVO DE RUTAS
                    
                    st.subheader("Comparativa Ruta por Ruta (Tus rutas)")
                    
                    fig_comp = crear_grafico_validacion_personal(
                        df_tus_rutas,
                        tiempos_reales,
                        tiempos_simulados
                    )
                    if fig_comp:
                        st.plotly_chart(fig_comp, use_container_width=True)
                    
                    
                    # 7. ANÁLISIS DE ERROR POR RUTA
                    
                    st.subheader("Análisis Detallado de Error por Ruta")
                    
                    df_errores = pd.DataFrame({
                        'ID Ruta': df_tus_rutas['id_ruta'].tolist(),
                        'Tiempo Real (min)': tiempos_reales,
                        'Tiempo Simulado (min)': tiempos_simulados,
                        'Error (min)': errores_ruta,
                        'Error %': [(e/r*100) if r > 0 else 0 for e, r in zip(errores_ruta, tiempos_reales)]
                    })
                    
                    df_errores = df_errores.sort_values('Error %', ascending=False)
                    
                    st.dataframe(
                        df_errores.round(2),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # 8. ESTADÍSTICAS ADICIONALES
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        top_errores = df_errores.nlargest(3, 'Error %')
                        if not top_errores.empty:
                            st.warning("Rutas con mayor error:")
                            for _, row in top_errores.iterrows():
                                st.write(f"• Ruta {int(row['ID Ruta'])}: {row['Error %']:.1f}%")
                    
                    with col2:
                        bottom_errores = df_errores.nsmallest(3, 'Error %')
                        if not bottom_errores.empty:
                            st.success("Rutas con menor error:")
                            for _, row in bottom_errores.iterrows():
                                st.write(f"• Ruta {int(row['ID Ruta'])}: {row['Error %']:.1f}%")
                    
                    with col3:
                        st.info("Resumen de errores:")
                        st.write(f"• Media: {df_errores['Error %'].mean():.1f}%")
                        st.write(f"• Mediana: {df_errores['Error %'].median():.1f}%")
                        st.write(f"• Máximo: {df_errores['Error %'].max():.1f}%")
                    
                    # 9. CONCLUSIÓN DE VALIDACIÓN
                    
                    st.divider()
                    
                    error_promedio = df_errores['Error %'].mean()
                    
                    if error_promedio < 5:
                        st.success(f"""
                        **Excelente validación de rutas!** 
                        
                        El modelo predice los tiempos con un error promedio del {error_promedio:.1f}%.
                        Se puede confiar plenamente en las predicciones para las rutas.
                        """)
                    elif error_promedio < 10:
                        st.warning(f"""
                        **Buena validación de rutas** 
                        
                        El modelo predice los tiempos con un error promedio del {error_promedio:.1f}%.
                        Considera revisar las rutas con mayor error para entender las desviaciones.
                        """)
                    else:
                        st.error(f"""
                        **Error significativo en rutas** 
                        
                        El modelo tiene un error promedio del {error_promedio:.1f}% en tus rutas.
                        **Posibles causas:**
                        - Hubo interrupciones no registradas ese día
                        - Las condiciones de trabajo fueron atípicas
                        - El modelo necesita recalibración con tus datos recientes
                        """)
                    
                    # 10. GUARDAR RESULTADOS DE VALIDACIÓN
                    
                    with st.expander("Guardar resultados de validación"):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                        
                        csv = df_errores.to_csv(index=False)
                        st.download_button(
                            label="Descargar validación (CSV)",
                            data=csv,
                            file_name=f"validacion_personal_{fecha_seleccionada}_{timestamp}.csv",
                            mime="text/csv",
                            key="download_validacion"
                        )
                        
                        st.info(f"""
                        **Contexto de esta validación:**
                        - Fecha: {fecha_seleccionada}
                        - Tus rutas: {len(rutas_tuyas)} de 87 totales
                        - Error promedio: {error_promedio:.1f}%
                        - MAE: {mae:.2f} minutos por ruta
                        """)
                
                else:
                    st.error("Error en la simulación de rutas")
                    
        except Exception as e:
            st.error(f"Error en validación: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("Selecciona una fecha de las jornadas y presiona 'Validar mis rutas'")
    
    with st.expander("¿Cómo funciona esta validación?"):
        st.markdown("""
        ### Validación Personalizada
        
        **¿Qué valida exactamente?**
        - Solo mis rutas (21-30 por día)
        
        **¿Por qué es importante?**
        - El dataset contiene solo mis registros
        - Cada operador tiene su propio ritmo y estilo
        - La validación debe reflejar mi realidad
        
        **¿Qué muestra?**
        - Comparación ruta por ruta de mis tiempos reales vs simulados
        - Error porcentual y absoluto por ruta
        - Identificación de rutas donde el modelo falla más
        
        **¿Cómo usar esta información?**
        - Si el error es <5%: Modelo confiable
        - Si el error es 5-10%: Revisar casos específicos
        - Si el error es >10%: Necesita ajuste personalizado
        """)
