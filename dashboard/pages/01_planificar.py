# pages/01_planificar.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import re
from pathlib import Path
import sys

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.loaders import load_models, predict_tiempo
from utils.visualizations import (
    crear_grafico_balanceo,
    crear_grafico_distribucion,
    crear_gauge_riesgo,
    mostrar_metricas_planificacion
)
from utils.planners import ejecutar_planificacion_simple
from src.data_loader import CargadorDatos

st.set_page_config(
    page_title="Planificar Jornada",
    page_icon="📋",
    layout="wide"
)

st.title("Planificar Jornada de Despachos")
st.markdown("---")

# Inicializar estado de sesión
if 'resultados_planificacion' not in st.session_state:
    st.session_state.resultados_planificacion = None

# Cargar modelos
models = load_models()

# Verificar que el cargador esté disponible

if 'cargador' not in models or models['cargador'] is None:
    try:
        models['cargador'] = CargadorDatos()
        st.success("Cargador de datos inicializado")
    except Exception as e:
        st.warning(f"No se pudo inicializar el cargador: {str(e)}")

# Configuración en sidebar
with st.sidebar:
    st.header("Configuración de la Jornada")
    
    # 1. Carga de rutas
    st.subheader("Carga de Rutas")
    opcion_carga = st.radio(
        "Método de carga:",
        ["Pegar IDs", "Cargar archivo"],
        key="opcion_carga"
    )
    
    rutas_ids = []
    
    if opcion_carga == "Pegar IDs":
        texto_rutas = st.text_area(
            "Pega los IDs de ruta (uno por línea o separados por coma):",
            height=150,
            placeholder="Ejemplo:\n12345, 67890, 11111\nO\n12345\n67890\n11111",
            key="texto_rutas"
        )
        if texto_rutas:
            numeros = re.findall(r'\d+', texto_rutas)
            rutas_ids = [int(n) for n in numeros]
            st.info(f"{len(rutas_ids)} rutas cargadas")
    else:
        archivo = st.file_uploader(
            "Carga un archivo .txt o .csv con los IDs",
            type=['txt', 'csv'],
            key="archivo_rutas"
        )
        if archivo:
            try:
                if archivo.name.endswith('.csv'):
                    df = pd.read_csv(archivo)
                    rutas_ids = df.iloc[:, 0].astype(int).tolist()
                else:
                    contenido = archivo.read().decode('utf-8')
                    numeros = re.findall(r'\d+', contenido)
                    rutas_ids = [int(n) for n in numeros]
                st.info(f"{len(rutas_ids)} rutas cargadas desde archivo")
            except Exception as e:
                st.error(f"Error al leer archivo: {str(e)}")
    
    st.divider()
    
    # 2. Parámetros de la jornada
    st.subheader("Personal")
    n_operadores = st.slider(
        "Número de operadores:",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="n_operadores"
    )
    
    st.subheader("Horario")
    hora_inicio = st.time_input(
        "Hora de inicio:",
        value=time(15, 0),
        key="hora_inicio"
    )
    
    meta_horas = st.slider(
        "Meta de horas (jornada objetivo):",
        min_value=6.0,
        max_value=12.0,
        value=8.0,
        step=0.5,
        key="meta_horas"
    )
    
    st.subheader("Simulación")
    n_simulaciones = st.slider(
        "Número de simulaciones:",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        key="n_simulaciones"
    )
    
    st.divider()
    
    # 3. Eventos del día
    st.subheader("Eventos Programados")
    agregar_evento = st.checkbox("Agregar reuniones/programas", key="agregar_evento")
    eventos = []
    
    if agregar_evento:
        col1, col2 = st.columns(2)
        with col1:
            hora_evento = st.time_input("Hora del evento:", value=time(17, 0), key="hora_evento")
            duracion_evento = st.number_input("Duración (minutos):", min_value=5, max_value=120, value=30, step=5, key="duracion_evento")
        with col2:
            # Convertir "Operador X" a número para el simulador
            opciones_afecta = [f"Operador {i+1}" for i in range(n_operadores)]
            afecta_seleccionados = st.multiselect(
                "Afecta a operadores:",
                options=opciones_afecta,
                default=opciones_afecta[:min(2, n_operadores)],
                key="afecta_evento"
            )
            if afecta_seleccionados:
                # Convertir nombres a números para el simulador
                afecta_numeros = [int(op.split()[1]) for op in afecta_seleccionados]
                eventos.append({
                    'hora': hora_evento.strftime('%H:%M'),
                    'duracion': duracion_evento,
                    'afecta': afecta_numeros,
                    'descripcion': f"Evento a las {hora_evento.strftime('%H:%M')}"
                })
                st.success(f"Evento agregado: {hora_evento.strftime('%H:%M')} - {duracion_evento}min")
    
    st.subheader("Interrupciones Aleatorias")
    prob_interrupcion = st.slider(
        "Probabilidad de interrupción (por hora):",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        format="%.0f%%",
        key="prob_interrupcion"
    )
    
    duracion_interrupcion = st.slider(
        "Duración media (minutos):",
        min_value=5,
        max_value=60,
        value=15,
        step=5,
        key="duracion_interrupcion"
    )
    
    st.divider()
    
    # Botón de ejecución
    ejecutar = st.button(
        "Ejecutar Planificación",
        type="primary",
        use_container_width=True,
        key="btn_ejecutar"
    )

# Área principal de resultados
if ejecutar:
    if not rutas_ids:
        st.warning("No hay rutas cargadas. Por favor carga los IDs de ruta.")
    else:
        with st.spinner("Ejecutando planificación..."):
            try:
                # Preparar configuración
                config = {
                    'rutas_ids': rutas_ids,
                    'n_operadores': n_operadores,
                    'hora_inicio': hora_inicio,
                    'meta_horas': meta_horas,
                    'n_simulaciones': n_simulaciones,
                    'eventos': eventos,
                    'prob_interrupcion': prob_interrupcion,
                    'duracion_interrupcion': duracion_interrupcion
                }
                
                # Ejecutar planificación usando el nuevo planner
                resultados = ejecutar_planificacion_simple(config, models)
                
                if resultados:
                    st.session_state.resultados_planificacion = resultados
                    
                    # Mostrar resultados
                    st.success("Planificación completada exitosamente!")
                    
                    # Métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    mostrar_metricas_planificacion(resultados, meta_horas, col1, col2, col3, col4)
                    
                    st.divider()
                    
                    # Gráficos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Balanceo de Cargas")
                        fig_balance = crear_grafico_balanceo(resultados.get('asignacion', pd.DataFrame()))
                        if fig_balance:
                            st.plotly_chart(fig_balance, use_container_width=True)
                        else:
                            st.info("No hay datos de asignación para mostrar")
                    
                    with col2:
                        st.subheader("Distribución del Makespan")
                        fig_dist = crear_grafico_distribucion(resultados)
                        if fig_dist:
                            st.plotly_chart(fig_dist, use_container_width=True)
                        else:
                            st.info("No hay datos de simulación para mostrar")
                    
                    # Gauge de riesgo
                    st.subheader("Medidor de Riesgo de Horas Extra")
                    prob_extra = resultados.get('prob_extra', 0.0)
                    fig_gauge = crear_gauge_riesgo(prob_extra)
                    if fig_gauge:
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Tabla de asignación
                    st.subheader("Tabla de Asignación")
                    asignacion = resultados.get('asignacion', pd.DataFrame())
                    
                    if not asignacion.empty:
                        # Formatear tabla
                        df_show = asignacion.copy()
                        if 'tiempo_estimado' in df_show.columns:
                            df_show['tiempo_estimado'] = df_show['tiempo_estimado'].round(1)
                        
                        st.dataframe(
                            df_show,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.divider()
                        
                        # Resumen por operador

                        st.subheader("Resumen de Carga por Operador")
                        
                        # Calcular estadísticas por operador
                        if 'operador' in asignacion.columns and 'tiempo_estimado' in asignacion.columns:
                            resumen_operador = asignacion.groupby('operador').agg({
                                'id_ruta': 'count',
                                'tiempo_estimado': ['sum', 'mean', 'min', 'max', 'std']
                            }).round(2)
                            
                            # Renombrar columnas
                            resumen_operador.columns = [
                                'N° Rutas',
                                'Tiempo Total (min)',
                                'Tiempo Promedio (min)',
                                'Tiempo Mínimo (min)',
                                'Tiempo Máximo (min)',
                                'Desv. Estándar (min)'
                            ]
                            
                            # Resetear índice para mejor visualización
                            resumen_operador = resumen_operador.reset_index()
                            resumen_operador['Tiempo Total (horas)'] = (resumen_operador['Tiempo Total (min)'] / 60).round(2)
                            
                            # Reordenar columnas
                            resumen_operador = resumen_operador[[
                                'operador',
                                'N° Rutas',
                                'Tiempo Total (min)',
                                'Tiempo Total (horas)',
                                'Tiempo Promedio (min)',
                                'Tiempo Mínimo (min)',
                                'Tiempo Máximo (min)',
                                'Desv. Estándar (min)'
                            ]]
                            
                            # Mostrar tabla de resumen
                            st.dataframe(
                                resumen_operador,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'operador': st.column_config.TextColumn('Operador', width='small'),
                                    'N° Rutas': st.column_config.NumberColumn('N° Rutas', format='%d'),
                                    'Tiempo Total (min)': st.column_config.NumberColumn('Tiempo Total (min)', format='%.1f'),
                                    'Tiempo Total (horas)': st.column_config.NumberColumn('Tiempo Total (horas)', format='%.2f'),
                                    'Tiempo Promedio (min)': st.column_config.NumberColumn('Tiempo Promedio (min)', format='%.1f'),
                                    'Tiempo Mínimo (min)': st.column_config.NumberColumn('Tiempo Mínimo (min)', format='%.1f'),
                                    'Tiempo Máximo (min)': st.column_config.NumberColumn('Tiempo Máximo (min)', format='%.1f'),
                                    'Desv. Estándar (min)': st.column_config.NumberColumn('Desv. Estándar (min)', format='%.2f')
                                }
                            )
                            
                            
                            # Tarjetas de resumen rápido por operador
                            
                            st.subheader("Resumen Rápido por Operador")
                            
                            # Crear columnas para cada operador
                            num_operadores = len(resumen_operador)
                            cols = st.columns(min(num_operadores, 4))
                            
                            for idx, (col, row) in enumerate(zip(cols, resumen_operador.iterrows())):
                                _, row_data = row
                                with col:
                                    # Determinar color y estado según carga
                                    carga_pct = (row_data['Tiempo Total (min)'] / (meta_horas * 60)) * 100
                                    if carga_pct < 80:
                                        color_estado = "🟢"
                                        estado = "Buena carga"
                                        color_fondo = "#d4edda"
                                        color_borde = "#28a745"
                                        color_texto = "#155724"
                                    elif carga_pct < 100:
                                        color_estado = "🟡"
                                        estado = "Carga óptima"
                                        color_fondo = "#fff3cd"
                                        color_borde = "#ffc107"
                                        color_texto = "#856404"
                                    else:
                                        color_estado = "🔴"
                                        estado = "Sobrecarga"
                                        color_fondo = "#f8d7da"
                                        color_borde = "#dc3545"
                                        color_texto = "#721c24"
                                    
                                    st.markdown(f"""
                                    <div style="
                                        padding: 15px;
                                        border-radius: 10px;
                                        border: 2px solid {color_borde};
                                        background-color: {color_fondo};
                                        margin: 5px 0;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    ">
                                        <h4 style="margin: 0; color: {color_texto}; text-align: center; font-weight: bold;">
                                            {row_data['operador']}
                                        </h4>
                                        <hr style="margin: 8px 0; border-color: {color_borde};">
                                        <p style="margin: 4px 0; font-size: 14px; color: {color_texto};">
                                            <b>Rutas:</b> {int(row_data['N° Rutas'])}
                                        </p>
                                        <p style="margin: 4px 0; font-size: 14px; color: {color_texto};">
                                            <b>Total:</b> {row_data['Tiempo Total (min)']:.1f} min 
                                            ({row_data['Tiempo Total (horas)']:.2f} h)
                                        </p>
                                        <p style="margin: 4px 0; font-size: 14px; color: {color_texto};">
                                            <b>Promedio:</b> {row_data['Tiempo Promedio (min)']:.1f} min/ruta
                                        </p>
                                        <p style="margin: 4px 0; font-size: 14px; color: {color_texto};">
                                            <b>Rango:</b> {row_data['Tiempo Mínimo (min)']:.0f} - 
                                            {row_data['Tiempo Máximo (min)']:.0f} min
                                        </p>
                                        <p style="margin: 4px 0; font-size: 14px; color: {color_texto};">
                                            <b>{color_estado} Carga:</b> {carga_pct:.1f}% de la meta
                                        </p>
                                        <p style="margin: 4px 0; font-size: 12px; color: {color_texto}; text-align: center; font-style: italic;">
                                            {estado}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            
                            # Métricas de balanceo
                            
                            st.divider()
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                max_carga = resumen_operador['Tiempo Total (min)'].max()
                                min_carga = resumen_operador['Tiempo Total (min)'].min()
                                diferencia = max_carga - min_carga
                                diferencia_pct = (diferencia / max_carga) * 100 if max_carga > 0 else 0
                                
                                st.metric(
                                    "Balanceo de Carga",
                                    f"{diferencia:.1f} min",
                                    f"{diferencia_pct:.1f}% de diferencia",
                                    delta_color="inverse" if diferencia_pct < 15 else "off"
                                )
                            
                            with col2:
                                carga_promedio = resumen_operador['Tiempo Total (min)'].mean()
                                st.metric(
                                    "Carga Promedio",
                                    f"{carga_promedio:.1f} min",
                                    f"{carga_promedio/60:.2f} horas"
                                )
                            
                            with col3:
                                total_rutas = resumen_operador['N° Rutas'].sum()
                                total_tiempo = resumen_operador['Tiempo Total (min)'].sum()
                                
                                st.metric(
                                    "Eficiencia",
                                    f"{total_rutas / len(resumen_operador):.1f} rutas/op",
                                    f"{total_tiempo / 60:.1f} horas totales"
                                )
                            
                            # SUGERENCIAS DE REBALANCEO
                            
                            if len(resumen_operador) > 1:
                                st.divider()
                                st.subheader("Sugerencias de Rebalanceo")
                                
                                max_op = resumen_operador.loc[resumen_operador['Tiempo Total (min)'].idxmax()]
                                min_op = resumen_operador.loc[resumen_operador['Tiempo Total (min)'].idxmin()]
                                
                                diferencia = max_op['Tiempo Total (min)'] - min_op['Tiempo Total (min)']
                                diferencia_pct = (diferencia / max_op['Tiempo Total (min)']) * 100
                                
                                if diferencia > 30:
                                    # Calcular cuántas rutas mover
                                    tiempo_promedio_ruta = min_op['Tiempo Promedio (min)']
                                    rutas_a_mover = max(1, int(diferencia / tiempo_promedio_ruta / 2))
                                    rutas_a_mover = min(rutas_a_mover, 5)  # Máximo 5 rutas
                                    
                                    nueva_diferencia = max(0, diferencia - rutas_a_mover * tiempo_promedio_ruta)
                                    nueva_carga_max = max_op['Tiempo Total (min)'] - rutas_a_mover * tiempo_promedio_ruta
                                    nueva_carga_min = min_op['Tiempo Total (min)'] + rutas_a_mover * tiempo_promedio_ruta
                                    
                                    st.warning(f"""
                                    **Desbalanceo detectado:**
                                    
                                    - **{max_op['operador']}** tiene **{diferencia:.1f} min** más que **{min_op['operador']}** 
                                      ({diferencia_pct:.1f}% de diferencia)
                                    
                                    **Recomendación:**
                                    - Mover **{rutas_a_mover} rutas** de **{max_op['operador']}** a **{min_op['operador']}**
                                    - Nuevo balanceo estimado:
                                      - {max_op['operador']}: {nueva_carga_max:.1f} min ({nueva_carga_max/60:.2f} h)
                                      - {min_op['operador']}: {nueva_carga_min:.1f} min ({nueva_carga_min/60:.2f} h)
                                      - Diferencia: {nueva_diferencia:.1f} min
                                    """)
                                    
                                    st.info("**Sugerencia:** Re-ejecuta la planificación con un rebalanceo manual o ajusta el número de operadores.")
                                else:
                                    st.success(f"""
                                    **Buen balanceo de carga:**
                                    
                                    - Diferencia entre operadores: **{diferencia:.1f} min** ({diferencia_pct:.1f}%)
                                    - Todos los operadores tienen cargas equilibradas
                                    - No se requieren ajustes de rebalanceo
                                    """)
                            
                            
                            # GRÁFICO DE BALANCEO MEJORADO
                            
                            st.subheader("Análisis de Carga por Operador")
                            
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            fig_balance_detallado = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=('Carga Total por Operador', 'Distribución de Tiempos'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            # Gráfico 1: Barras de carga total
                            colores_barras = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(resumen_operador)]
                            fig_balance_detallado.add_trace(
                                go.Bar(
                                    name='Carga Total',
                                    x=resumen_operador['operador'],
                                    y=resumen_operador['Tiempo Total (min)'],
                                    marker_color=colores_barras,
                                    text=[f"{t:.1f} min" for t in resumen_operador['Tiempo Total (min)']],
                                    textposition='outside'
                                ),
                                row=1, col=1
                            )
                            
                            # Línea de meta
                            meta_minutos = meta_horas * 60
                            fig_balance_detallado.add_hline(
                                y=meta_minutos,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Meta: {meta_horas}h",
                                row=1, col=1
                            )
                            
                            # Gráfico 2: Boxplot de tiempos por operador
                            tiempos_por_operador = []
                            for operador in resumen_operador['operador']:
                                tiempos_op = asignacion[asignacion['operador'] == operador]['tiempo_estimado']
                                tiempos_por_operador.append(tiempos_op)
                            
                            colores_box = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(tiempos_por_operador)]
                            
                            for i, (tiempos, color) in enumerate(zip(tiempos_por_operador, colores_box)):
                                fig_balance_detallado.add_trace(
                                    go.Box(
                                        y=tiempos,
                                        name=f'Op {i+1}',
                                        boxmean='sd',
                                        marker_color=color,
                                        legendgroup=f'box_{i}',
                                        showlegend=False
                                    ),
                                    row=1, col=2
                                )
                            
                            fig_balance_detallado.update_layout(
                                height=400,
                                showlegend=False,
                                margin=dict(l=50, r=50, t=50, b=50)
                            )
                            
                            fig_balance_detallado.update_yaxes(title_text="Tiempo (minutos)", row=1, col=1)
                            fig_balance_detallado.update_yaxes(title_text="Tiempo (minutos)", row=1, col=2)
                            
                            st.plotly_chart(fig_balance_detallado, use_container_width=True)
                            
                            
                            # BOTÓN DE DESCARGA
                            
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                            csv = df_show.to_csv(index=False)
                            st.download_button(
                                label="Descargar asignación (CSV)",
                                data=csv,
                                file_name=f"asignacion_{timestamp}.csv",
                                mime="text/csv",
                                key="download_planificacion"
                            )
                            
                            
                            # GUARDAR EN HISTORIAL
                            
                            try:
                                historial_path = Path(__file__).parent.parent / 'data' / 'historial.csv'
                                historial_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                df_guardar = df_show.copy()
                                df_guardar['fecha_planificacion'] = timestamp
                                df_guardar['n_operadores'] = n_operadores
                                df_guardar['meta_horas'] = meta_horas
                                df_guardar['prob_extra'] = prob_extra
                                df_guardar['makespan_simulado'] = resultados.get('makespan_simulado', 0)
                                
                                if historial_path.exists() and historial_path.stat().st_size > 0:
                                    try:
                                        historial = pd.read_csv(historial_path)
                                        historial = pd.concat([historial, df_guardar], ignore_index=True)
                                    except pd.errors.EmptyDataError:
                                        historial = df_guardar
                                else:
                                    historial = df_guardar
                                
                                historial.to_csv(historial_path, index=False)
                                st.success("💾 Planificación guardada en historial")
                            except Exception as e:
                                st.warning(f"No se pudo guardar en historial: {str(e)}")
                    
                    else:
                        st.warning("No se generó asignación")
                
                else:
                    st.error("Error en la planificación. Revisa los logs.")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:
    if st.session_state.resultados_planificacion:
        # Mostrar resultados anteriores si existen
        st.info("Mostrando última planificación realizada")
        resultados = st.session_state.resultados_planificacion
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        meta_horas = st.session_state.get('meta_horas', 8.0)
        mostrar_metricas_planificacion(resultados, meta_horas, col1, col2, col3, col4)
        
        st.divider()
        
        # Gráficos (resumidos)
        col1, col2 = st.columns(2)
        with col1:
            fig_balance = crear_grafico_balanceo(resultados.get('asignacion', pd.DataFrame()))
            if fig_balance:
                st.plotly_chart(fig_balance, use_container_width=True)
        
        with col2:
            fig_dist = crear_grafico_distribucion(resultados)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Configura los parámetros en la barra lateral y presiona 'Ejecutar Planificación'")
