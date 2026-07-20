# dashboard/utils/visualizations.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def crear_grafico_balanceo(asignacion):
    """Crea gráfico de balanceo de cargas"""
    if asignacion.empty:
        return None
    
    if 'operador' not in asignacion.columns or 'tiempo_estimado' not in asignacion.columns:
        return None
    
    # Calcular carga por operador
    carga_ops = asignacion.groupby('operador')['tiempo_estimado'].sum().reset_index()
    
    fig = px.bar(
        carga_ops,
        x='operador',
        y='tiempo_estimado',
        title='Carga de trabajo por operador',
        labels={'tiempo_estimado': 'Tiempo total (minutos)', 'operador': 'Operador'},
        color='tiempo_estimado',
        color_continuous_scale='Viridis',
        text='tiempo_estimado'
    )
    
    # Línea de meta
    tiempo_promedio = carga_ops['tiempo_estimado'].mean()
    fig.add_hline(
        y=tiempo_promedio,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio: {tiempo_promedio:.1f} min"
    )
    
    fig.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def crear_grafico_distribucion(resultados):
    """Crea histograma de distribución del makespan"""
    tiempos = resultados.get('tiempos_simulados', [])
    
    if not tiempos:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=tiempos,
        nbinsx=30,
        name='Frecuencia',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Estadísticas
    media = np.mean(tiempos)
    mediana = np.median(tiempos)
    p5 = np.percentile(tiempos, 5)
    p95 = np.percentile(tiempos, 95)
    
    # Líneas de referencia
    fig.add_vline(
        x=media,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Media: {media:.1f} min",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=mediana,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Mediana: {mediana:.1f} min",
        annotation_position="bottom"
    )
    
    fig.add_vline(
        x=p5,
        line_dash="dot",
        line_color="green",
        annotation_text=f"P5: {p5:.1f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=p95,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"P95: {p95:.1f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Distribución del Makespan Simulado',
        xaxis_title='Makespan (minutos)',
        yaxis_title='Frecuencia',
        height=400,
        showlegend=False
    )
    
    return fig

def crear_gauge_riesgo(prob_extra):
    """Crea un medidor de riesgo (gauge chart)"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_extra * 100,
        title={'text': "Riesgo de Horas Extra", 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 30], 'color': "yellow"},
                {'range': [30, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    # Añadir texto descriptivo
    if prob_extra < 0.1:
        status = "🟢 BAJO RIESGO"
    elif prob_extra < 0.3:
        status = "🟡 RIESGO MODERADO"
    else:
        status = "🔴 ALTO RIESGO"
    
    fig.add_annotation(
        x=0.5,
        y=-0.2,
        text=status,
        showarrow=False,
        font=dict(size=16)
    )
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    
    return fig

def crear_grafico_validacion(df_real, tiempos_reales, tiempos_simulados):
    """
    Crea gráfico comparativo para validación.
    
    CORREGIDO: Las rutas se muestran como etiquetas categóricas.
    """
    # Asegurar que ambas listas tengan la misma longitud
    n = min(len(tiempos_reales), len(tiempos_simulados))
    tiempos_reales = tiempos_reales[:n]
    tiempos_simulados = tiempos_simulados[:n]
    
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Comparativa por Ruta', 'Distribución de Errores'),
                        vertical_spacing=0.15)
    
    # CORRECCIÓN: Usar números de ruta como strings (categorías)
    rutas = [str(i+1) for i in range(n)]
    
    # Gráfico superior: barras comparativas
    fig.add_trace(
        go.Bar(
            name='Real',
            x=rutas,
            y=tiempos_reales,
            marker_color='lightblue',
            hovertemplate='<b>Ruta %{x}</b><br>Real: %{y:.1f} min<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Simulado',
            x=rutas,
            y=tiempos_simulados,
            marker_color='lightcoral',
            hovertemplate='<b>Ruta %{x}</b><br>Simulado: %{y:.1f} min<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico inferior: distribución de errores
    errores = [abs(r - s) for r, s in zip(tiempos_reales, tiempos_simulados)]
    
    fig.add_trace(
        go.Histogram(
            x=errores,
            nbinsx=20,
            marker_color='purple',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Línea de media en el histograma
    media_error = np.mean(errores)
    fig.add_vline(
        x=media_error,
        line_dash="dash",
        line_color="red",
        annotation_text=f"MAE: {media_error:.2f} min",
        row=2, col=1
    )
    
    # ============================================================
    # CORRECCIÓN: Eje X como categorías
    # ============================================================
    fig.update_xaxes(
        title_text="Ruta",
        type='category',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Error (minutos)",
        row=2, col=1
    )
    
    fig.update_yaxes(title_text="Tiempo (minutos)", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Comparativa Real vs Simulado",
        barmode='group'
    )
    
    return fig

# ============================================================
# FUNCIÓN mostrar_metricas_planificacion (AGREGADA)
# ============================================================

def mostrar_metricas_planificacion(resultados, meta_horas, col1, col2, col3, col4):
    """Muestra métricas principales en cards"""
    
    makespan_planificado = resultados.get('makespan_planificado', 0)
    makespan_simulado = resultados.get('makespan_simulado', 0)
    prob_extra = resultados.get('prob_extra', 0.0)
    
    # Determinar color según probabilidad
    if prob_extra < 0.1:
        color = "🟢"
        veredicto = "VIABLE"
    elif prob_extra < 0.3:
        color = "🟡"
        veredicto = "REQUIERE ATENCIÓN"
    else:
        color = "🔴"
        veredicto = "ALTO RIESGO"
    
    with col1:
        st.metric(
            "Makespan Planificado",
            f"{makespan_planificado:.1f} min",
            f"{makespan_planificado/60:.1f} horas"
        )
    
    with col2:
        st.metric(
            "Makespan Simulado",
            f"{makespan_simulado:.1f} min",
            f"{makespan_simulado/60:.1f} horas"
        )
    
    with col3:
        st.metric(
            "Prob. Horas Extra",
            f"{prob_extra*100:.1f}%",
            f"{color}"
        )
    
    with col4:
        st.metric(
            "Veredicto",
            veredicto,
            f"Meta: {meta_horas} horas"
        )

# ============================================================
# FUNCIÓN crear_grafico_validacion_personal (CORREGIDA)
# ============================================================

def crear_grafico_validacion_personal(df_tus_rutas, tiempos_reales, tiempos_simulados):
    """
    Crea gráfico comparativo específico para tus rutas personales.
    
    CORREGIDO: Los IDs de ruta se muestran como etiquetas categóricas,
    no como una escala numérica.
    """
    n = min(len(tiempos_reales), len(tiempos_simulados))
    
    if n == 0:
        return None
    
    # Preparar datos - IDs de ruta como strings (categorías)
    rutas = df_tus_rutas['id_ruta'].astype(str).tolist()[:n]
    tiempos_reales = tiempos_reales[:n]
    tiempos_simulados = tiempos_simulados[:n]
    errores = [abs(r - s) for r, s in zip(tiempos_reales, tiempos_simulados)]
    error_pct = [(e/r*100) if r > 0 else 0 for e, r in zip(errores, tiempos_reales)]
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            'Comparativa de Tiempos (Tus Rutas)',
            'Error Absoluto por Ruta',
            'Error Porcentual por Ruta'
        ),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Gráfico 1: Barras comparativas
    fig.add_trace(
        go.Bar(
            name='Tiempo Real',
            x=rutas,
            y=tiempos_reales,
            marker_color='lightblue',
            text=[f'{t:.1f}' for t in tiempos_reales],
            textposition='outside',
            hovertemplate='<b>Ruta %{x}</b><br>Tiempo Real: %{y:.1f} min<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Tiempo Simulado',
            x=rutas,
            y=tiempos_simulados,
            marker_color='lightcoral',
            text=[f'{t:.1f}' for t in tiempos_simulados],
            textposition='outside',
            hovertemplate='<b>Ruta %{x}</b><br>Tiempo Simulado: %{y:.1f} min<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Error absoluto
    colors_abs = ['green' if e < 3 else 'orange' if e < 6 else 'red' for e in errores]
    fig.add_trace(
        go.Bar(
            name='Error Absoluto',
            x=rutas,
            y=errores,
            marker_color=colors_abs,
            text=[f'{e:.1f} min' for e in errores],
            textposition='outside',
            hovertemplate='<b>Ruta %{x}</b><br>Error: %{y:.1f} min<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Línea de referencia MAE
    mae = np.mean(errores)
    fig.add_hline(
        y=mae,
        line_dash="dash",
        line_color="red",
        annotation_text=f"MAE: {mae:.2f} min",
        row=2, col=1
    )
    
    # Gráfico 3: Error porcentual
    colors_pct = ['green' if p < 5 else 'orange' if p < 10 else 'red' for p in error_pct]
    fig.add_trace(
        go.Bar(
            name='Error Porcentual',
            x=rutas,
            y=error_pct,
            marker_color=colors_pct,
            text=[f'{p:.1f}%' for p in error_pct],
            textposition='outside',
            hovertemplate='<b>Ruta %{x}</b><br>Error: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Línea de referencia error promedio
    error_prom = np.mean(error_pct)
    fig.add_hline(
        y=error_prom,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio: {error_prom:.1f}%",
        row=3, col=1
    )
    
    # CORRECCIÓN: Eje X como categorías
    fig.update_xaxes(
        title_text="ID Ruta",
        type='category',
        tickangle=-45,
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="ID Ruta",
        type='category',
        tickangle=-45,
        row=2, col=1
    )
    fig.update_xaxes(
        title_text="ID Ruta",
        type='category',
        tickangle=-45,
        row=3, col=1
    )
    
    fig.update_yaxes(title_text="Tiempo (minutos)", row=1, col=1)
    fig.update_yaxes(title_text="Error (minutos)", row=2, col=1)
    fig.update_yaxes(title_text="Error (%)", row=3, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text=f"Validación de tus rutas personales ({len(rutas)} rutas)",
        barmode='group',
        margin=dict(l=50, r=50, t=80, b=100)
    )
    
    return fig

# ============================================================
# FUNCIÓN crear_grafico_validacion_operadores (NUEVA - Opcional)
# ============================================================

def crear_grafico_validacion_operadores(df_tus_rutas, tiempos_reales, tiempos_simulados):
    """
    Versión alternativa que muestra IDs de ruta reales en lugar de índices.
    """
    n = min(len(tiempos_reales), len(tiempos_simulados))
    
    if n == 0:
        return None
    
    # Usar IDs de ruta reales como etiquetas
    rutas = df_tus_rutas['id_ruta'].astype(str).tolist()[:n]
    tiempos_reales = tiempos_reales[:n]
    tiempos_simulados = tiempos_simulados[:n]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rutas,
        y=tiempos_reales,
        mode='markers+lines',
        name='Tiempo Real',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', dash='solid')
    ))
    
    fig.add_trace(go.Scatter(
        x=rutas,
        y=tiempos_simulados,
        mode='markers+lines',
        name='Tiempo Simulado',
        marker=dict(size=10, color='red'),
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Comparativa de Tiempos por Ruta (Tus Rutas)',
        xaxis_title='ID Ruta',
        yaxis_title='Tiempo (minutos)',
        xaxis=dict(type='category'),
        height=500,
        showlegend=True
    )
    
    return fig
