
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Historial de Planificaciones",
    page_icon="📈",
    layout="wide"
)

st.title("Historial de Planificaciones")
st.markdown("---")

# Ruta del archivo de historial
historial_path = Path(__file__).parent.parent / 'data' / 'historial.csv'

# Cargar historial
if historial_path.exists():
    df_historial = pd.read_csv(historial_path)
    
    if not df_historial.empty:
        st.success(f"{len(df_historial)} registros en el historial")
        
        # Filtros
        st.subheader("Filtrar Historial")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'fecha_planificacion' in df_historial.columns:
                fechas_disponibles = sorted(df_historial['fecha_planificacion'].unique(), reverse=True)
                fecha_filtro = st.selectbox(
                    "Fecha de planificación:",
                    options=["Todas"] + fechas_disponibles,
                    key="filtro_fecha"
                )
            else:
                fecha_filtro = "Todas"
        
        with col2:
            if 'n_operadores' in df_historial.columns:
                operadores = sorted(df_historial['n_operadores'].unique())
                operador_filtro = st.selectbox(
                    "Número de operadores:",
                    options=["Todos"] + [str(op) for op in operadores],
                    key="filtro_operadores"
                )
            else:
                operador_filtro = "Todos"
        
        with col3:
            if 'meta_horas' in df_historial.columns:
                metas = sorted(df_historial['meta_horas'].unique())
                meta_filtro = st.selectbox(
                    "Meta de horas:",
                    options=["Todas"] + [str(m) for m in metas],
                    key="filtro_meta"
                )
            else:
                meta_filtro = "Todas"
        
        # Aplicar filtros
        df_filtrado = df_historial.copy()
        
        if fecha_filtro != "Todas" and 'fecha_planificacion' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['fecha_planificacion'] == fecha_filtro]
        
        if operador_filtro != "Todos" and 'n_operadores' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['n_operadores'] == int(operador_filtro)]
        
        if meta_filtro != "Todas" and 'meta_horas' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['meta_horas'] == float(meta_filtro)]
        
        # Mostrar resumen
        st.subheader(f"Resumen ({len(df_filtrado)} registros)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'tiempo_estimado' in df_filtrado.columns:
                st.metric("Tiempo promedio", f"{df_filtrado['tiempo_estimado'].mean():.1f} min")
            else:
                st.metric("Registros", len(df_filtrado))
        
        with col2:
            if 'n_operadores' in df_filtrado.columns:
                st.metric("Operadores promedio", f"{df_filtrado['n_operadores'].mean():.1f}")
        
        with col3:
            if 'meta_horas' in df_filtrado.columns:
                st.metric("Meta promedio", f"{df_filtrado['meta_horas'].mean():.1f} h")
        
        # Tabla de historial
        st.subheader("Registros Detallados")
        
        # Seleccionar columnas a mostrar
        columnas_mostrar = []
        columnas_disponibles = df_filtrado.columns.tolist()
        
        for col in ['fecha_planificacion', 'id_ruta', 'operador', 'tiempo_estimado', 'n_operadores', 'meta_horas']:
            if col in columnas_disponibles:
                columnas_mostrar.append(col)
        
        if not columnas_mostrar:
            columnas_mostrar = columnas_disponibles[:5]
        
        st.dataframe(
            df_filtrado[columnas_mostrar],
            use_container_width=True,
            hide_index=True
        )
        
        # Opciones de exportación
        st.subheader("Exportar Historial")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Exportar a CSV
            csv = df_filtrado.to_csv(index=False)
            st.download_button(
                label="Descargar Historial (CSV)",
                data=csv,
                file_name=f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_historial"
            )
        
        with col2:
            # Exportar a Excel (si está disponible)
            try:
                import openpyxl
                from io import BytesIO
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_filtrado.to_excel(writer, sheet_name='Historial', index=False)
                
                st.download_button(
                    label="Descargar Historial (Excel)",
                    data=output.getvalue(),
                    file_name=f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_historial_excel"
                )
            except:
                pass
        
        # Estadísticas adicionales
        with st.expander("Estadísticas Avanzadas"):
            st.subheader("Estadísticas por Operador")
            
            if 'operador' in df_filtrado.columns and 'tiempo_estimado' in df_filtrado.columns:
                stats_operador = df_filtrado.groupby('operador')['tiempo_estimado'].agg([
                    'count', 'mean', 'sum', 'std'
                ]).round(2)
                stats_operador.columns = ['Cantidad', 'Promedio', 'Total', 'Desv. Estándar']
                st.dataframe(stats_operador, use_container_width=True)
            
            st.subheader("Distribución de Tiempos")
            
            if 'tiempo_estimado' in df_filtrado.columns:
                import plotly.express as px
                fig = px.histogram(
                    df_filtrado,
                    x='tiempo_estimado',
                    title='Distribución de Tiempos Estimados',
                    labels={'tiempo_estimado': 'Tiempo (minutos)'},
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Limpiar historial
        with st.expander("Administrar Historial"):
            if st.button("Limpiar Historial", type="secondary"):
                if st.checkbox("Confirmar eliminación de todo el historial"):
                    df_historial_clean = pd.DataFrame()
                    df_historial_clean.to_csv(historial_path, index=False)
                    st.success("Historial limpiado exitosamente")
                    st.rerun()
    
    else:
        st.info("El historial está vacío. Comienza a planificar para guardar registros.")

else:
    st.info("No hay historial guardado aún. Las planificaciones se guardarán automáticamente.")
    
    # Crear archivo de historial vacío
    historial_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(historial_path, index=False)
    st.success("Archivo de historial creado")
