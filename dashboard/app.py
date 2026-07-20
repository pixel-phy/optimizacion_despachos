
import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Configuración de la página - DEBE SER LA PRIMERA LLAMADA A STREAMLIT
st.set_page_config(
    page_title="Dashboard de Planificación de Despachos",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent))

# Título principal
st.title(" Planificador de Jornadas de Despachos")
st.markdown("---")

# Información del proyecto en la barra lateral
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/delivery.png", width=80)
    st.markdown("## Investigación de Operaciones")
    st.markdown("---")
    st.markdown("**Fase 4:** Dashboard Interactivo")
    st.markdown(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")
    st.markdown("### KPIs del Sistema")
    
    # Cargar métricas del modelo (si existen)
    try:
        from utils.loaders import load_models
        models = load_models()
        if models and 'metadata' in models:
            metadata = models['metadata']
            st.metric(" R² del modelo", f"{metadata.get('r2', 0.80):.2f}")
            st.metric(" MAE", f"{metadata.get('mae', 2.86):.2f} min")
            st.metric(" Registros", f"{metadata.get('n_registros', 584)}")
    except:
        pass
    
    st.markdown("---")
    st.caption("Navega usando las pestañas superiores")

# Instrucciones iniciales
st.markdown("""
### Bienvenido al Dashboard de Planificación

Este sistema te permite planificar jornadas de despacho utilizando:
- **Random Forest** para predicción de tiempos
- **Optimización con PuLP** para asignación de rutas
- **Simulación Monte Carlo** para análisis de riesgo

#### Cómo usar:
1. Ve a la pestaña **📋 Planificar** para crear una nueva planificación
2. Usa la pestaña **🔍 Validar** para comparar con datos históricos
3. Revisa el **📈 Historial** de planificaciones anteriores

---
""")

# Mostrar estado de los modelos
try:
    from utils.loaders import load_models, load_available_dates
    
    with st.spinner(" Verificando modelos..."):
        models = load_models()
        if models:
            st.success(" Modelos cargados correctamente")
            fechas = load_available_dates()
            st.info(f" {len(fechas)} fechas disponibles para validación")
        else:
            st.error(" Error al cargar los modelos")
except Exception as e:
    st.error(f" Error: {str(e)}")
