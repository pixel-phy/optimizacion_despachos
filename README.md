# Optimización de Rendimiento en Bodega

## Descripción

Este es un proyecto personal de Investigación de Operaciones y Ciencia de datos, en el que pretendo optimizar mi rendimiento como despachador en el equipo de logística. El objetivo es maximizar mi productividad (productos/hora) y contribuir de mejor manera al equipo y amenizar mi ambiente laboral.

El proyecto se desarrollará por fases.

## Objetivos

- **Fase 1**: Análisis y entendimiento de los datos extraídos de mi operación.
- **Fase 2**: Machine Learning para predecir mi rendimiento.
- **Fase 3**: Optimización de mi ritmo y estrategias de trabajo.
- **Fase 4**: Dashboard personal y portafolio profesional.

## Datos

- **Fuente**: Registros de despachos de bodega desde el (2026-05-26 hasta actualidad).
- **Variables**: fecha, id_ruta, tiempos, cant_productos, valor_ruta
- **Contexto**: Jornada diaria personal en la que las rutas son asignadas aleatoriamente por el sistema según la programación de pedidos.

## Herramientas utilizadas

- **Limpieza**: Python, Pandas, Numpy, SQlite.
- **Visualización**: Matplotlib, Seaborn.
- **Machine Learning**: Scikit-learn, XGBoost.
- **Optimización**: PuLP, Simpy.
- **Deep Learning**: TensorFlow/Keras 
- **Dashboard**: Streamlit

## Autor

[Luis Miguel Marín Cadavid]


## Instalación
```bash
git clone https://github.com/[tu-usuario]/bimbo_performance_optimization.git
cd bimbo_performance_optimization
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
