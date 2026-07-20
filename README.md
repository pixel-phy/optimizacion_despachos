# Sistema de Planificación de Despachos en Bodega

## Descripción

Este es un proyecto personal de **Investigación de Operaciones** y **Ciencia de Datos**, en el que pretendo optimizar mi rendimiento como despachador en el equipo de logística. El objetivo es maximizar mi productividad (productos/hora), contribuir de mejor manera al equipo y amenizar mi ambiente laboral mediante la toma y el análisis de los datos en jornadas.

El proyecto integra técnicas de **Machine Learning**, **Optimización Matemática** y **Simulación Monte Carlo** para:

- Predecir tiempos de preparación de rutas con alta precisión.
- Optimizar la asignación de rutas entre el equipo de Despachos.
- Simular escenarios realistas para la toma de decisiones.
- Validar predicciones contra jornadas reales del pasado.

**Todo el sistema está construído 100 sobre mis datos personales** de despacho, lo que garantiza que las predicciones y simulaciones reflejan mi ritmo y estilo de trabajo real. 

---

## Objetivo por Fase

- **Fase 1:** Análisis exploratorio, limpieza y entendimiento de los Datos.
- **Fase 2:** Machine Learning para predecir mi rendimiento.
- **Fase 3:** Optimización de mi ritmo y estrategias de trabajo. 
- **Fase 4:** Dashboard personal y portafolio profesional.

---
## Datos

- **Fuente:** Registros de despachos de bodega (2026-05-26 hasta actualidad)
- **Registros:** 604 rutas personales (27 jornadas)
- **Variables:** fecha, id_ruta, tiempos, cant_productos, valor_ruta
- **Contexto:** Jornada diaria personal en la que las rutas son asignadas aleatoriamente por el sistema según la programación de los pedidos y su prioridad. Los pedidos se dejan listos (en la tarde) para ser cargados por los conductores al día siguiente.

--- 

## Estructura del proyecto

```
optimizacion_despachos/
│
├── 📁 src/                          # Código fuente reutilizable
│   ├── data_loader.py               # Carga de datos y modelos
│   ├── models.py                    # Modelos ML (Random Forest, XGBoost, K-Means)
│   ├── optimization.py              # Optimización PuLP y simulación Monte Carlo
│   ├── preprocessing.py             # Limpieza y preprocesamiento de datos
│   └── visualization.py             # Funciones de visualización
│
├── 📁 notebooks/                    # Notebooks de análisis (3 fases)
│   ├── 01_eda_cleaning.ipynb        # Fase 1: EDA y limpieza
│   ├── 02_machine_learning.ipynb    # Fase 2: Entrenamiento de modelos
│   └── 03_optimizacion_simulacion.ipynb  # Fase 3: Optimización y simulación
│
├── 📁 dashboard/                    # Dashboard interactivo (Streamlit)
│   ├── app.py                       # Página principal
│   ├── pages/                       # Páginas del dashboard
│   │   ├── 01_planificar.py         # Planificación de jornadas
│   │   ├── 02_validar.py            # Validación retrospectiva
│   │   └── 03_historial.py          # Historial de planificaciones
│   ├── utils/                       # Utilidades del dashboard
│   │   ├── loaders.py               # Carga de modelos y datos
│   │   ├── planners.py              # Planificación de jornadas
│   │   └── visualizations.py        # Visualizaciones interactivas
│   └── data/                        # Datos del dashboard
│       └── historial.csv            # Historial de planificaciones
│
├── 📁 data/                         # Datos del proyecto
│   ├── raw/                         # Datos crudos
│   │   └── io_data_bimbo.xlsx       # Datos personales de despachos (604 registros)
│   ├── processed/                   # Datos procesados
│   │   ├── despachos_clean.csv      # Datos limpios
│   │   └── rendimiento.db           # Base de datos SQLite
│   └── simulaciones/                # Resultados de simulaciones
│
├── 📁 models/                       # Modelos entrenados (persistidos)
│   ├── random_forest_model.pkl      # Random Forest (8 features)
│   ├── xgboost_model.pkl            # XGBoost (respaldo)
│   ├── scaler.pkl                   # Scaler (8 features)
│   ├── kmeans_model.pkl             # K-Means clustering
│   └── metadatos.pkl                # Metadatos del proyecto
│
├── 📁 reports/                      # Reportes y figuras
│   └── figures/                     # Gráficos generados
│
├── 📁 tests/                        # Tests unitarios
├── requirements.txt                 # Dependencias del proyecto
├── LICENSE                          # Licencia MIT
├── run_dashboard.py                 # Script para ejecutar el dashboard
└── README.md                        # Este archivo


```
---

## Herramientas utilizadas

### Machine Learning & Deep Learning

| Librería | Propósito |
| ---------| ----------|
| `scikit-learn` | Random Forest, K-Means, preprocesamiento |
| `xgboost` | Modelo de boosting (respaldo) |
| `tensorflow/keras` | Deep Learning (exploratorio) |
| `joblib` | Persistencia de modelos |

### Optimización

| Librería | Propósito |
| ---------| ----------|
| `pulp` | Programación Lineal Entera (MILP) |
| `simpy` | Simulación de eventos discretos |

### Visualización

| Librería | Propósito |
| ---------| ----------|
| `matplotlib` | Gráficos estáticos |
| `seaborn` | Gráficos estadísticos |
| `plotlib` | Gráficos interactivos (dashboard) |

### Dashboard 

| Librería | Propósito |
| ---------| ----------|
| `streamlit` | Framework de dashboard interactivo |

### Datos

| Librería | Propósito |
| ---------| ----------|
| `pandas` | Manipulación de datos |
| `numpy` | Cálculos numéricos |
| `sqlite3` | Base de datos ligera |

---

## Autor

**Luis Miguel Marín Cadavid**

- Auxiliar de Despachos | Investigador de Operaciones
- Proyecto desarrollado como herramienta personal para optimización de jornadas de despacho
