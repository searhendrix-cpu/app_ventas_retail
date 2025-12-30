import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- PASO 1: Configuración ---
st.set_page_config(
    page_title="RetailMax Predictor de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# --- PASO 2: Carga de Datos y Modelo ---
@st.cache_data
def cargar_datos_demo():
    np.random.seed(42)
    tiendas = [f"Tienda_{i:02d}" for i in range(1, 21)]
    fechas = pd.date_range(start='2024-01-01', end='2024-03-15', freq='W')

    data = []
    for tienda in tiendas:
        for fecha in fechas:
            data.append({
                'tienda_id': tienda,
                'fecha': fecha,
                'ventas_semanales': np.random.normal(15000, 3000),
                'promocion_activa': np.random.choice([0, 1], p=[0.7, 0.3]),
                'inventario_inicial': np.random.normal(50000, 10000),
                'temperatura_promedio': np.random.normal(20, 8),
                'año': fecha.year,
                'mes': fecha.month,
                'semana': fecha.isocalendar().week,
                'dia_semana': fecha.dayofweek
            })
    return pd.DataFrame(data)

@st.cache_resource
def cargar_modelo_demo():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    df = cargar_datos_demo()
    feature_columns = [
        'promocion_activa', 'inventario_inicial', 'temperatura_promedio',
        'año', 'mes', 'semana', 'dia_semana'
    ]
    X = df[feature_columns]
    y = df['ventas_semanales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    return modelo, feature_columns

modelo, feature_columns = cargar_modelo_demo()
df_historico = cargar_datos_demo()

# --- PASO 3: Sidebar ---
def crear_sidebar():
    st.sidebar.markdown("## Configuración de Predicción")
    tiendas_disponibles = df_historico['tienda_id'].unique()
    tienda_seleccionada = st.sidebar.selectbox("Selecciona la tienda:", options=tiendas_disponibles)
    fecha_prediccion = st.sidebar.date_input("Fecha de predicción", value=datetime.now())
    promocion_activa = st.sidebar.radio("¿Habrá promoción activa?", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No", index=0)
    inventario_inicial = st.sidebar.slider("Inventario inicial:", min_value=20000, max_value=80000, value=50000, step=1000)
    temperatura_promedio = st.sidebar.slider("Temperatura promedio (°C):", min_value=-10, max_value=40, value=20, step=1)
    predecir = st.sidebar.button("Hacer Predicción", type="primary")

    return {
        'tienda': tienda_seleccionada,
        'fecha': fecha_prediccion,
        'promocion': promocion_activa,
        'inventario': inventario_inicial,
        'temperatura': temperatura_promedio,
        'predecir': predecir
    }

parametros = crear_sidebar()

# --- PASO 4: Predicción ---
def hacer_prediccion(parametros):
    fecha = pd.to_datetime(parametros['fecha'])
    datos_prediccion = pd.DataFrame({
        'promocion_activa': [parametros['promocion']],
        'inventario_inicial': [parametros['inventario']],
        'temperatura_promedio': [parametros['temperatura']],
        'año': [fecha.year],
        'mes': [fecha.month],
        'semana': [fecha.isocalendar().week],
        'dia_semana': [fecha.dayofweek]
    })

    prediccion = modelo.predict(datos_prediccion)[0]
    std_error = prediccion * 0.15
    intervalo_inferior = prediccion - 1.96 * std_error
    intervalo_superior = prediccion + 1.96 * std_error

    return {
        'prediccion': prediccion,
        'intervalo_inferior': max(0, intervalo_inferior),
        'intervalo_superior': intervalo_superior,
        'confianza': 95
    }

def mostrar_resultados_prediccion(resultado, parametros):
    st.markdown("## Resultados de la Predicción")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ventas Predichas", f"${resultado['prediccion']:,.0f}")
    col2.metric("Rango Mínimo", f"${resultado['intervalo_inferior']:,.0f}")
    col3.metric("Rango Máximo", f"${resultado['intervalo_superior']:,.0f}")

    media_historica = df_historico[df_historico['tienda_id'] == parametros['tienda']]['ventas_semanales'].mean()

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = resultado['prediccion'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': media_historica},
        gauge = {'axis': {'range': [None, 30000]}, 'bar': {'color': "darkblue"}}
    ))
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PASO 5: Análisis ---
def crear_analisis_comparativo(parametros, resultado):
    st.markdown("## Análisis Comparativo")
    datos_tienda = df_historico[df_historico['tienda_id'] == parametros['tienda']].copy().sort_values('fecha')

    fig_temporal = go.Figure()
    fig_temporal.add_trace(go.Scatter(x=datos_tienda['fecha'], y=datos_tienda['ventas_semanales'], name='Histórico'))
    fig_temporal.add_trace(go.Scatter(x=[parametros['fecha']], y=[resultado['prediccion']], mode='markers', marker=dict(color='red', size=12), name='Predicción'))
    st.plotly_chart(fig_temporal, use_container_width=True)

def crear_analisis_sensibilidad():
    st.markdown("## Análisis de Sensibilidad")
    escenarios = []
    for promo in [0, 1]:
        d = pd.DataFrame({'promocion_activa': [promo], 'inventario_inicial': [50000], 'temperatura_promedio': [20], 'año': [2024], 'mes': [3], 'semana': [12], 'dia_semana': [0]})
        escenarios.append({'Escenario': 'Con Promo' if promo else 'Sin Promo', 'Ventas': modelo.predict(d)[0]})

    fig = px.bar(pd.DataFrame(escenarios), x='Escenario', y='Ventas', title='Impacto Promoción')
    st.plotly_chart(fig, use_container_width=True)

# --- PASO 6: Main ---
def main():
    st.markdown('<h1 class="main-header">RetailMax Predictor de Ventas</h1>', unsafe_allow_html=True)

    if parametros['predecir']:
        with st.spinner('Generando predicción...'):
            resultado = hacer_prediccion(parametros)
        mostrar_resultados_prediccion(resultado, parametros)
        crear_analisis_comparativo(parametros, resultado)
        crear_analisis_sensibilidad()

        # Descarga
        df_desc = pd.DataFrame([{
            'Tienda': parametros['tienda'],
            'Prediccion': resultado['prediccion'],
            'Fecha': datetime.now()
        }])
        st.download_button("Descargar CSV", df_desc.to_csv(), "prediccion.csv")

    else:
        st.markdown("## Dashboard General")
        col1, col2 = st.columns(2)
        col1.metric("Promedio General", f"${df_historico['ventas_semanales'].mean():,.0f}")
        col2.metric("Total Ventas", f"${df_historico['ventas_semanales'].sum():,.0f}")

if __name__ == "__main__":
    main()
