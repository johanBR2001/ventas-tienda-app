import streamlit as st
import pandas as pd
import pickle

# Cargar modelo entrenado
with open("modelo_prediccion.pkl", "rb") as file:
    modelo = pickle.load(file)

# Cargar datos de prueba
df = pd.read_csv("ventas_tienda.csv")

# Título de la app
st.title("🛍️ Predicción de Ventas para Tiendas")

# Mostrar primeros datos
st.subheader("Datos de entrada")
st.write(df.head())

# Predicción
predicciones = modelo.predict(df[['precio_unitario', 'unidades_vendidas']])

# Mostrar resultados
df['ventas_estimadas'] = predicciones
st.subheader("Predicciones del Modelo")
st.write(df[['precio_unitario', 'unidades_vendidas', 'ventas_estimadas']])

# Gráfico simple
st.line_chart(df['ventas_estimadas'])
