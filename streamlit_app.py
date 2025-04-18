import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cargar modelo entrenado
modelo = joblib.load('modelo_prediccion.pkl')

st.title("Predicción y Registro de Ventas para Tienda")

menu = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["Predicción de ventas", "Registrar ventas reales", "Ver historial y métricas"]
)

# Archivo de historial
archivo_historial = "historial_predicciones.csv"

if menu == "Predicción de ventas":
    st.subheader("Ingresar datos para predicción")

    precio = st.number_input("Precio del producto", value=3.5)
    promocion = st.selectbox("¿Hay promoción?", ["No", "Sí"])
    dia_semana = st.selectbox("Día de la semana", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
    producto = st.selectbox("Producto", ["Leche", "Pan", "Refresco"])
    clima = st.selectbox("Clima del día", ["Soleado", "Nublado", "Lluvioso"])
    evento_especial = st.selectbox("¿Hay evento especial?", ["No", "Sí"])

    # Codificación
    promocion_val = 1 if promocion == "Sí" else 0
    dia_semana_val = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"].index(dia_semana)
    evento_val = 1 if evento_especial == "Sí" else 0

    producto_pan = 1 if producto == "Pan" else 0
    producto_refresco = 1 if producto == "Refresco" else 0

    clima_nublado = 1 if clima == "Nublado" else 0
    clima_lluvioso = 1 if clima == "Lluvioso" else 0

    if st.button("Predecir ventas"):
        entrada = np.array([[precio, promocion_val, dia_semana_val, evento_val,
                             producto_pan, producto_refresco, clima_nublado, clima_lluvioso]])
        prediccion = modelo.predict(entrada)[0]
        st.success(f"Predicción de ventas: {int(prediccion)} unidades")

        nueva_fila = pd.DataFrame([{
            "precio": precio,
            "promocion": promocion,
            "dia_semana": dia_semana,
            "producto": producto,
            "clima": clima,
            "evento_especial": evento_especial,
            "prediccion": int(prediccion),
            "ventas_reales": np.nan
        }])

        if os.path.exists(archivo_historial):
            historial = pd.read_csv(archivo_historial)
            historial = pd.concat([historial, nueva_fila], ignore_index=True)
        else:
            historial = nueva_fila

        historial.to_csv(archivo_historial, index=False)

elif menu == "Registrar ventas reales":
    st.subheader("Completa las ventas reales del día")
    if os.path.exists(archivo_historial):
        historial = pd.read_csv(archivo_historial)
        pendientes = historial[historial["ventas_reales"].isna()]
        if not pendientes.empty:
            seleccion = st.selectbox("Selecciona un registro pendiente:", pendientes.index.astype(str))
            seleccion = int(seleccion)
            fila = pendientes.loc[seleccion]
            st.write(f"""
                Producto: {fila['producto']} | Día: {fila['dia_semana']} | Clima: {fila['clima']}  
                Precio: {fila['precio']} | Promoción: {fila['promocion']} | Evento: {fila['evento_especial']}
            """)
            real = st.number_input("Ventas reales", min_value=0, step=1)
            if st.button("Guardar venta real"):
                historial.at[seleccion, "ventas_reales"] = real
                historial.to_csv(archivo_historial, index=False)
                st.success("Ventas reales guardadas")
        else:
            st.info("No hay registros pendientes de ventas reales.")
    else:
        st.warning("Aún no se han realizado predicciones.")

elif menu == "Ver historial y métricas":
    st.subheader("Historial completo")
    if os.path.exists(archivo_historial):
        historial = pd.read_csv(archivo_historial)
        st.dataframe(historial)

        completados = historial.dropna()
        if not completados.empty:
            mae = mean_absolute_error(completados["ventas_reales"], completados["prediccion"])
            rmse = np.sqrt(mean_squared_error(completados["ventas_reales"], completados["prediccion"]))

            st.write(f"**MAE (Error Absoluto Medio)**: {mae:.2f}")
            st.write(f"**RMSE (Raíz del Error Cuadrático Medio)**: {rmse:.2f}")
        else:
            st.info("Aún no hay suficientes datos para evaluar el modelo.")
    else:
        st.warning("No hay datos disponibles.")
