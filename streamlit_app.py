import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sistema de Ventas IA", layout="centered")

# Estilos personalizados
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #1E3A8A; }
        .stButton>button {
            color: white;
            background: linear-gradient(90deg, #2563EB, #1D4ED8);
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Modelo
modelo = joblib.load('modelo_prediccion.pkl')
archivo_historial = "historial_predicciones.csv"

st.title("📈 Predicción Inteligente de Ventas")

menu = st.sidebar.radio("🧭 Navegación", ["📊 Predecir ventas", "📝 Registrar ventas reales", "📚 Historial y métricas"])

if menu == "📊 Predecir ventas":
    st.header("📋 Ingreso de Datos")

    precio = st.number_input("💲 Precio del producto", value=3.5)
    promocion = st.selectbox("🎯 ¿Hay promoción?", ["No", "Sí"])
    dia_semana = st.selectbox("📅 Día de la semana", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
    producto = st.selectbox("🛍️ Producto", ["Leche", "Pan", "Refresco"])
    clima = st.selectbox("🌦️ Clima", ["Soleado", "Nublado", "Lluvioso"])
    evento_especial = st.selectbox("🎉 Evento especial", ["No", "Sí"])

    # Codificación
    promocion_val = 1 if promocion == "Sí" else 0
    dia_semana_val = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"].index(dia_semana)
    evento_val = 1 if evento_especial == "Sí" else 0
    producto_pan = 1 if producto == "Pan" else 0
    producto_refresco = 1 if producto == "Refresco" else 0
    clima_nublado = 1 if clima == "Nublado" else 0
    clima_lluvioso = 1 if clima == "Lluvioso" else 0

    if st.button("🔮 Predecir ventas"):
        entrada = np.array([[precio, promocion_val, dia_semana_val, evento_val,
                             producto_pan, producto_refresco, clima_nublado, clima_lluvioso]])
        prediccion = modelo.predict(entrada)[0]
        st.success(f"✅ Se estiman **{int(prediccion)} unidades** vendidas.")

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

        historial = pd.read_csv(archivo_historial) if os.path.exists(archivo_historial) else pd.DataFrame()
        historial = pd.concat([historial, nueva_fila], ignore_index=True)
        historial.to_csv(archivo_historial, index=False)

elif menu == "📝 Registrar ventas reales":
    st.header("✍️ Registro de Ventas Reales")
    if os.path.exists(archivo_historial):
        historial = pd.read_csv(archivo_historial)
        pendientes = historial[historial["ventas_reales"].isna()]
        if not pendientes.empty:
            seleccion = st.selectbox("📌 Selecciona un registro pendiente:", pendientes.index.astype(str))
            seleccion = int(seleccion)
            fila = pendientes.loc[seleccion]
            st.info(f"""**Producto:** {fila['producto']} | **Día:** {fila['dia_semana']}  
            **Clima:** {fila['clima']} | **Precio:** {fila['precio']} | **Promo:** {fila['promocion']} | **Evento:** {fila['evento_especial']}""")

            real = st.number_input("✏️ Ventas reales", min_value=0, step=1)
            if st.button("💾 Guardar"):
                historial.at[seleccion, "ventas_reales"] = real
                historial.to_csv(archivo_historial, index=False)
                st.success("✅ Venta real registrada correctamente.")
        else:
            st.info("🎉 No hay ventas pendientes por registrar.")
    else:
        st.warning("⚠️ No se han realizado predicciones aún.")

elif menu == "📚 Historial y métricas":
    st.header("📈 Historial y Métricas")
    if os.path.exists(archivo_historial):
        historial = pd.read_csv(archivo_historial)
        st.dataframe(historial, use_container_width=True)

        completados = historial.dropna()
        if not completados.empty:
            mae = mean_absolute_error(completados["ventas_reales"], completados["prediccion"])
            rmse = np.sqrt(mean_squared_error(completados["ventas_reales"], completados["prediccion"]))

            st.metric("📏 MAE", f"{mae:.2f}")
            st.metric("📉 RMSE", f"{rmse:.2f}")
        else:
            st.info("🔎 Se necesitan datos reales para calcular métricas.")
    else:
        st.warning("📂 No hay datos aún en el historial.")
