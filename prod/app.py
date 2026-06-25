import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
from utils import cargar_modelo, preprocesar_imagen, predecir

# 1. CONFIGURACIÓN DE PÁGINA EN MODO ANCHO (Evita el scroll infinito)
st.set_page_config(page_title="GIRSU Mendoza - Inteligencia Artificial", layout="wide")

st.title("♻️ Sistema Inteligente de Gestión de Residuos Urbanos")
st.write("Clasificación automatizada con Visión por Computadora bajo los lineamientos de la Ley Provincial 5.961 (GIRSU Mendoza).")

# Diccionario de Metadatos Educativos para la defensa ante el profesor
DETALLES_CONTENEDORES = {
    "VERDE": {
        "descripcion": "Materiales secos que pueden reingresar al circuito productivo gracias a los recuperadores urbanos.",
        "bolsa": "Bolsa transparente o de color verde. Siempre limpios y secos.",
        "ejemplos": "Cartón, botellas plásticas, frascos de vidrio, latas de aluminio, papel de diario.",
        "puntos": ["Punto Limpio Capital (Plaza Independencia)", "Punto Limpio Godoy Cruz (Parque Benegas)"],
        "coords": [[-32.8894, -68.8451], [-32.9342, -68.8491]]
    },
    "NEGRO": {
        "descripcion": "Residuos orgánicos o húmedos que no pueden ser recuperados y tienen como destino el relleno sanitario.",
        "bolsa": "Bolsa negra o gris común de residuos.",
        "ejemplos": "Restos de comida, cáscaras de fruta, yerba, pañales, servilletas usadas, cerámicos rotos.",
        "puntos": ["Recolección domiciliaria nocturna según cronograma municipal"],
        "coords": [[-32.8900, -68.8300]]  # Centro general
    },
    "AMARILLO": {
        "descripcion": "Residuos de Aparatos Eléctricos y Electrónicos (RAEE) y componentes con metales pesados altamente contaminantes.",
        "bolsa": "Disposición en caja o bolsa cerrada especial, llevar directamente al centro de recepción.",
        "ejemplos": "Pilas, baterías de celular, cargadores rotos, mouses, teclados, tubos fluorescentes.",
        "puntos": ["Punto Limpio RAEE Guaymallén", "Punto Especial Maipú Municipio"],
        "coords": [[-32.8952, -68.8101], [-32.9721, -68.7905]]
    },
    "MARRÓN": {
        "descripcion": "Fracción textil en desuso que requiere un procesamiento de acopio o donación para evitar la saturación de los vertederos.",
        "bolsa": "Bolsa cerrada para proteger los materiales de la humedad.",
        "ejemplos": "Ropa vieja, calzado gastado, sábanas, retazos de tela, alfombras.",
        "puntos": ["Centros de acopio textil municipales", "Contenedores especiales de cooperativas aliadas"],
        "coords": [[-32.9100, -68.8400]]
    }
}

# Carga segura del modelo
PATH_PESOS = os.path.join(os.path.dirname(__file__), "..", "dev", "modelo_final_ganador.pth")

if not os.path.exists(PATH_PESOS):
    st.error(f"⚠️ No se encontraron los pesos del modelo en la ruta esperada: {PATH_PESOS}")
else:
    model = cargar_modelo(PATH_PESOS)

    # 2. PESTAÑAS INFORMATIVAS INICIALES (Aporte UX exigido por el profesor)
    tab_principal, tab_guia_girsu = st.tabs(["🔍 Clasificador en Tiempo Real", "📖 Guía de Contenedores Mendoza"])

    with tab_guia_girsu:
        st.subheader("Guía Rápida de Separación en Origen")
        cols_guia = st.columns(4)
        col_icons = ["🟢", "⚫", "🟡", "🟤"]
        for i, (key, info) in enumerate(DETALLES_CONTENEDORES.items()):
            with cols_guia[i]:
                st.markdown(f"### {col_icons[i]} Contenedor {key}")
                st.caption(info["descripcion"])
                st.markdown(f"**袋️ Tipo de Bolsa:** {info['bolsa']}")

    with tab_principal:
        # Zona de carga
        uploaded_file = st.file_uploader("Subí una foto nítida del residuo urbano o usá la cámara", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.write("---")

            # 3. TRABAJO EN GRID DE COLUMNAS (Evita que la imagen rompa la pantalla)
            col_izq, col_der = st.columns([1, 1.2])

            with col_izq:
                st.markdown("### 📸 Vista previa")
                # Achicamos la imagen dándole un ancho fijo controlado para que no ocupe toda la pantalla
                st.image(image, caption="Muestra ingresada", width=320)
                
                st.markdown("### 📊 Vector de Probabilidades")
                with st.spinner("Procesando imagen con EfficientNet-B0..."):
                    tensor_img = preprocesar_imagen(image)
                    clase_predicha, probabilidades = predecir(model, tensor_img)

                # Desglose de confianza en barras de progreso
                for clase, prob in probabilidades.items():
                    st.write(f"{clase}: {prob*100:.2f}%")
                    st.progress(prob)

            with col_der:
                st.markdown("### 🎯 Veredicto del Sistema")
                
                # Alerta condicional según tipo de residuo (Verde éxito / Negro peligro o rechazo)
                if "NEGRO" in clase_predicha:
                    st.warning(f"**Resultado:** {clase_predicha}")
                    color_clave = "NEGRO"
                elif "VERDE" in clase_predicha:
                    st.success(f"**Resultado:** {clase_predicha}")
                    color_clave = "VERDE"
                elif "AMARILLO" in clase_predicha:
                    st.info(f"**Resultado:** {clase_predicha}")
                    color_clave = "AMARILLO"
                else:
                    st.error(f"**Resultado:** {clase_predicha}")
                    color_clave = "MARRÓN"

                # Extracción dinámica de metadatos pedagógicos del contenedor ganador
                meta = DETALLES_CONTENEDORES[color_clave]
                
                st.markdown(f"**📝 Descripción de la categoría:** {meta['descripcion']}")
                st.markdown(f"**🛍️ Protocolo de Embolsado:** {meta['bolsa']}")
                st.markdown(f"**💡 Ejemplos incluidos:** *{meta['ejemplos']}*")

                st.write("---")
                
                # 4. BOTÓN INTERACTIVO BAJO DEMANDA PARA DESPLEGAR MAPA (Exigencia del pliego)
                st.markdown("#### 🗺️ Geolocalización y Disposición")
                if st.button(f"Ver Centros de Recepción para Contenedor {color_clave}"):
                    st.write("**Ubicaciones sugeridas en el Gran Mendoza donde llevar este residuo:**")
                    for punto in meta["puntos"]:
                        st.write(f"📍 {punto}")
                    
                    # Creación de DataFrame de coordenadas reales aproximadas de Mendoza para el mapa de Streamlit
                    map_data = pd.DataFrame(
                        meta["coords"],
                        columns=['lat', 'lon']
                    )
                    st.map(map_data, zoom=11)