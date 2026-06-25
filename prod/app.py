import streamlit as st
from PIL import Image
import os
import pandas as pd
from utils import cargar_modelo, preprocesar_imagen, predecir, DETALLES_CONTENEDORES

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="GIRSU Mendoza - Inteligencia Artificial", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carga segura del modelo (Cacheada vía utils)
PATH_PESOS = os.path.join(os.path.dirname(_file_), "..", "dev", "modelo_final_ganador.pth")

# 2. PANEL LATERAL (SIDEBAR) - Mejora UX/UI
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Recycle_Symbol_%28Solid%29.svg/120px-Recycle_Symbol_%28Solid%29.svg.png", width=60)
    st.title("♻️ Carga de Residuos")
    st.write("Clasificación automatizada con EfficientNet-B0.")
    
    # Contexto geográfico
    municipio = st.selectbox("📍 Seleccioná tu Municipio", ["Godoy Cruz", "Capital", "Guaymallén", "Luján de Cuyo", "Maipú", "Las Heras"])
    
    st.divider()
    
    # El uploader se mueve al lateral para limpiar la vista principal
    uploaded_file = st.file_uploader("Subí una foto nítida del residuo", type=["jpg", "jpeg", "png"])
    
    st.divider()
    st.caption("⚖️ Sistema alineado a la Ley Provincial 5.961 y Ley GIRSU 9.659 de Mendoza.")

# 3. CUERPO PRINCIPAL
st.title("Sistema Inteligente de Gestión de Residuos Urbanos")
st.markdown("Herramienta educativa y de asistencia ciudadana para la separación en origen.")

if not os.path.exists(PATH_PESOS):
    st.error(f"⚠️ No se encontraron los pesos del modelo en la ruta: {PATH_PESOS}")
    model = None
else:
    model = cargar_modelo(PATH_PESOS)

if model:
    # PESTAÑAS INFORMATIVAS
    tab_principal, tab_guia_girsu = st.tabs(["🔍 Análisis en Tiempo Real", "📖 Guía Ley GIRSU Mendoza"])

    with tab_guia_girsu:
        st.subheader("Guía Rápida de Separación en Origen")
        cols_guia = st.columns(4)
        col_icons = ["🟢", "⚫", "🟡", "🟤"]
        for i, (key, info) in enumerate(DETALLES_CONTENEDORES.items()):
            with cols_guia[i]:
                st.markdown(f"### {col_icons[i]} Contenedor {key}")
                st.caption(info["descripcion"])
                st.markdown(f"*Impacto Legal:* {info['ley']}")
                st.markdown(f"*Tipo de Bolsa:* {info['bolsa']}")

    with tab_principal:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # GRID DE COLUMNAS ASIMÉTRICAS (Evita que la imagen rompa la pantalla)
            col_izq, col_der = st.columns([1, 1.4])

            with col_izq:
                st.markdown("### 📸 Muestra Analizada")
                # Imagen chica y controlada
                st.image(image, use_column_width=True)
                
                st.markdown("### 📊 Confianza de la Red")
                with st.spinner("Procesando descriptores con EfficientNet..."):
                    tensor_img = preprocesar_imagen(image)
                    clase_predicha, probabilidades = predecir(model, tensor_img)

                # Desglose de confianza en barras de progreso
                for clase, prob in probabilidades.items():
                    st.write(f"*{clase.split(' ')[0]}*: {prob*100:.1f}%")
                    st.progress(prob)

            with col_der:
                st.markdown("### 🎯 Veredicto y Asistencia Educativa")
                
                # Asignación de color clave
                if "NEGRO" in clase_predicha:
                    st.warning(f"Clasificación Predominante: *{clase_predicha}*")
                    color_clave = "NEGRO"
                elif "VERDE" in clase_predicha:
                    st.success(f"Clasificación Predominante: *{clase_predicha}*")
                    color_clave = "VERDE"
                elif "AMARILLO" in clase_predicha:
                    st.error(f"Clasificación Predominante: *{clase_predicha}*")
                    color_clave = "AMARILLO"
                else:
                    st.info(f"Clasificación Predominante: *{clase_predicha}*")
                    color_clave = "MARRÓN"

                # Extracción dinámica de metadatos pedagógicos del contenedor ganador
                meta = DETALLES_CONTENEDORES[color_clave]
                
                st.write(f"*📝 Descripción:* {meta['descripcion']}")
                st.info(f"*⚖️ Respaldo Ley GIRSU:* {meta['ley']}")
                st.write(f"*🛍️ Protocolo de Embolsado:* {meta['bolsa']}")
                st.write(f"*💡 Ejemplos incluidos:* {meta['ejemplos']}")
                st.warning(f"*👉 Acción en {municipio}:* Consultá el cronograma oficial de tu zona para la disposición de este residuo.")

                st.write("---")
                
                # MAPA BAJO DEMANDA
                st.markdown("#### 🗺️ Puntos de Recepción Sugeridos")
                if st.button(f"Ver Geolocalización para {color_clave}", use_container_width=True):
                    for punto in meta["puntos"]:
                        st.write(f"📍 {punto}")
                    
                    # Generación del mapa interactivo
                    map_data = pd.DataFrame(meta["coords"], columns=['lat', 'lon'])
                    st.map(map_data, zoom=11)
        else:
            # Estado de espera (cuando no hay imagen cargada)
            st.info("👈 Cargá una imagen desde el panel lateral izquierdo para que el modelo inicie la clasificación.")
            st.write("Al analizar la imagen, el sistema no solo identificará el tipo de residuo, sino que te brindará las directrices legales exactas para descartarlo correctamente en tu municipio, fomentando la Economía Circular.")