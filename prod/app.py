import streamlit as st
from PIL import Image
import os
import pandas as pd
import pydeck as pdk  # Motor de mapas interactivos con tooltips avanzados
import sys

# SINCRO DE RUTAS: Forzamos a la nube a mirar adentro de /prod para encontrar sus módulos
sys.path.append(os.path.dirname(__file__))

from utils import cargar_modelo, preprocesar_imagen, predecir, DETALLES_CONTENEDORES, PUNTOS_GEOLOCALIZADOS

# 1. CONFIGURACIÓN DE PÁGINA EN MODO ANCHO
st.set_page_config(
    page_title="GIRSU Mendoza - Inteligencia Artificial", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carga segura de la red neuronal
PATH_PESOS = os.path.join(os.path.dirname(__file__), "..", "dev", "modelo_final_ganador.pth")

# ==============================================================================
# 2. PANEL LATERAL (SIDEBAR) - CENTRALIZACIÓN DE ENTRADAS
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Recycle_Symbol_%28Solid%29.svg/120px-Recycle_Symbol_%28Solid%29.svg.png", width=50)
    st.title("⚙️ Configuración")
    st.write("Carga de muestras y contexto geográfico para la asistencia ciudadana.")
    
    st.divider()
    
    # Contexto geográfico dinámico
    municipio = st.selectbox(
        "📍 Seleccioná tu Municipio:", 
        ["Capital", "Godoy Cruz", "Guaymallén", "Luján de Cuyo", "Maipú", "Las Heras"]
    )
    
    st.divider()
    
    # Cargador de archivos en el lateral
    uploaded_file = st.file_uploader("📸 Subí una foto del residuo:", type=["jpg", "jpeg", "png"])
    
    st.divider()
    st.caption("⚖️ Sistema alineado a la Ley Provincial 5.961 y Ley GIRSU 9.659 de Mendoza.")

# ==============================================================================
# 3. CUERPO PRINCIPAL - TABLERO DE CONTROL (DASHBOARD)
# ==============================================================================
st.title("♻️ Sistema Inteligente de Gestión de Residuos Urbanos")
st.markdown("Plataforma de asistencia ciudadana impulsada por Aprendizaje Profundo (*Deep Learning*).")

if not os.path.exists(PATH_PESOS):
    st.error(f"⚠️ No se encontraron los pesos del modelo en la ruta: {PATH_PESOS}")
    model = None
else:
    model = cargar_modelo(PATH_PESOS)

if model:
    # Organización por pestañas para segmentar la densidad de información
    tab_principal, tab_guia_girsu = st.tabs(["🔍 Análisis en Tiempo Real", "📖 Guía de Contenedores y Leyes"])

    # --- PESTAÑA 2: GUÍA DIGITAL (Educativa) ---
    with tab_guia_girsu:
        st.subheader("Manual Digital de Clasificación en Origen")
        cols_guia = st.columns(4)
        col_icons = ["🟢", "⚫", "🟡", "🟤"]
        for i, (key, info) in enumerate(DETALLES_CONTENEDORES.items()):
            with cols_guia[i]:
                st.markdown(f"### {col_icons[i]} Contenedor {key}")
                st.caption(info["descripcion"])
                st.markdown(f"**袋️ Tipo de Bolsa:**\n*{info['bolsa']}*")
                st.markdown(f"**💡 Ejemplos:**\n{info['ejemplos']}")

    # --- PESTAÑA 1: PARTE DE INFERENCIA OPERATIVA ---
    with tab_principal:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # GRID DE COLUMNAS ASIMÉTRICAS
            col_izq, col_der = st.columns([1, 1.6])

            with col_izq:
                st.markdown("#### 📸 Muestra Recibida")
                # CORRECCIÓN UX: Imagen optimizada a un tamaño bien pequeño (150px)
                st.image(image, caption="Muestra de residuo", width=150)
                
                st.markdown("#### 📊 Confianza del Modelo")
                with st.spinner("Procesando descriptores con EfficientNet-B0..."):
                    tensor_img = preprocesar_imagen(image)
                    clase_predicha, probabilidades = predecir(model, tensor_img)

                # Desglose compacto de las probabilidades Softmax
                for clase, prob in probabilidades.items():
                    nombre_corto = clase.split(" ")[0]
                    st.write(f"**{nombre_corto}**: {prob*100:.1f}%")
                    st.progress(prob)

            with col_der:
                st.markdown("#### 🎯 Veredicto de Clasificación")
                
                # Asignación semántica de componentes visuales por color de residuo detectado
                if "NEGRO" in clase_predicha:
                    st.warning(f"**Contenedor Recomendado:** {clase_predicha}")
                    color_clave = "NEGRO"
                elif "VERDE" in clase_predicha:
                    st.success(f"**Contenedor Recomendado:** {clase_predicha}")
                    color_clave = "VERDE"
                elif "AMARILLO" in clase_predicha:
                    st.error(f"**Contenedor Recomendado:** {clase_predicha}")
                    color_clave = "AMARILLO"
                else:
                    st.info(f"**Contenedor Recomendado:** {clase_predicha}")
                    color_clave = "MARRÓN"

                meta = DETALLES_CONTENEDORES[color_clave]

                # DIVULGACIÓN PROGRESIVA: Información organizada en colapsables
                with st.expander("📋 Detalles de Disposición e Impacto Ambiental", expanded=True):
                    st.markdown(f"**📝 Descripción:** {meta['descripcion']}")
                    st.markdown(f"**🛍️ Protocolo de Embolsado:** {meta['bolsa']}")
                    st.markdown(f"**⚖️ Marco Legal:** {meta['ley']}")
                    st.markdown(f"**💡 Ítems Comunes:** {meta['ejemplos']}")

                with st.expander(f"🗺️ Guía de Puntos Verdes y Gestión de Residuos en {municipio}", expanded=True):
                    st.write(f"Mapa general de la red ecológica de **{municipio}**. Pasar el mouse por encima de los puntos para ver detalles:")
                    
                    # CORRECCIÓN ARQUITECTÓNICA: Recopilar TODOS los puntos del departamento sin filtrar por la IA
                    puntos_municipio = PUNTOS_GEOLOCALIZADOS.get(municipio, {})
                    lista_total_puntos = []
                    
                    for tacho_cat, puntos in puntos_municipio.items():
                        for p in puntos:
                            # Asignamos el color RGBA correspondiente a la naturaleza real de cada punto
                            if tacho_cat == "VERDE":
                                p_color = [40, 167, 69, 230]
                            elif tacho_cat == "AMARILLO":
                                p_color = [230, 126, 34, 230]
                            elif tacho_cat == "MARRÓN":
                                p_color = [139, 69, 19, 230]
                            else:
                                p_color = [50, 50, 50, 230]
                            
                            lista_total_puntos.append({
                                "name": p["name"],
                                "lat": p["lat"],
                                "lon": p["lon"],
                                "tipo": p["tipo"],
                                "categoria": tacho_cat,
                                "color_rgb": p_color
                            })
                    
                    if lista_total_puntos:
                        df_mapa = pd.DataFrame(lista_total_puntos)
                        
                        # Capa PyDeck con asignación de color dinámica mapeada desde la columna 'color_rgb'
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            df_mapa,
                            get_position="[lon, lat]",
                            get_color="color_rgb",         # Lee la lista RGBA de cada fila individualmente
                            get_radius=150,
                            radius_min_pixels=10,
                            radius_max_pixels=20,
                            pickable=True,
                        )
                        
                        view_state = pdk.ViewState(
                            latitude=df_mapa['lat'].mean(),
                            longitude=df_mapa['lon'].mean(),
                            zoom=12
                        )
                        
                        # Renderizado del mapa con Tooltip avanzado (Nombre, Categoría y Tipo de tratamiento)
                        st.pydeck_chart(pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip={"text": "🏢 Centro: {name}\n♻️ Contenedor: {categoria}\n🔧 Recibe: {tipo}"},
                            map_style=None
                        ))
                    else:
                        st.info(f"ℹ️ El municipio de {municipio} procesa sus residuos mediante rutas de recolección móvil programada. No cuenta con estaciones fijas de transferencia en nuestra base de datos.")
                    
                    st.divider()
                    st.caption(f"👉 **Acción Ciudadana:** Respetá los cronogramas de recolección de {municipio} para colaborar con el medio ambiente.")
        else:
            # Estado inicial limpio
            st.info("👈 Seleccioná tu municipio y cargá una fotografía desde el panel lateral izquierdo para iniciar el análisis inteligente.")
            
            # Tarjetas de presentación métricas del sistema
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label="Objetivo Provincial", value="GIRSU 2026")
                st.caption("Fomento de la separación en origen y erradicación de basurales a cielo abierto.")
            with c2:
                st.metric(label="Arquitectura Core", value="EfficientNet-B0")
                st.caption("Modelo de Red Neuronal Convolucional con Compound Scaling optimizado para inferencia rápida.")
            with c3:
                st.metric(label="Entorno de Inferencia", value="CPU Sincrónico")
                st.caption("Carga diferida mediante memoria cacheada para mitigar la latencia web.")