import streamlit as st
from PIL import Image
import os
from utils import cargar_modelo, preprocesar_imagen, predecir

#Configuración de página
st.set_page_config(page_title="Clasificador GIRSU Mendoza", layout="centered")

st.title("♻️ Clasificador de Residuos Urbanos")
st.write("Proyecto automatizado bajo los lineamientos de la Ley Provincial 5.961 (GIRSU Mendoza).")

#Ruta hacia los pesos ganadores (que están en dev/)
PATH_PESOS = os.path.join(os.path.dirname(file), "..", "dev", "modelo_final_ganador.pth")

#Verificar si los pesos existen antes de romper la app
if not os.path.exists(PATH_PESOS):
    st.error(f"⚠️ No se encontraron los pesos del modelo en la ruta esperada: {PATH_PESOS}")
else:
    # Cargamos el modelo utilizando la función cacheada de utils.py
    model = cargar_modelo(PATH_PESOS)

    # El usuario sube la imagen
    uploaded_file = st.file_uploader("Subí una foto nítida del residuo urbano", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

#Mostramos la imagen ingresada
        st.image(image, caption="Imagen ingresada por el usuario", use_container_width=True)

        st.write("---")
        st.subheader("📊 Resultado del Análisis")

        with st.spinner("Procesando imagen con la red EfficientNet-B0..."):
            # Preprocesar e inferir usando utils.py
            tensor_img = preprocesar_imagen(image)
            clase_predicha, probabilidades = predecir(model, tensor_img)

#Mostrar clase ganadora destacada
        st.success(f"Contenedor Destino Recomendado: {clase_predicha}")

#Mostrar las probabilidades en barras
        st.write("Probabilidades por clase:")
        for clase, prob in probabilities.items():
            st.write(f"{clase}: {prob*100:.2f}%")
            st.progress(prob)