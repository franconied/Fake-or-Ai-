import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Detector de Verdad - Mr. F")
st.title("¿Es Real o IA? 🤖")

# Límite de 2MB
MAX_FILE_SIZE = 2 * 1024 * 1024 

uploaded_file = st.file_uploader("Subí tu foto o video (máx 2MB)", type=['jpg', 'png', 'mp4'])

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("¡El archivo es muy pesado! Bajalo a menos de 2MB.")
    else:
        st.image(uploaded_file, caption='Archivo subido', use_column_width=True)
        
        with st.spinner('Analizando píxeles...'):
            # Aquí conectaríamos con la API de Hugging Face
            # (Necesitarás un Token gratis de huggingface.co)
            st.warning("RESULTADO: Altas probabilidades de ser FAKE / AI")
            st.progress(85) # Ejemplo de confianza
