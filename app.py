import streamlit as st
import requests
from PIL import Image
import io
import cv2
import tempfile
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="REAL OR FAKE", page_icon="🛡️", layout="wide")

# REEMPLAZA CON TU TOKEN REAL
API_TOKEN = st.secrets["HF_TOKEN"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# URLs de los dos "Cerebros"
MODEL_GENERAL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
MODEL_ROSTROS = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-Model"

def consultar_modelo(url, datos_binarios):
    try:
        resp = requests.post(url, headers=headers, data=datos_binarios)
        return resp.json()
    except:
        return None

def extraer_score(resultado):
    """Extrae el porcentaje de IA de la respuesta del modelo"""
    if isinstance(resultado, list) and len(resultado) > 0:
        for item in resultado:
            if item['label'].lower() in ['artificial', 'fake', 'label_1']:
                return item['score']
    return 0

# --- INTERFAZ ---
st.title("🛡️ REAL OR FAKE")
st.write("Double checked info.")

archivo = st.file_uploader("Subí una foto o video (2MB máx)", type=['jpg', 'jpeg', 'png', 'mp4'])

if archivo:
    img_final = None
    # Lógica de Video / Foto (Igual que antes)
    if archivo.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(archivo.read())
        vf = cv2.VideoCapture(tfile.name)
        ret, frame = vf.read()
        if ret:
            img_final = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img_final = Image.open(archivo)

    if img_final:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_final, caption="Archivo cargado", use_container_width=True)
        
        with col2:
            if st.button("EJECUTAR ANÁLISIS CRUZADO"):
                with st.spinner("Consultando múltiples redes neuronales..."):
                    # Preparar imagen para la API
                    buf = io.BytesIO()
                    img_final.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()

                    # Consulta en paralelo
                    res_gen = consultar_modelo(MODEL_GENERAL, img_bytes)
                    res_face = consultar_modelo
