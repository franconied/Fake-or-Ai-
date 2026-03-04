{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import requests\
\
# --- 1. CONFIGURACI\'d3N DE LA P\'c1GINA ---\
st.set_page_config(\
    page_title="VERITAS AI Detector",\
    page_icon="
\f1 \uc0\u55357 \u57057 \u65039 
\f0 ",\
    layout="centered"\
)\
\
# --- 2. EL MOTOR (CONEXI\'d3N CON LA IA) ---\
# REEMPLAZA EL TEXTO ABAJO POR TU TOKEN REAL\
API_TOKEN = "TU_API_TOKEN_ACA"\
API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"\
headers = \{"Authorization": f"Bearer \{API_TOKEN\}"\}\
\
def consultar_ia(datos_archivo):\
    response = requests.post(API_URL, headers=headers, data=datos_archivo)\
    return response.json()\
\
# --- 3. DISE\'d1O DE LA WEB (LANDING) ---\
st.markdown("<h1 style='text-align: center;'>
\f1 \uc0\u55357 \u57057 \u65039 
\f0  VERITAS AI</h1>", unsafe_allow_html=True)\
st.markdown("<p style='text-align: center;'>Herramienta de verificaci\'f3n de medios por Mr. F</p>", unsafe_allow_html=True)\
st.divider()\
\
# Selector de archivos (L\'edmite 2MB)\
archivo_subido = st.file_uploader("Sub\'ed una imagen (JPG, PNG) - M\'e1x 2MB", type=['jpg', 'jpeg', 'png'])\
\
if archivo_subido is not None:\
    # Validamos el peso de 2MB\
    if archivo_subido.size > 2 * 1024 * 1024:\
        st.error("
\f1 \uc0\u9888 \u65039 
\f0  El archivo es muy pesado. Sub\'ed algo de menos de 2MB.")\
    else:\
        # Mostramos la imagen\
        st.image(archivo_subido, caption="Imagen para analizar", use_container_width=True)\
        \
        if st.button("ANALIZAR AHORA"):\
            with st.spinner("Escaneando p\'edxeles en la nube..."):\
                try:\
                    # Enviamos la imagen a la IA\
                    resultado = consultar_ia(archivo_subido.getvalue())\
                    \
                    # Interpretamos la respuesta\
                    if isinstance(resultado, list) and len(resultado) > 0:\
                        # Buscamos el puntaje de 'artificial' o 'label_1'\
                        score = 0\
                        for item in resultado:\
                            if item['label'].lower() in ['artificial', 'fake', 'label_1']:\
                                score = item['score']\
                        \
                        porcentaje = round(score * 100, 2)\
                        \
                        # Mostramos el resultado final\
                        st.subheader(f"Probabilidad de IA: \{porcentaje\}%")\
                        \
                        if porcentaje > 60:\
                            st.error("
\f1 \uc0\u55357 \u57000 
\f0  RESULTADO: FAKE / GENERADO POR IA")\
                            st.info("Se detectaron patrones no naturales en la imagen.")\
                        else:\
                            st.success("
\f1 \uc0\u9989 
\f0  RESULTADO: PARECE REAL / ORIGINAL")\
                            st.info("No se hallaron rastros evidentes de generaci\'f3n sint\'e9tica.")\
                    else:\
                        st.warning("La IA est\'e1 despertando. Por favor, intent\'e1 de nuevo en 10 segundos.")\
                \
                except Exception as e:\
                    st.error(f"Hubo un error en la conexi\'f3n: \{e\}")\
\
st.divider()\
st.caption("Uso exclusivo para chequeo de material de producci\'f3n.")}