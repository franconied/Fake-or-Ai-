import streamlit as st
import requests
from PIL import Image
import io
import cv2
import tempfile
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(
    page_title="REAL OR FAKE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .real-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }
    .fake-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# REEMPLAZA CON TU TOKEN REAL
API_TOKEN = st.secrets.get("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# URLs de los dos modelos
MODEL_GENERAL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
MODEL_ROSTROS = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-Model"

def consultar_modelo(url, datos_binarios):
    """Consulta un modelo en Hugging Face y retorna el resultado"""
    try:
        resp = requests.post(url, headers=headers, data=datos_binarios, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"Error {resp.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout: La consulta tardó demasiado"}
    except Exception as e:
        return {"error": str(e)}

def extraer_score(resultado, modelo_tipo="general"):
    """
    Extrae el porcentaje de IA/FAKE de la respuesta del modelo
    Retorna (score, label, confianza)
    """
    if isinstance(resultado, dict) and "error" in resultado:
        return None, "Error", resultado.get("error")
    
    if isinstance(resultado, list) and len(resultado) > 0:
        scores = {}
        for item in resultado:
            if isinstance(item, dict) and "label" in item:
                label = item['label'].lower()
                score = item.get('score', 0)
                scores[label] = score
        
        # Buscar etiquetas de AI/Fake
        if modelo_tipo == "general":
            fake_keywords = ['artificial', 'fake', 'label_1', 'ai']
        else:
            fake_keywords = ['fake', 'deepfake', 'manipulated', 'label_1']
        
        for key in fake_keywords:
            if key in scores:
                return scores[key], key, f"{scores[key]*100:.1f}%"
        
        # Si no encontramos fake, retornar el score más alto
        if scores:
            max_label = max(scores, key=scores.get)
            return scores[max_label], max_label, f"{scores[max_label]*100:.1f}%"
    
    return 0, "desconocido", "N/A"

def determinar_veredicto(score_general, score_rostros):
    """
    Combina los scores de ambos modelos para un veredicto final
    """
    score_promedio = (score_general + score_rostros) / 2
    
    if score_promedio >= 0.7:
        return "🚨 PROBABLEMENTE FALSO", "fake-box", score_promedio
    elif score_promedio >= 0.4:
        return "⚠️ DUDOSO", "result-box", score_promedio
    else:
        return "✅ PROBABLEMENTE REAL", "real-box", score_promedio

# --- INTERFAZ PRINCIPAL ---
st.title("🛡️ REAL OR FAKE")
st.write("**Detecta imágenes y videos deepfake usando IA avanzada**")
st.write("Cargá una foto o video y obtendrás un análisis cruzado de múltiples modelos de inteligencia artificial.")

# Validar token
if not API_TOKEN:
    st.error("❌ Token de Hugging Face no configurado. Agrega `HF_TOKEN` a tus secrets.")
    st.stop()

# --- CARGA DE ARCHIVO ---
col_upload, col_info = st.columns([3, 1])

with col_upload:
    archivo = st.file_uploader(
        "📤 Subí una foto o video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'],
        help="Máximo 50MB. Los videos serán analizados en el primer frame"
    )

with col_info:
    st.info("💡 **Modelo General** + **Detector de Rostros** = Análisis Cruzado")

if archivo:
    # Validar tamaño
    file_size_mb = len(archivo.getvalue()) / (1024 * 1024)
    
    if file_size_mb > 50:
        st.error(f"❌ Archivo demasiado grande ({file_size_mb:.1f}MB). Máximo: 50MB")
        st.stop()
    
    img_final = None
    
    # --- EXTRACCIÓN DE FRAME ---
    with st.spinner("Procesando archivo..."):
        try:
            if archivo.type == "video/mp4" or archivo.type == "video/quicktime" or archivo.type == "video/x-msvideo":
                # Video: extraer primer frame
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(archivo.read())
                tfile.close()
                
                vf = cv2.VideoCapture(tfile.name)
                ret, frame = vf.read()
                vf.release()
                
                if ret:
                    img_final = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    archivo_tipo = f"📹 VIDEO (frame: 0s)"
                else:
                    st.error("❌ No se pudo procesar el video")
                    st.stop()
            else:
                # Imagen
                img_final = Image.open(archivo)
                archivo_tipo = "🖼️ IMAGEN"
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {str(e)}")
            st.stop()
    
    if img_final:
        # --- VISTA PREVIA ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vista previa")
            st.image(img_final, use_container_width=True)
            st.caption(f"{archivo_tipo} | {archivo.name}")
        
        with col2:
            st.subheader("Opciones")
            st.write(f"📁 Archivo: **{archivo.name}**")
            st.write(f"📊 Tamaño: **{file_size_mb:.2f} MB**")
            st.write(f"🎯 Tipo: **{archivo_tipo}**")
            
            # Botón de análisis
            analizar = st.button(
                "🔍 EJECUTAR ANÁLISIS CRUZADO",
                type="primary",
                use_container_width=True
            )
            
            if analizar:
                # --- ANÁLISIS ---
                with st.spinner("Consultando múltiples redes neuronales... esto puede tomar 30-60 segundos"):
                    # Preparar imagen para API
                    buf = io.BytesIO()
                    img_final.save(buf, format="JPEG", quality=90)
                    img_bytes = buf.getvalue()
                    
                    # Crear placeholder para progreso
                    progress_placeholder = st.empty()
                    
                    # Consulta en paralelo
                    progress_placeholder.info("⏳ Modelo General (Detección de IA)...")
                    res_gen = consultar_modelo(MODEL_GENERAL, img_bytes)
                    
                    progress_placeholder.info("⏳ Modelo de Rostros (Detección de Deepfake)...")
                    res_face = consultar_modelo(MODEL_ROSTROS, img_bytes)
                    
                    progress_placeholder.empty()
                
                # --- PROCESAR RESULTADOS ---
                score_gen, label_gen, conf_gen = extraer_score(res_gen, "general")
                score_face, label_face, conf_face = extraer_score(res_face, "rostro")
                
                # Manejo de errores
                if score_gen is None or score_face is None:
                    st.error("❌ Error en los modelos. Intenta nuevamente.")
                    if score_gen is None:
                        st.write(f"Modelo General: {conf_gen}")
                    if score_face is None:
                        st.write(f"Modelo Rostros: {conf_face}")
                else:
                    # --- VEREDICTO FINAL ---
                    veredicto, box_class, score_final = determinar_veredicto(score_gen, score_face)
                    
                    st.markdown(f"""
                    <div class="{box_class}">
                        <h2 style="margin-top: 0;">VEREDICTO FINAL</h2>
                        <h1 style="margin-bottom: 0;">{veredicto}</h1>
                        <p style="font-size: 16px; margin-top: 10px;">
                            <strong>Confianza General: {score_final*100:.1f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # --- DETALLES DE CADA MODELO ---
                    col_m1, col_m2 = st.columns(2)
                    
                    with col_m1:
                        st.subheader("🧠 Modelo General (IA)")
                        st.markdown(f"""
                        <div class="metric-card">
                            <p><strong>Resultado:</strong> {label_gen.upper()}</p>
                            <p><strong>Confianza:</strong> {conf_gen}</p>
                            <p style="color: #667eea; font-size: 12px; margin-top: 10px;">
                                Detecta si la imagen fue creada por IA
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Barra de progreso
                        st.progress(score_gen, text=f"{score_gen*100:.1f}% IA/Fake")
                    
                    with col_m2:
                        st.subheader("👤 Modelo de Rostros (Deepfake)")
                        st.markdown(f"""
                        <div class="metric-card">
                            <p><strong>Resultado:</strong> {label_face.upper()}</p>
                            <p><strong>Confianza:</strong> {conf_face}</p>
                            <p style="color: #667eea; font-size: 12px; margin-top: 10px;">
                                Detecta si el rostro fue manipulado o sintético
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Barra de progreso
                        st.progress(score_face, text=f"{score_face*100:.1f}% Deepfake")
                    
                    st.divider()
                    
                    # --- INFORMACIÓN ÚTIL ---
                    with st.expander("ℹ️ ¿Cómo funciona esto?"):
                        st.write("""
                        **Análisis Cruzado**: Tu imagen es analizada por dos modelos de IA diferentes:
                        
                        1. **Modelo General**: Detecta patrones típicos de imágenes creadas por IA (colores, texturas, artefactos)
                        2. **Modelo de Rostros**: Se especializa en detectar rostros manipulados, deepfakes y caras sintéticas
                        
                        El resultado final combina ambos análisis para darte un veredicto más preciso.
                        
                        **⚠️ Limitaciones**:
                        - Ningún sistema es 100% preciso
                        - Imágenes de muy baja calidad pueden dar resultados inexactos
                        - Los modelos mejoran constantemente
                        - Usa esto como referencia, no como prueba definitiva
                        """)
                    
                    # --- EXPORTAR RESULTADOS (opcional) ---
                    with st.expander("📊 Datos técnicos"):
                        st.json({
                            "modelo_general": {
                                "label": label_gen,
                                "score": float(score_gen),
                                "confianza": conf_gen
                            },
                            "modelo_rostros": {
                                "label": label_face,
                                "score": float(score_face),
                                "confianza": conf_face
                            },
                            "veredicto_final": veredicto,
                            "score_promedio": float(score_final)
                        })
else:
    st.info("👆 Carga una imagen o video para comenzar el análisis")
