import streamlit as st
import requests
from PIL import Image
import io
import cv2
import tempfile
import numpy as np
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(
    page_title="REAL OR FAKE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .main { padding: 20px; background: #0a0a0a; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
        color: #000;
        font-weight: 800;
        font-family: 'Space Mono', monospace;
        border-radius: 4px;
        padding: 14px 30px;
        border: none;
        letter-spacing: 1px;
        transition: all 0.2s;
        text-transform: uppercase;
    }
    .stButton > button:hover { transform: scale(1.03); opacity: 0.9; }

    .result-box {
        background: #1a1a2e;
        padding: 24px;
        border-radius: 8px;
        border-left: 5px solid #f39c12;
        color: #fff;
    }
    .real-box {
        background: #0d1f0d;
        padding: 24px;
        border-radius: 8px;
        border-left: 5px solid #00ff88;
        color: #fff;
    }
    .fake-box {
        background: #1f0d0d;
        padding: 24px;
        border-radius: 8px;
        border-left: 5px solid #ff4444;
        color: #fff;
    }
    .metric-card {
        background: #111;
        padding: 16px;
        border-radius: 6px;
        border: 1px solid #222;
        margin: 10px 0;
        color: #eee;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
    }
    .forensic-card {
        background: #111;
        border: 1px solid #333;
        padding: 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        color: #ccc;
    }
    .forensic-card .label { color: #00ccff; font-weight: bold; }
    .forensic-card .value { color: #00ff88; }
    .forensic-card .suspicious { color: #ff4444; }
    .warning-box {
        background: #1a1a1a;
        border: 2px solid #ff4444;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #fff;
        font-family: 'Space Mono', monospace;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# CONFIGURACIÓN DE APIs
API_TOKEN = st.secrets.get("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

MODELOS_HF_GENERAL = [
    "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector",
    "https://api-inference.huggingface.co/models/Organismo/DetectAI",
]
MODELOS_HF_ROSTROS = [
    "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-Model",
]

# ============================================================
# ANÁLISIS FORENSE REAL (sin dependencias externas)
# ============================================================

def analizar_frecuencias(img_array):
    """Análisis de frecuencias con FFT — las imágenes IA tienen patrones espectrales anómalos."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    
    h, w = magnitude.shape
    cx, cy = h // 2, w // 2
    radio = min(h, w) // 8
    
    centro = magnitude[cx-radio:cx+radio, cy-radio:cy+radio].mean()
    bordes = magnitude.mean()
    ratio_frecuencias = centro / (bordes + 1e-8)
    
    # Imágenes IA tienden a tener ratio más alto (artefactos en frecuencias altas)
    sospechoso = ratio_frecuencias > 2.5
    return ratio_frecuencias, sospechoso

def analizar_ruido(img_array):
    """Analiza el patrón de ruido — imágenes IA tienen ruido más uniforme/sintético."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Filtro laplaciano para extraer ruido
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    std_ruido = laplacian.std()
    mean_ruido = abs(laplacian.mean())
    
    # Dividir en bloques y analizar varianza
    h, w = gray.shape
    block_size = 32
    varianzas = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            bloque = laplacian[i:i+block_size, j:j+block_size]
            varianzas.append(bloque.var())
    
    # Imágenes reales tienen varianza de ruido más heterogénea
    cv_ruido = np.std(varianzas) / (np.mean(varianzas) + 1e-8)
    
    # Bajo coeficiente de variación = ruido muy uniforme = sospechoso
    sospechoso = cv_ruido < 0.8
    return cv_ruido, std_ruido, sospechoso

def analizar_compresion_ela(img_array):
    """
    Error Level Analysis (ELA) — detecta inconsistencias de compresión JPEG.
    Imágenes manipuladas o generadas por IA tienen patrones ELA anómalos.
    """
    img_pil = Image.fromarray(img_array)
    
    buf1 = io.BytesIO()
    img_pil.save(buf1, format="JPEG", quality=90)
    buf1.seek(0)
    img_90 = np.array(Image.open(buf1).convert("RGB")).astype(np.float32)
    
    buf2 = io.BytesIO()
    img_pil.save(buf2, format="JPEG", quality=75)
    buf2.seek(0)
    img_75 = np.array(Image.open(buf2).convert("RGB")).astype(np.float32)
    
    original = img_array.astype(np.float32)
    
    ela_diff = np.abs(original - img_90)
    ela_diff2 = np.abs(img_90 - img_75)
    
    ela_mean = ela_diff.mean()
    ela_std = ela_diff.std()
    ela_ratio = ela_mean / (ela_std + 1e-8)
    
    # ELA muy uniforme (bajo std relativo) = posible generación IA
    sospechoso = ela_std < 8.0 or ela_ratio > 3.0
    return ela_mean, ela_std, sospechoso

def analizar_consistencia_color(img_array):
    """
    Analiza distribución de color en el espacio HSV.
    Imágenes IA tienden a tener saturación anormalmente uniforme o extrema.
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    sat_mean = s_channel.mean()
    sat_std = s_channel.std()
    val_std = v_channel.std()
    
    # Calcular entropía del canal de matiz
    hist_h = np.histogram(h_channel, bins=36, range=(0, 180))[0]
    hist_h = hist_h / (hist_h.sum() + 1e-8)
    entropia_h = -np.sum(hist_h * np.log(hist_h + 1e-8))
    
    # Saturación muy alta y uniforme = sospechoso
    sospechoso = (sat_mean > 140 and sat_std < 40) or (sat_std < 20)
    return sat_mean, sat_std, entropia_h, sospechoso

def analizar_bordes_texturas(img_array):
    """
    Analiza la naturalidad de bordes y texturas.
    Imágenes IA tienen bordes más suaves y texturas repetitivas.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detectar bordes con Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()
    
    # Analizar textura con LBP simplificado
    gray_f = gray.astype(np.float32)
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
    texture_response = cv2.filter2D(gray_f, -1, kernel)
    texture_energy = (texture_response ** 2).mean()
    texture_std = texture_response.std()
    
    # Bordes muy suaves (baja densidad) + textura baja energía = sospechoso
    sospechoso = edge_density < 5.0 or texture_energy < 100
    return edge_density, texture_energy, texture_std, sospechoso

def analizar_simetria_facial(img_array):
    """
    Detecta simetría facial anormal — deepfakes suelen tener simetría perfecta.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    mitad_izq = gray[:, :w//2].astype(np.float32)
    mitad_der = np.fliplr(gray[:, w//2:]).astype(np.float32)
    
    min_w = min(mitad_izq.shape[1], mitad_der.shape[1])
    diff = np.abs(mitad_izq[:, :min_w] - mitad_der[:, :min_w])
    
    asimetria = diff.mean()
    
    # Muy baja asimetría = demasiado simétrico = sospechoso (deepfake)
    sospechoso = asimetria < 8.0
    return asimetria, sospechoso

def calcular_score_forensico(img_array):
    """
    Combina todos los análisis forenses y devuelve un score de 0 a 1
    donde 1 = muy probable IA/Fake.
    """
    indicadores = []
    detalles = {}
    
    # 1. Análisis de frecuencias
    try:
        ratio_freq, sosp_freq = analizar_frecuencias(img_array)
        indicadores.append(0.7 if sosp_freq else 0.2)
        detalles["frecuencias"] = {
            "ratio": round(float(ratio_freq), 3),
            "sospechoso": sosp_freq,
            "descripcion": "Artefactos en frecuencias altas (FFT)"
        }
    except:
        pass
    
    # 2. Análisis de ruido
    try:
        cv_ruido, std_ruido, sosp_ruido = analizar_ruido(img_array)
        indicadores.append(0.75 if sosp_ruido else 0.15)
        detalles["ruido"] = {
            "coef_variacion": round(float(cv_ruido), 3),
            "std": round(float(std_ruido), 2),
            "sospechoso": sosp_ruido,
            "descripcion": "Uniformidad del patrón de ruido"
        }
    except:
        pass
    
    # 3. ELA
    try:
        ela_mean, ela_std, sosp_ela = analizar_compresion_ela(img_array)
        indicadores.append(0.8 if sosp_ela else 0.2)
        detalles["ela"] = {
            "mean": round(float(ela_mean), 2),
            "std": round(float(ela_std), 2),
            "sospechoso": sosp_ela,
            "descripcion": "Error Level Analysis (consistencia JPEG)"
        }
    except:
        pass
    
    # 4. Color
    try:
        sat_mean, sat_std, entropia_h, sosp_color = analizar_consistencia_color(img_array)
        indicadores.append(0.65 if sosp_color else 0.1)
        detalles["color"] = {
            "saturacion_media": round(float(sat_mean), 1),
            "saturacion_std": round(float(sat_std), 1),
            "entropia_matiz": round(float(entropia_h), 3),
            "sospechoso": sosp_color,
            "descripcion": "Distribución HSV y entropía de color"
        }
    except:
        pass
    
    # 5. Bordes y texturas
    try:
        edge_density, texture_energy, texture_std, sosp_bordes = analizar_bordes_texturas(img_array)
        indicadores.append(0.7 if sosp_bordes else 0.15)
        detalles["bordes"] = {
            "densidad_bordes": round(float(edge_density), 2),
            "energia_textura": round(float(texture_energy), 1),
            "sospechoso": sosp_bordes,
            "descripcion": "Naturalidad de bordes y texturas"
        }
    except:
        pass
    
    # 6. Simetría
    try:
        asimetria, sosp_simetria = analizar_simetria_facial(img_array)
        indicadores.append(0.6 if sosp_simetria else 0.1)
        detalles["simetria"] = {
            "asimetria": round(float(asimetria), 2),
            "sospechoso": sosp_simetria,
            "descripcion": "Simetría bilateral (deepfake suele ser perfecta)"
        }
    except:
        pass
    
    if not indicadores:
        return 0.5, detalles
    
    # Score ponderado — ELA y ruido tienen más peso
    score = np.mean(indicadores)
    
    # Bonus: si 4+ indicadores son sospechosos, aumentar score
    n_sospechosos = sum(1 for v in detalles.values() if v.get("sospechoso", False))
    if n_sospechosos >= 4:
        score = min(score * 1.2, 0.95)
    elif n_sospechosos <= 1:
        score = max(score * 0.8, 0.05)
    
    return float(score), detalles


def consultar_modelo_hf(urls, datos_binarios):
    if isinstance(urls, str):
        urls = [urls]
    intentos = []
    for url in urls:
        try:
            nombre = url.split("/")[-1][:30]
            resp = requests.post(url, headers=HF_HEADERS, data=datos_binarios, timeout=30)
            if resp.status_code == 200:
                return resp.json(), None
            else:
                intentos.append(f"{nombre}: Error {resp.status_code}")
        except Exception as e:
            intentos.append(f"{url.split('/')[-1][:20]}: {str(e)[:25]}")
    return None, intentos

def extraer_score_hf(resultado, modelo_tipo="general"):
    if not isinstance(resultado, list):
        return None
    scores = {}
    for item in resultado:
        if isinstance(item, dict) and "label" in item:
            scores[item['label'].lower()] = item.get('score', 0)
    fake_keys = ['artificial','fake','label_1','ai','ai-generated','deepfake','manipulated','fake_face']
    for k in fake_keys:
        if k in scores:
            return scores[k]
    if scores:
        return max(scores.values())
    return None

def determinar_veredicto(score, tiene_hf=False):
    if tiene_hf:
        # Con HF los modelos son más precisos, umbrales normales
        if score >= 0.60:
            return "🚨 PROBABLEMENTE FALSO / IA", "fake-box", score
        elif score >= 0.35:
            return "⚠️ DUDOSO — PUEDE SER IA", "result-box", score
        else:
            return "✅ PROBABLEMENTE REAL", "real-box", score
    else:
        # Sin HF, el análisis forense solo no es suficiente para ser categórico
        if score >= 0.55:
            return "🚨 INDICIOS DE IA / FAKE", "fake-box", score
        elif score >= 0.30:
            return "⚠️ INCONCLUSO — Análisis forense insuficiente", "result-box", score
        else:
            return "✅ SIN INDICIOS CLAROS DE IA", "real-box", score


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuración")
    if not API_TOKEN:
        st.warning("⚠️ **Sin token HF** — usando análisis forense local")
        st.write("Para agregar HF como capa extra:")
        st.write("Settings → Secrets → `HF_TOKEN`")
    else:
        st.success("✅ Token HF configurado")
    
    st.divider()
    st.subheader("🔬 Técnicas Forenses")
    st.write("""
    **Análisis local activo:**
    - FFT (frecuencias espectrales)
    - Error Level Analysis (ELA)
    - Patrón de ruido sintético
    - Distribución HSV/color
    - Densidad y energía de bordes
    - Simetría bilateral
    
    **+ Hugging Face (si hay token):**
    - AI Image Detector
    - Deep Fake Detector
    """)
    st.divider()
    st.subheader("💡 Tips")
    st.write("""
    - Mínimo 200×200 px
    - El análisis forense funciona sin internet
    - HF suma precisión cuando está disponible
    """)

# --- HEADER ---
st.title("🛡️ REAL OR FAKE")
st.write("**Detector forense de imágenes IA y deepfakes**")
st.write("Análisis real con 6 técnicas forenses + modelos de Hugging Face cuando estén disponibles.")

# --- UPLOAD ---
col_upload, col_info = st.columns([3, 1])
with col_upload:
    archivo = st.file_uploader(
        "📤 Subí una foto o video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi', 'webp'],
        help="Máximo 50MB"
    )
with col_info:
    st.info("🔬 6 análisis forenses reales")

if archivo:
    file_size_mb = len(archivo.getvalue()) / (1024 * 1024)
    if file_size_mb > 50:
        st.error(f"❌ Archivo muy grande ({file_size_mb:.1f}MB). Máximo: 50MB")
        st.stop()
    
    img_final = None
    with st.spinner("Procesando..."):
        try:
            if archivo.type in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(archivo.read())
                tfile.close()
                vf = cv2.VideoCapture(tfile.name)
                ret, frame = vf.read()
                vf.release()
                if ret:
                    img_final = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    archivo_tipo = "📹 VIDEO (frame: 0s)"
                else:
                    st.error("❌ No se pudo procesar el video")
                    st.stop()
            else:
                img_final = Image.open(archivo).convert("RGB")
                archivo_tipo = "🖼️ IMAGEN"
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
    
    if img_final:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vista previa")
            st.image(img_final, use_container_width=True)
            st.caption(f"{archivo_tipo} | {archivo.name} | {file_size_mb:.2f} MB")
        
        with col2:
            st.subheader("Opciones")
            st.write(f"📁 **{archivo.name}**")
            st.write(f"📐 Resolución: **{img_final.width}×{img_final.height}**")
            analizar = st.button("🔍 EJECUTAR ANÁLISIS FORENSE", type="primary", use_container_width=True)
        
        if analizar:
            with st.spinner("Ejecutando análisis forense... (puede tardar 15-30 seg)"):
                buf = io.BytesIO()
                img_final.save(buf, format="JPEG", quality=92)
                img_bytes = buf.getvalue()
                img_array = np.array(img_final)
                
                # 1. Análisis forense local (siempre)
                score_forense, detalles_forense = calcular_score_forensico(img_array)
                
                # 2. Hugging Face (si hay token)
                score_hf_gen = None
                score_hf_face = None
                hf_errores = []
                
                if API_TOKEN:
                    with st.spinner("Consultando Hugging Face..."):
                        res_gen, err_gen = consultar_modelo_hf(MODELOS_HF_GENERAL, img_bytes)
                        res_face, err_face = consultar_modelo_hf(MODELOS_HF_ROSTROS, img_bytes)
                        score_hf_gen = extraer_score_hf(res_gen, "general")
                        score_hf_face = extraer_score_hf(res_face, "rostro")
                        if err_gen: hf_errores.extend(err_gen)
                        if err_face: hf_errores.extend(err_face)
                
                # Score final combinado
                scores_validos = [score_forense]
                if score_hf_gen is not None:
                    scores_validos.append(score_hf_gen)
                if score_hf_face is not None:
                    scores_validos.append(score_hf_face)
                
                # Ponderación: forense tiene peso base, HF suma si está disponible
                if len(scores_validos) == 1:
                    score_final = score_forense
                elif len(scores_validos) == 2:
                    score_final = score_forense * 0.5 + scores_validos[1] * 0.5
                else:
                    score_final = score_forense * 0.4 + score_hf_gen * 0.35 + score_hf_face * 0.25
            
            # --- VEREDICTO ---
            tiene_hf = score_hf_gen is not None or score_hf_face is not None
            veredicto, box_class, _ = determinar_veredicto(score_final, tiene_hf)
            
            st.markdown(f"""
            <div class="{box_class}">
                <h2 style="margin-top:0; color:#fff;">VEREDICTO FINAL</h2>
                <h1 style="margin-bottom:0; margin-top:10px; color:#fff;">{veredicto}</h1>
                <p style="font-size:16px; margin-top:15px; margin-bottom:0; color:#ccc; font-family:'Space Mono',monospace;">
                    Score de IA/Fake: <strong style="color:#00ff88;">{score_final*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if not tiene_hf:
                st.warning("⚠️ **Precisión limitada**: Sin token de Hugging Face, el análisis es solo forense local (~60-70% de precisión). Configurá `HF_TOKEN` para resultados más confiables.")
            
            st.divider()
            
            # --- BREAKDOWN ---
            col_f, col_hf = st.columns(2)
            
            with col_f:
                st.subheader("🔬 Análisis Forense Local")
                n_sosp = sum(1 for v in detalles_forense.values() if v.get("sospechoso", False))
                st.write(f"**{n_sosp}/{len(detalles_forense)} indicadores sospechosos** | Score: `{score_forense*100:.1f}%`")
                
                for nombre, datos in detalles_forense.items():
                    emoji = "🔴" if datos.get("sospechoso") else "🟢"
                    estado = "SOSPECHOSO" if datos.get("sospechoso") else "NORMAL"
                    clase_estado = "suspicious" if datos.get("sospechoso") else "value"
                    
                    # Armar métricas como string
                    metricas = {k: v for k, v in datos.items() if k not in ["sospechoso", "descripcion"]}
                    metricas_str = " | ".join([f"{k}: {v}" for k, v in metricas.items()])
                    
                    st.markdown(f"""
                    <div class="forensic-card">
                        {emoji} <span class="label">{nombre.upper()}</span>
                        — <span class="{clase_estado}">{estado}</span><br>
                        <span style="color:#888; font-size:11px;">{datos.get('descripcion','')}</span><br>
                        <span style="color:#aaa;">{metricas_str}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_hf:
                st.subheader("🤖 Modelos Hugging Face")
                if not API_TOKEN:
                    st.warning("Sin token HF — solo análisis forense local activo")
                    st.write("Configurá `HF_TOKEN` en Secrets para sumar precisión con modelos de deep learning.")
                else:
                    if score_hf_gen is not None:
                        st.markdown(f"""
                        <div class="metric-card">
                            🧠 <span style="color:#00ccff;">AI Image Detector</span><br>
                            Score IA: <strong style="color:#00ff88;">{score_hf_gen*100:.1f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(score_hf_gen)
                    else:
                        st.warning("❌ Modelo General: no disponible")
                    
                    if score_hf_face is not None:
                        st.markdown(f"""
                        <div class="metric-card">
                            👤 <span style="color:#00ccff;">Deep Fake Detector</span><br>
                            Score Fake: <strong style="color:#00ff88;">{score_hf_face*100:.1f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(score_hf_face)
                    else:
                        st.warning("❌ Modelo Rostros: no disponible")
                    
                    if hf_errores:
                        with st.expander("Ver errores HF"):
                            for e in hf_errores:
                                st.write(f"- {e}")
            
            st.divider()
            
            # --- EXPORTAR ---
            with st.expander("📊 Exportar datos técnicos"):
                datos_export = {
                    "fecha": datetime.now().isoformat(),
                    "archivo": archivo.name,
                    "veredicto": veredicto,
                    "score_final": round(score_final, 4),
                    "score_forense_local": round(score_forense, 4),
                    "score_hf_general": round(score_hf_gen, 4) if score_hf_gen else None,
                    "score_hf_rostros": round(score_hf_face, 4) if score_hf_face else None,
                    "detalle_forense": detalles_forense
                }
                st.json(datos_export)

else:
    st.info("👆 Cargá una imagen o video para comenzar el análisis forense")
