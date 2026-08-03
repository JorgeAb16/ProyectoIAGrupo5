import json
import os
from pathlib import Path

import cv2
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

# Carga HF_TOKEN (y otras variables) desde un archivo .env en desarrollo
# local. En Streamlit Community Cloud no existe este archivo, por lo que
# el token se toma de st.secrets (ver obtener_hf_token más abajo).
load_dotenv()

# ----------------------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# ----------------------------------------------------------------------
IMG_SIZE = (384, 384)

# Umbrales de confianza específicos por clase (reemplaza la constante global)
# <--- CAMBIO: Se añade el diccionario de umbrales por clase
CLASS_THRESHOLDS = {
    "Acne": 0.4,
    "Actinic_Keratosis": 0.45,
    "Cancer_Piel": 0.7,
    "Candidiasis": 0.6,
    "Dano_Solar": 0.8,
    "Eczema": 0.75,
    "Infestaciones_Picaduras": 0.9,
    "Lunares_Moles": 0.8,
    "Piel_Normal": 0.3,
    "Psoriasis": 0.7,
    "Queratosis_Seborreica": 0.65,
    "Tinea_Hongos": 0.7,
    "Verrugas": 0.3
}

# Umbral de "porcentaje de piel visible" (heurística clásica de color, no el
# modelo): si una foto tiene muy pocos píxeles con tono de piel, probablemente
# no es una foto útil para este clasificador (es un objeto, una pantalla, etc).
# Es un umbral experimental — bájalo si te da demasiados falsos positivos con
# fotos de piel legítimas (lesiones muy pigmentadas, mala iluminación, etc).
PIEL_MIN_PORCENTAJE = 0.12

GOOGLE_DRIVE_FILE_ID = "1DdNyswDjGhUCIxmA6-Q9HZyf37xCWmva"
MODEL_PATH = "skin_disease_model.keras"
CLASS_NAMES_PATH = "clases.json"

# ----------------------------------------------------------------------
# CONFIGURACIÓN DEL AGENTE DE PREGUNTAS (Hugging Face Inference Providers)
# ----------------------------------------------------------------------
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def obtener_hf_token() -> str:
    """
    Busca el token de Hugging Face primero en variables de entorno (.env,
    útil en local) y luego en st.secrets (usado en Streamlit Community Cloud).
    """
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return ""


# Preprocesamiento real usado en el entrenamiento (confirmado en el notebook):
# EfficientNetV2S. A diferencia de EfficientNet v1 (que no reescala los pixeles),
# EfficientNetV2 SI los reescala a rango -1..1.
preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input


# ----------------------------------------------------------------------
# TRADUCCIÓN AL ESPAÑOL DE LAS 12 CLASES
# ----------------------------------------------------------------------
LABELS_ES = {
    "Acne": "Acné",
    "Actinic_Keratosis": "Queratosis actínica",
    "Candidiasis": "Candidiasis",
    "Cancer_Piel": "Cáncer de piel",
    "Dano_Solar": "Daño solar",
    "Eczema": "Eccema",
    "Infestaciones_Picaduras": "Infestaciones y picaduras",
    "Lunares_Moles": "Lunares (nevos)",
    "Piel_Normal": "Piel normal / sin hallazgos",
    "Psoriasis": "Psoriasis",
    "Queratosis_Seborreica": "Queratosis seborreica",
    "Tinea_Hongos": "Tiña / hongos",
    "Verrugas": "Verrugas",
}

# Nombre exacto de la clase "sin hallazgos" tal como aparece en clases.json,
# usado para identificar cuándo el modelo predice piel normal en vez de
# inferirlo por heurística.
CLASE_PIEL_NORMAL = "Piel_Normal"


def nombre_legible(clase_original: str) -> str:
    if clase_original in LABELS_ES:
        return LABELS_ES[clase_original]
    return clase_original.replace("_", " ").title()


# ----------------------------------------------------------------------
# TOKENS DE DISEÑO
# ----------------------------------------------------------------------
BG = "#EEF3F0"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F5F8F6"
BORDER = "#DCE7E2"
INK = "#13322D"
INK_SOFT = "#4C6B63"
INK_FAINT = "#7C948C"
TEAL = "#0E6E63"
TEAL_DEEP = "#0B4A42"
TEAL_TINT = "#E4F0EC"
AMBER = "#B5651D"
AMBER_BG = "#FBEEE3"
CORAL = "#E15B4C"

# Paleta de acento por rango de diferencial: cada puesto del ranking recibe
# un color distinto, para que la lista se lea de un vistazo (patrón de
# leyenda diagnóstica) en vez de una fila monocroma repetida.
ACENTOS_RANGO = ["#0E6E63", "#C23B5B", "#B5651D", "#5C6BC0", "#2E86AB", "#7C5CBF"]

ESPECTRO = ["#F5DEC6", "#E8C39C", "#D2A379", "#B47F55", "#8C5A3A", "#5A3826"]
ESPECTRO_CSS = ", ".join(ESPECTRO)

st.set_page_config(
    page_title="DermIA Honduras — Detección de enfermedades de la piel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html {{ color-scheme: light; }}

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stBottomBlockContainer"] {{
        background-color: {BG} !important;
        color: {INK} !important;
        font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    }}
    /* Textura de puntos sutil (referencia a una hoja de calibración clínica) */
    [data-testid="stAppViewContainer"] {{
        background-image: radial-gradient(circle, #D8E6DF 1px, transparent 1px) !important;
        background-size: 22px 22px !important;
    }}
    .block-container {{
        background-color: transparent !important;
        color: {INK} !important;
    }}
    [data-testid="stHeader"] {{ background-color: transparent !important; }}
    .block-container {{ padding-top: 0 !important; padding-bottom: 3rem; max-width: 1120px; }}

    h1, h2, h3 {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
    .stMarkdown, .stCaption, label, p, span {{ color: {INK}; }}

    /* ============ HERO A TODO LO ANCHO ============ */
    /* Técnica de "full bleed": se sale del contenedor centrado de Streamlit */
    .hero {{
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        width: 100vw;
        padding: 3.4rem 1.5rem 3rem 1.5rem;
        margin-bottom: 2.4rem;
        background: linear-gradient(115deg, {ESPECTRO_CSS});
        overflow: hidden;
    }}
    .hero::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(11,74,66,0.15) 0%, rgba(11,74,66,0.78) 78%, {TEAL_DEEP} 100%);
    }}
    .hero-inner {{
        position: relative;
        z-index: 1;
        max-width: 1120px;
        margin: 0 auto;
        text-align: center;
    }}
    .marca {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        justify-content: center;
    }}
    .marca-titulo {{
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 3.6rem;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 24px rgba(0,0,0,0.18);
    }}
    .marca-badge {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {TEAL_DEEP};
        background: #FFFFFF;
        border-radius: 999px;
        padding: 0.24rem 0.7rem;
        margin-left: 0.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .sub-header {{
        font-size: 1.12rem;
        color: #E4F3EE;
        text-align: center;
        max-width: 640px;
        margin: 0.7rem auto 0 auto;
        line-height: 1.6;
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {TEAL};
        margin-bottom: 0.5rem;
    }}

    @media (prefers-reduced-motion: reduce) {{
        * {{ animation: none !important; transition: none !important; }}
    }}

    a:focus-visible, button:focus-visible, [role="button"]:focus-visible,
    input:focus-visible, textarea:focus-visible {{
        outline: 2px solid {TEAL} !important;
        outline-offset: 2px !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {TEAL_DEEP} !important;
        border-right: none;
    }}
    [data-testid="stSidebar"] * {{ color: #EAF3F0 !important; }}
    [data-testid="stSidebar"] .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.2rem;
    }}
    [data-testid="stSidebar"] .sidebar-brand-name {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.35rem;
        letter-spacing: -0.01em;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] .sidebar-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9CC4BB !important;
        margin-bottom: 1.4rem;
    }}
    [data-testid="stSidebar"] .sidebar-section-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9CC4BB !important;
        margin: 1.5rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.14);
        padding-top: 1.2rem;
    }}
    [data-testid="stSidebar"] .sidebar-step {{
        font-size: 0.86rem;
        line-height: 1.55;
        color: #D7E8E3 !important;
        margin-bottom: 0.55rem;
    }}
    [data-testid="stSidebar"] .sidebar-step b {{ color: #FFFFFF !important; }}
    [data-testid="stSidebar"] .sidebar-legal {{
        font-size: 0.78rem;
        line-height: 1.55;
        color: #A9CAC3 !important;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 0.85rem 0.95rem;
        margin-top: 0.6rem;
    }}
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{ background-color: #FFFFFF !important; }}
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {{ background-color: #6FBBAC !important; }}
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {{ color: #A9CAC3 !important; }}

    .upload-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 1.6rem 1.7rem 1.3rem 1.7rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04), 0 8px 24px -12px rgba(19,50,45,0.12);
        margin-bottom: 0.3rem;
    }}
    .upload-card-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.9rem;
    }}
    .upload-card-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: {INK};
    }}
    .upload-card-hint {{
        font-size: 0.82rem;
        color: {INK_FAINT};
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE_ALT} !important;
        border: 1.5px dashed #A9CDC3 !important;
        border-radius: 14px !important;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {TEAL} !important;
        background: {TEAL_TINT} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: {INK_SOFT} !important; }}
    [data-testid="baseButton-secondary"] {{
        border-radius: 8px !important;
        border-color: {TEAL} !important;
        color: {TEAL_DEEP} !important;
    }}

    /* Selector "Subir archivo" / "Usar cámara" */
    .upload-card [data-testid="stRadio"] {{ margin-bottom: 0.9rem; }}
    .upload-card [data-testid="stRadio"] label {{
        background: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 999px;
        padding: 0.3rem 0.9rem !important;
        margin-right: 0.4rem;
        transition: border-color 0.15s ease, background 0.15s ease;
    }}
    .upload-card [data-testid="stRadio"] label:has(input:checked) {{
        background: {TEAL_TINT};
        border-color: {TEAL};
    }}
    .upload-card [data-testid="stRadio"] [role="radiogroup"] {{ gap: 0.3rem; }}

    /* Vista de cámara, mismo lenguaje visual que el dropzone de archivos */
    [data-testid="stCameraInput"] {{
        border: 1.5px dashed #A9CDC3 !important;
        border-radius: 14px !important;
        padding: 0.6rem !important;
        background: {SURFACE_ALT} !important;
    }}
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {{ border-radius: 10px !important; }}
    [data-testid="stCameraInput"] button {{
        border-radius: 8px !important;
        border-color: {TEAL} !important;
        color: {TEAL_DEEP} !important;
    }}

    .empty-state {{
        background: {SURFACE};
        border: 1px dashed {BORDER};
        border-radius: 20px;
        padding: 2.6rem 1.5rem;
        text-align: center;
        color: {INK_FAINT};
        margin-top: 0.5rem;
    }}
    .empty-state-icon {{ font-size: 1.8rem; margin-bottom: 0.6rem; }}
    .empty-state-title {{ font-weight: 600; color: {INK_SOFT}; font-size: 0.98rem; margin-bottom: 0.25rem; }}
    .empty-state-body {{ font-size: 0.86rem; max-width: 360px; margin: 0 auto; line-height: 1.5; }}

    .image-frame {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 0.7rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04), 0 8px 24px -12px rgba(19,50,45,0.12);
    }}
    .image-frame img {{ border-radius: 13px !important; }}

    .result-card {{
        background: {SURFACE};
        border-radius: 14px;
        padding: 0.85rem 1.1rem 0.85rem 1rem;
        margin: 0.55rem 0;
        border: 1px solid {BORDER};
        border-left: 5px solid var(--acento, {BORDER});
        display: flex;
        align-items: center;
        gap: 0.95rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.03);
        transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
        animation: aparecer 0.35s ease both;
    }}
    .result-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -10px rgba(19,50,45,0.25);
    }}
    .result-card.principal {{
        border-color: var(--acento, {TEAL});
        background: linear-gradient(180deg, color-mix(in srgb, var(--acento, {TEAL}) 10%, {SURFACE}) 0%, {SURFACE} 70%);
        box-shadow: 0 6px 18px -10px rgba(19,50,45,0.3);
    }}
    @keyframes aparecer {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .rank-badge {{
        flex-shrink: 0;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #FFFFFF;
        background: var(--acento, {INK_FAINT});
    }}

    .disease-name {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.02rem;
        font-weight: 600;
        color: {INK};
    }}
    .rank-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--acento, {INK_FAINT});
        letter-spacing: 0.07em;
        margin-bottom: 0.05rem;
    }}

    .prob-track {{
        width: 100%;
        height: 4px;
        border-radius: 4px;
        background: #E2E9E5;
        margin-top: 0.4rem;
        overflow: hidden;
    }}
    .prob-fill {{ height: 100%; border-radius: 4px; }}

    .gauge {{
        position: relative;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .gauge-inner {{
        width: 39px;
        height: 39px;
        border-radius: 50%;
        background: {SURFACE};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        font-weight: 600;
        color: {INK};
    }}

    .triage {{
        background: {AMBER_BG};
        border-left: 4px solid {AMBER};
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        color: #6B3B10;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }}
    .piel-normal {{
        background: {TEAL_TINT};
        border-left: 4px solid {TEAL};
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        color: {TEAL_DEEP};
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }}
    .disclaimer {{
        background: {SURFACE_ALT};
        border-left: 4px solid {TEAL};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: {INK_SOFT};
        margin-top: 1.7rem;
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    /* ============ EXPLICACIÓN VISUAL (GRAD-CAM) ============ */
    .gradcam-frame {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 0.6rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04);
    }}
    .gradcam-frame img {{ border-radius: 10px !important; }}
    .gradcam-caption {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: {INK_SOFT};
        text-align: center;
        margin-top: 0.5rem;
    }}

    .chat-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 1.4rem 1.5rem 0.4rem 1.5rem;
        margin-top: 1.6rem;
        box-shadow: 0 1px 2px rgba(19,50,45,0.04), 0 8px 24px -12px rgba(19,50,45,0.12);
    }}
    .chat-card-head {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.2rem;
    }}
    .chat-card-title {{
        font-weight: 600;
        font-size: 1.0rem;
        color: {INK};
    }}
    .chat-card-sub {{
        font-size: 0.83rem;
        color: {INK_FAINT};
        margin-bottom: 0.9rem;
    }}
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        padding: 0.35rem 0 !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: {SURFACE_ALT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 0.65rem 0.95rem !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
        background: {TEAL} !important;
        border: none !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {{
        color: #FFFFFF !important;
    }}
    [data-testid="stChatInput"] {{
        border-radius: 12px !important;
        border: 1px solid {BORDER} !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {TEAL} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background: {SURFACE_ALT} !important;
        color: {INK} !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    .streamlit-expanderHeader {{
        color: {INK} !important;
        background: {SURFACE_ALT} !important;
        border-radius: 10px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }}
    [data-testid="stSlider"] [role="slider"] {{ background-color: {TEAL} !important; }}
    [data-testid="stSlider"] > div > div > div > div {{ background-color: {TEAL} !important; }}
    [data-testid="stAlert"] {{ border-radius: 12px !important; }}

    footer {{ visibility: hidden; }}
    .app-footer {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 2.4rem;
        padding-top: 1.1rem;
        border-top: 1px solid {BORDER};
        font-size: 0.8rem;
        color: {INK_FAINT};
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# FILTRO DE ENTRADA: ¿esto parece piel? (heurística clásica de color,
# no el modelo — sirve para avisar si suben una foto de otra cosa)
# ----------------------------------------------------------------------
def porcentaje_pixeles_piel(imagen_pil) -> float:
    """Heurística clásica de detección de color de piel en espacio YCrCb.
    No reemplaza al modelo, solo sirve como filtro rápido de entrada:
    si casi nada de la imagen tiene tono de piel, probablemente subieron
    una foto de otra cosa (un objeto, una pantalla, un paisaje, etc)."""
    img = imagen_pil.convert("RGB").resize((150, 150))
    arr = np.array(img)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mascara = (y > 60) & (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)
    return float(mascara.mean())


# ----------------------------------------------------------------------
# TEST-TIME AUGMENTATION (misma técnica del notebook de entrenamiento)
# ----------------------------------------------------------------------
def predecir_con_tta(modelo, entrada_preprocesada):
    """Promedia la predicción de la imagen original y su espejo horizontal.
    El flip se aplica DESPUÉS de preprocess_input, lo cual es válido porque
    es puramente espacial y no interactúa con el escalado de píxeles."""
    probs_original = modelo.predict(entrada_preprocesada, verbose=0)
    entrada_flip = tf.image.flip_left_right(entrada_preprocesada)
    probs_flip = modelo.predict(entrada_flip, verbose=0)
    return (probs_original + probs_flip) / 2.0


# ----------------------------------------------------------------------
# GRAD-CAM (misma técnica del notebook de entrenamiento)
# ----------------------------------------------------------------------
def obtener_submodelo_base(modelo_completo):
    """Encuentra el sub-modelo EfficientNetV2S embebido dentro del modelo
    completo (fue agregado como una sola capa al construir el modelo)."""
    for layer in modelo_completo.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            return layer
    raise ValueError("No se encontró el sub-modelo base (EfficientNetV2S) dentro del modelo cargado.")


def encontrar_ultima_capa_conv(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(m.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No se encontró una capa convolucional en el modelo base.")


def calcular_gradcam(modelo_completo, base_model, ultima_capa_conv, entrada_preprocesada):
    """Recrea el forward pass del modelo completo en dos partes (base +
    capas de clasificación) para poder capturar los gradientes de la
    última capa convolucional respecto a la clase predicha."""
    grad_model = tf.keras.models.Model(
        [base_model.input], [base_model.get_layer(ultima_capa_conv).output, base_model.output]
    )
    capas_clasificacion = modelo_completo.layers[-5:]  # GAP, Dropout, Dense, Dropout, Dense(softmax)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(entrada_preprocesada)
        x = base_output
        for capa in capas_clasificacion:
            x = capa(x)
        pred_idx = tf.argmax(x[0])
        canal_clase = x[:, pred_idx]

    grads = tape.gradient(canal_clase, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def superponer_heatmap(imagen_pil, heatmap, img_size):
    arr = np.array(imagen_pil.convert("RGB").resize(img_size)).astype("uint8")
    heatmap_resized = cv2.resize(heatmap, img_size)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superpuesta = cv2.addWeighted(arr, 0.55, heatmap_color, 0.45, 0)
    return Image.fromarray(superpuesta)


# ----------------------------------------------------------------------
# DESCARGA Y CARGA DEL MODELO (desde Google Drive)
# ----------------------------------------------------------------------
def descargar_modelo_si_hace_falta():
    if Path(MODEL_PATH).exists():
        return
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource(show_spinner="Descargando y cargando el modelo (puede tardar la primera vez)...")
def cargar_modelo():
    descargar_modelo_si_hace_falta()
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        clases = json.load(f)

    # Grad-CAM es opcional: si la arquitectura cargada no coincide con lo
    # esperado (base EfficientNetV2S + 5 capas de clasificación), la app
    # sigue funcionando normal, solo sin la explicación visual.
    try:
        base_model = obtener_submodelo_base(modelo)
        ultima_capa_conv = encontrar_ultima_capa_conv(base_model)
    except Exception:
        base_model, ultima_capa_conv = None, None

    return modelo, clases, base_model, ultima_capa_conv


def preprocesar_imagen(imagen_pil):
    img = imagen_pil.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = tf.image.resize(arr, IMG_SIZE)  # misma interpolacion (bilineal) que en el entrenamiento
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def render_gauge(prob: float, color_relleno: str) -> str:
    """Medidor circular hecho con conic-gradient, coloreado con el acento
    correspondiente al rango del diferencial."""
    grados = max(4, min(360, round(prob * 360)))
    return (
        f'<div class="gauge" style="background: conic-gradient(from -90deg, '
        f'{color_relleno} {grados}deg, #E2E9E5 {grados}deg 360deg);">'
        f'<div class="gauge-inner">{prob * 100:.0f}%</div></div>'
    )


def render_result_card(rank: int, clase: str, prob: float, es_principal: bool) -> str:
    acento = ACENTOS_RANGO[(rank - 1) % len(ACENTOS_RANGO)]
    clase_card = "result-card principal" if es_principal else "result-card"
    return (
        f'<div class="{clase_card}" style="--acento:{acento};">'
        f'{render_gauge(prob, acento)}'
        f'<div style="flex:1;">'
        f'<div class="rank-label">DIFERENCIAL {rank:02d}</div>'
        f'<div class="disease-name">{nombre_legible(clase)}</div>'
        f'<div class="prob-track"><div class="prob-fill" '
        f'style="width:{prob * 100:.0f}%; background:{acento};"></div></div>'
        f'</div>'
        f'<div class="rank-badge">{rank:02d}</div>'
        f'</div>'
    )


# ----------------------------------------------------------------------
# AGENTE DE PREGUNTAS (Hugging Face Inference Providers)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def obtener_cliente_hf():
    token = obtener_hf_token()
    if not token:
        return None
    return InferenceClient(api_key=token, provider="auto")


def construir_instruccion_sistema(clase_detectada: str, confianza: float, es_piel_normal: bool = False) -> str:
    """
    Instrucción de sistema para el agente: se le informa la afección
    detectada por el modelo de visión, y se le pide comportarse como un
    asistente educativo que da recomendaciones de cuidado concretas,
    no como un médico ni como un bot que solo repite "ve al doctor".
    """
    if es_piel_normal:
        contexto_deteccion = (
            f"El modelo de visión por computadora analizó una imagen y la clasificó como "
            f"'{nombre_legible(CLASE_PIEL_NORMAL)}' con un {confianza * 100:.0f}% de confianza "
            "(no encontró señales de ninguna de las condiciones que reconoce).\n\n"
        )
    else:
        contexto_deteccion = (
            f"El modelo de visión por computadora analizó una imagen y detectó, "
            f"como diagnóstico diferencial más probable, "
            f"'{nombre_legible(clase_detectada)}' con un {confianza * 100:.0f}% "
            "de confianza.\n\n"
        )

    return (
        "Eres un asistente educativo de salud de la piel, integrado en "
        "DermIA Honduras, una herramienta de análisis preliminar de "
        "imágenes dermatológicas orientada al contexto hondureño (clima "
        "tropical, alta radiación UV, alta exposición solar).\n\n"
        f"{contexto_deteccion}"
        "Tu función es responder en español las preguntas del usuario sobre este resultado, "
        "dando información realmente útil y concreta — no solo remitirlo al médico. "
        "Para cada pregunta, cuando aplique, incluye:\n"
        "- Qué es la condición y por qué ocurre (causas y factores de riesgo comunes, "
        "incluyendo los relevantes al clima tropical hondureño cuando corresponda: sudoración, "
        "humedad, radiación UV, etc.).\n"
        "- Cuidados generales concretos y accionables: rutina de limpieza apropiada, tipo de "
        "productos a buscar o evitar (por ejemplo 'no comedogénico', 'sin fragancia', "
        "'con óxido de zinc'), hábitos de protección solar, cambios de higiene o de hábitos, "
        "y qué evitar (rascar, exprimir, exponerse al sol sin protección, etc.).\n"
        "- Señales de alarma específicas de ESA condición que justifican ir a un dermatólogo "
        "pronto (por ejemplo: cambios de tamaño/color/forma en un lunar, sangrado, dolor "
        "creciente, fiebre, falta de mejora tras varias semanas de cuidado en casa) — en vez "
        "de recomendar la consulta como respuesta genérica a todo.\n\n"
        "Sé específico y práctico, como lo sería un buen folleto educativo de salud, no evasivo. "
        "Responde de forma clara, organizada (usa viñetas cuando ayude) y profesional, sin ser "
        "excesivamente largo.\n\n"
        "Reglas importantes (no negociables):\n"
        "- NUNCA confirmes un diagnóstico definitivo, ni confirmes con certeza que la piel está "
        "sana: dilo como lo que es, un resultado preliminar de un modelo de IA.\n"
        "- NUNCA recomiendes medicamentos con receta, dosis, ni nombres de fármacos específicos "
        "(puedes mencionar categorías generales de venta libre, como 'protector solar SPF 30+' "
        "o 'humectante sin fragancia', sin recetar nada).\n"
        "- Menciona acudir a un dermatólogo cuando exista una señal de alarma real para esa "
        "condición, cuando el usuario lo pida explícitamente, o al cierre si la condición lo "
        "amerita — no la repitas como muletilla en cada respuesta.\n"
        "- Si te preguntan algo fuera del tema de la piel o la afección detectada, responde "
        "brevemente pero reorienta la conversación hacia el propósito de la herramienta."
    )


def responder_pregunta_agente(cliente: InferenceClient, historial: list, pregunta: str) -> str:
    """
    Envía la pregunta del usuario junto con el historial de la conversación
    al modelo de chat de Hugging Face y retorna la respuesta en texto.
    """
    mensajes = historial + [{"role": "user", "content": pregunta}]
    respuesta = cliente.chat.completions.create(
        model=HF_CHAT_MODEL,
        messages=mensajes,
        max_tokens=500,
        temperature=0.4,
    )
    return respuesta.choices[0].message.content


# ----------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------
def render_sidebar(num_clases: int) -> int:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<svg width="26" height="26" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="15" cy="15" r="10" stroke="#FFFFFF" stroke-width="2.5"/>'
            '<line x1="22.5" y1="22.5" x2="30" y2="30" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
            '</svg>'
            '<span class="sidebar-brand-name">DermIA Honduras</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-tag">Análisis dermatológico con IA</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">Cómo funciona</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-step"><b>1.</b> Sube una foto nítida de la lesión.</div>'
            '<div class="sidebar-step"><b>2.</b> Se verifica que la imagen tenga piel visible.</div>'
            '<div class="sidebar-step"><b>3.</b> El modelo (EfficientNetV2S) la analiza en segundos.</div>'
            '<div class="sidebar-step"><b>4.</b> Revisa los diagnósticos diferenciales y el mapa de calor Grad-CAM.</div>'
            '<div class="sidebar-step"><b>5.</b> Pregúntale al asistente lo que quieras saber sobre el resultado.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Preferencias</div>', unsafe_allow_html=True)
        top_k = st.slider(
            "Diagnósticos diferenciales a mostrar",
            min_value=1,
            max_value=min(10, num_clases),
            value=3,
        )

        st.markdown('<div class="sidebar-section-title">Aviso legal</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-legal">Esta herramienta tiene fines educativos y demostrativos. '
            'No constituye un diagnóstico médico ni sustituye la evaluación de un '
            'dermatólogo.</div>',
            unsafe_allow_html=True,
        )

        return top_k


def main():
    st.markdown(
        '<div class="hero"><div class="hero-inner">'
        '<div class="marca">'
        '<svg width="38" height="38" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="15" cy="15" r="10" stroke="#FFFFFF" stroke-width="2.5"/>'
        '<line x1="22.5" y1="22.5" x2="30" y2="30" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>'
        '</svg>'
        '<span class="marca-titulo">DermIA Honduras</span>'
        '<span class="marca-badge">Beta</span>'
        '</div>'
        '<div class="sub-header">Análisis preliminar de afecciones de piel comunes en el contexto '
        'hondureño (clima tropical, alta radiación UV) mediante inteligencia artificial. '
        'Una herramienta de apoyo, no un reemplazo del criterio médico.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    try:
        modelo, class_names, base_model, ultima_capa_conv = cargar_modelo()
    except Exception as e:
        st.error(
            "No se pudo descargar o cargar el modelo desde Google Drive. "
            "Revisa que el archivo siga compartido como 'Cualquier persona con el enlace' "
            f"y que '{CLASS_NAMES_PATH}' exista junto a app.py.\n\nError: {e}"
        )
        st.stop()

    top_k = render_sidebar(len(class_names))

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-card-head">'
        '<span class="upload-card-title">Sube una imagen de la lesión</span>'
        '<span class="upload-card-hint" id="upload-hint">Formatos JPG, JPEG o PNG</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    modo_entrada = st.radio(
        "Fuente de la imagen",
        options=["Subir archivo", "Usar cámara"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if modo_entrada == "Subir archivo":
        archivo = st.file_uploader(
            "Arrastra o selecciona una imagen",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    else:
        st.caption("Se te pedirá permiso del navegador para usar la cámara.")
        archivo = st.camera_input(
            "Toma una foto de la lesión",
            label_visibility="collapsed",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if archivo is not None:
        imagen = Image.open(archivo)

        col1, col2 = st.columns([1, 1.3], gap="large")

        with col1:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(imagen, caption="Imagen cargada", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        pct_piel = porcentaje_pixeles_piel(imagen)
        if pct_piel < PIEL_MIN_PORCENTAJE:
            st.markdown(
                f'<div class="triage">🔍 <strong>Esta imagen tiene muy poco tono de piel visible '
                f'({pct_piel:.0%}).</strong> Este clasificador está entrenado solo para fotos de '
                f'lesiones cutáneas — si subiste otra cosa (un objeto, una pantalla, etc.), el '
                f'resultado no va a ser confiable. Verifica la foto antes de continuar.</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Analizando imagen..."):
            entrada = preprocesar_imagen(imagen)
            predicciones = predecir_con_tta(modelo, entrada)[0]

        top_idx = np.argsort(predicciones)[-top_k:][::-1]
        confianza_principal = float(predicciones[top_idx[0]])
        clase_top = class_names[top_idx[0]]

        # <--- CAMBIO: Obtener el umbral específico para la clase predicha
        umbral_clase = CLASS_THRESHOLDS.get(clase_top, 0.5)
        umbral_piel_normal = CLASS_THRESHOLDS.get(CLASE_PIEL_NORMAL, 0.3)

        es_piel_normal = (
            clase_top == CLASE_PIEL_NORMAL and confianza_principal >= umbral_piel_normal
        )

        with col2:
            st.markdown('<div class="eyebrow">Resultado</div>', unsafe_allow_html=True)

            if es_piel_normal:
                st.markdown(
                    f'<div class="piel-normal">🌿 <strong>No se detectaron hallazgos relevantes '
                    f'({confianza_principal * 100:.0f}% de confianza).</strong> El modelo clasificó '
                    'esta imagen como piel sin señales de las condiciones que reconoce. Esto '
                    '<strong>no reemplaza una revisión médica</strong> — si notas cambios, '
                    'crecimiento, sangrado o algo que te preocupe, consulta a un dermatólogo de '
                    'todas formas.</div>',
                    unsafe_allow_html=True,
                )
            elif confianza_principal < umbral_clase:   # <--- CAMBIO: comparar con umbral específico
                st.markdown(
                    f'<div class="triage">⚠️ <strong>Confianza baja '
                    f'({confianza_principal * 100:.0f}%).</strong> Por debajo del umbral mínimo '
                    f'({umbral_clase * 100:.0f}%) para un diagnóstico confiable de '
                    f'<strong>{nombre_legible(clase_top)}</strong>. '   # <--- CAMBIO: se indica la clase
                    f'Se recomienda <strong>consultar directamente con un dermatólogo</strong>.</div>',
                    unsafe_allow_html=True,
                )

            for i, idx in enumerate(top_idx):
                clase = class_names[idx]
                prob = float(predicciones[idx])
                st.markdown(
                    render_result_card(i + 1, clase, prob, es_principal=(i == 0)),
                    unsafe_allow_html=True,
                )

        if base_model is not None:
            st.markdown(
                '<div class="eyebrow" style="margin-top:1.8rem;">Explicación visual (Grad-CAM)</div>',
                unsafe_allow_html=True,
            )
            try:
                with st.spinner("Generando mapa de calor..."):
                    heatmap = calcular_gradcam(modelo, base_model, ultima_capa_conv, entrada)
                    superpuesta = superponer_heatmap(imagen, heatmap, IMG_SIZE)

                col_g1, col_g2 = st.columns(2, gap="medium")
                with col_g1:
                    st.markdown('<div class="gradcam-frame">', unsafe_allow_html=True)
                    st.image(imagen.convert("RGB").resize(IMG_SIZE), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="gradcam-caption">IMAGEN ORIGINAL</div>', unsafe_allow_html=True)
                with col_g2:
                    st.markdown('<div class="gradcam-frame">', unsafe_allow_html=True)
                    st.image(superpuesta, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="gradcam-caption">ZONAS QUE MÁS INFLUYERON EN LA PREDICCIÓN</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.caption("No se pudo generar la explicación visual para esta imagen.")

        st.markdown(
            '<div class="disclaimer">Este resultado es generado por un modelo de inteligencia '
            'artificial con fines educativos y demostrativos. '
            '<strong>No constituye un diagnóstico médico.</strong> '
            'Ante cualquier duda, consulta a un dermatólogo.</div>',
            unsafe_allow_html=True,
        )

        clase_principal = clase_top

        if st.session_state.get("clase_activa") != clase_principal:
            st.session_state["clase_activa"] = clase_principal
            st.session_state["mensajes_chat"] = [
                {
                    "role": "system",
                    "content": construir_instruccion_sistema(
                        clase_principal, confianza_principal, es_piel_normal
                    ),
                }
            ]

        st.markdown('<div class="chat-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chat-card-head">'
            '<span class="chat-card-title">💬 Preguntas sobre este resultado</span>'
            '</div>'
            f'<div class="chat-card-sub">Pregúntale al asistente sobre '
            f'{nombre_legible(clase_principal).lower()}: causas, cuidados y cuándo consultar '
            'a un especialista.</div>',
            unsafe_allow_html=True,
        )

        cliente_hf = obtener_cliente_hf()

        if cliente_hf is None:
            st.info(
                "Para habilitar el asistente de preguntas, configura la variable "
                "`HF_TOKEN` (token de Hugging Face) en tu archivo `.env` local "
                "o en los secrets de Streamlit."
            )
        else:
            for mensaje in st.session_state["mensajes_chat"]:
                if mensaje["role"] == "system":
                    continue
                avatar = "🧑" if mensaje["role"] == "user" else "🔬"
                with st.chat_message(mensaje["role"], avatar=avatar):
                    st.markdown(mensaje["content"])

            pregunta = st.chat_input(
                f"Pregunta algo sobre {nombre_legible(clase_principal).lower()}..."
            )

            if pregunta:
                st.session_state["mensajes_chat"].append({"role": "user", "content": pregunta})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(pregunta)

                with st.chat_message("assistant", avatar="🔬"):
                    with st.spinner("Pensando..."):
                        try:
                            respuesta_texto = responder_pregunta_agente(
                                cliente_hf, st.session_state["mensajes_chat"][:-1], pregunta
                            )
                        except Exception as e:
                            respuesta_texto = (
                                "No fue posible obtener respuesta del asistente en este "
                                f"momento. Detalle técnico: {e}"
                            )
                    st.markdown(respuesta_texto)

                st.session_state["mensajes_chat"].append(
                    {"role": "assistant", "content": respuesta_texto}
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">🩺</div>'
            '<div class="empty-state-title">Aún no hay ninguna imagen cargada</div>'
            '<div class="empty-state-body">Sube una fotografía nítida de la piel para '
            'recibir un análisis preliminar y poder conversar con el asistente sobre '
            'el resultado.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="app-footer">'
        '<span>DermIA Honduras · EfficientNetV2S · Modelo alojado en Google Drive</span>'
        '<span>Herramienta educativa — no reemplaza consulta médica</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
