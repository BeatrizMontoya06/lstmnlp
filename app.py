"""
NEO-CYBER CHRONOS // K-LSTM VAPOR-CORE 2006
Interfaz Autónoma de Síntesis Literaria & Neuronal
"""

import streamlit as st
import numpy as np
import json
import time
import os

# Configuración de página con identidad Y2K / Cyber-Retro
st.set_page_config(
    page_title="CHRONOS-2006 // Neural Core",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS Radicales: Desestructurado, Y2K Cyber-Retro & Neoperreo Glow ─────
st.markdown("""
<style>
    /* Fondo con texturas e identidad cyberpunk / 2000s */
    .stApp {
        background: linear-gradient(135deg, #0d0221 0%, #190535 35%, #0f172a 100%);
        background-attachment: fixed;
    }

    html, body, [class*="css"] {
        font-family: 'Courier New', Courier, monospace;
        color: #00ffcc;
    }

    /* Ocultar elementos estándar de Streamlit para cambiar la estructura por completo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Contenedor tipo Terminal / Cyberdeck flotante */
    .cyber-container {
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid #ff007f;
        border-radius: 4px;
        padding: 25px;
        box-shadow: 0 0 20px rgba(255, 0, 127, 0.4), inset 0 0 15px rgba(0, 255, 204, 0.2);
        margin-bottom: 25px;
    }

    /* Cabecera Brutalista / Retro-Futurista */
    .brutal-header {
        background: linear-gradient(90deg, #ff007f, #7928ca, #0070f3);
        padding: 4px;
        border-radius: 6px;
        margin-bottom: 25px;
    }
    
    .brutal-inner {
        background: #09090b;
        padding: 20px;
        border-radius: 4px;
        text-align: center;
    }

    .brutal-inner h1 {
        font-family: 'Impact', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 2px;
        color: #ffffff;
        text-shadow: 3px 3px #ff007f, -3px -3px #00ffcc;
        margin: 0;
    }

    .brutal-inner p {
        color: #00ffcc;
        font-size: 0.95rem;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Pantalla de Texto Generado (Estilo CRT / Matrix / Cyberpunk) */
    .crt-output {
        background: #020617;
        border: 2px solid #00ffcc;
        border-radius: 6px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 1.15rem;
        line-height: 1.7;
        color: #00ffcc;
        box-shadow: inset 0 0 15px rgba(0, 255, 204, 0.3);
        min-height: 180px;
        text-shadow: 0 0 5px rgba(0, 255, 204, 0.5);
    }

    /* Estilo para los botones */
    .stButton > button {
        background: linear-gradient(45deg, #ff007f, #7928ca);
        color: white;
        border: 2px solid #00ffcc;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        text-transform: uppercase;
        border-radius: 4px;
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.5);
        transition: 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #00ffcc, #0070f3);
        color: #000;
        border-color: #ff007f;
    }

    /* Insignias flotantes */
    .cyber-badge {
        display: inline-block;
        background: #ff007f;
        color: white;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        border: 1px solid #00ffcc;
        box-shadow: 2px 2px 0px #00ffcc;
    }
</style>
""", unsafe_allow_html=True)


# ── Funciones de Carga y Procesamiento Neuronal ───────────────────────────────

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)


def is_embedding_model(model):
    first = model.layers[0]
    return hasattr(first, "input_dim") or first.__class__.__name__ == "Embedding"


def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))


def prepare_input(window, char_to_idx, vocab_size, use_embedding):
    indices = [char_to_idx.get(c, 0) for c in window]
    if use_embedding:
        return np.array([indices], dtype=np.int32)
    else:
        x = np.array(indices, dtype=np.float32) / float(vocab_size)
        return x.reshape(1, len(window), 1)


def generate_full_text(model, seed_text, char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars=200, temperature=0.8):
    use_emb = is_embedding_model(model)
    seed_text = seed_text.lower()
    if len(seed_text) < seq_length:
        seed_text = seed_text.rjust(seq_length)
    seed_text = seed_text[-seq_length:]
    seed_text = "".join(c if c in char_to_idx else " " for c in seed_text)

    generated = ""
    window = list(seed_text)

    for _ in range(n_chars):
        x = prepare_input(window[-seq_length:], char_to_idx, vocab_size, use_emb)
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temperature)]
        generated += next_char
        window.append(next_char)

    return generated


# ── Sidebar Modificado Radicalmente ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔌 CHRONOS DECK // CONFIG")
    st.markdown('<span class="cyber-badge">STATUS: ONLINE</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### [01] INGESTA DE PESOS")
    model_file    = st.file_uploader("Archivo Binario (.keras / .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("Vocabulario JSON (.json)", type=["json"])

    st.markdown("---")
    st.markdown("#### [02] MATRIZ TÉRMICA")
    temperature = st.slider(
        "Nivel de Entropía (T)", min_value=0.1, max_value=2.0,
        value=0.8, step=0.05
    )
    n_chars = st.slider("Longitud de Trama (Chars)", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("#### [03] BANCO DE SEMILLAS")
    seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "con estas razones perdia",
        "el hidalgo tomo la espada",
    ]
    selected_seed = st.selectbox("Semillas pre-cargadas:", ["(personalizada)"] + seeds)


# ── Cabecera Principal Desestructurada ────────────────────────────────────────
st.markdown("""
<div class="brutal-header">
    <div class="brutal-inner">
        <h1>CHRONOS-LSTM // 2006 CORE</h1>
        <p>[ SISTEMA AUTÓNOMO DE SÍNTESIS LITERARIA Y RECURRENCIA NEURONAL ]</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Organización en una Interfaz de Flujo Único (Sin Tabs Tradicionales) ──────

st.markdown('<div class="cyber-container">', unsafe_allow_html=True)
st.markdown("### ⚡ CONSOLA DE OPERACIÓN DE TEXTO")

# Distribución asimétrica de controles principales
col_left, col_right = st.columns([1.2, 1.8], gap="large")

with col_left:
    st.markdown("#### ⚙️ Parámetros de Entrada")
    default_text = selected_seed if selected_seed != "(personalizada)" else ""
    seed_input = st.text_area(
        "Inyección de Semilla Inicial:",
        value=default_text, height=140,
        placeholder="Escribe la frase base..."
    )
    
    execute_btn = st.execute_btn = st.button("🚀 EJECUTAR SECUENCIA NEURONAL", use_container_width=True)
    
    if model_file and metadata_file:
        st.success("STATUS: Modelo conectado externamente.")
    else:
        st.warning("STATUS: Modo Simulación Activa (Carga archivos en el sidebar).")

with col_right:
    st.markdown("#### 📺 Monitor CRT de Salida")
    terminal_output = st.empty()
    terminal_output.markdown(
        '<div class="crt-output">> Sistema en espera de estímulo... Ingrese texto semilla e inicie la secuencia.</div>',
        unsafe_allow_html=True
    )

if execute_btn:
    if not seed_input.strip():
        st.error("ERROR: El búfer de semilla está vacío.")
    elif model_file and metadata_file:
        ext = "keras" if model_file.name.endswith(".keras") else "h5"
        with open(f"/tmp/model.{ext}", "wb") as f:
            f.write(model_file.read())
        with open("/tmp/metadata.json", "wb") as f:
            f.write(metadata_file.read())

        model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

        if err:
            st.error(f"Fallo crítico al compilar: {err}")
        else:
            with st.spinner("PROCESANDO TENSORES..."):
                generated = generate_full_text(
                    model, seed_input,
                    meta["char_to_idx"], meta["idx_to_char"],
                    meta["seq_length"], meta["vocab_size"],
                    n_chars, temperature
                )
            terminal_output.markdown(
                f'<div class="crt-output">{generated}</div>',
                unsafe_allow_html=True
            )
    else:
        with st.spinner("SIMULANDO NÚCLEO..."):
            time.sleep(0.4)
        simulations = {
            0.3: "en un lugar de la mancha de cuyo nombre no quiero acordarme no ha mucho tiempo que vivia un hidalgo de la mancha con adarga antigua",
            0.8: "en un lugar de la mancha donde el viento golpea las rocas antiguas y el caballero levanta su espada hacia un cielo eléctrico",
            1.5: "en un lugar de la mancha xilofon cibernetico cruzo dimensiones paralelas mientras los molinos escupían codigo binario",
        }
        closest_t = min(simulations.keys(), key=lambda k: abs(k - temperature))
        terminal_output.markdown(
            f'<div class="crt-output">{simulations[closest_t]}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)


# ── Sección Inferior: Módulos de Información Desplegados ──────────────────────
col_sec1, col_sec2 = st.columns(2, gap="large")

with col_sec1:
    st.markdown('<div class="cyber-container">', unsafe_allow_html=True)
    st.markdown("### 📊 DIAGNÓSTICO DE RED")
    st.markdown("""
    * **Arquitectura:** Red Neuronal Recurrente (LSTM)
    * **Objetivo:** Predicción probabilística caracter a caracter ($P(s_{t} | s_{t-1})$).
    * **Optimización:** Entrenada para capturar dependencias de largo plazo evitando el desvanecimiento del gradiente.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col_sec2:
    st.markdown('<div class="cyber-container">', unsafe_allow_html=True)
    st.markdown("### 🎛️ CONTROL DE TEMPERATURA")
    st.markdown("""
    * **T Baja (< 0.5):** Comportamiento determinista, repetitivo y sumamente lógico.
    * **T Media (0.8 - 1.0):** Balance dinámico ideal para narrativa fluida.
    * **T Alta (> 1.2):** Alta entropía, generación experimental y alucinación creativa pura.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Pie terminal definitivo
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #ff007f; font-family: Courier New; font-size: 0.8rem;'>"
    "// CHRONOS-2006 // INTERFAZ MULTIMODAL Y AGENTES DE IA // STREAMLIT DEPLOY //"
    "</div>",
    unsafe_allow_html=True
)
