"""
CYBER-VISTA 2006 // SYSTEM CORE: NEURAL-PULSE
Plataforma de Creación e Inferencia de Textos LSTM
"""

import streamlit as st
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="CYBER-VISTA 2006 // NEURAL-PULSE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Estética Radical: Cyber-Y2K & Frutiger Aero (Reestructuración Total) ────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1128 0%, #1c3144 50%, #0077b6 100%);
        background-attachment: fixed;
    }
    
    html, body, [class*="css"] {
        font-family: 'Verdana', Geneva, sans-serif;
        color: #e0fbfc;
    }

    /* Ocultar elementos estándar innecesarios y rediseñar cabecera absoluta */
    header {visibility: hidden;}
    
    .hyper-header {
        background: rgba(10, 25, 47, 0.85);
        border: 2px solid #00f5d4;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 0 25px rgba(0, 245, 212, 0.4), inset 0 0 15px rgba(0, 245, 212, 0.2);
        text-align: center;
        margin-bottom: 20px;
    }

    .hyper-header h1 {
        font-family: 'Impact', sans-serif;
        font-size: 3rem;
        color: #00f5d4;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
        text-shadow: 2px 2px 0px #ff007f, 4px 4px 10px rgba(0,0,0,0.8);
    }

    .hyper-header p {
        color: #98c1d9;
        font-size: 1rem;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* Paneles modulares flotantes estilo HUD futurista 2006 */
    .hud-panel {
        background: rgba(27, 38, 59, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid #3d5a80;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        margin-bottom: 15px;
    }

    .hud-panel h3 {
        color: #ff007f;
        font-size: 1.2rem;
        border-bottom: 1px dashed #3d5a80;
        padding-bottom: 6px;
        margin-top: 0;
    }

    /* Caja de salida de texto estilo terminal / visor holográfico */
    .terminal-output {
        background: #020817;
        border: 2px solid #00f5d4;
        border-radius: 10px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 1.15rem;
        color: #00f5d4;
        line-height: 1.6;
        min-height: 220px;
        box-shadow: inset 0 0 15px rgba(0, 245, 212, 0.2);
    }

    /* Botones de acción total */
    .stButton>button {
        background: linear-gradient(90deg, #ff007f, #7209b7);
        color: white;
        border: 2px solid #f72585;
        border-radius: 8px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.4);
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #7209b7, #00f5d4);
        border-color: #00f5d4;
        color: #020817;
    }
</style>
""", unsafe_allow_html=True)


# ── Funciones de Procesamiento ────────────────────────────────────────────────

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


# ── Cabecera Única y Radical ──────────────────────────────────────────────────
st.markdown("""
<div class="hyper-header">
    <h1>⚡ CYBER-VISTA 2006 // NEURAL-PULSE ⚡</h1>
    <p>SISTEMA CENTRAL DE SÍNTESIS LITERARIA RECURRENTE (MOTOR LSTM v4.2)</p>
</div>
""", unsafe_allow_html=True)


# ── Estructura de Control Unificada en Pantalla Principal (Sin Sidebar Tradicional) ──
col_top1, col_top2, col_top3 = st.columns([1, 1, 1])

with col_top1:
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown("### 📂 1. Ingesta de Núcleo")
    mod_file = st.file_uploader("Modelo Neuronal (.keras / .h5)", type=["keras", "h5"])
    meta_file = st.file_uploader("Metadatos de Vocabulario (.json)", type=["json"])
    st.markdown('</div>', unsafe_allow_html=True)

with col_top2:
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown("### 🎛️ 2. Parámetros Cuánticos")
    temp_val = st.slider("Temperatura de Creatividad", 0.1, 2.0, 0.8, 0.05)
    length_val = st.slider("Caracteres de Salida", 50, 400, 200, 50)
    st.markdown('</div>', unsafe_allow_html=True)

with col_top3:
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown("### 🌱 3. Semilla de Arranque")
    preset_seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "el hidalgo tomo la espada"
    ]
    chosen_seed = st.selectbox("Banco de frases:", ["(Personalizada)"] + preset_seeds)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Área de Interacción Dinámica Central ──────────────────────────────────────
st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
c_left, c_right = st.columns([1, 1], gap="large")

with c_left:
    st.markdown("### ✍️ Consola de Entrada de Texto")
    default_text = chosen_seed if chosen_seed != "(Personalizada)" else ""
    user_seed = st.text_area("Escribe tu premisa literaria:", value=default_text, height=140)
    trigger_gen = st.button("🚀 DISPARAR INFERENCIA NEURONAL", use_container_width=True)

with c_right:
    st.markdown("### 🖥️ Visor de Resultados Holográfico")
    visor_slot = st.empty()
    visor_slot.markdown(
        '<div class="terminal-output">> [SISTEMA EN ESPERA DE COMANDO...] <br>> Ingrese parámetros y ejecute el núcleo para iniciar la secuencia.</div>',
        unsafe_allow_html=True
    )

if trigger_gen:
    if not user_seed.strip():
        st.error("⚠️ ERROR: La semilla de entrada no puede estar vacía.")
    elif mod_file and meta_file:
        ext = "keras" if mod_file.name.endswith(".keras") else "h5"
        with open(f"/tmp/model.{ext}", "wb") as f:
            f.write(mod_file.read())
        with open("/tmp/metadata.json", "wb") as f:
            f.write(meta_file.read())

        net_model, metadata_content, load_err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

        if load_err:
            st.error(f"Error crítico en el modelo: {load_err}")
        else:
            with st.spinner("⚡ Procesando tensores de red recurrente..."):
                final_output = generate_full_text(
                    net_model, user_seed,
                    metadata_content["char_to_idx"], metadata_content["idx_to_char"],
                    metadata_content["seq_length"], metadata_content["vocab_size"],
                    length_val, temp_val
                )
            visor_slot.markdown(f'<div class="terminal-output">{final_output}</div>', unsafe_allow_html=True)
    else:
        with st.spinner("⚡ Simulando flujo de red neuronal 2006..."):
            time.sleep(0.4)
        sim_data = {
            0.3: "en un lugar de la mancha de cuyo nombre no quiero acordarme no ha mucho tiempo que vivia un hidalgo de los de lanza en astillero",
            0.8: "en un lugar de la mancha donde el sol calienta los campos de batalla el caballero alzo su espada mirando el horizonte infinito",
            1.5: "en un lugar de la mancha xilofon cosmico navego velozmente hacia las altas cumbres de la fantasia digital"
        }
        matched_key = min(sim_data.keys(), key=lambda k: abs(k - temp_val))
        visor_slot.markdown(f'<div class="terminal-output">{sim_data[matched_key]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Bloque Informativo Inferior (Sección de Documentación Interactiva) ──────────
st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
st.markdown("### 📊 TELEMETRÍA Y ARQUITECTURA DEL SISTEMA")
tab_a, tab_b, tab_c = st.tabs(["🧠 ¿Qué es una RNN?", "⚙️ Ventajas LSTM", "🌡️ Control de Temperatura"])

with tab_a:
    st.markdown("Las Redes Neuronales Recurrentes (RNN) permiten procesar secuencias de datos manteniendo un ciclo de retroalimentación en su estado oculto, ideal para modelar lenguaje natural.")

with tab_b:
    st.markdown("Las celdas LSTM resuelven el problema del desvanecimiento del gradiente mediante tres compuertas lógicas especializadas (Olvido, Entrada y Salida), permitiendo retener dependencias a largo plazo.")

with tab_c:
    st.markdown("La temperatura altera la distribución de probabilidad de la capa softmax. Valores bajos generan respuestas conservadoras y lógicas; valores altos potencian la creatividad caótica.")

st.markdown('</div>', unsafe_allow_html=True)
