"""
AeroQuill™ - Neural Scriptor Studio 2006
Aplicación Streamlit - Estética Frutiger Aero / Y2K
"""

import streamlit as st
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="AeroQuill™ Studio 2006",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos Frutiger Aero / Y2K ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general estilo Windows Vista / Aero */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #a1c4fd 0%, #c2e9fb 50%, #d4fc79 100%);
        background-attachment: fixed;
    }

    /* Contenedores translúcidos tipo Glassmorphism / Aero */
    div.stMarkdown, .css-1r6slb0, div[data-testid="stVerticalBlock"] > div {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Título principal con estilo brillante de los 2000 */
    .main-title {
        font-size: 2.8rem; 
        font-weight: 800;
        font-family: 'Trebuchet MS', 'Lucida Sans Unicode', sans-serif;
        background: linear-gradient(180deg, #ffffff 0%, #2b74e4 50%, #003366 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        text-align: center;
        margin-bottom: 0px;
    }
    
    .subtitle { 
        color: #002b49; 
        font-size: 1.1rem; 
        text-align: center;
        font-weight: bold;
        text-shadow: 0px 1px 1px rgba(255,255,255,0.8);
        margin-bottom: 2rem;
    }

    /* Cajas de texto generado con estilo de burbuja brillante */
    .generated-text {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px; 
        padding: 1.8rem;
        font-family: 'Georgia', serif; 
        font-size: 1.1rem;
        line-height: 1.8; 
        color: #1a252c;
        border: 2px solid #ffffff;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2), inset 0 0 10px rgba(255, 255, 255, 0.8);
        min-height: 140px;
    }

    /* Cajas informativas estilo retro-futurista */
    .info-box {
        background: linear-gradient(135deg, rgba(230,244,255,0.9) 0%, rgba(190,227,248,0.9) 100%);
        border-radius: 12px; 
        padding: 1.2rem;
        border: 1px solid #7cb9e8;
        font-size: 0.9rem;
        color: #003366;
        box-shadow: inset 0 1px 3px rgba(255,255,255,0.8);
    }
    
    .badge-y2k {
        display: inline-block;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ── Funciones de Carga y Generación ───────────────────────────────────────────

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


# ── Sidebar Estilo Panel de Control 2006 ──────────────────────────────────────

with st.sidebar:
    st.markdown("### 🖥️ AeroQuill Control Center")
    st.markdown('<span class="badge-y2k">Vista v2.06 Edición</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### 📂 Carga de Núcleo (Modelo)")
    model_file    = st.file_uploader("Archivo de red (.keras / .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("Metadatos neuronales (.json)", type=["json"])

    st.markdown("---")
    st.markdown("#### 🎛️ Motor de Temperatura")

    temperature = st.slider(
        "Nivel de Creatividad (T)", min_value=0.1, max_value=2.0,
        value=0.8, step=0.05,
        help="Frío = Lógico y conservador | Caliente = Innovador y caótico"
    )

    if temperature < 0.5:
        st.caption("🧊 Modo Frío: Coherente, estructurado, repetitivo.")
    elif temperature < 1.0:
        st.caption("💧 Modo Templado: Óptimo balance natural.")
    elif temperature < 1.4:
        st.caption("🌿 Modo Cálido: Alta inventiva literaria.")
    else:
        st.caption("🔥 Modo Caótico: Generación experimental pura.")

    n_chars = st.slider("Longitud de salida (Caracteres)", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("#### 🌱 Semillas Históricas")
    seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "con estas razones perdia",
        "el hidalgo tomo la espada",
    ]
    selected_seed = st.selectbox("Banco de frases precargadas:", ["(personalizada)"] + seeds)

    st.markdown("---")
    st.markdown("""<div class="info-box">
    <b>Estado del Sistema:</b> Red LSTM entrenada con textos clásicos del Quijote. Optimizada para entornos de escritorio virtuales y web de alta velocidad.
    </div>""", unsafe_allow_html=True)


# ── Main / Interfaz Principal ─────────────────────────────────────────────────

st.markdown('<h1 class="main-title">AeroQuill™ Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Neural Scriptor & Text Synthesis Suite · Frutiger Aero Experience</p>', unsafe_allow_html=True)
st.markdown("---")

# Banner decorativo tipo panel retro de bienvenida
col_banner1, col_banner2, col_banner3 = st.columns(3)
col_banner1.metric("Entorno Gráfico", "AeroGlass™", "Activo")
col_banner2.metric("Motor Neuronal", "LSTM Recurrente", "v1.4")
col_banner3.metric("Conectividad Cloud", "Streamlit Hub", "Estable")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["✨ Estación de Escritura", "🌡️ Comparativa Térmica", "📜 Biblioteca Teórica"])

# ── Tab 1: Generar ────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### ✍️ Panel de Entrada Semilla")
        default = selected_seed if selected_seed != "(personalizada)" else ""
        seed_input = st.text_area(
            "Escribe o selecciona tu inspiración inicial:",
            value=default, height=120,
            placeholder="Ejemplo: en un lugar de la mancha...",
        )
        gen_btn = st.button("🚀 Iniciar Síntesis Neuronal", type="primary", use_container_width=True)

        if model_file and metadata_file:
            st.success("✔ Conectado a archivos de modelo personalizados.")
        else:
            st.warning("⚠️ Sin archivos subidos: El sistema operará en **Modo Demostración Simulada**.")

    with col2:
        st.markdown("### 🖥️ Pantalla de Salida")
        output = st.empty()
        output.markdown(
            '<div class="generated-text"><em style="color:#789">Esperando señal de procesamiento... Ingrese su semilla y presione iniciar.</em></div>',
            unsafe_allow_html=True
        )

    if gen_btn:
        if not seed_input.strip():
            st.error("⚠️ Error de sintaxis: Por favor escriba un texto semilla válido.")
        elif model_file and metadata_file:
            ext = "keras" if model_file.name.endswith(".keras") else "h5"
            with open(f"/tmp/model.{ext}", "wb") as f:
                f.write(model_file.read())
            with open("/tmp/metadata.json", "wb") as f:
                f.write(metadata_file.read())

            model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

            if err:
                st.error(f"Error crítico al cargar el modelo: {err}")
            else:
                char_to_idx = meta["char_to_idx"]
                idx_to_char = meta["idx_to_char"]
                vocab_size  = meta["vocab_size"]
                seq_length  = meta["seq_length"]

                with st.spinner("🔄 Procesando secuencias neuronales en la nube..."):
                    texto = generate_full_text(
                        model, seed_input,
                        char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars, temperature
                    )

                output.markdown(
                    f'<div class="generated-text">{texto}</div>',
                    unsafe_allow_html=True
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Caracteres", len(texto))
                m2.metric("Temperatura", f"{temperature:.2f}")
                m3.metric("Palabras Est.", len(texto.split()))
        else:
            # Demo interactiva sin modelo
            with st.spinner("🔄 Simulando red neuronal de época..."):
                time.sleep(0.6)
            demos = {
                0.3: "en un lugar de la mancha de cuyo nombre no quiero acordarme no ha mucho tiempo que vivia un hidalgo de la mancha",
                0.8: "en un lugar de la mancha donde el sol caia sobre las piedras del camino el caballero alzo su espada y miro al horizonte",
                1.5: "en un lugar de la mancha xilofon caballeo partio con su rocin hacia las montanas del sueno donde los gigantes bailaban",
            }
            closest = min(demos.keys(), key=lambda k: abs(k - temperature))
            output.markdown(
                f'<div class="generated-text">{demos[closest]}</div>',
                unsafe_allow_html=True
            )


# ── Tab 2: Comparar temperaturas ──────────────────────────────────────────────
with tab2:
    st.markdown("### 🌡️ Laboratorio de Espectro Térmico")
    st.markdown("Analiza cómo reacciona la creatividad del núcleo ante diferentes niveles de temperatura simultáneamente.")

    compare_seed = st.text_input("Frase base para comparativa:", value="en un lugar de la mancha de cuyo nombre")
    n_cmp = st.slider("Longitud de muestra", 50, 200, 100, key="ncmp")
    cmp_btn = st.button("⚡ Ejecutar Comparativa Global", type="primary")

    if cmp_btn and model_file and metadata_file:
        ext = "keras" if model_file.name.endswith(".keras") else "h5"
        with open(f"/tmp/model.{ext}", "wb") as f:
            f.write(model_file.getvalue())
        with open("/tmp/metadata.json", "wb") as f:
            f.write(metadata_file.getvalue())

        model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

        if model and meta:
            temps = [0.3, 0.6, 0.9, 1.2, 1.5]
            cols  = st.columns(len(temps))
            labels = ["Frío", "Templado 1", "Templado 2", "Caliente", "Extremo"]

            with st.spinner("Generando matriz de muestras..."):
                for col, temp, label in zip(cols, temps, labels):
                    text = generate_full_text(
                        model, compare_seed,
                        meta["char_to_idx"], meta["idx_to_char"],
                        meta["seq_length"], meta["vocab_size"],
                        n_cmp, temp
                    )
                    with col:
                        st.markdown(f"**T={temp} ({label})**")
                        st.markdown(
                            f'<div style="background:rgba(255,255,255,0.9); border-radius:10px; padding:0.8rem; font-family:Georgia,serif; font-size:0.85rem; line-height:1.6; min-height:160px; border-top:4px solid #0072ff; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">{text}</div>',
                            unsafe_allow_html=True
                        )
    elif cmp_btn:
        st.warning("⚠️ Carga tus archivos de modelo en el panel izquierdo para habilitar la comparativa real.")


# ── Tab 3: Teoria ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📚 Manual de Arquitectura y Teoría Neuronal (Edición 2006)")

    with st.expander("📖 ¿Qué es una Red Neuronal Recurrente (RNN)?", expanded=True):
        st.markdown("""
Una **RNN** procesa secuencias conservando un estado oculto que funciona como memoria a corto plazo:
