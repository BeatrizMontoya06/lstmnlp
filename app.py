"""
NEO-AERO LABS™ // Generador Cuántico Quijote-LSTM 2006
Aplicación Streamlit - Estética Frutiger Aero / Web Retro 2000s
"""

import streamlit as st
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="NEO-AERO LABS™ // LSTM Studio 2006",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos CSS Extremos: Frutiger Aero, Glossy, Y2K & Retro 2000s ─────────────
st.markdown("""
<style>
    /* Fondo con degradado tridimensional clásico de Windows Vista / Frutiger Aero */
    .stApp {
        background: radial-gradient(circle at 50% 20%, #dff0fe 0%, #8ec5fc 40%, #e0c3fc 100%);
        background-attachment: fixed;
    }

    /* Tipografías generales limpias pero con estilo system */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0b2545;
    }

    /* Cabecera principal estilo Banner Retro 2000s */
    .retro-header {
        background: linear-gradient(180deg, #ffffff 0%, #b8d9fb 50%, #4a90e2 100%);
        border: 2px solid #ffffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 51, 102, 0.25), inset 0 2px 5px rgba(255, 255, 255, 0.9);
        margin-bottom: 25px;
    }

    .retro-header h1 {
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 2.4rem;
        font-weight: 900;
        background: linear-gradient(180deg, #004080 0%, #001f3f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 3px rgba(255, 255, 255, 0.8);
        margin: 0;
    }

    .retro-header p {
        color: #133863;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 5px;
        letter-spacing: 0.5px;
    }

    /* Tarjetas de contenido con estilo de cristal brillante (Glassmorphism Aero) */
    .aero-card {
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 20px;
        border: 2px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15), inset 0 0 12px rgba(255, 255, 255, 0.7);
        margin-bottom: 20px;
    }

    /* Caja para el texto generado con aspecto de documento antiguo/futurista */
    .generated-output-box {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        border-radius: 12px;
        padding: 20px;
        font-family: 'Georgia', serif;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #1d2d44;
        border: 2px inset #b0c4de;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.08);
        min-height: 160px;
    }

    /* Badges / Etiquetas flotantes Y2K */
    .badge-status {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Marquesina simulada inferior */
    .marquee-footer {
        background: #002b5b;
        color: #76c7c0;
        padding: 6px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        text-align: center;
        border: 1px solid #4a90e2;
        box-shadow: inset 0 0 8px rgba(0,0,0,0.5);
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


# ── Barra Lateral (Sidebar) - Panel de Control Técnico ────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ NeoAero Control Panel")
    st.markdown('<span class="badge-status">Sistema Activo v2.6</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### 📂 Ingesta de Pesos Neuronales")
    model_file    = st.file_uploader("Cargar Red LSTM (.keras / .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("Cargar Vocabulario (.json)", type=["json"])

    st.markdown("---")
    st.markdown("#### 🌡️ Matriz de Temperatura")
    temperature = st.slider(
        "Nivel de Caos / Creatividad", min_value=0.1, max_value=2.0,
        value=0.8, step=0.05,
        help="Valores bajos = Lógico/Literal | Valores altos = Experimental/Alucinatorio"
    )

    if temperature < 0.5:
        st.caption("🧊 Modo Estricto: Predicciones altamente probables y repetitivas.")
    elif temperature < 1.0:
        st.caption("💧 Modo Equilibrado: Coherencia literaria natural.")
    elif temperature < 1.4:
        st.caption("🌿 Modo Creativo: Variaciones estilísticas avanzadas.")
    else:
        st.caption("🔥 Modo Psicodélico: Generación caótica de alto riesgo.")

    n_chars = st.slider("Bloque de caracteres de salida", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("#### 📜 Semillas de Arranque")
    seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "con estas razones perdia",
        "el hidalgo tomo la espada",
    ]
    selected_seed = st.selectbox("Banco de inspiración rápida:", ["(personalizada)"] + seeds)

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.7); padding: 10px; border-radius: 8px; font-size: 0.8rem; border: 1px solid #b0c4de;">
    <b>Utilidad del Sistema:</b> Estación de pruebas para procesamiento de lenguaje natural basada en secuencias temporales recurrentes.
    </div>
    """, unsafe_allow_html=True)


# ── Cabecera Principal de la Página (Con Identidad y Propósito) ────────────────
st.markdown("""
<div class="retro-header">
    <h1>🌐 NEO-AERO LABS™ // LSTM Scriptor Studio</h1>
    <p>Plataforma Interactiva de Inteligencia Artificial para la Síntesis de Texto Neuronal Carácter a Carácter</p>
</div>
""", unsafe_allow_html=True)

# Mini barra de estado con widgets llenos de cosas útiles
col_w1, col_w2, col_w3, col_w4 = st.columns(4)
col_w1.metric("Arquitectura", "Red LSTM", "Deep Learning")
col_w2.metric("Dataset Base", "El Quijote", "1605 Classics")
col_w3.metric("Interfaz", "Frutiger Aero", "Glass Theme")
col_w4.metric("Conexión Cloud", "Streamlit Hub", "Online")

st.markdown("<br>", unsafe_allow_html=True)

# Pestañas principales cargadas de funciones
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Estación de Síntesis", 
    "🧪 Laboratorio Térmico", 
    "📊 Panel de Diagnóstico", 
    "📖 Manual del Sistema"
])

# ── Tab 1: Estación de Síntesis (Principal) ────────────────────────────────────
with tab1:
    st.markdown('<div class="aero-card">', unsafe_allow_html=True)
    col_in1, col_in2 = st.columns([1, 1], gap="large")

    with col_in1:
        st.markdown("### ✍️ 1. Configuración de Semilla")
        default_val = selected_seed if selected_seed != "(personalizada)" else ""
        seed_input = st.text_area(
            "Introduce el texto inicial para guiar a la red neuronal:",
            value=default_val, height=130,
            placeholder="Ejemplo: en un lugar de la mancha de cuyo nombre...",
        )
        gen_action = st.button("⚡ Ejecutar Generación Neuronal", type="primary", use_container_width=True)

        if model_file and metadata_file:
            st.success("✔ Conectado a archivos personalizados externos.")
        else:
            st.info("💡 Sugerencia: Carga tus pesos en el panel izquierdo para procesamiento real, de lo contrario funcionará en modo simulación interactiva.")

    with col_in2:
        st.markdown("### 🖥️ 2. Consola de Resultados")
        output_slot = st.empty()
        output_slot.markdown(
            '<div class="generated-output-box"><em style="color:#666">Esperando comandos de entrada en la consola... Haga clic en ejecutar.</em></div>',
            unsafe_allow_html=True
        )

    if gen_action:
        if not seed_input.strip():
            st.error("⚠️ Error crítico: El campo de semilla no puede estar vacío.")
        elif model_file and metadata_file:
            ext = "keras" if model_file.name.endswith(".keras") else "h5"
            with open(f"/tmp/model.{ext}", "wb") as f:
                f.write(model_file.read())
            with open("/tmp/metadata.json", "wb") as f:
                f.write(metadata_file.read())

            model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

            if err:
                st.error(f"Error al compilar el modelo: {err}")
            else:
                char_to_idx = meta["char_to_idx"]
                idx_to_char = meta["idx_to_char"]
                vocab_size  = meta["vocab_size"]
                seq_length  = meta["seq_length"]

                with st.spinner("🔄 Procesando tensores y calculando probabilidades..."):
                    generated_text = generate_full_text(
                        model, seed_input,
                        char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars, temperature
                    )

                output_slot.markdown(
                    f'<div class="generated-output-box">{generated_text}</div>',
                    unsafe_allow_html=True
                )

                metric_c1, metric_c2, metric_c3 = st.columns(3)
                metric_c1.metric("Longitud Generada", f"{len(generated_text)} chars")
                metric_c2.metric("Temperatura Aplicada", f"{temperature:.2f}")
                metric_c3.metric("Palabras Estimadas", len(generated_text.split()))
        else:
            with st.spinner("🔄 Simulando red neuronal de alta velocidad..."):
                time.sleep(0.5)
            simulated_responses = {
                0.3: "en un lugar de la mancha de cuyo nombre no quiero acordarme no ha mucho tiempo que vivia un hidalgo de los de lanza en astillero",
                0.8: "en un lugar de la mancha donde el sol calienta los campos polvorientos el caballero empuno su adarga antigua mirando el horizonte",
                1.5: "en un lugar de la mancha xilofon fantastico corrio velozmente hacia las altas cumbres mientras los molinos cantaban canciones de cristal",
            }
            closest_key = min(simulated_responses.keys(), key=lambda k: abs(k - temperature))
            output_slot.markdown(
                f'<div class="generated-output-box">{simulated_responses[closest_key]}</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 2: Laboratorio Térmico (Comparativa simultánea) ────────────────────────
with tab2:
    st.markdown('<div class="aero-card">', unsafe_allow_html=True)
    st.markdown("### 🌡️ Comparativa de Espectro Térmico Multivariable")
    st.markdown("Evalúa de manera simultánea cómo reacciona el motor neuronal ante 5 temperaturas distintas usando la misma semilla.")

    cmp_seed_input = st.text_input("Frase base para la comparativa térmica:", value="en un lugar de la mancha de cuyo nombre")
    cmp_length = st.slider("Longitud de muestra por hilo", 40, 150, 90, key="slider_cmp_len")
    run_cmp_btn = st.button("⚡ Ejecutar Banco de Pruebas Térmicas", type="primary")

    if run_cmp_btn and model_file and metadata_file:
        ext = "keras" if model_file.name.endswith(".keras") else "h5"
        with open(f"/tmp/model.{ext}", "wb") as f:
            f.write(model_file.getvalue())
        with open("/tmp/metadata.json", "wb") as f:
            f.write(metadata_file.getvalue())

        model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

        if model and meta:
            test_temps = [0.3, 0.6, 0.9, 1.2, 1.5]
            grid_cols = st.columns(len(test_temps))
            labels_desc = ["Frío (0.3)", "Templado (0.6)", "Ideal (0.9)", "Caliente (1.2)", "Extremo (1.5)"]

            with st.spinner("Ejecutando matriz de inferencia paralela..."):
                for col_elem, t_val, lbl in zip(grid_cols, test_temps, labels_desc):
                    output_text = generate_full_text(
                        model, cmp_seed_input,
                        meta["char_to_idx"], meta["idx_to_char"],
                        meta["seq_length"], meta["vocab_size"],
                        cmp_length, t_val
                    )
                    with col_elem:
                        st.markdown(f"**{lbl}**")
                        st.markdown(
                            f'<div style="background:rgba(255,255,255,0.9); border-radius:10px; padding:10px; font-family:Georgia,serif; font-size:0.8rem; line-height:1.5; min-height:180px; border-top:4px solid #0072ff; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">{output_text}</div>',
                            unsafe_allow_html=True
                        )
    elif run_cmp_btn:
        st.warning("⚠️ Debe cargar los archivos de pesos del modelo (.keras/.json) en la barra lateral para activar la comparativa real.")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 3: Panel de Diagnóstico ────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="aero-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Panel de Control y Estadísticas del Núcleo")
    st.markdown("Monitoreo en tiempo real de los parámetros de hardware virtual asignados a la sesión actual.")

    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Memoria RAM Asignada", "512 MB", "Virtualizada")
    d_col2.metric("Estado del Servidor Cloud", "Streamlit Host", "99.9% Uptime")
    d_col3.metric("Latencia de Inferencia", "12 ms", "Óptima")

    st.markdown("---")
    st.markdown("#### ⚙️ Registro de Actividad del Sistema")
    st.code("""
[INFO] 2006-10-29 23:00:12 - Inicializando entorno virtual Frutiger Aero...
[INFO] 2006-10-29 23:00:13 - Cargando librerías base de TensorFlow y Keras Backend.
[SUCCESS] 2006-10-29 23:00:15 - Módulos de interfaz gráfica renderizados correctamente.
[READY] Sistema preparado para recibir flujos de texto por lotes.
    """, language="text")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 4: Manual del Sistema ──────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="aero-card">', unsafe_allow_html=True)
    st.markdown("### 📖 Documentación Técnica y Fundamentos Teóricos")

    with st.expander("1. ¿Qué es una Red Neuronal Recurrente (RNN)?", expanded=True):
        st.markdown(
            "Una Red Neuronal Recurrente procesa datos secuenciales manteniendo un estado oculto interno "
            "que actúa como memoria a corto plazo, permitiendo conectar información previa con los eventos actuales."
        )

    with st.expander("2. ¿Cómo mitigan las celdas LSTM el desvanecimiento de gradiente?"):
        st.markdown(
            "Las redes LSTM introducen celdas de memoria protegidas por compuertas especializadas:\n"
            "- **Puerta de Olvido:** Filtra y elimina datos irrelevantes del pasado.\n"
            "- **Puerta de Entrada:** Selecciona qué información nueva vale la pena almacenar.\n"
            "- **Puerta de Salida:** Regula qué parte del estado interno se proyecta como resultado."
        )

    with st.expander("3. Propósito de la Temperatura en el Muestreo Softmax"):
        st.markdown(
            "La función de temperatura modifica la escala de logits antes de la distribución de probabilidad, "
            "permitiendo modular el grado de predictibilidad y originalidad en el texto generado."
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página con estilo estético retro / marquesina
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="marquee-footer">
    🌟 NEO-AERO LABS™ // Desarrollado para Interfaces Multimodales y Agentes de Inteligencia Artificial (Edición 2006) 🌟
</div>
""", unsafe_allow_html=True)
