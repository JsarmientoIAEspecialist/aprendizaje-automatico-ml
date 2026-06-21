"""
app.py — Interfaz de chat de "Lex" (Streamlit).

Pasos 5 a 7 de RAG + bucle conversacional:
    5. Recupera los fragmentos relevantes desde ChromaDB.
    6. Construye el prompt aumentado (contexto + pregunta).
    7. Groq genera la respuesta basada SOLO en ese contexto.
Al final de cada respuesta se citan las páginas consultadas.

Uso:
    streamlit run app.py
(Requiere haber ejecutado antes:  python ingestar.py)
"""

import os

import streamlit as st

from rag_core import (
    DEFAULT_K,
    LLM_MODEL,
    PERSIST_DIR,
    cargar_embeddings,
    cargar_llm,
    cargar_vector_store,
    es_respuesta_sin_info,
    responder,
)

st.set_page_config(page_title="Lex — Asistente del Código de Tránsito", page_icon="⚖️")


# --- Carga de recursos (cacheada para no recargar en cada interacción) ------
@st.cache_resource(show_spinner="Cargando modelo y base de conocimiento...")
def inicializar():
    embeddings = cargar_embeddings()
    vector_store = cargar_vector_store(embeddings)
    llm = cargar_llm()
    return vector_store, llm


# --- Cabecera ---------------------------------------------------------------
st.title("⚖️ Lex")
st.caption(
    "Asistente legal especializado en el **Código Nacional de Tránsito de Colombia** "
    "(Ley 769 de 2002). Responde únicamente con el contenido del documento y cita las "
    "páginas consultadas."
)

# Si no existe la base vectorial, avisar y detener.
if not os.path.exists(os.path.join(PERSIST_DIR, "index.faiss")):
    st.error(
        "⚠️ La base de conocimiento no existe todavía.\n\n"
        "Ejecuta primero en la terminal:  `python ingestar.py`"
    )
    st.stop()

try:
    vector_store, llm = inicializar()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# --- Barra lateral ----------------------------------------------------------
with st.sidebar:
    st.header("Sobre Lex")
    st.markdown(
        "- **Documento:** Código Nacional de Tránsito (Ley 769 de 2002)\n"
        "- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (local)\n"
        "- **LLM:** Llama 3.3 70B (vía HuggingFace Router)\n"
        "- **Base vectorial:** FAISS"
    )
    k = st.slider("Fragmentos a recuperar (k)", min_value=3, max_value=10, value=DEFAULT_K)
    st.divider()
    st.markdown(
        "**Ejemplos de preguntas:**\n"
        "- ¿Qué es la inmovilización de un vehículo?\n"
        "- ¿Cuáles son las causales de suspensión de la licencia?\n"
        "- ¿Qué documentos debo portar para conducir?"
    )
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

# --- Historial de la conversación (bucle conversacional) --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("fragmentos"):
            with st.expander("🔍 Fragmentos consultados (trazabilidad)"):
                for i, frag in enumerate(msg["fragmentos"], 1):
                    pag = frag["page"]
                    st.markdown(f"**[{i}] Pág. {pag}**")
                    st.caption(frag["texto"])

# Entrada del usuario.
if pregunta := st.chat_input("Escribe tu pregunta sobre el Código de Tránsito..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Lex está consultando el Código de Tránsito..."):
            resultado = responder(pregunta, vector_store, llm, k=k)

        respuesta = resultado["respuesta"]
        paginas = resultado["paginas"]

        # Citar páginas al final (requisito de la actividad).
        if paginas and not es_respuesta_sin_info(respuesta):
            cita = ", ".join(str(p) for p in paginas)
            respuesta_final = f"{respuesta}\n\n---\n📄 **Páginas consultadas:** {cita}"
        else:
            respuesta_final = respuesta

        st.markdown(respuesta_final)

        # Guardar fragmentos en forma serializable para el historial.
        # La página se muestra 1-indexada (PyPDFLoader la numera desde 0),
        # igual que en la cita "Páginas consultadas".
        def _pag(f):
            p = f.metadata.get("page")
            return p + 1 if isinstance(p, int) else "?"

        fragmentos_ser = [
            {
                "page": _pag(f),
                "texto": f.page_content[:400] + ("..." if len(f.page_content) > 400 else ""),
            }
            for f in resultado["fragmentos"]
        ]
        with st.expander("🔍 Fragmentos consultados (trazabilidad)"):
            for i, frag in enumerate(fragmentos_ser, 1):
                st.markdown(f"**[{i}] Pág. {frag['page']}**")
                st.caption(frag["texto"])

    st.session_state.messages.append(
        {"role": "assistant", "content": respuesta_final, "fragmentos": fragmentos_ser}
    )
