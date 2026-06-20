"""
rag_core.py — Lógica compartida del sistema RAG "Lex".

Aquí viven las piezas reutilizables por `ingestar.py` (que construye la base
vectorial) y por `app.py` (la interfaz de chat). Mantener todo en un solo lugar
evita duplicar configuración y garantiza que la ingesta y la consulta usen
EXACTAMENTE el mismo modelo de embeddings.
"""

import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# --- Configuración central (un solo lugar para cambiar parámetros) ----------

# Modelo de embeddings local (CPU, multilingüe, ~120 MB la primera vez).
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Carpeta donde FAISS persiste el índice vectorial en disco (index.faiss + index.pkl).
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
_INDEX_FILE = "index.faiss"   # índice vectorial (FAISS)
_STORE_FILE = "index.pkl"     # textos + metadatos (docstore)

# LLM vía el router de inferencia de HuggingFace (API compatible con OpenAI).
# El sufijo ":groq" hace que HF enrute la petición a Groq (llama-3.3-70b-versatile).
HF_ROUTER_URL = "https://router.huggingface.co/v1"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct:groq"

# Cuántos fragmentos recuperar por pregunta (más fragmentos = mejor recall).
DEFAULT_K = 8

# Identidad del asistente (nombre + rol) y reglas de comportamiento.
PROMPT_TEMPLATE = """Eres Lex, un asistente legal especializado en el Código Nacional \
de Tránsito de Colombia (Ley 769 de 2002).

Reglas que debes seguir SIEMPRE:
1. Responde ÚNICAMENTE con la información del contexto proporcionado más abajo.
2. Si la respuesta no está en el contexto, responde EXACTAMENTE: \
"No encontré información sobre esto en el Código Nacional de Tránsito."
3. No inventes datos, artículos ni cifras que no aparezcan en el contexto.
4. Responde en español, de forma clara y concisa, y cuando sea posible menciona \
el número de artículo.
5. NO menciones la palabra "fragmento" ni números de fragmento en tu respuesta; \
escribe como si hablaras directamente con el usuario.

Contexto recuperado del documento:
{context}

Pregunta del usuario: {question}

Respuesta de Lex:"""


def cargar_embeddings() -> HuggingFaceEmbeddings:
    """Carga el modelo de embeddings local (mismo en ingesta y consulta)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def guardar_vector_store(vector_store: FAISS) -> None:
    """Persiste el índice FAISS a disco de forma segura en Windows.

    No usamos `FAISS.save_local()` porque internamente llama a
    `faiss.write_index()` (C++), que NO admite rutas con caracteres no-ASCII en
    Windows (y este proyecto vive bajo ".../Aprendizaje automático..."). En su
    lugar serializamos el índice a bytes y lo escribimos con `open()` de Python,
    que sí maneja rutas Unicode.
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)
    indice_bytes = faiss.serialize_index(vector_store.index).tobytes()
    with open(os.path.join(PERSIST_DIR, _INDEX_FILE), "wb") as f:
        f.write(indice_bytes)
    with open(os.path.join(PERSIST_DIR, _STORE_FILE), "wb") as f:
        pickle.dump((vector_store.docstore, vector_store.index_to_docstore_id), f)


def cargar_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Abre el índice vectorial ya construido en disco (formato propio)."""
    with open(os.path.join(PERSIST_DIR, _INDEX_FILE), "rb") as f:
        indice = faiss.deserialize_index(np.frombuffer(f.read(), dtype=np.uint8).copy())
    with open(os.path.join(PERSIST_DIR, _STORE_FILE), "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    return FAISS(
        embedding_function=embeddings,
        index=indice,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def cargar_llm() -> ChatOpenAI:
    """Configura el LLM vía el router de HuggingFace (compatible con OpenAI).

    Requiere HF_TOKEN en el entorno/.env.
    """
    load_dotenv()
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError(
            "Falta HF_TOKEN. Copia .env.example a .env y pega tu token de HuggingFace "
            "(lo obtienes gratis en https://huggingface.co/settings/tokens)."
        )
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        api_key=api_key,
        base_url=HF_ROUTER_URL,
    )


def _formatear_contexto(fragmentos) -> str:
    """Une los fragmentos recuperados etiquetando la página de cada uno."""
    return "\n\n---\n\n".join(
        f"[Fragmento {i + 1} — Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(fragmentos)
    )


def _paginas_consultadas(fragmentos) -> list[int]:
    """Páginas únicas (1-indexadas) de los fragmentos usados, ordenadas.

    PyPDFLoader numera las páginas desde 0, por eso sumamos 1 para mostrar el
    número real de página del PDF.
    """
    paginas = set()
    for doc in fragmentos:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            paginas.add(page + 1)
    return sorted(paginas)


def responder(pregunta: str, vector_store: FAISS, llm: ChatOpenAI, k: int = DEFAULT_K) -> dict:
    """Ejecuta el ciclo RAG completo para una pregunta.

    Devuelve un dict con la respuesta del LLM, las páginas consultadas
    (para citarlas) y los fragmentos recuperados (para trazabilidad).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    fragmentos = retriever.invoke(pregunta)

    contexto = _formatear_contexto(fragmentos)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    mensaje = prompt.invoke({"context": contexto, "question": pregunta})
    respuesta = llm.invoke(mensaje).content

    return {
        "respuesta": respuesta,
        "paginas": _paginas_consultadas(fragmentos),
        "fragmentos": fragmentos,
    }
