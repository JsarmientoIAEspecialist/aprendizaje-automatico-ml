"""
ingestar.py — Construye la base de conocimiento de Lex (se ejecuta UNA vez).

Pipeline (pasos 1 a 4 de RAG):
    1. Carga el PDF del Código Nacional de Tránsito.
    2. Lo divide en fragmentos (chunks).
    3. Genera embeddings locales de cada fragmento.
    4. Los almacena en una base vectorial ChromaDB persistente en disco.

Uso:
    python ingestar.py
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core import (
    EMBEDDING_MODEL,
    PERSIST_DIR,
    cargar_embeddings,
    guardar_vector_store,
)

# Importamos FAISS aquí porque solo la ingesta crea el índice.
from langchain_community.vectorstores import FAISS

PDF_PATH = os.path.join(os.path.dirname(__file__), "pdf", "codigo_nacional_transito.pdf")


def main() -> None:
    print("=" * 65)
    print("  LEX — Ingesta del Código Nacional de Tránsito")
    print("=" * 65)

    # --- PASO 1: Cargar el documento PDF -----------------------------------
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"No se encontró el PDF en: {PDF_PATH}\n"
            "Coloca el archivo 'codigo_nacional_transito.pdf' dentro de la carpeta 'pdf/'."
        )

    print(f"\n[1/4] Cargando PDF: {os.path.basename(PDF_PATH)}")
    documentos = PyPDFLoader(PDF_PATH).load()
    print(f"      Páginas cargadas: {len(documentos)}")

    # Verificación de capa de texto: si el PDF es un escaneo, no habrá texto.
    texto_total = sum(len(d.page_content.strip()) for d in documentos)
    if texto_total == 0:
        raise RuntimeError(
            "El PDF no contiene texto extraíble (parece un escaneo en imagen). "
            "Consigue una versión del documento con capa de texto."
        )
    print(f"      Caracteres de texto detectados: {texto_total:,}")

    # --- PASO 2: Dividir en fragmentos (chunking) --------------------------
    print("\n[2/4] Dividiendo en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"      Fragmentos generados: {len(chunks)}")
    print(f"      Factor de expansión:  {len(chunks) / len(documentos):.1f}x")

    # --- PASO 3: Generar embeddings locales --------------------------------
    print(f"\n[3/4] Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    print("      (la primera vez se descargan ~120 MB; corre en CPU)")
    embeddings = cargar_embeddings()

    # --- PASO 4: Almacenar en la base vectorial FAISS ----------------------
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"\n[4/4] Índice anterior eliminado: {PERSIST_DIR}")
    else:
        print("\n[4/4] Creando base vectorial nueva...")

    print(f"      Indexando {len(chunks)} fragmentos en FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persistir el índice a disco (genera index.faiss + index.pkl).
    guardar_vector_store(vector_store)

    total = vector_store.index.ntotal

    # Consulta de prueba para verificar que el índice quedó utilizable.
    prueba = vector_store.similarity_search("licencia de conducción", k=1)
    print(f"      Índice verificado (consulta de prueba devolvió {len(prueba)} resultado).")

    print("\n" + "=" * 65)
    print("  [OK] Base de conocimiento construida correctamente")
    print("=" * 65)
    print(f"  Ubicación:            {PERSIST_DIR}")
    print(f"  Fragmentos indexados: {total}")
    print(f"  Modelo embeddings:    {EMBEDDING_MODEL} (local, CPU, 384 dim)")
    print("\n  Ahora puedes lanzar el chat con:  streamlit run app.py")


if __name__ == "__main__":
    main()
