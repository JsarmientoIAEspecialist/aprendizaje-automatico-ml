# ⚖️ Lex — Asistente RAG sobre el Código Nacional de Tránsito

Asistente conversacional que responde, en lenguaje natural, preguntas sobre el
**Código Nacional de Tránsito de Colombia (Ley 769 de 2002)**. Las respuestas se
fundamentan **únicamente** en el texto del documento y **citan las páginas
consultadas**, de modo que toda respuesta es verificable.

> **Asistente:** *Lex* — asistente legal especializado en el Código Nacional de Tránsito.

---

## 👤 Autor y fecha

- **Estudiante:** Juan Sebastián Sarmiento Aunta
- **Curso:** Aprendizaje Automático (Machine Learning)
- **Actividad:** Sistema RAG sobre documento especializado
- **Fecha:** 19 de junio de 2026

---

## 📄 Documento seleccionado y justificación

**Documento:** Código Nacional de Tránsito de Colombia — **Ley 769 de 2002**
(versión consolidada con sus modificaciones, **195 páginas**), descargada del
Normograma oficial de la Cancillería de Colombia.

**¿Por qué este documento?**

- Es una **norma legal real, pública y oficial**, con más de 30 páginas de
  contenido técnico-jurídico (195 en total), tal como exige la actividad.
- Tiene **alto impacto cotidiano**: cualquier conductor, peatón o ciclista se ve
  afectado por sus reglas, pero casi nadie lo lee completo por su extensión y
  lenguaje jurídico.
- Está **estructurado por artículos**, lo que facilita recuperar fragmentos
  precisos y citar la fuente exacta — ideal para demostrar el valor de un RAG.
- Permite formular preguntas concretas en lenguaje natural ("¿qué documentos debo
  portar?", "¿cuándo me pueden inmovilizar el carro?") cuya respuesta exacta está
  enterrada en el texto.

---

## 🎯 Persona usuaria objetivo y caso de uso

**Usuaria objetivo:** un **conductor particular o aspirante a licencia de
conducción** (también útil para estudiantes de academias de conducción y agentes
de tránsito) sin formación jurídica.

**Caso de uso:** resolver dudas puntuales sobre normas de tránsito **sin tener
que leer las 195 páginas** de la ley. Por ejemplo, antes de un viaje el usuario
pregunta "¿qué documentos debo portar para conducir?" y obtiene una respuesta
precisa con la página exacta del Código donde se establece, lista para verificar.

---

## ❓ Cinco preguntas y sus respuestas generadas por el sistema

> _(Respuestas reales producidas por Lex; cada una indica las páginas del PDF
> consultadas.)_

### 1. ¿Cuáles son los requisitos para obtener la licencia de conducción?

Para obtener por primera vez una licencia de conducción para vehículos de servicio
diferente del servicio público, se deben cumplir los siguientes requisitos, según el
**ARTÍCULO 19**:

1. Saber leer y escribir.
2. Tener 16 años cumplidos.
3. Aprobar un examen teórico-práctico de conducción para vehículos particulares que
   realizarán los organismos de tránsito de acuerdo con la reglamentación que expida
   el Ministerio de Transporte.

📄 **Páginas consultadas:** 31, 33, 35, 41, 42, 45, 46

---

### 2. ¿Qué documentos debe portar el conductor para transitar?

Los documentos que debe portar el conductor para transitar son: documento de
identidad, licencia de conducción, licencia de tránsito, Seguro Obligatorio de
Accidentes de Tránsito (SOAT) y certificado de revisión técnico-mecánica y de gases.
Sin embargo, las autoridades de tránsito pueden verificar estos documentos a través de
los Sistemas de Información establecidos, sin necesidad de que el conductor los porte
físicamente.

📄 **Páginas consultadas:** 7, 31, 34, 40, 55, 88, 172, 175

---

### 3. ¿Cuáles son las causales de suspensión o cancelación de la licencia de conducción?

La **suspensión** de la licencia de conducción se puede dar por disposición de las
autoridades de tránsito basada en la imposibilidad transitoria física o mental para
conducir, soportado en un certificado médico o en el examen de aptitud física, mental o
de coordinación expedido por un Centro de Reconocimiento de Conductores legalmente
habilitado (artículo 26).

La **cancelación** de la licencia de conducción se puede dar por: (1) imposibilidad
permanente física o mental para conducir, soportada en un certificado médico;
(2) decisión judicial; (3) muerte del titular; y (4) reincidencia al conducir en estado
de embriaguez o bajo el efecto de drogas alucinógenas. Además, la reincidencia en un
tercer grado de embriaguez es causal de cancelación definitiva de la licencia.

📄 **Páginas consultadas:** 47, 49, 50, 183

---

### 4. ¿Qué sanciones hay por conducir en estado de embriaguez?

Las sanciones varían según el grado de embriaguez (artículo 152):

- Entre 20 y 39 mg de etanol/100 ml de sangre: suspensión de la licencia entre seis (6)
  y doce (12) meses.
- Primer grado (40 a 99 mg): suspensión de la licencia entre uno (1) y tres (3) años.
- Segundo grado (100 a 149 mg): suspensión entre dos (2) y tres (3) años, más la
  obligación de prestar veinte (20) horas de servicios comunitarios.
- Tercer grado: suspensión entre tres (3) y diez (10) años, más un curso de
  sensibilización sobre alcoholemia y drogadicción de mínimo cuarenta (40) horas.

Si se trata de conductores de servicio público, transporte escolar o instructores de
conducción, la multa y el período de suspensión se duplican.

📄 **Páginas consultadas:** 139, 147, 177, 179, 181, 182, 183

---

### 5. ¿Qué es un vehículo de servicio diplomático o consular?

Un vehículo de servicio diplomático o consular es un vehículo automotor destinado al
servicio de funcionarios diplomáticos o consulares.

📄 **Páginas consultadas:** 11, 13, 14, 16, 20, 58, 62

---

### ➕ Demostración del comportamiento seguro (pregunta fuera del documento)

**Pregunta:** ¿Cuál es la receta del ajiaco santafereño?

**Respuesta de Lex:** *"No encontré información sobre esto en el Código Nacional de
Tránsito."*

> Esto demuestra que el asistente **no inventa**: cuando la respuesta no está en el
> documento, lo dice explícitamente (en estos casos no cita páginas).

---

## ⚙️ Cómo funciona (pipeline RAG)

```
PDF (Código de Tránsito)
   │  ingestar.py
   ▼
1. Carga del documento        → PyPDFLoader
2. División en fragmentos      → RecursiveCharacterTextSplitter (500 / 50)
3. Embeddings locales          → paraphrase-multilingual-MiniLM-L12-v2 (CPU)
4. Almacenamiento vectorial    → FAISS (índice persistente en disco)
   │  app.py (por cada pregunta)
   ▼
5. Recuperación semántica      → top-k fragmentos relevantes
6. Prompt aumentado            → contexto + pregunta + reglas de "Lex"
7. Generación de respuesta     → Llama 3.3 70B (vía HuggingFace Router → Groq)
   └─► Respuesta + páginas citadas + fragmentos (trazabilidad)
```

**Reglas del asistente (prompt):** Lex responde solo con el contexto recuperado;
si la información no está en el documento responde exactamente *"No encontré
información sobre esto en el Código Nacional de Tránsito"*; y nunca inventa.

---

## 🚀 Instrucciones para ejecutar el sistema

### Requisitos
- Python 3.10 o superior.
- Un **token de HuggingFace** (gratuito en <https://huggingface.co/settings/tokens>, tipo *Read*).

### 1. Clonar y entrar a la carpeta del proyecto
```bash
cd proyecto-rag-transito
```

### 2. Crear el entorno virtual e instalar dependencias
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```
> ℹ️ La primera instalación descarga `torch` (~200 MB) y, al ejecutar por primera
> vez, el modelo de embeddings (~120 MB). Todo corre en **CPU**, sin GPU.

### 3. Configurar el token
```bash
# Copia la plantilla y edita el archivo .env
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```
Abre `.env` y pega tu token de HuggingFace:
```
HF_TOKEN=hf_tu_token_real
```

### 4. Construir la base de conocimiento (una sola vez)
```bash
python ingestar.py
```
Esto carga el PDF, lo divide en fragmentos, genera los embeddings y crea la base
vectorial `chroma_db/`.

### 5. Lanzar el asistente (chat web)
```bash
streamlit run app.py
```
Se abrirá en el navegador (por defecto <http://localhost:8501>). Escribe tus
preguntas en el cuadro de chat.

---

## 📁 Estructura del proyecto

```
proyecto-rag-transito/
├── pdf/
│   └── codigo_nacional_transito.pdf   # documento fuente (Ley 769 de 2002)
├── rag_core.py        # lógica compartida (embeddings, LLM, prompt, ciclo RAG)
├── ingestar.py        # construye la base vectorial (pasos 1–4)
├── app.py             # chat Streamlit (pasos 5–7 + bucle conversacional)
├── requirements.txt
├── .env.example       # plantilla de configuración
├── .gitignore
└── README.md
```
> `faiss_index/` y `.env` **no** se versionan (se generan/configuran localmente).

---

## 🧰 Tecnologías

| Componente | Herramienta |
|---|---|
| Carga de PDF | `pypdf` / LangChain `PyPDFLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` (local) |
| Base vectorial | `FAISS` |
| LLM | `Llama 3.3 70B` vía **HuggingFace Router** (API compatible con OpenAI; enruta a Groq) |
| Interfaz | `Streamlit` |

---

## ⚠️ Limitaciones

- Responde solo sobre el **texto del documento cargado**; no conoce jurisprudencia
  ni normas posteriores que no estén en el PDF.
- La calidad depende de la **recuperación**: si la respuesta está repartida en
  muchos artículos, puede recuperar solo una parte (ajustable con el parámetro `k`).
- **Definiciones del glosario (Artículo 2):** como ese artículo agrupa decenas de
  términos muy cortos en pocos fragmentos, las preguntas tipo *"¿qué es la
  inmovilización?"* a veces no recuperan la definición exacta y el asistente responde
  *"No encontré información…"*. Es preferible esto a inventar, pero es una limitación
  real de la búsqueda semántica sobre texto tipo diccionario.
- El LLM puede reformular el texto; las **páginas citadas** son la fuente
  fiable para verificar la respuesta literal.

---

## 🎥 Video de demostración

🔗 _Enlace al video (3–5 min): **(pendiente de agregar)**_
