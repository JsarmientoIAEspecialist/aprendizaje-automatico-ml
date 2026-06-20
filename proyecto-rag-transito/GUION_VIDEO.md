# 🎬 Guion del video — Lex (RAG sobre el Código de Tránsito)

Duración objetivo: **3 a 5 minutos**. Antes de grabar deja listo:
- La terminal abierta en la carpeta `proyecto-rag-transito` con el venv activado.
- El navegador con la app (`streamlit run app.py`) ya cargada en `http://localhost:8501`.
- (Opcional) El PDF del Código de Tránsito abierto para mostrar una página citada.

---

## ⏱️ (0:00 – 0:30) Presentación

**Qué mostrar:** tu cara o la pantalla con el título del proyecto / README.

**Qué decir:**
> "Hola, soy **Juan Sebastián Sarmiento**. Para la actividad de sistemas RAG construí
> **Lex**, un asistente conversacional sobre el **Código Nacional de Tránsito de
> Colombia, la Ley 769 de 2002**. Elegí este documento porque es una norma real,
> pública y de 195 páginas que afecta a cualquier conductor, pero que casi nadie lee
> completa por su extensión y lenguaje jurídico. Es perfecto para demostrar el valor
> de un RAG: preguntar en lenguaje natural y obtener la respuesta exacta con su página."

---

## ⏱️ (0:30 – 1:30) Ejecución del sistema y explicación de cada paso

**Qué mostrar:** la terminal ejecutando `python ingestar.py` (o su salida ya generada).

**Qué decir (mientras corre):**
> "El sistema tiene dos partes. La primera, `ingestar.py`, construye la base de
> conocimiento una sola vez y hace cuatro pasos:
> 1. **Carga** el PDF del Código de Tránsito — son 195 páginas.
> 2. Lo **divide en fragmentos** pequeños: 1.145 trozos de unos 500 caracteres.
> 3. Convierte cada fragmento en **embeddings** —vectores numéricos— con un modelo
>    local que corre en mi CPU, sin enviar el documento a internet.
> 4. **Almacena** esos vectores en una base vectorial FAISS, que queda guardada en disco."

**Luego muestra:** la terminal con `streamlit run app.py` y el navegador abriéndose.
> "La segunda parte es `app.py`, la interfaz de chat hecha con Streamlit. Aquí es donde
> conversamos con Lex."

---

## ⏱️ (1:30 – 3:30) Demostración del asistente

**Qué mostrar:** el chat de Lex. Escribe las preguntas en vivo.

**Pregunta 1 (dentro del documento):**
> Escribe: *"¿Cuáles son los requisitos para obtener la licencia de conducción?"*

Mientras responde, explica:
> "Lex toma mi pregunta, busca los fragmentos más parecidos en la base vectorial, se
> los pasa al modelo de lenguaje —Llama 3.3 en Groq— y este responde **usando solo
> ese contexto**. Fíjense que al final **cita las páginas consultadas**: aquí me dice
> que viene del Artículo 19, en las páginas 41 y 42."

**(Opcional)** Despliega *"🔍 Fragmentos consultados"* para mostrar la trazabilidad:
> "Y puedo ver exactamente qué fragmentos del documento usó, así la respuesta es
> verificable, no una caja negra."

**Pregunta 2 (fuera del documento) — la más importante de mostrar:**
> Escribe: *"¿Cuál es la receta del ajiaco santafereño?"*

> "Y esto es clave: cuando pregunto algo que NO está en el documento, Lex **no
> inventa**. Responde explícitamente: *'No encontré información sobre esto en el
> Código Nacional de Tránsito'*. Eso es lo que hace confiable a un asistente legal."

**(Si hay tiempo) Pregunta 3:**
> *"¿Qué sanciones hay por conducir en estado de embriaguez?"* — muestra una respuesta
> rica con los grados de alcoholemia y la cita al Artículo 152.

---

## ⏱️ (3:30 – 4:00) Conclusión

**Qué decir:**
> "Para cerrar: lo que más aprendí es que la **calidad de un RAG depende sobre todo de
> la recuperación** —de encontrar el fragmento correcto—, no solo del modelo de
> lenguaje. Y una **limitación** que descubrí: con las definiciones del glosario del
> Artículo 2, que agrupa decenas de términos muy cortos, a veces no recupera la
> definición exacta y prefiere decir que no la encontró. Es una limitación real de la
> búsqueda semántica sobre texto tipo diccionario, pero prefiero eso a que invente.
> Gracias."

---

### ✅ Checklist antes de subir el video
- [ ] Se ve tu nombre y el documento elegido (0:30).
- [ ] Se ve el sistema ejecutándose y se explican los pasos (1 min).
- [ ] Se ve al menos una pregunta respondida **con páginas citadas** (2 min).
- [ ] Se muestra el caso "no encontré información".
- [ ] Conclusión con 1 aprendizaje + 1 limitación (30 seg).
- [ ] Pega el enlace del video en el `README.md` (sección 🎥).
