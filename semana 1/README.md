# Semana 1 — Aprendizaje Automático (Machine Learning)

Scripts de la primera semana del curso de aprendizaje autónomo de Machine Learning.

## Archivos

- **`prueba_entorno.py`** — Verifica la configuración del entorno: detecta si se está usando un entorno virtual y comprueba la conexión a Internet.
- **`app_gemini.py`** — Ejemplo de consulta a la API de Google Gemini (`gemini-2.5-flash`) usando una clave guardada en variables de entorno.

## Requisitos

```bash
pip install google-genai python-dotenv requests
```

## Configuración

`app_gemini.py` lee la clave de la API desde un archivo `.env` (no incluido en el repositorio por seguridad):

```env
GEMINI_API_KEY=tu_clave_aqui
```

## Uso

```bash
python prueba_entorno.py
python app_gemini.py
```
