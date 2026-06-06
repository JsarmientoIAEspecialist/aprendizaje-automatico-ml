import os
import sys
from google import genai
from dotenv import load_dotenv

# La consola de Windows usa cp1252; forzamos UTF-8 para imprimir emojis/acentos
sys.stdout.reconfigure(encoding="utf-8")

# Carga de variables de entorno desde el archivo .env
load_dotenv()

clave_api = os.getenv("GEMINI_API_KEY")

# Inicializa el cliente de Gemini con la clave API
cliente = genai.Client(api_key=clave_api)


def ejecutar_consulta():
    print("⚡ Ejecutando consulta a Gemini...")

    try:
        respuesta = cliente.models.generate_content(
            model="gemini-2.5-flash",
            contents="Preséntate como experto en ML y responde a esta pregunta: ¿Cuáles son las mejores prácticas para entrenar un modelo de machine learning? en tres renglones"
        )
        print("Respuesta de Gemini:")
        print(respuesta.text)
    except Exception as e:
        print("Error al ejecutar la consulta:", e)


if __name__ == "__main__":
    ejecutar_consulta()
