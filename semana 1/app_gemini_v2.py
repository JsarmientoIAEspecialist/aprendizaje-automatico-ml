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
    # Conversación de ejemplo entre un cliente y un agente de servicio al cliente
    texto = """
        Cliente: Hola, tengo un problema con mi pedido. No llegó a tiempo y necesito una solución.
        Agente: Lamento mucho la demora en la entrega de su pedido. Permítame verificar el estado de su pedido y encontrar una solución para usted. ¿Podría proporcionarme su número de pedido?
        Cliente: Sí, mi número de pedido es 12345.
        Agente: Gracias por proporcionar su número de pedido. He verificado el estado de su pedido y parece que hubo un retraso en el envío. Me disculpo por los inconvenientes.
        Cliente: Prefiero que me envíen un nuevo pedido lo antes posible.
        Agente: Entendido. Voy a procesar el envío de un nuevo pedido para usted de inmediato. Debería recibir una confirmación de envío en breve.
    """

    prompt = f"""Resume la conversación entre el cliente y el agente de servicio al cliente, en 4 puntos clave y al final indica si se resolvió el problema:

    {texto}
    """

    try:
        respuesta = cliente.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        print("Respuesta de Gemini:")
        print(respuesta.text)
    except Exception as e:
        print("Error al ejecutar la consulta:", e)


if __name__ == "__main__":
    ejecutar_consulta()
