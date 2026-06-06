import requests
import sys
import os

def verificar_configuracion():
    print("Verificando configuración del entorno...")

    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("🐍 Entorno virtual detectado.")
    else:
        print("⚠️ No se detectó un entorno virtual. Se recomienda usar uno para evitar conflictos de dependencias.")

    try:
        respuesta = requests.get("https://www.google.com")
        if respuesta.status_code == 200:
            print("🐍 Conexión a Internet verificada.")
    except requests.RequestException as e:
        print("❌ Error al verificar conexión")

if __name__ == "__main__":
    verificar_configuracion()
