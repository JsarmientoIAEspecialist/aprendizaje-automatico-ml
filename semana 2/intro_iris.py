import sklearn
import pandas as pd
import matplotlib
from sklearn.datasets import load_iris

# ─── 1. Versiones de librerías ───────────────────────────────────────────────
print("=" * 55)
print("  VERSIONES DE LIBRERIAS")
print("=" * 55)
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  pandas       : {pd.__version__}")
print(f"  matplotlib   : {matplotlib.__version__}")

# ─── 2. Cargar Iris y convertir a DataFrame ───────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[t] for t in iris.target]

# ─── 3. Primeras filas ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DATASET IRIS - primeras 5 filas (head)")
print("=" * 55)
print(df.head().to_string(index=True))

# ─── 4. Preguntas conceptuales ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PREGUNTAS CONCEPTUALES")
print("=" * 55)

print("""
[Tarea T]
Si quisieramos predecir la especie de una flor, la tarea T
seria una CLASIFICACION MULTICLASE:
  -> Dado un vector de 4 medidas morfologicas
     (longitud/ancho del sepalo y el petalo),
     asignar la flor a una de las 3 especies:
     setosa, versicolor o virginica.

[Filas y columnas]
  * Cada FILA es una muestra individual (una flor medida).
    El dataset tiene 150 flores en total.

  * Las primeras 4 COLUMNAS son las caracteristicas (features):
      - sepal length (cm)
      - sepal width  (cm)
      - petal length (cm)
      - petal width  (cm)
    Son las variables de entrada que el modelo usara para aprender.

  * La columna 'species' es la etiqueta objetivo (target / label):
    lo que el modelo debe aprender a predecir.
""")
