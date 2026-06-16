"""
Taller K-means en 3D (k=2) - Ciencia de Datos
Ejecucion manual paso a paso + implementacion con scikit-learn.

Requisitos:  pip install numpy scikit-learn matplotlib
Ejecucion:   python kmeans_taller.py
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------
# 1. Conjunto de datos
# ---------------------------------------------------------------
clientes = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
X = np.array([
    [40, 20, 25],   # C1
    [42, 22, 27],   # C2
    [25, 80, 45],   # C3
    [23, 77, 43],   # C4
    [60, 15, 30],   # C5
    [62, 18, 33],   # C6
], dtype=float)


def distancia_euclidiana(p, q):
    """Distancia euclidiana en 3D: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)"""
    return np.sqrt(np.sum((p - q) ** 2))


# ---------------------------------------------------------------
# 2. K-means manual (mismo procedimiento que a mano)
# ---------------------------------------------------------------
def kmeans_manual(X, centroides_iniciales, max_iter=10):
    M = np.array(centroides_iniciales, dtype=float)
    asignaciones = None
    for it in range(1, max_iter + 1):
        print(f"\n===== ITERACION {it} =====")
        print(f"Centroide 1 = {M[0]}")
        print(f"Centroide 2 = {M[1]}")
        print(f"{'Cliente':<8}{'Dist C1':>10}{'Dist C2':>10}{'Cluster':>10}")
        nuevas = []
        for i, c in enumerate(clientes):
            d1 = distancia_euclidiana(X[i], M[0])
            d2 = distancia_euclidiana(X[i], M[1])
            cl = 1 if d1 <= d2 else 2
            nuevas.append(cl)
            print(f"{c:<8}{d1:>10.4f}{d2:>10.4f}{cl:>10}")
        nuevas = np.array(nuevas)

        # Recalcular centroides
        nuevos_centroides = np.array([X[nuevas == k].mean(axis=0) for k in (1, 2)])
        print(f"Nuevo Centroide 1 = {nuevos_centroides[0]}")
        print(f"Nuevo Centroide 2 = {nuevos_centroides[1]}")

        if asignaciones is not None and np.array_equal(nuevas, asignaciones):
            print(">>> CONVERGENCIA: las asignaciones no cambian.")
            M = nuevos_centroides
            asignaciones = nuevas
            break
        asignaciones = nuevas
        M = nuevos_centroides
    return asignaciones, M


init = [[40, 20, 25],   # C1
        [25, 80, 45]]   # C3
asign_manual, centroides_manual = kmeans_manual(X, init)

print("\n========== RESULTADO MANUAL ==========")
for k in (1, 2):
    miembros = [clientes[i] for i in range(len(clientes)) if asign_manual[i] == k]
    print(f"Cluster {k}: {miembros}  -> centroide {centroides_manual[k-1]}")

# ---------------------------------------------------------------
# 3. K-means con scikit-learn
# ---------------------------------------------------------------
init_sk = np.array([[40, 20, 25], [25, 80, 45]], dtype=float)
km = KMeans(n_clusters=2, init=init_sk, n_init=1, max_iter=300, random_state=0)
labels_sk = km.fit_predict(X)

print("\n========== RESULTADO SKLEARN ==========")
print("Etiquetas:", labels_sk)            # 0/1 (orden segun init)
print("Centroides:\n", km.cluster_centers_)
print("Inercia (SSE):", km.inertia_)

# ---------------------------------------------------------------
# 4. Comparacion
# ---------------------------------------------------------------
# sklearn: 0 -> cluster 1 ; 1 -> cluster 2  (porque init va en ese orden)
equiv = {0: 1, 1: 2}
sk_como_manual = np.array([equiv[l] for l in labels_sk])
print("\n========== COMPARACION ==========")
print("Manual :", asign_manual)
print("Sklearn:", sk_como_manual)
print("Coinciden 100%:", np.array_equal(asign_manual, sk_como_manual))

# ---------------------------------------------------------------
# 5. Grafico 3D
# ---------------------------------------------------------------
col = {1: '#2563eb', 2: '#dc2626'}
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for l in (1, 2):
    m = asign_manual == l
    ax.scatter(X[m, 0], X[m, 1], X[m, 2], c=col[l], s=90,
               edgecolors='k', label=f'Cluster {l}')
ax.scatter(centroides_manual[:, 0], centroides_manual[:, 1], centroides_manual[:, 2],
           c='black', marker='X', s=220, label='Centroides')
for i, c in enumerate(clientes):
    ax.text(X[i, 0] + 1, X[i, 1] + 1, X[i, 2] + 1, c, fontsize=9)
ax.set_xlabel('X1'); ax.set_ylabel('X2'); ax.set_zlabel('X3')
ax.set_title('K-means 3D (k=2)')
ax.legend()
plt.tight_layout()
plt.savefig('kmeans_3d.png', dpi=150)
plt.show()
