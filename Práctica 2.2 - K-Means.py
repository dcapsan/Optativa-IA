# K-Means sobre el conjunto Iris
# El dataset Iris contiene 150 muestras con 3 especies reales, pero sin etiquetas de cluster.
# K-Means intenta agruparlas a partir de las características de las flores.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, adjusted_rand_score, classification_report

iris = load_iris()
datos = iris.data[:, :2]  # solo longitud y anchura del sépalo para visualizar

tabla = pd.DataFrame(datos, columns=['sepal_length', 'sepal_width'])
tabla['species'] = iris.target
tabla['species_name'] = iris.target_names[iris.target]

# K-Means con 2 clusters
modelo_k2 = KMeans(n_clusters=2, random_state=0, n_init=10)
modelo_k2.fit(datos)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(datos[:, 0], datos[:, 1], c=modelo_k2.labels_, cmap='viridis')
plt.title('Agrupación con 2 grupos (K=2)')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Con 4 clusters
modelo_k4 = KMeans(n_clusters=4, random_state=0, n_init=10)
modelo_k4.fit(datos)

plt.subplot(1, 3, 2)
plt.scatter(datos[:, 0], datos[:, 1], c=modelo_k4.labels_, cmap='viridis')
plt.title('Agrupación con 4 grupos (K=4)')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Con 5 clusters
modelo_k5 = KMeans(n_clusters=5, random_state=0, n_init=10)
modelo_k5.fit(datos)

plt.subplot(1, 3, 3)
plt.scatter(datos[:, 0], datos[:, 1], c=modelo_k5.labels_, cmap='viridis')
plt.title('Agrupación con 5 grupos (K=5)')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

plt.tight_layout()
plt.show()

# ¿Qué datos reales llegan sin etiquetar en el mundo profesional?
# En el entorno profesional, la inmensa mayoría de los datos se generan de forma masiva y sin etiquetar
# debido al alto coste humano de clasificarlos manualmente. Algunos ejemplos:
# - Segmentación de clientes
# - Detección de fraude y anomalías
# - Sistemas de recomendación
# - Agrupación de documentos y tickets de soporte

# Práctica - Comparación
modelo_k3 = KMeans(n_clusters=3, random_state=0, n_init=10)
modelo_k3.fit(datos)
tabla['cluster'] = modelo_k3.labels_

comparacion = pd.crosstab(
    tabla['species_name'],
    tabla['cluster'],
    rownames=['Especie Real'],
    colnames=['Cluster K-Means']
)
print('\n=== MATRIZ DE CONFUSIÓN ===')
print(comparacion)

# Los labels de K-Means no tienen un orden fijo, por eso hay que mapear cada cluster a la especie dominante.
cluster_a_especie = tabla.groupby('cluster')['species'].agg(lambda x: x.mode().iloc[0]).to_dict()
predicciones_mapeadas = tabla['cluster'].map(cluster_a_especie)

acierto = accuracy_score(tabla['species'], predicciones_mapeadas)
ari = adjusted_rand_score(tabla['species'], tabla['cluster'])

print('\n=== EVALUACIÓN NUMÉRICA ===')
print(f'Precisión global (Accuracy): {acierto * 100:.2f}%')
print(f'Adjusted Rand Index (ARI):    {ari:.4f}')

print('\n=== REPORTE DETALLADO POR ESPECIE ===')
print(classification_report(tabla['species'], predicciones_mapeadas, target_names=iris.target_names))