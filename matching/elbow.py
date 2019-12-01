import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

data = pd.read_csv("db_comparables.csv")
data = data.loc[data['Judet'] == 'Bucuresti']

X = pd.DataFrame(data, columns=['Latitudine','Longitudine', 'Pret'])

ELBOW_MIN_CLUSTERS = 1
ELBOW_MAX_CLUSTERS = 150
wcss = []
for i in range(ELBOW_MIN_CLUSTERS, ELBOW_MAX_CLUSTERS):
    print(i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(ELBOW_MIN_CLUSTERS, ELBOW_MAX_CLUSTERS), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
