import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('flats_for_clustering.tsv', sep='\t')


z_threshold = 3  # prog z-score dla odstajacych wart, usuwanie odstajacych wart
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    z_scores = (data[col] - data[col].mean()) / data[col].std()
    data = data[abs(z_scores) <= z_threshold]

# kodowanie kolumn tekstowych 
for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column].fillna('missing'))

# uzup brak wart num
data.fillna(data.mean(numeric_only=True), inplace=True)

# normalizacja
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# k-srednie
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# dodawanie etykiet klastrow
data['Cluster'] = kmeans_labels

# redukcja wymiarow PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Wykres punktowy
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('Klastry danych po redukcji wymiarów PCA')
plt.xlabel('Pierwsza składowa PCA')
plt.ylabel('Druga składowa PCA')
plt.colorbar(label='Cluster')
plt.show()
