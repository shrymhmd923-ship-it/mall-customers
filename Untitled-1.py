
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Load data
df = pd.read_csv("Data.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# 2. Select features
features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]
X = df[features].copy()

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find the best K
k_values = [2, 3, 4, 5, 6, 7, 8]
inertias = []
silhouette_scores = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

best_k = k_values[
    silhouette_scores.index(max(silhouette_scores))
]

print("\nSilhouette Scores:")
for k, score in zip(k_values, silhouette_scores):
    print(f"K={k}: {score:.3f}")

print("\nBest K:", best_k)

# 5. Elbow Method
plt.figure(figsize=(7, 4))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# 6. Silhouette Score plot
plt.figure(figsize=(7, 4))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by K")
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# 7. Final K-Means model
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Profiles:")
print(df.groupby("Cluster")[features].mean().round(2))

# 8. Visualize clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"K-Means Customer Segmentation (K={best_k})")
plt.tight_layout()
plt.show()

# 9. Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=best_k)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

print("\nHierarchical Cluster Sizes:")
print(df["Hierarchical_Cluster"].value_counts().sort_index())

# 10. Dendrogram
linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster / Sample Group")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
