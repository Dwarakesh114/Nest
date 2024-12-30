import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
from wordcloud import WordCloud
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data
print("Loading and preprocessing data...")
data = pd.read_csv("usecase_1_.csv")  # Replace with the actual file path
data = data.drop_duplicates(subset=['Brief Summary']).dropna(subset=['Brief Summary'])
data = data[data['Brief Summary'].str.len() > 10]
study_texts = data['Brief Summary']

print(f"Total records after preprocessing: {len(study_texts)}")

# 2. Initialize Embedding Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 3. Generate Embeddings in Batches
print("Generating embeddings...")
batch_size = 256  # Adjust batch size based on your hardware
embeddings = []
for i in tqdm(range(0, len(study_texts), batch_size)):
    batch = study_texts.iloc[i:i + batch_size].tolist()
    embeddings.append(model.encode(batch, show_progress_bar=False))
embeddings = np.vstack(embeddings)  # Combine all batches

# Save embeddings to avoid recalculating
np.save("embeddings.npy", embeddings)

# 4. Perform Clustering
print("Performing clustering...")
num_clusters = 5
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1024)
clusters = kmeans.fit_predict(embeddings)

# Assign clusters back to the DataFrame
data['cluster'] = clusters

# 5. Evaluate Clustering (Using Downsampling)
print("Evaluating clustering...")
sample_size = 10000  # Adjust this size for faster evaluation
if len(embeddings) > sample_size:
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    sample_clusters = clusters[sample_indices]
else:
    sample_embeddings = embeddings
    sample_clusters = clusters

# Silhouette Score
silhouette_avg = silhouette_score(sample_embeddings, sample_clusters)
print(f"Silhouette Score (sampled): {silhouette_avg:.4f}")

# Davies-Bouldin Index
db_score = davies_bouldin_score(sample_embeddings, sample_clusters)
print(f"Davies-Bouldin Index: {db_score:.4f}")

# 6. Dimensionality Reduction for Visualization
print("Reducing dimensions for visualization...")
umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
reduced_embeddings = umap.fit_transform(embeddings)

# Plot Clusters
plt.figure(figsize=(12, 8))
for cluster in range(num_clusters):
    cluster_points = reduced_embeddings[data['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
plt.legend()
plt.title("Cluster Visualization of Clinical Studies")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()

# 7. Generate Word Clouds for Each Cluster
print("Generating word clouds...")
for cluster in range(num_clusters):
    cluster_texts = " ".join(study_texts[data['cluster'] == cluster])
    wordcloud = WordCloud(background_color='white', max_words=100, width=800, height=400).generate(cluster_texts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Cluster {cluster}")
    plt.show()

# 8. Save Results
print("Saving results...")
data.to_csv("clustered_clinical_trials.csv", index=False)
print("Results saved to 'clustered_clinical_trials.csv'")
