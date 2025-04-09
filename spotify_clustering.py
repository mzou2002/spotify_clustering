import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


spotify_df = pd.read_csv("Spotify_2024_Global_Streaming_Data.csv")

features = [
    'Monthly Listeners (Millions)',
    'Total Streams (Millions)',
    'Avg Stream Duration (Min)',
    'Streams Last 30 Days (Millions)',
    'Skip Rate (%)',
    'Genre',
    'Platform Type'
]
spotify_subset = spotify_df[features].copy()

numeric_features = [
    'Monthly Listeners (Millions)',
    'Total Streams (Millions)',
    'Avg Stream Duration (Min)',
    'Streams Last 30 Days (Millions)',
    'Skip Rate (%)'
]
categorical_features = ['Genre', 'Platform Type']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])


kmeans_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])
kmeans_pipeline.fit(spotify_subset)
# Add the cluster labels as a new column
spotify_subset['Cluster'] = kmeans_pipeline.named_steps['kmeans'].labels_


X_processed = preprocessor.fit_transform(spotify_subset.drop(columns='Cluster'))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
spotify_subset['PCA1'] = X_pca[:, 0]
spotify_subset['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=spotify_subset, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=70)
plt.title('PCA Projection of Spotify Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

target_genres = ['K-pop', 'R&B', 'Reggaeton']
total_tracks = len(spotify_subset)

for genre in target_genres:
    genre_tracks = spotify_subset[spotify_subset['Genre'] == genre]
    total_genre = len(genre_tracks)
    print(f"Total {genre} tracks: {total_genre}")
    
    P_genre = total_genre / total_tracks if total_tracks > 0 else 0
    print(f"P({genre}) = {P_genre}")
    
    for cluster_id, group in spotify_subset.groupby("Cluster"):
        cluster_genre_count = group[group['Genre'] == genre].shape[0]
        P_cluster_and_genre = cluster_genre_count / total_tracks
        P_cluster_given_genre = P_cluster_and_genre / P_genre if P_genre > 0 else 0
        print(f"P(Cluster {cluster_id:02d} | {genre}): {P_cluster_given_genre}")
    
