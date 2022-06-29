from typing import List

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def get_masader_clusters(embeddings: List[List[float]]) -> List[int]:
    reduced_embeddings = compute_reduced_embeddings(embeddings)

    return compute_clusters(reduced_embeddings)


def compute_clusters(embeddings: List[List[float]]) -> List[int]:
    clustering = KMeans(n_clusters=15).fit(embeddings)

    return clustering.labels_.tolist()


def compute_reduced_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_data = tsne_model.fit_transform(embeddings)

    return (tsne_data - tsne_data.min()).tolist()
