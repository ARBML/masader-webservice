from sklearn.cluster import KMeans
import pandas as pd


def get_cluster_data(tsne_data):

    clustering = KMeans(n_clusters=15).fit(tsne_data)
    return pd.DataFrame(clustering.labels_.reshape(len(tsne_data), 1)).to_json(orient='split')
