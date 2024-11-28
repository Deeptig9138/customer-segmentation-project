from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(df, n_clusters):
    """ Perform KMeans clustering and return labels and model. """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df)
    return labels, kmeans

def evaluate_clustering(df, labels):
    """ Evaluate clustering using silhouette score. """
    return silhouette_score(df, labels)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/customers_preprocessed.csv")
    
    labels, model = perform_kmeans_clustering(df, n_clusters=5)
    score = evaluate_clustering(df, labels)
    print(f"Silhouette Score: {score}")