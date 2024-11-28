import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_clusters(df, labels):
    """ Visualize clusters in a 2D PCA plot. """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis")
    plt.title("Customer Segments (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()