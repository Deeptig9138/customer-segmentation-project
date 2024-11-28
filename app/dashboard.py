import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("data/customers_preprocessed.csv")

st.write("Sample Data:", df.head())

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

kmeans = KMeans(n_clusters=10, random_state=42)

labels = kmeans.fit_predict(df[numerical_columns])

if len(labels) == len(df):
    df['Cluster'] = labels
    st.write("Clustering Complete!")
else:
    st.error(f"Length mismatch between labels ({len(labels)}) and DataFrame ({len(df)})")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[numerical_columns])

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
plt.title("K-Means Clustering with 10 Clusters (PCA 2D Projection)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(label='Cluster')
st.pyplot(plt)

st.write("Cluster Centers (PCA projection):")
st.write(kmeans.cluster_centers_)
