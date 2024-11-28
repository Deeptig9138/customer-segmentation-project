# 🌟 Customer Segmentation Project 📊

Welcome to the **Customer Segmentation Project**! This project uses **KMeans clustering** to identify customer segments and provides an interactive **Streamlit** dashboard for data visualization. Explore the code, analyze clusters, and gain insights into customer behaviors. 🛍️

---

## 📁 Project Structure
```
customer-segmentation-project/
├── data/                            # Directory for data files
│   ├── customers.csv                # Raw dataset
│   ├── customers_preprocessed.csv   # Preprocessed dataset
├── src/
│   ├── data_preprocessing.py        # Data cleaning, encoding, and scaling script
│   ├── clustering.py                # KMeans clustering and evaluation script
│   ├── utils.py                     # Utility functions for data handling
│   ├── visualization.py             # PCA and cluster visualization functions
├── app/
│   ├── dashboard.py                 # Streamlit dashboard for interactive visualization
├── case study/
│   ├── R&D(ML)_DeeptiGupta.pdf      # Case study documentation
├── images/                          # Images for Readme documentation
│   ├── preprocessing.png            # Visualization of preprocessing steps
│   ├── clustering.png               # Example of clustering output
│   ├── dashboard.png                # Streamlit dashboard view
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🔧 Setup and Installation

1. **Clone the Repository**:  
   ```
   git clone https://github.com/Deeptig9138/customer-segmentation-project.git
   cd customer-segmentation-project
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```
   streamlit run app/dashboard.py
   ```

---

## 📝 Project Workflow

### 1. Data Preprocessing 🧹
The data_preprocessing.py script handles missing values, encodes categorical variables, and scales numerical data using StandardScaler. It saves the cleaned data for clustering.

**Sample Code**:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')
```

**Output**: A preprocessed CSV file saved in the data/ folder.

![Preprocessing](https://github.com/Deeptig9138/customer-segmentation-project/blob/main/images/preprocessing.png)

### 2. Clustering 🤖
The clustering.py script applies KMeans clustering and evaluates the performance using the Silhouette Score.

**Sample Code**:
```
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df)
    return labels, kmeans
```

![Clustering](https://github.com/Deeptig9138/customer-segmentation-project/blob/main/images/clustering.png)

### 3. Interactive Dashboard 📈
Use Streamlit in dashboard.py to visualize clusters interactively. The dashboard includes PCA plots for dimensionality reduction and cluster interpretation.

**Dashboard Features**:
- View sample data.
- Adjust the number of clusters dynamically.
- Visualize clusters in a 2D PCA plot.

**Run the dashboard** using the command below:
```
streamlit run app/dashboard.py
```

![Dashboard](https://github.com/Deeptig9138/customer-segmentation-project/blob/main/images/dashboard.png)
![Dashboard1](https://github.com/Deeptig9138/customer-segmentation-project/blob/main/images/dashboard1.png)

## 📊 Visualizing Clusters
The visualization.py script leverages PCA for dimensionality reduction and uses Seaborn for beautiful scatter plots.

**Sample Visualization Code**:
```
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df, labels):
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis")
    plt.show()
```

## ⚙️ Utility Functions
The utils.py script contains helper functions for saving and loading data:
- **save_data**: Save data to a CSV file.
- **load_processed_data**: Load preprocessed data.

---

## 📂 Case Study 📖
A detailed **Case Study** is provided in the case study folder.
- Explains the problem statement
- Answer the given questions in the task 1.

---

## 🎯 Key Results
- Silhouette Score: Measure of cluster quality.
- PCA Visualization: Easy interpretation of clusters in 2D space.
- Interactive Dashboard: Real-time cluster exploration.

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Matplotlib & Seaborn

---

## 🤝 Contributions
Feel free to submit pull requests or report issues! Contributions are always welcome. 🙌
💡 *Thank you for exploring the Customer Segmentation Project!* 💡
