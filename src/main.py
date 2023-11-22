import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

with open(r"C:\Users\amand\Documents\GitHub\music-analysis\data\spotify-2023.csv", 'r') as f:
    df = pd.read_csv(f)
    
    print(df.head())
    
selected_attributes = ['streams', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

dataframe = df[selected_attributes]
dataframe['streams'] = pd.to_numeric(dataframe['streams'], errors='coerce')
dataframe = dataframe.dropna()

# Drop 'streams' for standardization
dataframe_no_streams = dataframe.drop('streams', axis=1)

# Standardize the data
scaler = StandardScaler()
scaled_data_no_streams = scaler.fit_transform(dataframe_no_streams)

X = dataframe.values
mu = np.mean(X, axis=0)

Y = X - mu
# Apply PCA for dimensionality reduction
U, S, VT = np.linalg.svd(scaled_data_no_streams)

V = np.transpose(VT)
Z = np.dot(scaled_data_no_streams, V)

PC1 = V[:, 0]
PC2 = V[:, 1]

#Find the attributes that are primarily represented by the first and second PC
attributes_for_PC1 = dataframe.columns[np.argsort(np.abs(PC1))[::-1]]
attributes_for_PC2 = dataframe.columns[np.argsort(np.abs(PC2))[::-1]]
print("Attributes primarily represented by the first PC: " + attributes_for_PC1[0])
print("Attributes primarily represented by the second PC: " + attributes_for_PC2[0])

# Combine with 'streams' for plotting
Z_with_streams = np.column_stack((dataframe['streams'].values, Z))

scatter_pc1_pc2 = plt.figure()
plt.scatter(Z_with_streams[:, 1], Z_with_streams[:, 2], c=Z_with_streams[:, 0], cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection onto the First Two Principal Components with Streams')
plt.colorbar(label='Streams')
plt.show()