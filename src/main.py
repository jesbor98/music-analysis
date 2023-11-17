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
    
# Convert a specific column to integers (replace 'your_column_name' with the actual column name)
#df['streams'] = pd.to_numeric(df['streams'], errors='coerce').astype('Int64')    
    
# Selecting relevant features for clustering
features = df.iloc[:, 17:]  # Adjust the index based on your dataset

# Handling missing values
features = features.fillna(features.mean())

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Determine the optimal number of clusters
optimal_clusters = 3  # You can adjust this based on your preference

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['cluster'] = kmeans.fit_predict(principal_components)

# Plot the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    cluster_points = principal_components[df['cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}')

plt.title('K-means Clustering of Objects')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# # Data Summary
# data_summary = df.describe()

# # Display the cleaned DataFrame without the index
# df.reset_index(drop=True, inplace=True)

# df.info()
# df.head()

# # # Check for missing data again to verify that all null values have been replaced
# # missing_data = df.isnull().sum()
# # df.head()

# # # Convert columns to appropriate data types
# # df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
# # df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')
# # df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'], errors='coerce')
# # df['key'] = pd.to_numeric(df['key'], errors='coerce')

# # # Select the relevant columns for the heatmap
# # attributes_to_compare = ['streams', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
# #                          'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'bpm', 
# #                          'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%',
# #                          'liveness_%', 'speechiness_%']

# # features = df[attributes_to_compare]

# # # Compute the correlation matrix
# # correlation_matrix = features.corr()

# # # Create a heatmap
# # plt.figure(figsize=(20, 18))
# # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# # plt.title('Correlation Heatmap of Song Characteristics')
# # plt.show()

# selected_attributes = ['streams', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

# dataframe = df[selected_attributes]
# dataframe['streams'] = pd.to_numeric(dataframe['streams'], errors='coerce')
# dataframe = dataframe.dropna()

# # Drop 'streams' for standardization
# dataframe_no_streams = dataframe.drop('streams', axis=1)

# # Standardize the data
# scaler = StandardScaler()
# scaled_data_no_streams = scaler.fit_transform(dataframe_no_streams)

# X = dataframe.values
# mu = np.mean(X, axis=0)

# Y = X - mu
# # Apply PCA for dimensionality reduction
# U, S, VT = np.linalg.svd(scaled_data_no_streams)




# V = np.transpose(VT)
# Z = np.dot(scaled_data_no_streams, V)

# PC1 = V[:, 0]
# PC2 = V[:, 1]

# #Find the attributes that are primarily represented by the first and second PC
# attributes_for_PC1 = dataframe.columns[np.argsort(np.abs(PC1))[::-1]]
# attributes_for_PC2 = dataframe.columns[np.argsort(np.abs(PC2))[::-1]]
# print("Attributes primarily represented by the first PC:" + attributes_for_PC1[0])
# print("Attributes primarily represented by the second PC:" + attributes_for_PC2[0])

# # Combine with 'streams' for plotting
# Z_with_streams = np.column_stack((dataframe['streams'].values, Z))

# scatter_pc1_pc2 = plt.figure()
# plt.scatter(Z_with_streams[:, 1], Z_with_streams[:, 2], c=Z_with_streams[:, 0], cmap='viridis', alpha=0.7)
# plt.xlabel('PC1 ' + attributes_for_PC1[0])
# plt.ylabel('PC2 ' + attributes_for_PC2[0])
# plt.title('Projection onto the First Two Principal Components with Streams')
# plt.colorbar(label='Streams')
# plt.show()