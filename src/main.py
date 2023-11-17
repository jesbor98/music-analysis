
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

with open(r"C:\Users\jessi\OneDrive\Dokument\GitHub\music-analysis\data\spotify-2023.csv", 'r') as f:
    df = pd.read_csv(f)
    
    print(df.head())


# Data Summary
data_summary = df.describe()

# Display the cleaned DataFrame without the index
df.reset_index(drop=True, inplace=True)

df.info()
df.head()

# # Check for missing data again to verify that all null values have been replaced
# missing_data = df.isnull().sum()
# df.head()

# # Convert columns to appropriate data types
# df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
# df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')
# df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'], errors='coerce')
# df['key'] = pd.to_numeric(df['key'], errors='coerce')

# # Select the relevant columns for the heatmap
# attributes_to_compare = ['streams', 'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
#                          'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'bpm', 
#                          'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%',
#                          'liveness_%', 'speechiness_%']

# features = df[attributes_to_compare]

# # Compute the correlation matrix
# correlation_matrix = features.corr()

# # Create a heatmap
# plt.figure(figsize=(20, 18))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# plt.title('Correlation Heatmap of Song Characteristics')
# plt.show()

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

# Combine with 'streams' for plotting
Z_with_streams = np.column_stack((dataframe['streams'].values, Z))

scatter_pc1_pc2 = plt.figure()
plt.scatter(Z_with_streams[:, 1], Z_with_streams[:, 2], c=Z_with_streams[:, 0], cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection onto the First Two Principal Components with Streams')
plt.colorbar(label='Streams')
plt.show()

PC1 = V[:, 0]

#Find the attributes that are primarily represented by the first PC
attributes_for_PC1 = dataframe.columns[np.argsort(np.abs(PC1))[::-1]]
print("Attributes primarily represented by the first PC:")
print(attributes_for_PC1)

#Here starts the second part of the question
#Here we calculate the projection of data onto PC2
projection_onto_PC2 = np.dot(Y, VT[:, 1])

#Identify the observation with the largest positive projection
observation_with_largest_positive_projection = np.argmax(projection_onto_PC2)

#Identify the observation with the largest negative projection
observation_with_largest_negative_projection = np.argmin(projection_onto_PC2)

#Now we can display the projections
print("\nLargest positive projection onto PC2:", projection_onto_PC2[observation_with_largest_positive_projection])
print("Largest negative projection onto PC2:", projection_onto_PC2[observation_with_largest_negative_projection])