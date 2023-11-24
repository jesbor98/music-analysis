import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree

with open(r"./data/spotify-2023.csv", 'r') as f:
    df = pd.read_csv(f)
    
    print(df.head())
    
#-----------Here starts PCA with all attributes------------------
selected_attributes = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

dataframe = df[selected_attributes]
dataframe = dataframe.dropna()

# Standardize the data
mu = np.mean(dataframe, axis=0)
X = dataframe - mu

# Perform PCA
U, S, VT = np.linalg.svd(X)
V = np.transpose(VT)
Z = np.dot(X, V)

PC1 = V[:, 0]
PC2 = V[:, 1]

#Find the attributes that are primarily represented by the first and second PC
attributes_for_PC1 = dataframe.columns[np.argsort(np.abs(PC1))[::-1]]
attributes_for_PC2 = dataframe.columns[np.argsort(np.abs(PC2))[::-1]]
print("Attributes primarily represented by PC1: " + attributes_for_PC1[0])
print("Attributes primarily represented by PC2: " + attributes_for_PC2[0])

scatter_pc1_pc2 = plt.figure()
plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection onto the First Two Principal Components')
plt.show()

#-----------------------------

#-----------Here starts valence vs liveness plot-----------------

# Select the attributes for the scatter plot
attribute_x = 'acousticness_%'
attribute_y = 'valence_%'

#scatter_plot_AB = plt.figure()
dataframe.plot.scatter(x=attributes_for_PC1[0], y=attributes_for_PC2[0], label='Acousticness vs. Valence', color='blue')

# Calculate the line of best fit (linear regression)
fit = np.polyfit(df[attribute_x], df[attribute_y], 1)
line = np.polyval(fit, df[attribute_x])

# Plot the line of best fit
plt.plot(df[attribute_x], line, color='red', label='Linear Fit')
plt.title('Acousticness vs. Valence')
#plt.xlim(-10, 120)
#plt.ylim(-20, 200)
plt.legend()
plt.show()


#-----------------------------

#-----------Decision Tree-----------------
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
dtc_attr_streams_only = ['streams']
dtc_attr = ['streams', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
X_dtc_attr_streams_only = df[dtc_attr_streams_only]
X_dtc_attr = df[dtc_attr]
y_dtc_attr = df['mode'].ravel()

feature_names_list_streams_only = X_dtc_attr_streams_only.columns.tolist()
feature_names_list = X_dtc_attr.columns.tolist()

mode_counts = df['mode'].value_counts()
mode_counts = mode_counts.sort_values(ascending=False)
print(mode_counts)

dtc_streams_only = DecisionTreeClassifier(criterion='gini', min_samples_split=15)
dtc = DecisionTreeClassifier(criterion='gini', min_samples_split=15)
dtc_streams_only.fit(X_dtc_attr_streams_only, y_dtc_attr)
dtc.fit(X_dtc_attr, y_dtc_attr)

plt.figure(figsize=(100, 100))
plot_tree(dtc_streams_only, feature_names=feature_names_list_streams_only, class_names=['Major', 'Minor'], filled=True, rounded=True, impurity=True, fontsize=8)
plt.show()

plt.figure(figsize=(100, 100))
plot_tree(dtc, feature_names=feature_names_list, class_names=['Major', 'Minor'], filled=True, rounded=True, impurity=True, fontsize=8)
plt.show()

#------------------------------------

#---------Print leaf node with most streams------------

# Assuming dtc_streams_only is your trained decision tree model
leaf_indices = dtc_streams_only.apply(X_dtc_attr_streams_only)

# Find the index corresponding to the instance with the most streams
index_most_streams = X_dtc_attr_streams_only['streams'].idxmax()

# Find the leaf node for the instance with the most streams
leaf_node_most_streams = leaf_indices[index_most_streams]

print(f"The leaf node for the instance with the most streams is: {leaf_node_most_streams}")

#----------------------------

#----------Print number of samples in the leaf node we got------------------

leaf_node_index = 330

# Access the decision tree structure
tree_structure = dtc_streams_only.tree_

# Find the number of samples for the specified leaf node
num_samples_in_leaf = tree_structure.n_node_samples[leaf_node_index]

print(f"The number of samples in leaf node {leaf_node_index} is: {num_samples_in_leaf}")

#---------------------------------------------

#-------------Print the top 12 most streamed songs------------------

# Sort the DataFrame by 'streams' in descending order and take the top 10 rows
top_12_streamed_songs = df.sort_values(by='streams', ascending=False).head(12)

# Print the top 10 most streamed songs
print(top_12_streamed_songs[['track_name', 'artist(s)_name', 'streams', 'mode']].to_string(index=False))

#-----------------------------------------------

#-----------Print top 12 songs with attributes in order------------

# Iterate through the top 12 most streamed songs
for index, row in top_12_streamed_songs.iterrows():
    # Extract relevant information
    track_name = row['track_name']
    artist_name = row['artist(s)_name']
    streams = row['streams']
    
    # Extract and sort the specified attributes in descending order
    attributes = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
    sorted_attributes = sorted([(attr, row[attr]) for attr in attributes], key=lambda x: x[1], reverse=True)
    
    # Print the information for each song
    print(f"Track: {track_name} - Artist: {artist_name} - Streams: {streams}")
    for attr, value in sorted_attributes:
        print(f"{attr}: {value}")
    print('\n' + '-'*50 + '\n')