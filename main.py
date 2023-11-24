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
decision_tree_attributes = ['streams', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
X_spotify = df[decision_tree_attributes]
y_spotify = df['mode'].ravel()

#attribute_names = [name[0] for name in df['attributeNames'][0]]
#class_names = [name[0][0] for name in df['classNames']]
feature_names_list = X_spotify.columns.tolist()

mode_counts = df['mode'].value_counts()
mode_counts = mode_counts.sort_values(ascending=False)
print(mode_counts)

dtc_spotify = DecisionTreeClassifier(criterion='gini', min_samples_split=100)
dtc_spotify.fit(X_spotify, y_spotify)

plt.figure(figsize=(100, 100))
plot_tree(dtc_spotify, feature_names=feature_names_list, class_names=['Minor', 'Major'], filled=True, rounded=True, impurity=True, fontsize=8)

plt.show()