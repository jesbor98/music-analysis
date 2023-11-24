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
    
#-----------Here starts PCA with all attributes------------------
selected_attributes = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

dataframe = df[selected_attributes]
dataframe = dataframe.dropna()

# Standardize the data


X = dataframe.values
mu = np.mean(X, axis=0)

Y = X - mu
# Apply PCA for dimensionality reduction
U, S, VT = np.linalg.svd(Y)

V = np.transpose(VT)
Z = np.dot(Y, V)

PC1 = V[:, 0]
PC2 = V[:, 1]

#Find the attributes that are primarily represented by the first and second PC
attributes_for_PC1 = dataframe.columns[np.argsort(np.abs(PC1))[::-1]]
attributes_for_PC2 = dataframe.columns[np.argsort(np.abs(PC2))[::-1]]
print("Attributes primarily represented by the first PC: " + attributes_for_PC1[0])
print("Attributes primarily represented by the second PC: " + attributes_for_PC2[0])

scatter_pc1_pc2 = plt.figure()
plt.scatter(Z[:, 1], Z[:, 2], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection onto the First Two Principal Components with Streams')
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