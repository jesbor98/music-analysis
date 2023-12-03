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
print("Attribute 1 primarily represented by PC1: " + attributes_for_PC1[0])
print("Attribute 1 primarily represented by PC2: " + attributes_for_PC2[0])
print("Attribute 2 primarily represented by PC1: " + attributes_for_PC1[1])
print("Attribute 2 primarily represented by PC2: " + attributes_for_PC2[1])
print("Attribute 3 primarily represented by PC1: " + attributes_for_PC1[2])
print("Attribute 3 primarily represented by PC2: " + attributes_for_PC2[2])

scatter_pc1_pc2 = plt.figure()
plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection onto the First Two Principal Components')
plt.show()

#-----------------------------

#-----------Decision Tree-----------------
split_percentage = 0.7
split_index = int(len(df) * split_percentage)

df_train = df.iloc[:split_index]

dtc_attr = ['valence_%', 'acousticness_%', 'energy_%', 'danceability_%']
X_dtc_attr = df_train[dtc_attr]
y_dtc_attr = df_train['mode'].ravel()

feature_names_list = X_dtc_attr.columns.tolist()

mode_counts = df_train['mode'].value_counts()
mode_counts = mode_counts.sort_values(ascending=False)
print(mode_counts)

dtc = DecisionTreeClassifier(criterion='gini', min_samples_split=15)
dtc.fit(X_dtc_attr, y_dtc_attr)

plt.figure(figsize=(100, 100))
plot_tree(dtc, feature_names=feature_names_list, class_names=['Major', 'Minor'], filled=True, rounded=True, impurity=True, fontsize=8)
plt.show()

#------------------------------------------------

#---------Confusion Matrix------------
df_test = df.iloc[split_index:]

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Keep track of sampled songs to avoid duplicates
sampled_songs = set()

for _ in range(50):
    # Randomly sample a row from df_test that hasn't been sampled before
    remaining_samples = df_test[~df_test.index.isin(sampled_songs)]
    if remaining_samples.empty:
        break
    random_sample = remaining_samples.sample(n=1)

    # Extract attributes for prediction
    X_random_sample = random_sample[dtc_attr]

    # Predict the mode for the random sample
    predicted_mode = dtc.predict(X_random_sample)[0]

    # Update confusion matrix variables
    actual_mode = random_sample['mode'].values[0]
    if actual_mode == 'Major' and predicted_mode == 'Major':
        true_positive += 1
    elif actual_mode == 'Minor' and predicted_mode == 'Minor':
        true_negative += 1
    elif actual_mode == 'Major' and predicted_mode == 'Minor':
        false_negative += 1
    elif actual_mode == 'Minor' and predicted_mode == 'Major':
        false_positive += 1

    # Add the sampled song to the set
    sampled_songs.add(random_sample.index[0])

# Create and print confusion matrix
conf_matrix = pd.DataFrame([[true_positive, false_negative], [false_positive, true_negative]],
                            index=['Actual Major', 'Actual Minor'],
                            columns=['Predicted Major', 'Predicted Minor'])
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap='Blues')
plt.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], conf_matrix.columns)
plt.yticks([0, 1], conf_matrix.index)
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix.iloc[i, j]), ha='center', va='center', color='black', fontsize=12)
plt.show()
# SKRIV KOMMENTAR HÄR ANGÅENDE ATT VI HADE PROBLEM MED CONF MATRIXEN

#--------------------------------------