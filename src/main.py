
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import csv


with open(r"C:\Users\jessi\OneDrive\Dokument\GitHub\music-analysis\data\spotify-2023.csv", 'r') as f:
    df = pd.read_csv(f)
    
    print(df.head())


# Data Summary
data_summary = df.describe()

# Display the cleaned DataFrame without the index
df.reset_index(drop=True, inplace=True)

df.info()
df.head()

# Check for missing data again to verify that all null values have been replaced
missing_data = df.isnull().sum()
df.head()


# Select the relevant columns for the heatmap
features = df[['valence_%', 'energy_%', 'acousticness_%', 'danceability_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%', 'bpm']]

# Compute the correlation matrix
correlation_matrix = features.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Song Characteristics')
plt.show()






# class DataReader:
#     def __init__(self, file_path):
#         self.file_path = file_path

#     def load_data(self):
#         try:
#             data = pd.read_csv(self.file_path)
#             return data
#         except FileNotFoundError:
#             print(f"Error: File not found at {self.file_path}")
#             return None

# if __name__ == "__main__":
#     file_path = "data/spotify-2023.csv"
#     data_reader = DataReader(file_path)
#     data = data_reader.load_data()

#     if data is not None:
#         # Print the entire DataFrame
#         print(data.to_string())
#     else:
#         print("Failed to load data.")
