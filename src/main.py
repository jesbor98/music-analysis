
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn
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
