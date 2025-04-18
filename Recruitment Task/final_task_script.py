# Anurag Anand
# RA2411003010760 SRM K1 , aa9004@srmist.edu.in
# TASK 1: TITANIC DATASET HANDLING


## to run replace the path for csv file with your system path and the output path to get the final edited csv file , pre-ran and availble on my github repo already

# LOADING DATA
import pandas as pd

# Load dataset from CSV
# putt your system path here for csv file, also available on my github repo
file_path = r"D:\ACM SIGKDD R&D RECRUITMENT TASK\Task 1 Titanic Dataset Handling\tested.csv" 
df = pd.read_csv(file_path)
print("\n Dataset Loaded")
print(f"Initial Shape: {df.shape}")
print(df.head())

# DATA INSPECTION

print("\n\n DATASET INSPECTION:")
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe(include='all'))
print("\nMissing Values Before Handling:")
print(df.isnull().sum())


#  DATA SELECTION

# Select relevant columns (drop 'Cabin' due to missing values)
selected_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[selected_columns]
print("\n\n Selected Columns:")
print(f"New Shape: {df.shape}")
print(df.head())

#  FILTERING 

#  Filter rows to retain specific data (e.g., passengers in Pclass 1 or 2)
print("\n\n FILTERING DATA...")
filter_condition = df['Pclass'].isin([1, 2])  # Example: Keep First and Second Class passengers
df = df[filter_condition]
print(f"Shape After Filtering: {df.shape}")
print("\nFiltered Data Samples:")
print(df.head())

#  HANDLING MISSING DATA

# Fill missing values for Age, Fare, and Embarked
print("\n\n HANDLING MISSING VALUES...")
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Mode for Embarked
df['Age'] = df['Age'].fillna(df['Age'].median())  # Median for Age
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Median for Fare
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# VALIDATION & EXPORT

# Assert no missing values remain
assert df['Age'].isnull().sum() == 0, " Age has missing values!"
assert df['Embarked'].isnull().sum() == 0, " Embarked has missing values!"
assert df['Fare'].isnull().sum() == 0, " Fare has missing values!"

# Save cleaned dataset
#or use absolute path using the 3 below lines  if current directory causes issues in the system being used to run
#save_path = r"C:\Users\anura\OneDrive\Desktop\final_titanic_dataset.csv" 
#df.to_csv(save_path, index=False)
#print(f"\n\n FINAL DATASET SAVED TO: {save_path}")


#comment these two lines below if using absolute path 
df.to_csv("final_titanic_dataset.csv", index=False)
print("\n\n FINAL DATASET SAVED AS 'final_titanic_dataset.csv'")

# Final summary
print("\n\n FINAL DATASET VERIFICATION:")
print(f"Shape: {df.shape}")
print(df.describe(include='all'))

