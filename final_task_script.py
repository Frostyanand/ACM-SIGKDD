import pandas as pd

# Load the dataset
file_path = r"D:\ACM SIGKDD R&D RECRUITMENT TASK\Task 1 Titanic Dataset Handling\tested.csv"  # replace with the correct csv file path here
df = pd.read_csv(file_path)
print(" Dataset Loaded")
print(f"Shape: {df.shape}")
print(df.head())

# Inspect Data
print("\n Dataset Information:")
print(df.info())
print("\n Summary Statistics:")
print(df.describe())
print("\n Missing Values Before Handling:")
print(df.isnull().sum())

# Select Relevant Columns (Cabin removed due to excessive missing values)
selected_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[selected_columns]
print("\n Selected Columns:")
print(df.head())

# Handle Missing Values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing Embarked values with mode
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill missing Age with median (robust to outliers)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Fill missing Fare with median

# Convert Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df.drop(columns=['SibSp', 'Parch'], inplace=True)
print("\n Feature Engineering Completed")

# Verify missing values after fixing
print("\n Missing Values After Fix:")
print(df.isnull().sum())

# Final Assertions (Critical for validation)
assert df['Age'].isnull().sum() == 0, "❌ Age has missing values!"
assert df['Embarked'].isnull().sum() == 0, "❌ Embarked has missing values!"
assert df['Fare'].isnull().sum() == 0, "❌ Fare has missing values!"

# Save Cleaned Data
df.to_csv("final_titanic_dataset.csv", index=False)
print("\n Final cleaned dataset saved as 'final_titanic_dataset.csv'")

# Final Summary
print("\n🔍 Final Dataset Verification:")
print(f"Shape: {df.shape}")
print(df.describe(include='all'))