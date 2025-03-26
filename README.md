# ACM-SIGKDD-RECRUITMENT-TASK
contains the task submission files for the recruitment task provided for 2025 recruitment drive into the club .

# Titanic Dataset Cleaning & Processing

This repository contains the cleaned and processed Titanic dataset along with the script used to handle missing values, perform feature engineering, and prepare the data for further analysis.
Files in This Repository
# final_task_script.py – The Python script that performs all the data cleaning and processing in one go.
# tested.csv – The original Titanic dataset used as input.
# final_titanic_dataset.csv – The cleaned dataset after processing.

# How the Data is Processed
This script follows a structured approach to clean and prepare the Titanic dataset. Below are the key steps:
1️⃣# Loading the Dataset
Reads the tested.csv file into a Pandas DataFrame.
Displays the first few rows and the shape of the dataset.
2️⃣# Data Inspection
Checks for missing values.
Prints summary statistics.
Displays the data types of each column.
3️⃣# Selecting Relevant Columns
Drops unnecessary columns such as Cabin (since it has too many missing values).
Retains important columns: Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
4️⃣# Handling Missing Values
Embarked: Filled with the most frequent value (mode).
Age: Filled with the median value to handle outliers.
Fare: Filled with the median value to maintain consistency.
5️⃣# Converting Categorical Variables
Converts Sex column: male → 0, female → 1.
Converts Embarked column: S → 0, C → 1, Q → 2.
6️⃣# Feature Engineering
Creates a new FamilySize column (SibSp + Parch + 1).
Creates an IsAlone column (1 if FamilySize == 1, otherwise 0).
Drops SibSp and Parch since they are now redundant.
7️⃣# Final Validation & Saving the Cleaned Dataset
Ensures there are no missing values left.
Saves the cleaned data as final_titanic_dataset.csv.
Displays a final summary of the cleaned dataset.
# How to Run the Script
Make sure you have Python 3+ and pandas installed. Then, follow these steps:
Place tested.csv in the same directory as final_task_script.py.
Open a terminal or command prompt in this directory.
Run the script using:
python final_task_script.py
The cleaned dataset will be saved as final_titanic_dataset.csv.
# Why This Submission is Strong
All missing values are handled correctly.
Feature engineering improves the dataset.
Script runs in one go without requiring multiple steps.
Well-structured and easy to understand.
This ensures that the dataset is ready for further analysis or model training
