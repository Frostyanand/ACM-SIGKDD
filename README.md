# ACM-SIGKDD-RECRUITMENT-TASK
Submission for the 2025 recruitment drive into the ACM SIGKDD club.

# Titanic Dataset Cleaning & Processing

This repository contains the cleaned and processed Titanic dataset along with the script used to handle missing values, filter data, and prepare it for analysis.

---

## Files in This Repository
- `final_task_script.py` – Python script for data cleaning, filtering, and processing.
- `tested.csv` – Original Titanic dataset (input).
- `final_titanic_dataset.csv` – Cleaned dataset (output).

---

## How the Data is Processed
The script follows a structured approach:
1. **Loading the Dataset**  
   - Reads `tested.csv` into a Pandas DataFrame.  
   - Displays initial shape and sample rows.

2. **Data Inspection**  
   - Checks missing values.  
   - Prints summary statistics and data types.  

3. **Selecting Relevant Columns**  
   - Drops `Cabin` (excessive missing values).  
   - Retains: `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.  

4. **Filtering Data**  
   - Keeps passengers in **Pclass 1 or 2** (business decision to focus on higher-class passengers).  

5. **Handling Missing Values**  
   - `Embarked`: Filled with mode (most frequent value).  
   - `Age` & `Fare`: Filled with median (robust to outliers).  

6. **Final Validation & Export**  
   - Ensures no missing values remain.  
   - Saves cleaned dataset to `final_titanic_dataset.csv`.  

---

## How to Run the Script
1. **Prerequisites**:  
   - Python 3+ and pandas installed.  
   - Place `tested.csv` in your working directory.  

2. **Steps**:  
   - Update the CSV path in the script if needed (line: `file_path = ...`).  
   - Run:  
     ```bash
     python "Anurag Anand Final Task Script.py"
     ```
   - The cleaned dataset will save to your **current working directory** (or a custom path if specified).  

---

## Why This Submission Stands Out
-  **Complete Task Coverage**: Handles all required steps (loading, inspection, selection, filtering, missing values).  
-  **Transparent Filtering**: Focuses on Pclass 1/2 passengers for targeted analysis.  
-  **Robust Missing Value Handling**: Uses median/mode to preserve data integrity.  
-  **One-Click Execution**: Script runs end-to-end with clear outputs.  

---

**Note**: Replace the CSV path in the script with your local path if needed.  
