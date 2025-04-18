# Model 1 : 
#  Used Car Price Prediction - Regression Model

This project aims to build a regression model to **predict the selling price of used cars** using the CarDekho dataset. It involves understanding, cleaning, and preprocessing the dataset before training the model.

---

##  Dataset Description

The dataset contains listings of used cars along with their technical and categorical attributes, sourced from CarDekho. It includes **4,334 entries** and **8 columns**, each representing a specific feature relevant to car resale value.
## Dataset Info:
Rows: 4340
Columns: 8
Duplicates: 763 rows →  Should be removed
Missing values: None  No nulls

Data type mix: Contains numerical and categorical features
###  Column Overview

| Column Name     | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `name`          | Full name of the car (Make, Model, Variant) – mostly unstructured text      |
| `year`          | Year of manufacture                                                        |
| `selling_price` | **Target column** – price (in INR) at which the car is listed              |
| `km_driven`     | Number of kilometers driven                                                 |
| `fuel`          | Type of fuel used – Petrol, Diesel, CNG, etc.                              |
| `seller_type`   | Type of seller – Individual, Dealer, or Trustmark Dealer                   |
| `transmission`  | Transmission type – Manual or Automatic                                    |
| `owner`         | Ownership status – First, Second, Third, etc.                              |

---

### Column Data-Type 

| Column         | Description                             | Data Type |
|----------------|-----------------------------------------|-----------|
| `name`         | Car model name                          | object    |
| `year`         | Year of manufacture                     | int       |
| `selling_price`| ✅ **Target**: Price of car (in INR)     | int       |
| `km_driven`    | Kilometers driven                       | int       |
| `fuel`         | Type of fuel (e.g., Petrol, Diesel)     | object    |
| `seller_type`  | Individual / Dealer / Trustmark         | object    |
| `transmission` | Manual / Automatic                      | object    |
| `owner`        | Ownership history (First, Second, etc.) | object    |

##  Data Preprocessing Steps

### 1.  Dropped Duplicates 
The dataset contains 763 duplicate rows.
- **Why?** Duplicate records lead to data leakage and model bias.
- **Action:** Removed all duplicate rows based on exact matches.

---

### 2.  Outlier Detection and Handling
- The `selling_price` column showed **severe right-skew**, with a few listings priced extremely high (₹50L+).
- Price distribution:

  | Price Range      | Cars |
  |------------------|------|
  | < ₹1 Lakh        | 318  |
  | ₹1L – ₹3L         | 1,274|
  | ₹3L – ₹5L         | 851  |
  | ₹5L – ₹10L        | 888  |
  | ₹10L – ₹20L       | 190  |
  | ₹20L – ₹50L       | 53   |
  | > ₹50L           | **3**|

- **Action:** Listings priced above ₹20L were considered outliers and **dropped** to prevent skewing model training.

---

### 3.  Encoded Categorical Variables

| Column        | Type         | Encoding Strategy                            |
|---------------|--------------|-----------------------------------------------|
| `fuel`        | Nominal      | One-Hot Encoding                             |
| `seller_type` | Nominal      | One-Hot Encoding                             |
| `transmission`| Binary       | Label Encoding (Manual=0, Automatic=1)       |
| `owner`       | Ordinal-ish  | Custom Mapping (First Owner=0, Second=1...)  |

- **Why?** Machine learning models require numeric input. These transformations preserve category information while making it usable for the model.

#### One-Hot Encoding
Used when the column has no natural order (just categories), like fuel types or seller types.

fuel
-----
Petrol
Diesel
CNG

After One-Hot Encoding, it becomes:
fuel_Petrol  fuel_Diesel  fuel_CNG
     1            0           0
     0            1           0
     0            0           1
Used for:
fuel → ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
seller_type → ['Individual', 'Dealer', 'Trustmark Dealer']

####  Label Encoding (Binary Category)
Use when a column has only two categories, and there's no order between them.

Example: For transmission:

transmission
-------------
Manual
Automatic

We can convert to -
Manual    → 0  
Automatic → 1
Works for binary columns like transmission.

#### Custom Encoding (Ordinal)
Use when the categories have some logical order (like ownership: 1st owner, 2nd owner…).

Example:
For owner:

owner
---------------------
First Owner          → 0
Second Owner         → 1
Third Owner          → 2
Fourth & Above Owner → 3
Test Drive Car       → 4 (or can handle separately)

---

### 4.  Feature Engineering: `car_age`
- Created a new column:  
	Newer cars usually cost more. Calculating car_age = current_year - year is more meaningful.

### 3.  Drop Columns:
Column	Reason
year	After we convert to car_age, year is no longer needed
name	Too many unique values (1491), hard to encode meaningfully .

### Feature Matrix (X) and Target Vector (y)
X = All columns except selling_price
y = The selling_price column

## 🧼 Clean/Transform km_driven
No nulls, but range is wide (1 to 8+ lakh km)
Apply log transformation to reduce outlier skew:
df['km_driven'] = np.log1p(df['km_driven'])

### Train-Test Split
Split your data into:

80% training data (used to train the model)
20% test data (used to evaluate the model's performance)
