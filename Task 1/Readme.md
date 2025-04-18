# Model 1 : 
#  Used Car Price Prediction - Regression Model

This project aims to build a regression model to **predict the selling price of used cars** using the CarDekho dataset. It involves understanding, cleaning, and preprocessing the dataset before training the model.

---

##  Dataset Description

The dataset contains listings of used cars along with their technical and categorical attributes, sourced from CarDekho. It includes **4,334 entries** and **8 columns**, each representing a specific feature relevant to car resale value.

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

##  Data Preprocessing Steps

### 1.  Dropped Duplicates
- **Why?** Duplicate records lead to data leakage and model bias.
- **Action:** Removed all duplicate rows based on exact matches.

---

### 2.  Dropped Irregular or Invalid Rows
- Rows with manufacturing years earlier than 1900 or extremely low `km_driven` values were considered **invalid**.
- These rare anomalies can confuse the model and were thus **removed**.

---

### 3.  Outlier Detection and Handling
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

### 4.  Encoded Categorical Variables

| Column        | Type         | Encoding Strategy                            |
|---------------|--------------|-----------------------------------------------|
| `fuel`        | Nominal      | One-Hot Encoding                             |
| `seller_type` | Nominal      | One-Hot Encoding                             |
| `transmission`| Binary       | Label Encoding (Manual=0, Automatic=1)       |
| `owner`       | Ordinal-ish  | Custom Mapping (First Owner=0, Second=1...)  |

- **Why?** Machine learning models require numeric input. These transformations preserve category information while making it usable for the model.

---

### 5.  Feature Engineering: `car_age`
- Created a new column:  
