# Student Depression Prediction Model

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
  - [Categorical Data Encoding](#categorical-data-encoding)
  - [Feature Engineering](#feature-engineering)
  - [Data Cleaning](#data-cleaning)
- [Feature Selection and Scaling](#feature-selection-and-scaling)
- [Model Development](#model-development)
  - [Model Selection](#model-selection)
  - [Training Process](#training-process)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
  - [Performance Metrics](#performance-metrics)
  - [Cross-Validation](#cross-validation)
  - [Feature Importance](#feature-importance)
- [Results and Insights](#results-and-insights)
- [Ethical Considerations](#ethical-considerations)
- [Limitations and Future Work](#limitations-and-future-work)
- [Technical Implementation](#technical-implementation)
- [Conclusion](#conclusion)
- [References](#references)

## Project Overview

This project develops a machine learning model to predict depression in students based on various factors such as academic performance, lifestyle habits, social factors, and demographic information. Depression among students is a significant concern in educational institutions worldwide, affecting academic performance, social interactions, and overall quality of life. Early identification of at-risk students can facilitate timely intervention and support.

Our model aims to provide an objective tool that can help identify students who may be at risk of depression, enabling proactive support measures from educational institutions. This predictive model is not intended to replace professional diagnosis but rather to serve as a screening tool to guide mental health resource allocation and support services.

The project follows a comprehensive data science workflow: exploratory data analysis, preprocessing, feature engineering, model selection, training, and evaluation. We employ various statistical and machine learning techniques to build a robust predictive model with high accuracy and sensitivity.

## Dataset Description

The dataset used in this project contains information about 27,901 students with 18 different attributes covering various aspects of student life and mental health indicators:

| Feature | Description | Data Type |
|---------|-------------|-----------|
| id | Unique identifier for each student | Integer |
| Gender | Gender of the student (Male/Female) | Categorical |
| Age | Age of the student | Numerical (Float) |
| City | City of residence | Categorical |
| Profession | Student or other profession | Categorical |
| Academic Pressure | Level of academic pressure (0-5 scale) | Numerical |
| Work Pressure | Level of work pressure (0-5 scale) | Numerical |
| CGPA | Cumulative Grade Point Average (0-10 scale) | Numerical |
| Study Satisfaction | Level of satisfaction with studies (0-5 scale) | Numerical |
| Job Satisfaction | Level of satisfaction with job (0-5 scale) | Numerical |
| Sleep Duration | Daily sleep duration category | Categorical |
| Dietary Habits | Quality of dietary habits | Categorical |
| Degree | Level of education | Categorical |
| Suicidal Thoughts | Whether the student has had suicidal thoughts | Binary |
| Work/Study Hours | Daily hours dedicated to work/study | Numerical |
| Financial Stress | Level of financial stress (1-5 scale) | Numerical |
| Family History of Mental Illness | Family history of mental illness | Binary |
| Depression | Whether the student is diagnosed with depression (target variable) | Binary |

The dataset provides a comprehensive view of factors that may contribute to depression in students, including academic pressures, lifestyle habits, and personal circumstances. The target variable, "Depression," is binary (0 for not depressed, 1 for depressed), making this a binary classification problem.

## Exploratory Data Analysis

### Initial Data Exploration

Our first step involved understanding the basic properties of the dataset:

```python
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
df.info()
```

This revealed the dataset contained 27,901 rows and 18 columns with various data types: numeric (float64, int64) and categorical (object). All columns had complete data with no null values initially detected.

### Statistical Summary

The statistical analysis provided insights into the distribution and range of numerical features:

```python
df.describe(include='all')
```

Key findings:
- Age range: 18-59 years (mean: 25.8)
- Academic Pressure: 0-5 scale (mean: 3.14)
- CGPA: 0-10 scale (mean: 7.66)
- Most students reported low Work Pressure (mean: 0.00043)
- Work/Study Hours ranged from 0-12 hours (mean: 7.16)

### Class Distribution Analysis

The target variable "Depression" showed an imbalanced distribution:
- Depressed: 58.55% (16,304 students)
- Not Depressed: 41.45% (11,541 students)

This slight imbalance was noted for consideration during model training and evaluation.

### Feature Exploration

#### Categorical Features Analysis

For categorical features, we examined unique values and frequencies:

```python
for col in categorical_columns:
    print(f"{col} - Unique values:")
    print(df[col].unique())
    print(df[col].value_counts())
```

Interesting findings:
- Gender: 55.72% Male, 44.28% Female
- Cities: 52 unique values, with some invalid entries
- Sleep Duration: 5 categories, with "Less than 5 hours" being most common (29.78%)
- Dietary Habits: 4 categories with "Unhealthy" being most common (37.0%)
- Suicidal Thoughts: 63.28% reported "Yes"

#### Numerical Features Analysis

For numerical features, we examined distributions and correlations:

```python
# Distribution plots
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
# Correlation matrix
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

Key correlations:
- Moderate positive correlation (0.32) between Academic Pressure and Financial Stress
- Negative correlation (-0.28) between Study Satisfaction and Academic Pressure
- Weak positive correlation (0.21) between CGPA and Study Satisfaction

### Data Quality Assessment

During EDA, we identified several data quality issues:
1. Invalid city names (e.g., names like "3.0", "M.Tech" appearing in the City column)
2. Inconsistent formatting in categorical columns (e.g., varying case, extra spaces)
3. Quotes in text fields (e.g., 'Less than 5 hours')
4. Missing values represented as "?" in Financial Stress column
5. Outliers in some numerical columns

These insights guided our data preprocessing strategy to ensure data quality and reliability for model development.

## Data Preprocessing

Based on our exploratory data analysis, we implemented a comprehensive preprocessing pipeline to prepare the data for modeling.

### Handling Missing Values

While the dataset appeared complete initially, deeper analysis revealed masked missing values:

```python
# Checking for '?' values in Financial Stress column
(df['Financial Stress'] == '?').sum()
```

We found 3 records with "?" in the Financial Stress column, which were treated as missing values:

```python
# Replace '?' with NaN
df['Financial Stress'] = df['Financial Stress'].replace('?', np.nan)

# Convert to float
df['Financial Stress'] = df['Financial Stress'].astype(float)

# Drop rows with missing Financial Stress (only 3 rows)
df = df[df['Financial Stress'].notna()].copy()
```

### Categorical Data Encoding

#### Gender Encoding

Gender was encoded using Label Encoding since it's a binary feature:

```python
from sklearn.preprocessing import LabelEncoder

# Clean and standardize
df['Gender'] = df['Gender'].str.strip().str.capitalize()

# Encode
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
# Output: {'Female': 0, 'Male': 1}
```

#### Sleep Duration Processing

Sleep Duration required special handling due to quotes and inconsistent formatting:

```python
# Remove quotes and standardize
df['Sleep Duration'] = df['Sleep Duration'].str.replace("'", "").str.strip().str.lower()

# Drop rows with 'others' (only 18 rows)
df = df[df['Sleep Duration'] != 'others'].copy()

# Map to ordinal values
sleep_map = {
    'less than 5 hours': 1,
    '5-6 hours': 2,
    '7-8 hours': 3,
    'more than 8 hours': 4
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)
```

#### Dietary Habits Encoding

Dietary Habits were mapped to ordinal values representing quality:

```python
# Clean and standardize
df['Dietary Habits'] = df['Dietary Habits'].str.strip().str.capitalize()

# Drop 'Others' (only 12 rows)
df = df[df['Dietary Habits'] != 'Others'].copy()

# Map to ordinal scale
diet_map = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)
```

#### Suicidal Thoughts and Family History Encoding

Binary categorical variables were encoded using Label Encoder:

```python
# Suicidal thoughts
col = 'Have you ever had suicidal thoughts ?'
df[col] = df[col].str.strip().str.capitalize()
le_suicide = LabelEncoder()
df[col] = le_suicide.fit_transform(df[col])
# Output: {'No': 0, 'Yes': 1}

# Family history
col = 'Family History of Mental Illness'
df[col] = df[col].str.strip().str.capitalize()
le_family = LabelEncoder()
df[col] = le_family.fit_transform(df[col])
# Output: {'No': 0, 'Yes': 1}
```

### Feature Engineering

#### Degree Grouping

We created a new feature by grouping degrees into broader categories:

```python
undergrad = {'BA', 'BSC', 'BCA', 'BBA', 'B.COM', 'B.PHARM', 'BE', 'B.ED', 'B.ARCH', 'LLB', 'BHM', 'B.TECH'}
postgrad = {'MA', 'MBA', 'MSC', 'MCA', 'M.COM', 'M.ED', 'M.PHARM', 'ME', 'M.TECH', 'MD', 'MHM', 'LLM'}
others = {'OTHERS', "'CLASS 12'", 'PHD'}

def map_degree(degree):
    if degree in undergrad:
        return 'Undergraduate'
    elif degree in postgrad:
        return 'Postgraduate'
    else:
        return 'Other'

df['Degree Grouped'] = df['Degree'].apply(map_degree)
df['Degree Grouped'] = LabelEncoder().fit_transform(df['Degree Grouped'])
# Output: {'Other': 0, 'Postgraduate': 1, 'Undergraduate': 2}
```

### Data Cleaning

#### City Cleaning

The City column required extensive cleaning due to invalid entries:

```python
# Standardize format
df['City'] = df['City'].str.strip().str.lower()

# Define valid cities
valid_cities = {
    'visakhapatnam', 'bangalore', 'srinagar', 'varanasi', 'jaipur', 'pune', 'thane',
    'chennai', 'nagpur', 'nashik', 'vadodara', 'kalyan', 'rajkot', 'ahmedabad',
    'kolkata', 'mumbai', 'lucknow', 'indore', 'surat', 'ludhiana', 'bhopal',
    'meerut', 'agra', 'ghaziabad', 'hyderabad', 'vasai-virar', 'kanpur', 'patna',
    'faridabad', 'delhi'
}

# Mark invalid cities as NaN
df['City'] = df['City'].apply(lambda x: x if x in valid_cities else np.nan)

# Drop rows with invalid cities (26 rows)
df = df[df['City'].notna()].copy()
```

#### Profession Analysis and Removal

After analyzing the Profession column, we found it was highly imbalanced with 99.8% being "student":

```python
print(df['Profession'].value_counts())
# student: 27,814 (99.8%)
# other professions: <1%

# Since this is specifically about student depression and almost all are students, 
# we dropped this column as it provided little predictive value
df.drop(columns=['Profession'], inplace=True)
```

#### ID Column Removal

The ID column was removed as it had no predictive value:

```python
df.drop(columns=['id'], inplace=True)
```

## Feature Selection and Scaling

### One-Hot Encoding for City

City was one-hot encoded to convert it to a format suitable for machine learning:

```python
# One-hot encode city (drop_first to avoid multicollinearity)
city_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cities = city_encoder.fit_transform(df[['City']])
city_columns = [f"City_{city}" for city in city_encoder.categories_[0][1:]]
city_df = pd.DataFrame(encoded_cities, columns=city_columns)

# Add encoded columns to main dataframe
df = pd.concat([df.drop(columns=['City']), city_df], axis=1)
```

### Feature Correlation Analysis

We performed correlation analysis to identify relationships between features:

```python
numeric_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                   'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 
                   'Financial Stress']

correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Numeric Features')
plt.show()
```

Key insights from correlation analysis:
- Academic Pressure and Financial Stress showed moderate correlation (0.32)
- Study Satisfaction negatively correlated with Academic Pressure (-0.28)
- CGPA showed positive correlation with Study Satisfaction (0.21)
- No features showed strong enough correlation to warrant removal due to multicollinearity

### Feature Scaling

Numeric features were standardized to ensure equal contribution to the model:

```python
from sklearn.preprocessing import StandardScaler

features_to_scale = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                     'Study Satisfaction', 'Job Satisfaction', 
                     'Work/Study Hours', 'Financial Stress']

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
```

### Final Feature Set Preparation

The final feature set was prepared by removing unnecessary columns:

```python
# Drop the original Degree column (keep the grouped version)
X = df.drop(columns=['Depression', 'Degree'])
y = df['Depression']
```

The final dataset contained:
- 27,845 rows (after cleaning)
- Over 35 columns (after one-hot encoding)
- Target variable: Depression (binary: 0 or 1)

## Model Development

### Train-Test Split

We split the data into training (80%) and testing (20%) sets, ensuring balanced class distribution:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
```

### Model Selection

We implemented and compared multiple classification algorithms to identify the best performer:

1. **Logistic Regression**: A linear model for binary classification
2. **Random Forest**: An ensemble of decision trees
3. **Gradient Boosting**: A boosting algorithm that builds trees sequentially
4. **Support Vector Machine (SVM)**: A powerful algorithm for classification with margin optimization

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}
```

### Training Process

Each model was trained using the same training dataset, with consistent methodology:

```python
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics (accuracy, AUC, etc.)
    accuracy = accuracy_score(y_test, y_pred)
    
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = "N/A"
    
    # Additional metrics and visualizations...
    
    return model, accuracy, auc
```

### Hyperparameter Tuning

For the Random Forest model (which showed the best initial performance), we conducted hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
```

The best hyperparameters were:
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2

## Model Evaluation

### Performance Metrics

We evaluated each model using comprehensive metrics:

1. **Accuracy**: Percentage of correct predictions
2. **Precision**: True positives divided by predicted positives
3. **Recall**: True positives divided by actual positives
4. **F1 Score**: Harmonic mean of precision and recall
5. **AUC-ROC**: Area Under the Receiver Operating Characteristic curve

Results for the best model (Random Forest):

| Metric | Value |
|--------|-------|
| Accuracy | 0.8976 |
| Precision | 0.8854 |
| Recall | 0.9236 |
| F1 Score | 0.9041 |
| AUC-ROC | 0.9417 |

```python
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Depressed', 'Depressed'],
            yticklabels=['Not Depressed', 'Depressed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
```

### Cross-Validation

We performed 5-fold cross-validation to ensure model robustness:

```python
cv_scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='roc_auc')
print(f"5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
```

Cross-validation results:
- Individual fold scores: [0.9384, 0.9402, 0.9397, 0.9431, 0.9418]
- Mean CV Score: 0.9406

The consistent performance across folds indicates model stability and generalizability.

### Feature Importance

We analyzed feature importance from the Random Forest model to understand which factors have the strongest influence on depression prediction:

```python
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importance (Random Forest)')
plt.show()
```

Top 10 most important features:
1. Suicidal Thoughts (0.2137)
2. Financial Stress (0.1423)
3. Academic Pressure (0.0915)
4. Sleep Duration (0.0872)
5. Study Satisfaction (0.0803)
6. CGPA (0.0762)
7. Work/Study Hours (0.0715)
8. Dietary Habits (0.0693)
9. Age (0.0524)
10. Family History of Mental Illness (0.0483)

## Results and Insights

### Model Comparison

We compared all models to determine the most effective approach:

| Model | Accuracy | AUC-ROC | F1 Score |
|-------|----------|---------|----------|
| Logistic Regression | 0.8423 | 0.9065 | 0.8618 |
| Random Forest | 0.8976 | 0.9417 | 0.9041 |
| Gradient Boosting | 0.8891 | 0.9379 | 0.8967 |
| SVM | 0.8357 | 0.8974 | 0.8553 |

The Random Forest model outperformed other algorithms across all metrics, showing superior accuracy, AUC-ROC, and F1 scores.

### Key Findings

1. **Psychological Factors**: Suicidal thoughts emerged as the strongest predictor of depression, highlighting the critical link between these mental health conditions.

2. **Socioeconomic Factors**: Financial stress ranked as the second most important predictor, underscoring the significant impact of economic hardship on mental health.

3. **Academic Factors**: Academic pressure and study satisfaction were highly influential, confirming that academic experiences substantially affect students' mental well-being.

4. **Lifestyle Factors**: Sleep duration and dietary habits showed significant importance, emphasizing the connection between physical health behaviors and mental health outcomes.

5. **Performance Indicators**: CGPA (academic performance) was an important predictor, though less influential than psychological and stress-related factors.

### Practical Applications

The model's predictions can be utilized to:

1. **Early Intervention**: Identify at-risk students before clinical symptoms become severe.
2. **Resource Allocation**: Help educational institutions allocate mental health resources more effectively.
3. **Tailored Support Programs**: Design targeted support programs addressing the most influential risk factors.
4. **Awareness Initiatives**: Focus mental health awareness campaigns on key risk factors identified by the model.

## Ethical Considerations

Throughout this project, we prioritized ethical considerations in developing and applying the depression prediction model:

### Privacy and Confidentiality

- All data was anonymized with no personally identifiable information.
- The model is intended as a screening tool, not for sharing individual predictions without consent.

### Bias and Fairness

- We assessed the model for potential biases across demographic groups.
- Gender and city distributions were analyzed to ensure fair predictions across different populations.

### Appropriate Use

- The model is designed as a supplementary screening tool, not a replacement for professional diagnosis.
- Clear documentation emphasizes the model's limitations and appropriate implementation contexts.

### Transparency

- Model methodology, limitations, and performance metrics are transparently documented.
- Feature importance analysis provides interpretability of predictions.

## Limitations and Future Work

### Current Limitations

1. **Self-Reported Data**: The dataset relies on self-reported information, which may contain inaccuracies or reporting biases.

2. **Cross-Sectional Nature**: The data represents a single point in time, limiting understanding of how depression risk factors evolve.

3. **Geographic Constraints**: The dataset predominantly features Indian cities, potentially limiting generalizability to other regions.

4. **Limited Contextual Information**: The dataset lacks certain potentially relevant factors like social support systems, exercise habits, or substance use.

5. **Binary Classification**: The model predicts depression as binary (yes/no) rather than assessing severity or specific depression types.

### Future Directions

1. **Longitudinal Analysis**: Incorporate time-series data to understand how risk factors and depression evolve over a student's academic journey.

2. **Additional Features**: Include more comprehensive lifestyle and behavioral factors like social media usage, exercise habits, and substance use.

3. **Multi-class Classification**: Extend the model to predict depression severity levels or specific depression subtypes.

4. **Explainable AI Techniques**: Implement more advanced explainability methods like SHAP (SHapley Additive exPlanations) values for better feature interpretation.

5. **Intervention Effectiveness**: Develop companion models to predict which interventions might be most effective for specific student profiles.

6. **Cross-Cultural Validation**: Test and adapt the model across diverse cultural and geographic contexts to improve generalizability.

## Technical Implementation

### Environment and Dependencies

The project was implemented using Python 3.8 with the following key libraries:

```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
matplotlib==3.5.1
seaborn==0.11.2
joblib==1.1.0
```

### Code Structure

The project is organized into the following components:

1. **Data Loading and Exploration**: Initial loading and exploratory analysis of the dataset.
2. **Data Preprocessing**: Cleaning, encoding, and feature engineering.
3. **Feature Selection and Scaling**: Preparing features for modeling.
4. **Model Training**: Implementation and training of multiple classification models.
5. **Model Evaluation**: Comprehensive evaluation and comparison of model performance.
6. **Feature Importance Analysis**: Understanding feature contributions to predictions.
7. **Model Persistence**: Saving the best model for future use.

### Model Deployment

The best-performing model (Random Forest) was saved for deployment:

```python
import joblib
joblib.dump(best_rf_model, 'student_depression_model.pkl')
```

For inference, the model can be loaded and used as follows:

```python
# Load the model
loaded_model = joblib.load('student_depression_model.pkl')

# Preprocess new data using the same pipeline
# ...

# Make predictions
predictions = loaded_model.predict(new_data)
probabilities = loaded_model.predict_proba(new_data)[:, 1]
```

## Conclusion

This project successfully developed a high-performing machine learning model for predicting depression risk in students, achieving 89.76% accuracy and 94.17% AUC-ROC score. The Random Forest algorithm emerged as the most effective approach, outperforming other classification models.

Our analysis revealed that psychological factors (particularly suicidal thoughts), financial stress, and academic pressures were the most influential predictors of depression in students. Lifestyle factors like sleep duration and dietary habits also played significant roles.

The model provides a valuable tool for educational institutions to identify students at risk of depression and allocate resources more effectively. By focusing on the most significant risk factors identified in our analysis, targeted intervention programs can be developed to support student mental health.

While the model demonstrates strong predictive performance, it should be used as a complementary screening tool alongside professional assessment, respecting privacy considerations and acknowledging its limitations. Future work should focus on incorporating more diverse data sources, longitudinal tracking, and cross-cultural validation to further enhance the model's utility and generalizability.
