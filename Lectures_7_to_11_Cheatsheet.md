# Python Data Science Cheatsheet (Lectures 7-11)
## Exams Provide Step-by-Step Instructions - Use This as Reference

---

## **LECTURE 7: INTRODUCTION TO STATISTICS**

### **1. Descriptive Statistics**

#### **Mean & Median**
```python
# For a specific column or subset
mean_val = df['column_name'].mean()
median_val = df['column_name'].median()

# For groups
df.groupby('category')['column_name'].agg(['mean', 'median'])

# For filtered data
subset = df[df['condition']]
mean_val = subset['column_name'].mean()
```

#### **Quantiles**
```python
import numpy as np

# Quartiles (0, 0.25, 0.5, 0.75, 1)
np.quantile(df['column'], [0, 0.25, 0.5, 0.75, 1])

# Quintiles
np.quantile(df['column'], [0, 0.2, 0.4, 0.6, 0.8, 1])

# Deciles
np.quantile(df['column'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
```

#### **Variance & Standard Deviation**
```python
# For a single column
variance = df['column'].var()
std_dev = df['column'].std()

# For groups
df.groupby('category')['column'].agg(['var', 'std'])
```

#### **Histograms**
```python
import matplotlib.pyplot as plt
%matplotlib inline

# Simple histogram
df['column'].hist()
plt.show()

# Histogram with bins
df['column'].hist(bins=10)
plt.show()

# Grouped histograms
import seaborn as sns
sns.displot(data=df, x='column', col='category', col_wrap=2, bins=9)
plt.show()
```

---

### **2. Outlier Detection (IQR Method)**

```python
import numpy as np

# Calculate by group if needed
grouped_data = df.groupby('category')['numeric_column'].sum()

# Calculate quartiles and IQR
q1 = np.quantile(grouped_data, 0.25)
q3 = np.quantile(grouped_data, 0.75)
iqr = q3 - q1

# Calculate cutoffs
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# Find outliers
outliers = grouped_data[(grouped_data < lower) | (grouped_data > upper)]
print("Outliers:")
print(outliers)

# Find non-outliers
non_outliers = grouped_data[(grouped_data >= lower) & (grouped_data <= upper)]
```

---

### **3. Probability & Distributions**

#### **Discrete Probability Distribution**
```python
# Count occurrences and calculate probabilities
counts = df['category'].value_counts()
probs = counts / df.shape[0]
print(probs)
```

#### **Continuous Uniform Distribution**
```python
from scipy.stats import uniform

# Probability less than X (min=0, max=30)
prob_less_than_5 = uniform.cdf(5, 0, 30)
print(f"P(X < 5): {prob_less_than_5}")

# Probability greater than X
prob_greater_than_5 = 1 - uniform.cdf(5, 0, 30)
print(f"P(X > 5): {prob_greater_than_5}")

# Probability between A and B
prob_between = uniform.cdf(20, 0, 30) - uniform.cdf(10, 0, 30)
print(f"P(10 < X < 20): {prob_between}")
```

#### **Expected Value (for Discrete Distributions)**
```python
# Create probability distribution
size_dist = df['column'].value_counts() / df.shape[0]
size_dist = size_dist.reset_index()
size_dist.columns = ['value', 'probability']

# Calculate expected value
expected_value = np.sum(size_dist['value'] * size_dist['probability'])
print(f"Expected Value: {expected_value}")

# Probability of condition
prob_condition = np.sum(size_dist[size_dist['value'] >= threshold]['probability'])
```

---

### **4. Sampling**

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(24)

# Sample without replacement
sample_without = df.sample(5)

# Sample with replacement
sample_with = df.sample(5, replace=True)

# Sampling distribution of mean
sample_means = []
for i in range(100):
    sample = df['column'].sample(20, replace=True)
    sample_means.append(sample.mean())

# Plot sampling distribution
import pandas as pd
pd.Series(sample_means).hist()
plt.show()
```

---

### **5. Correlation**

```python
import seaborn as sns

# Scatterplot
sns.scatterplot(x='col1', y='col2', data=df)
plt.show()

# Scatterplot with trendline
sns.lmplot(x='col1', y='col2', data=df, ci=None)
plt.show()

# Calculate correlation coefficient
correlation = df['col1'].corr(df['col2'])
print(f"Correlation: {correlation}")
```

---

### **6. Data Transformations**

```python
# Log transformation (for right-skewed data)
df['log_column'] = np.log(df['numerical_column'])

# Square root transformation
df['sqrt_column'] = np.sqrt(df['numerical_column'])

# Square/cube transformations
df['squared'] = df['column'] ** 2
df['cubed'] = df['column'] ** 3
```

---

## **LECTURE 8: REGRESSION**

### **1. Visualizing Relationships**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot
sns.scatterplot(x='independent_var', y='dependent_var', data=df)
plt.show()

# Scatter plot with regression line
sns.regplot(x='independent_var', y='dependent_var', data=df, ci=None)
plt.show()

# Scatter plot with transparency
sns.regplot(x='independent_var', y='dependent_var', data=df, 
            ci=None, scatter_kws={'alpha': 0.5})
plt.show()
```

---

### **2. Simple Linear Regression**

```python
from statsmodels.formula.api import ols

# Create model: dependent ~ independent
model = ols('dependent_var ~ independent_var', data=df)
model_result = model.fit()

# Print model parameters (intercept and slope)
print(model_result.params)

# Get model summary (R-squared, p-values, etc.)
print(model_result.summary())
```

---

### **3. Regression with Categorical Variables**

```python
# Numeric coding for categories (no intercept)
model = ols('dependent_var ~ categorical_var + 0', data=df)
model_result = model.fit()
print(model_result.params)

# With intercept (baseline category omitted)
model = ols('dependent_var ~ categorical_var', data=df)
model_result = model.fit()
print(model_result.params)
```

---

### **4. Making Predictions**

```python
import numpy as np
import pandas as pd

# Create explanatory data
explanatory_data = pd.DataFrame({'independent_var': np.arange(0, 11)})

# Use model to predict
predictions = model_result.predict(explanatory_data)

# Create prediction dataframe
prediction_data = explanatoryatory_data.assign(
    dependent_var = predictions
)
print(prediction_data)
```

---

### **5. Non-Linear Relationships**

#### **Cubic Transformation**
```python
# Add cubed variable
df['var_cubed'] = df['var'] ** 3

# Fit model with cubed variable
model = ols('dependent_var ~ var_cubed', data=df)
model_result = model.fit()
print(model_result.params)

# Predict with transformation
explanatory_data = pd.DataFrame({
    'var_cubed': np.arange(10, 41, 5) ** 3,
    'var': np.arange(10, 41, 5)
})
predictions = model_result.predict(explanatory_data)
```

#### **Square Root Transformation**
```python
# Add sqrt variable
df['sqrt_var'] = np.sqrt(df['var'])

# Fit model
model = ols('dependent_var ~ sqrt_var', data=df)
model_result = model.fit()
print(model_result.params)

# Predict
explanatory_data = pd.DataFrame({
    'sqrt_var': np.sqrt(np.arange(0, 81, 10) ** 2)
})
predictions = model_result.predict(explanatory_data)
```

---

### **6. Visualizing Predictions**

```python
import seaborn as sns

# Plot original data
sns.regplot(x='var', y='dependent_var', data=df, ci=None)

# Overlay predictions
if 'var_cubed' in explanatory_data.columns:
    sns.scatterplot(data=prediction_data, x='var', y='dependent_var', 
                   color='red', marker='s')
else:
    sns.scatterplot(data=prediction_data, x='var', y='dependent_var', 
                   color='red', marker='s')
plt.show()
```

---

## **LECTURE 9: K-NEAREST NEIGHBORS (KNN)**

### **1. Basic KNN Setup**

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv('data.csv')

# Select features and target
X = df[['feature1', 'feature2']].values
y = df['target'].values

# Create and fit KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
```

---

### **2. Making Predictions with KNN**

```python
import numpy as np

# New data to predict
X_new = np.array([[val1, val2],
                  [val3, val4],
                  [val5, val6]])

# Make predictions
y_pred = knn.predict(X_new)
print("Predictions:", y_pred)
```

---

### **3. Train/Test Split**

```python
from sklearn.model_selection import train_test_split

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 30% for testing
    random_state=21,  # For reproducibility
    stratify=y       # Maintain class proportions
)

# Fit on training data
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Evaluate on test data
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

---

### **4. Finding Optimal k**

```python
import numpy as np
import matplotlib.pyplot as plt

neighbors = np.arange(1, 26)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

# Plot results
plt.figure(figsize=(8, 6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, list(train_accuracies.values()), label='Training Accuracy')
plt.plot(neighbors, list(test_accuracies.values()), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# Find best k
best_k = max(test_accuracies, key=test_accuracies.get)
print(f"Best k: {best_k}")
```

---

### **5. Using All Features**

```python
# Drop target column
X = df.drop('target_column', axis=1).values
y = df['target_column'].values

# Check shapes
print(X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {knn.score(X_test, y_test)}")
```

---

## **LECTURE 10: ADVANCED ML TECHNIQUES**

### **1. Cross-Validation**

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np

# Define K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create model
model = LinearRegression()

# Perform cross-validation
cv_scores = cross_val_score(
    model, X_train, y_train, 
    cv=kf, 
    scoring='neg_mean_squared_error'
)

# RMSE of each fold
rmse_scores = np.sqrt(-cv_scores)
print(f"RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {rmse_scores.mean()}")
```

---

### **2. Handling Missing Values**

```python
from sklearn.impute import SimpleImputer

# For categorical data
imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

# For numerical data (default: mean)
imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)

# Other strategies: 'median', 'most_frequent', 'constant'
```

---

### **3. Encoding Categorical Variables**

#### **Label Encoding**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_train_cat = le.fit_transform(X_train_cat)
X_test_cat = le.transform(X_test_cat)

# Reshape if needed
X_train_cat = X_train_cat.reshape(-1, 1)
X_test_cat = X_test_cat.reshape(-1, 1)
```

#### **One-Hot Encoding**
```python
# For pandas DataFrame
df_dummies = pd.get_dummies(df['column'], drop_first=True)

# Concat to original dataframe
df_final = pd.concat([df, df_dummies], axis=1)
df_final = df_final.drop('column', axis=1)
```

---

### **4. Combining Preprocessed Data**

```python
import numpy as np

# Combine numerical and categorical features
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)

print(X_train.shape, X_test.shape)
```

---

### **5. Using Pipelines**

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Define pipeline steps
steps = [
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
]

# Create pipeline
pipeline = Pipeline(steps)

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {pipeline.score(X_test, y_test)}")
```

---

### **6. Linear Regression for Prediction**

```python
from sklearn.linear_model import LinearRegression

# Create model
reg = LinearRegression()

# Fit to data (X must be 2D)
X = X.reshape(-1, 1)
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

# Get coefficients
print(f"Intercept: {reg.intercept_}")
print(f"Coefficients: {reg.coef_}")
```

---

## **LECTURE 11: COMPLETE ML PIPELINE**

### **1. Data Loading & Initial Exploration**

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load data without headers if needed
df = pd.read_csv('data.csv', header=None)

# Explore data
print(df.head())
print(df.describe())
print(df.info())
print(df.tail())
```

---

### **2. Missing Value Handling**

```python
import numpy as np

# Replace '?' or other placeholders with NaN
df = df.replace('?', np.nan)

# Check missing values
print(df.isna().sum())

# Impute numeric columns with mean
df.loc[[row_indices]] = df.loc[[row_indices]].fillna(
    df.mean(numeric_only=True)
)

# Impute categorical columns with mode
for col in df.columns:
    if df[col].dtypes == 'object':
        df = df.fillna(df[col].value_counts().index[0])

# Verify no missing values
print(df.isna().sum())
```

---

### **3. Label Encoding for Categorical Data**

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode each object column
for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])
```

---

### **4. Train/Test Split (Before Preprocessing)**

```python
from sklearn.model_selection import train_test_split

# Drop unnecessary features
df = df.drop([col11, col13], axis=1)

# Convert to numpy array
data = df.values

# Separate features and target
X, y = data[:, 0:12], data[:, 13]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

---

### **5. Proper Split-Then-Preprocess Approach**

```python
# Split BEFORE preprocessing
train_df, test_df = train_test_split(
    df, test_size=0.33, random_state=42
)

# Handle missing on train and test separately
train_df = train_df.replace('?', np.nan)
test_df = test_df.replace('?', np.nan)

# Impute using TRAIN statistics
test_df.fillna(train_df.mean(numeric_only=True), inplace=True)
train_df.fillna(train_df.mean(numeric_only=True), inplace=True)

# Verify
print(train_df.isnull().sum())
print(test_df.isnull().sum())
```

---

### **6. One-Hot Encoding**

```python
# Convert categorical to dummies
train_df = pd.get_dummies(train_df, dtype=int)
test_df = pd.get_dummies(test_df)

# Align test columns with train
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

print(train_df.columns)
```

---

### **7. Feature Scaling (MinMaxScaler)**

```python
from sklearn.preprocessing import MinMaxScaler

# Separate features and labels
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.ravel()
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.ravel()

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)  # Use transform only on test data

print(rescaledX_train.shape)
print(rescaledX_test.shape)
```

---

### **8. Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

# Create classifier
logreg = LogisticRegression(max_iter=1000)

# Fit to training data
logreg.fit(rescaledX_train, y_train)

# Make predictions
y_pred = logreg.predict(rescaledX_test)

# Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

---

### **9. K-Nearest Neighbors**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# With default k=5
knn = KNeighborsClassifier()
knn.fit(rescaledX_train, y_train)

y_pred_knn = knn.predict(rescaledX_test)

accuracy = knn.score(rescaledX_test, y_test)
print(f"Accuracy (k=5): {accuracy}")

print(confusion_matrix(y_test, y_pred_knn))
```

---

### **10. Finding Best k - Elbow Method**

```python
import numpy as np
import matplotlib.pyplot as plt

error_rates = []
k_range = range(1, 41)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(rescaledX_train, y_train)
    y_pred_temp = knn_temp.predict(rescaledX_test)
    error_rates.append(np.mean(y_pred_temp != y_test))

# Find best k
best_k_elbow = k_range[np.argmin(error_rates)]
min_error = min(error_rates)

print(f"Best k (Elbow): {best_k_elbow}")
print(f"Min error rate: {min_error:.4f}")

# Train with best k
knn_elbow = KNeighborsClassifier(n_neighbors=best_k_elbow)
knn_elbow.fit(rescaledX_train, y_train)
accuracy_elbow = knn_elbow.score(rescaledX_test, y_test)

print(f"Accuracy with k={best_k_elbow}: {accuracy_elbow:.4f}")
```

---

### **11. Finding Best k - GridSearchCV**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Create parameter grid
param_grid = {'n_neighbors': range(1, 41)}

# Initialize GridSearchCV
knn_gscv = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit to find best parameters
knn_gscv.fit(rescaledX_train, y_train)

# Get best parameters
best_k_cv = knn_gscv.best_params_['n_neighbors']
best_score_cv = knn_gscv.best_score_

print(f"Best k: {best_k_cv}")
print(f"Best CV accuracy: {best_score_cv:.4f}")

# Evaluate on test set
best_knn = KNeighborsClassifier(n_neighbors=best_k_cv)
best_knn.fit(rescaledX_train, y_train)
test_accuracy = best_knn.score(rescaledX_test, y_test)

print(f"Test accuracy: {test_accuracy:.4f}")
y_pred_best = best_knn.predict(rescaledX_test)
print(confusion_matrix(y_test, y_pred_best))
```

---

### **12. Hyperparameter Tuning - Multiple Parameters**

```python
# Parameter grid with multiple parameters
param_grid = {
    'n_neighbors': np.arange(1, 11),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]  # 1=Manhattan, 2=Euclidean
}

# Grid search
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

# Fit
grid.fit(rescaledX_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
```

---

## **METRICS & EVALUATION**

### **Confusion Matrix**
```
                Predicted
                Positive  Negative
Actual Positive    TP        FN
       Negative    FP        TN

TP: True Positive
TN: True Negative
FP: False Positive (Type I error)
FN: False Negative (Type II error)
```

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### **Classification Report**
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

## **COMMON DATA CLEANING PATTERNS**

### **1. Dropping Rows with Specific Values**
```python
# Drop rows where column equals 0
df = df[df['column'] != 0]

# Drop rows with missing values in specific columns
df = df.dropna(subset=['col1', 'col2'])
```

### **2. Converting Categories to Binary**
```python
import numpy as np

# Convert categorical to binary (0/1)
df['binary_col'] = np.where(df['category'] == 'DesiredValue', 1, 0)
```

### **3. Checking Data Correlations**
```python
# Correlation matrix (only numeric)
print(df.corr(numeric_only=True))

# Single correlation
print(df['col1'].corr(df['col2']))
```

---

## **QUICK REFERENCE FOR EXAMS**

| Task | Library/Function |
|------|----------------|
| Mean/Median | `.mean()`, `.median()` |
| Quantiles | `np.quantile()`, `.quantile()` |
| Standard Deviation | `.std()`, `.var()` |
| Histogram | `.hist()`, `sns.histplot()` |
| Correlation | `.corr()` |
| Regression | `ols()` from statsmodels |
| KNN | `KNeighborsClassifier()` |
| Train/Test Split | `train_test_split()` |
| Cross-Validation | `cross_val_score()`, `KFold()` |
| Missing Values | `SimpleImputer()` |
| Encoding | `LabelEncoder()`, `pd.get_dummies()` |
| Scaling | `MinMaxScaler()`, `StandardScaler()` |
| Grid Search | `GridSearchCV()` |
| Logistic Regression | `LogisticRegression()` |
| Confusion Matrix | `confusion_matrix()` |
| Classification Report | `classification_report()` |

---

**Remember:**
- Set random seeds for reproducibility: `np.random.seed()`, `random_state=`
- Always fit on training data, transform both train and test
- Split BEFORE preprocessing to avoid data leakage
- Use `stratify=y` in train_test_split for classification
- Fit scalers on training data only, not on test data
