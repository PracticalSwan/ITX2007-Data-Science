# Final Exam Cheatsheet - Quick Reference
**Exams provide step-by-step instructions - use this for syntax only**

---

## **LECTURE 7: STATISTICS**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
%matplotlib inline
```

---

### **Descriptive Statistics**
```python
# Mean & Median
df['col'].mean()
df['col'].median()
df.groupby('category')['col'].agg(['mean', 'median'])

# Quantiles
np.quantile(df['col'], [0, 0.25, 0.5, 0.75, 1])  # Quartiles
np.quantile(df['col'], [0, 0.2, 0.4, 0.6, 0.8, 1])  # Quintiles
np.quantile(df['col'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Deciles

# Variance & SD
df['col'].var()
df['col'].std()
df.groupby('cat')['col'].agg(['var', 'std'])

# Histograms
df['col'].hist()
df['col'].hist(bins=10)
sns.displot(data=df, x='col', col='cat', col_wrap=2, bins=9)
```

### **Outlier Detection (IQR)**
```python
q1 = np.quantile(data, 0.25)
q3 = np.quantile(data, 0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = data[(data < lower) | (data > upper)]
```

### **Probability**
```python
# Discrete distribution
counts = df['col'].value_counts()
probs = counts / df.shape[0]

# Expected value
size_dist = df['col'].value_counts() / df.shape[0]
size_dist = size_dist.reset_index()
size_dist.columns = ['value', 'prob']
expected_value = np.sum(size_dist['value'] * size_dist['prob'])

# Continuous uniform (min=0, max=30)
prob_less_than = uniform.cdf(x, 0, 30)
prob_greater_than = 1 - uniform.cdf(x, 0, 30)
prob_between = uniform.cdf(b, 0, 30) - uniform.cdf(a, 0, 30)
```

### **Sampling**
```python
np.random.seed(seed_num)

# Without replacement
df.sample(n)

# With replacement
df.sample(n, replace=True)
```

### **Correlation**
```python
# Scatterplot
sns.scatterplot(x='col1', y='col2', data=df)

# With trendline
sns.lmplot(x='col1', y='col2', data=df, ci=None)

# Correlation coefficient
df['col1'].corr(df['col2'])
```

### **Transformations**
```python
# Log, sqrt, square, cube
# Note: Use abs() or add constant for log to handle negative/zero values
df['log_col'] = np.log(df['col'].abs() + 1)  # +1 to avoid log(0)
df['sqrt_col'] = np.sqrt(df['col'].abs())
df['squared'] = df['col'] ** 2
df['cubed'] = df['col'] ** 3
```

---

## **LECTURE 8: REGRESSION**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
```

---

### **Visualizing Relationships**
```python
import seaborn as sns
sns.scatterplot(x='x', y='y', data=df)
sns.regplot(x='x', y='y', data=df, ci=None)
sns.regplot(x='x', y='y', data=df, ci=None, scatter_kws={'alpha': 0.5})
```

### **Linear Regression**
```python
# Simple regression
model = ols('y ~ x', data=df)
model_result = model.fit()
print(model_result.params)  # Intercept and slope
print(model_result.summary())

# Categorical variable (no intercept)
model = ols('y ~ cat_var + 0', data=df)
model_result = model.fit()
print(model_result.params)

# Categorical variable (with intercept)
model = ols('y ~ cat_var', data=df)
model_result = model.fit()
```

### **Making Predictions**
```python
# Create explanatory data
explanatory_data = pd.DataFrame({'x': np.arange(start, stop, step)})

# Predict
predictions = model_result.predict(explanatory_data)

# Create prediction dataframe
prediction_data = explanatory_data.assign(y = predictions)
```

### **Non-Linear Relationships**
```python
# CUBIC transformation
df['x_cubed'] = df['x'] ** 3
model = ols('y ~ x_cubed', data=df).fit()

explanatory_data = pd.DataFrame({
    'x_cubed': np.arange(10, 41, 5) ** 3,
    'x': np.arange(10, 41, 5)
})
predictions = model.predict(explanatory_data)

# SQUARE ROOT transformation
df['sqrt_x'] = np.sqrt(df['x'])
model = ols('y ~ sqrt_x', data=df).fit()

explanatory_data = pd.DataFrame({
    'sqrt_x': np.sqrt(np.arange(0, 81, 10) ** 2),
    'x': np.arange(0, 81, 10) ** 2
})

# FOURTH ROOT transformation
df['qdrt_n'] = df['n'] ** 0.25
model = ols('qdrt_y ~ qdrt_x', data=df).fit()

# Back-transform
prediction_data['n'] = prediction_data['qdrt_n'] ** 4
```

### **Model Evaluation**
```python
r_squared = model_result.rsquared
rse = np.sqrt(model_result.mse_resid)
```

---

## **LECTURE 9: KNN CLASSIFICATION**

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

### **Setup**
```python
X = df[['feat1', 'feat2']].values
y = df['target'].values

# Create KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
```

### **Predict New Data**
```python
X_new = np.array([[val1, val2], [val3, val4]])
y_pred = knn.predict(X_new)
```

### **Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,           # proportion for test
    random_state=21,          # seed
    stratify=y                # maintain class balance
)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
```

### **Find Optimal k**
```python
neighbors = np.arange(1, 26)
train_acc = {}
test_acc = {}

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc[k] = knn.score(X_train, y_train)
    test_acc[k] = knn.score(X_test, y_test)

# Plot
plt.plot(neighbors, list(train_acc.values()), label='Train')
plt.plot(neighbors, list(test_acc.values()), label='Test')
plt.legend()
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Best k
best_k = max(test_acc, key=test_acc.get)
```

### **All Features**
```python
X = df.drop('target', axis=1).values
y = df['target'].values
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

---

## **LECTURE 10: ADVANCED ML**

```python
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report
import numpy as np
```

### **Cross-Validation**
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

cv_scores = cross_val_score(model, X_train, y_train, cv=kf,
                           scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print(f"Mean RMSE: {rmse_scores.mean()}")
```

### **Missing Values**
```python
# Categorical - most frequent
imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

# Numeric - mean (default)
imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
```

### **Encoding**
```python
# Label encoding
le = LabelEncoder()
X_train_cat = le.fit_transform(X_train_cat)
X_test_cat = le.transform(X_test_cat)
X_train_cat = X_train_cat.reshape(-1, 1)
X_test_cat = X_test_cat.reshape(-1, 1)

# One-hot encoding
df_dummies = pd.get_dummies(df['col'], drop_first=True)
df_final = pd.concat([df, df_dummies], axis=1)
df_final = df_final.drop('col', axis=1)
```

### **Combine Preprocessed Data**
```python
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)
```

### **Pipelines**
```python
steps = [
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=100000))
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {pipeline.score(X_test, y_test)}")
```

### **Linear Regression**
```python
reg = LinearRegression()
X = X.reshape(-1, 1)  # Must be 2D
reg.fit(X, y)
predictions = reg.predict(X)

print(f"Intercept: {reg.intercept_}")
print(f"Coefficients: {reg.coef_}")
print(f"R^2: {reg.score(X_test, y_test)}")
```

### **Evaluations**
```python
# RMSE
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

---

## **LECTURE 11: COMPLETE ML PIPELINE**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
```

---

### **Data Loading & Exploration**
```python
df = pd.read_csv('file.csv', header=None)
print(df.head())
print(df.describe())
print(df.info())
print(df.tail())
```

### **Missing Values**
```python
# Replace placeholders
df = df.replace('?', np.nan)

# Check missing
print(df.isna().sum())

# Impute numeric
df.loc[[rows]] = df.loc[[rows]].fillna(df.mean(numeric_only=True))

# Impute categorical (mode)
for col in df.columns:
    if df[col].dtypes == 'object':
        mode_val = df[col].value_counts().index[0]
        df[col] = df[col].fillna(mode_val)
```

### **PROPER APPROACH: Split THEN Preprocess**
```python
# 1. Split FIRST
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)

# 2. Handle missing SEPARATELY
train_df = train_df.replace('?', np.nan)
test_df = test_df.replace('?', np.nan)

train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
test_df.fillna(train_df.mean(numeric_only=True), inplace=True)  # Use TRAIN stats

# 3. One-hot encode
train_df = pd.get_dummies(train_df, dtype=int)
test_df = pd.get_dummies(test_df)

test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# 4. Separate features and target
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.ravel()
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.ravel()

# 5. Scale (fit on TRAIN only)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)  # Transform only!
```

### **Logistic Regression**
```python
logreg = LogisticRegression(max_iter=1000)
logreg.fit(rescaledX_train, y_train)
y_pred = logreg.predict(rescaledX_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
```

### **KNN Classification**
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(rescaledX_train, y_train)
y_pred_knn = knn.predict(rescaledX_test)
print(knn.score(rescaledX_test, y_test))
print(confusion_matrix(y_test, y_pred_knn))
```

### **Finding Best k - Elbow Method**
```python
error_rates = []
k_range = range(1, 41)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(rescaledX_train, y_train)
    y_pred_temp = knn_temp.predict(rescaledX_test)
    error_rates.append(np.mean(y_pred_temp != y_test))

best_k = k_range[np.argmin(error_rates)]

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(rescaledX_train, y_train)
print(knn_best.score(rescaledX_test, y_test))
```

### **Finding Best k - GridSearchCV**
```python

param_grid = {'n_neighbors': range(1, 41)}
knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
knn_gscv.fit(rescaledX_train, y_train)

best_k = knn_gscv.best_params_['n_neighbors']
best_score = knn_gscv.best_score_

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(rescaledX_train, y_train)
print(best_knn.score(rescaledX_test, y_test))
```

### **Hyperparameter Tuning - Multiple Params**
```python
param_grid = {
    'n_neighbors': np.arange(1, 11),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]  # 1=Manhattan, 2=Euclidean
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(rescaledX_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

---

## **QUICK REFERENCE TABLE**

| Task | Function/Library |
|------|-----------------|
| Load data | `pd.read_csv()` |
| Mean/Median | `.mean()`, `.median()` |
| Quantiles | `np.quantile()`, `.quantile()` |
| Variance/SD | `.var()`, `.std()` |
| Histogram | `.hist()`, `sns.displot()` |
| Scatter plot | `sns.scatterplot()`, `sns.regplot()` |
| Correlation | `.corr()` |
| Regression | `ols()` from statsmodels |
| KNN | `KNeighborsClassifier()` |
| Train/Test Split | `train_test_split(stratify=y)` |
| Cross-Validation | `cross_val_score()`, `KFold()` |
| Missing Values | `SimpleImputer()` |
| Label Encode | `LabelEncoder()` |
| One-Hot Encode | `pd.get_dummies()` |
| Scale (MinMax) | `MinMaxScaler()` |
| Scale (Standard) | `StandardScaler()` |
| Log Regression | `LogisticRegression()` |
| Linear Regression | `LinearRegression()` |
| Pipeline | `Pipeline()` |
| Grid Search | `GridSearchCV()` |
| Confusion Matrix | `confusion_matrix()` |
| Classification Report | `classification_report()` |

---

## **CRITICAL REMINDERS**

1. **Set random seeds**: `np.random.seed()`, `random_state=`
2. **Split BEFORE preprocessing** (Lecture 11) - prevents data leakage
3. **Fit on training data only**, transform on both train and test
4. **Use stratify=y** for classification splits
5. **Scalers**: `fit_transform()` on train, `transform()` only on test
6. **GridSearchCV**: Fit on train data, evaluate on test separately
7. **Pipelines**: Combine imputation, scaling, modeling

---

**REMEMBER: Exams give step-by-step instructions - this is for quick syntax reference only!**
