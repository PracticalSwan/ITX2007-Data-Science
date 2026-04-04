# Final Exam Cheatsheet - Quick Reference

This version is synced to the current repo state on **2026-04-04** and includes the workflows used in:

- `final_practice_L7.ipynb` to `final_practice_L12.ipynb`
- `final_files/ITX2007_6726077_SithuWinSan_Final_2_2025_Q1.ipynb`
- `final_files/ITX2007_6726077_SithuWinSan_Final_2_2025_Q2.ipynb`

Use this when you need fast syntax during the exam. The longer explanations are in `Lectures_7_to_12_Cheatsheet.md`.

## Fast Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.cluster import KMeans
```

## 1. Statistics and EDA

### Load and inspect

```python
df = pd.read_csv("file.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
```

### Descriptive stats

```python
df["col"].mean()
df["col"].median()
df["col"].quantile([0.25, 0.5, 0.75])
df["col"].var()
df["col"].std()
```

### Grouped stats

```python
df.groupby("group_col")["value_col"].agg(["mean", "median", "min", "max"])
```

### Histogram and scatter

```python
df["col"].hist(bins=20)
plt.show()

sns.scatterplot(data=df, x="x_col", y="y_col")
plt.show()

sns.regplot(data=df, x="x_col", y="y_col")
plt.show()
```

### IQR outliers

```python
q1 = df["col"].quantile(0.25)
q3 = df["col"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[(df["col"] < lower) | (df["col"] > upper)]
```

### Correlation

```python
df["x_col"].corr(df["y_col"])
df.corr(numeric_only=True)
```

### Probability and sampling

```python
from scipy.stats import uniform

uniform.cdf(7, loc=0, scale=30)
1 - uniform.cdf(7, loc=0, scale=30)

np.random.seed(42)
sample = df.sample(n=10, replace=False)
boot = df.sample(frac=1, replace=True)
```

## 2. Regression

### Simple OLS

```python
model = ols("y ~ x", data=df).fit()
print(model.params)
print(model.summary())
```

### Categorical regression

```python
model = ols("y ~ category_col", data=df).fit()
print(model.summary())
```

### Predictions

```python
new_x = pd.DataFrame({"x": np.arange(10, 60, 5)})
preds = model.predict(new_x)
print(preds)
```

### Plot regression line

```python
sns.regplot(data=df, x="x", y="y")
plt.show()
```

### RMSE

```python
y_pred = model.predict(df[["x"]])
rmse = np.sqrt(mean_squared_error(df["y"], y_pred))
print(rmse)
```

### Transform features if needed

```python
df["x_sqrt"] = np.sqrt(df["x"])
df["x_cubed"] = df["x"] ** 3

model = ols("y ~ x_sqrt", data=df).fit()
```

## 3. KNN Classification

### Basic setup

```python
X = df[["feature_1", "feature_2"]]
y = df["target"]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
```

### Train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

### Search for best `k`

```python
scores = {}

for k in range(1, 26):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    scores[k] = model.score(X_test, y_test)

best_k = max(scores, key=scores.get)
best_score = scores[best_k]
print(best_k, best_score)
```

### Plot score by `k`

```python
plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()
```

### Predict new rows

```python
X_new = np.array([[35.0, 17.5, 10.1, 1]])
pred = knn.predict(X_new)
print(pred)
```

## 4. Preprocessing and Pipelines

### Replace placeholders and inspect nulls

```python
df = df.replace("?", np.nan)
print(df.isna().sum())
```

### Safe split-first workflow

```python
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
```

### Numeric and categorical imputation

```python
train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
test_df.fillna(train_df.mean(numeric_only=True), inplace=True)

for col in train_df.columns:
    if train_df[col].dtype == "object":
        mode_val = train_df[col].mode(dropna=True)[0]
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)
```

### Encoding

```python
encoder = LabelEncoder()
df["encoded_col"] = encoder.fit_transform(df["category_col"])
```

```python
train_df = pd.get_dummies(train_df, dtype=int)
test_df = pd.get_dummies(test_df, dtype=int)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
```

### Scaling

```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Logistic Regression

```python
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Pipeline

```python
steps = [
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=100000)),
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
```

### Cross-validation

```python
kf = KFold(n_splits=6, shuffle=True, random_state=5)
scores = cross_val_score(LinearRegression(), X, y, cv=kf)
print(scores.mean(), scores.std())
```

### GridSearchCV

```python
param_grid = {"n_neighbors": range(1, 41)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

## 5. K-Means Clustering

### Basic K-Means

```python
model = KMeans(n_clusters=3, n_init="auto")
labels = model.fit_predict(samples)
print(labels)
```

### Predict new samples

```python
new_labels = model.predict(new_samples)
print(new_labels)
```

### Elbow method

```python
ks = range(1, 10)
inertias = []

for k in ks:
    temp_model = KMeans(n_clusters=k, n_init="auto")
    temp_model.fit(samples)
    inertias.append(temp_model.inertia_)

plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()
```

### Cluster visualization

```python
plt.scatter(samples[:, 0], samples[:, 1], c=labels, alpha=0.6)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="D", s=60)
plt.show()
```

### Standardize before clustering

```python
pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init="auto"))
pipeline.fit(samples)
labels = pipeline.predict(samples)
```

### Crosstab evaluation

```python
df_eval = pd.DataFrame({"labels": labels, "true_class": true_labels})
print(pd.crosstab(df_eval["labels"], df_eval["true_class"]))
```

## 6. Current Repo Final Exam Patterns

### Q1 in `final_files/ITX2007_6726077_SithuWinSan_Final_2_2025_Q1.ipynb`

```python
train_1 = pd.read_csv("train_1.csv", index_col=0)
train_2 = pd.read_csv("train_2.csv", index_col=0)
train = pd.concat([train_1, train_2], ignore_index=True)
```

The current repo notebook then:

- checks dtypes and null counts
- fills several categorical fields with mode
- fills selected numeric fields such as `LoanAmount`
- label-encodes the loan dataset categories
- uses a heatmap and scatterplot for inspection
- fits `LoanAmount ~ ApplicantIncome` with `ols`

### Q2 in `final_files/ITX2007_6726077_SithuWinSan_Final_2_2025_Q2.ipynb`

```python
telecom_churn = pd.read_csv("telecom_churn.csv", index_col=0)

X = telecom_churn[
    ["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]
]
y = telecom_churn["churn"]
```

The current repo notebook then:

- compares multiple `test_size` values
- searches across `k` values
- trains a final `KNeighborsClassifier`
- predicts churn on `new_data.csv`

## 7. Quick Metrics Reference

| Task | Main call |
|------|-----------|
| Mean | `df["col"].mean()` |
| Quantiles | `df["col"].quantile([0.25, 0.5, 0.75])` |
| Correlation | `df["x"].corr(df["y"])` |
| OLS regression | `ols("y ~ x", data=df).fit()` |
| Train/test split | `train_test_split(X, y, test_size=0.3, stratify=y)` |
| KNN | `KNeighborsClassifier(n_neighbors=k)` |
| Accuracy | `accuracy_score(y_test, y_pred)` |
| Confusion matrix | `confusion_matrix(y_test, y_pred)` |
| RMSE | `np.sqrt(mean_squared_error(y_test, y_pred))` |
| Grid search | `GridSearchCV(model, param_grid, cv=5)` |
| K-Means | `KMeans(n_clusters=3, n_init="auto")` |
| Elbow metric | `model.inertia_` |
| Clustering comparison | `pd.crosstab(labels, true_labels)` |

## Critical Reminders

1. Split before preprocessing whenever the task is supervised learning.
2. Fit imputers and scalers on training data only.
3. Use `stratify=y` for classification.
4. Align dummy columns between train and test after one-hot encoding.
5. Use `max_iter=` if Logistic Regression does not converge.
6. Standardize before K-Means when feature scales are different.
7. Label plots and print the metric the question actually asks for.
8. If the question asks for interpretation, answer in words after the code.

## Last-Minute Exam Flow

1. Load the data.
2. Print `head()`, `info()`, `describe()`, and null counts.
3. Identify whether the task is stats, regression, classification, preprocessing, or clustering.
4. Use the matching block from this cheatsheet.
5. Print the final result clearly.
