# Python Data Science Cheatsheet (Lectures 7-12)

This reference is synced to the current repo state on **2026-04-04**.

It matches these notebooks and practice files:

- `Lecture7_Data Sets - Intro2Statistics/6726077_SithuWinSan_L7.ipynb`
- `Lecture8 - Regression/6726077_SithuWinSan_L8.ipynb`
- `Lecture9_DataSets - ML Part 1/6726077_SithuWinSan_L9.ipynb`
- `Lecture10_DataSets - ML Part 1/6726077_SithuWinSan_L10.ipynb`
- `Lecture11_DataSets - Credit Card Predictor/6726077_SithuWinSan_L11.ipynb`
- `Lecture12_DataSets - Unsupervised ML/6726077_SithuWinSan_L12.ipynb`
- `final_practice_L7.ipynb` to `final_practice_L12.ipynb`
- `final_files/ITX2007_6726077_SithuWinSan_Final_2_2025_Q1.ipynb`
- `final_files/ITX2007_6726077_SithuWinSan_Final_2_2025_Q2.ipynb`

## Quick Map

| Lecture | Main topic | Main notebook focus | Typical datasets |
|---------|------------|---------------------|------------------|
| 7 | Statistics | descriptive stats, sampling, inference | `food_consumption.csv`, `world_happiness.csv`, `amir_deals.csv` |
| 8 | Regression | linear models with `statsmodels` | `fish.csv`, `taiwan_real_estate2.csv`, `ad_conversion.csv` |
| 9 | KNN and train/test workflow | classification setup and evaluation | `churn_df.csv`, `diabetes.csv`, `sales_df.csv` |
| 10 | Preprocessing and validation | imputation, encoding, pipelines, CV | `music.csv`, `music_unclean.csv`, `telecom_churn_clean.csv` |
| 11 | Credit card prediction | split-then-preprocess, scaling, GridSearchCV | `cc_approvals.data` |
| 12 | K-Means clustering | elbow method, crosstab, scaled clustering | `points.csv`, `new_points.csv`, `seeds.csv`, `wine.data`, `fish.csv` |

## Core Imports

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.cluster import KMeans
```

## Lecture 7: Statistics

### Load and inspect

```python
df = pd.read_csv("food_consumption.csv")
print(df.head())
print(df.info())
print(df.describe())
```

### Descriptive statistics

```python
df["co2_emission"].mean()
df["co2_emission"].median()
df["co2_emission"].quantile([0, 0.25, 0.5, 0.75, 1.0])
df["co2_emission"].var()
df["co2_emission"].std()
```

### Grouped summaries

```python
df.groupby("food_category")["co2_emission"].agg(["mean", "median", "min", "max"])
df.groupby("food_category")["co2_emission"].quantile([0.25, 0.5, 0.75])
```

### Histograms

```python
df["co2_emission"].hist(bins=20)
plt.xlabel("CO2 emission")
plt.ylabel("Frequency")
plt.show()

sns.histplot(data=df, x="co2_emission", hue="food_category", bins=20, kde=False)
plt.show()
```

### IQR outlier detection

```python
q1 = df["co2_emission"].quantile(0.25)
q3 = df["co2_emission"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[(df["co2_emission"] < lower) | (df["co2_emission"] > upper)]
clean_df = df[(df["co2_emission"] >= lower) & (df["co2_emission"] <= upper)]
```

### Probability distributions

```python
sales_counts = amir_deals["num_users"].value_counts(normalize=True).sort_index()
expected_value = (sales_counts.index * sales_counts.values).sum()
```

```python
from scipy.stats import uniform

uniform.cdf(7, loc=0, scale=30)
1 - uniform.cdf(7, loc=0, scale=30)
uniform.cdf(18, loc=0, scale=30) - uniform.cdf(10, loc=0, scale=30)
```

### Sampling

```python
np.random.seed(42)
sample = df.sample(n=10, replace=False)
boot = df.sample(frac=1, replace=True)
```

### Correlation and scatter plots

```python
sns.scatterplot(data=world_happiness, x="gdp_per_cap", y="life_ladder")
plt.show()

sns.regplot(data=world_happiness, x="gdp_per_cap", y="life_ladder")
plt.show()

world_happiness["gdp_per_cap"].corr(world_happiness["life_ladder"])
```

### Transformations

```python
df["log_col"] = np.log(df["value"])
df["sqrt_col"] = np.sqrt(df["value"])
df["square_col"] = df["value"] ** 2
df["cube_col"] = df["value"] ** 3
```

Use `np.log1p()` or add a constant if zeros are present.

### Lecture 7 reminders

- Use grouped summaries when the question compares categories.
- Use `replace=True` only for bootstrap-style resampling.
- Use IQR bounds on the column you are judging, not the full dataframe.
- Always label axes on plots.

## Lecture 8: Regression

### Load regression data

```python
fish = pd.read_csv("fish.csv")
real_estate = pd.read_csv("taiwan_real_estate2.csv")
ads = pd.read_csv("ad_conversion.csv")
```

### Visualize relationships

```python
sns.scatterplot(data=fish, x="length_cm", y="mass_g", hue="species")
plt.show()

sns.regplot(data=fish, x="length_cm", y="mass_g", scatter_kws={"alpha": 0.6})
plt.show()
```

### Simple linear regression with `statsmodels`

```python
mdl_mass_vs_length = ols("mass_g ~ length_cm", data=fish).fit()
print(mdl_mass_vs_length.params)
print(mdl_mass_vs_length.summary())
```

### Regression with categorical variables

```python
mdl_mass_vs_species = ols("mass_g ~ species", data=fish).fit()
print(mdl_mass_vs_species.summary())
```

### Predictions

```python
explanatory_data = pd.DataFrame({"length_cm": np.arange(10, 60, 5)})
prediction_data = explanatory_data.assign(
    mass_g=mdl_mass_vs_length.predict(explanatory_data)
)
print(prediction_data)
```

### Non-linear transformations

```python
fish["length_cubed"] = fish["length_cm"] ** 3
mdl_mass_vs_length_cubed = ols("mass_g ~ length_cubed", data=fish).fit()
```

```python
ads["sqrt_n_impressions"] = np.sqrt(ads["n_impressions"])
mdl_click_vs_sqrt_impression = ols("n_clicks ~ sqrt_n_impressions", data=ads).fit()
```

```python
ads["qdrt_n_impressions"] = ads["n_impressions"] ** 0.25
mdl_click_vs_qdrt_impression = ols("n_clicks ~ qdrt_n_impressions", data=ads).fit()
```

### Back-transform predictions

```python
pred = mdl_click_vs_qdrt_impression.predict(explanatory_data)
pred_original_scale = pred ** 4
```

### Plot predictions

```python
plt.scatter(fish["length_cm"], fish["mass_g"], alpha=0.6)
plt.plot(
    explanatory_data["length_cm"],
    prediction_data["mass_g"],
    color="red",
)
plt.show()
```

### Regression evaluation

```python
print(mdl_mass_vs_length.rsquared)
print(np.sqrt(mdl_mass_vs_length.mse_resid))
```

### Lecture 8 reminders

- `ols("y ~ x", data=df)` uses a formula string.
- Use `model.summary()` for coefficients, p-values, and R-squared.
- Transform the explanatory variable first, then fit on the transformed feature.
- Back-transform predictions if the question wants the original unit.

## Lecture 9: KNN and train/test workflow

### Basic KNN

```python
churn = pd.read_csv("churn_df.csv")
X = churn[["account_length", "customer_service_calls"]]
y = churn["churn"]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
```

### Predict new rows

```python
X_new = np.array([[12, 1], [45, 3], [180, 0]])
predictions = knn.predict(X_new)
print(predictions)
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

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

### Search for best `k`

```python
scores = {}

for k in range(1, 21):
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
plt.xticks(range(1, 21))
plt.show()
```

### Use more features

```python
X = churn.drop("churn", axis=1)
y = churn["churn"]

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

### Lecture 9 reminders

- Use `stratify=y` for classification.
- Keep `X` as a DataFrame when selecting multiple features.
- Use `.score()` for a quick accuracy check.
- Search across `k` values instead of guessing one.

## Lecture 10: Preprocessing, validation, and pipelines

### Cross-validation

```python
X = sales_df[["TV", "radio", "social_media"]]
y = sales_df["sales"]

kf = KFold(n_splits=6, shuffle=True, random_state=5)
reg = LinearRegression()

cv_scores = cross_val_score(reg, X, y, cv=kf)
print(cv_scores)
print(cv_scores.mean())
print(cv_scores.std())
```

### Missing-value imputation

```python
imp_num = SimpleImputer(strategy="mean")
X_num = imp_num.fit_transform(X_num)

imp_cat = SimpleImputer(strategy="most_frequent")
X_cat = imp_cat.fit_transform(X_cat)
```

### Label encoding

```python
encoder = LabelEncoder()
music["genre_label"] = encoder.fit_transform(music["genre"])
```

### One-hot encoding

```python
music_dummies = pd.get_dummies(music, columns=["genre"], drop_first=True)
```

### Combine arrays manually

```python
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)
```

### Build a pipeline

```python
steps = [
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=100000)),
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))
```

### Evaluate classification models

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
```

### Evaluate regression models

```python
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
```

### Lecture 10 reminders

- Use CV for a more stable performance estimate.
- Impute before scaling.
- `LabelEncoder` is for label-like categorical values, not full multi-column feature frames.
- Pipelines help keep preprocessing and modeling together.

## Lecture 11: Credit card prediction pipeline

### Load `cc_approvals.data`

```python
cc_apps = pd.read_csv("cc_approvals.data", header=None)
print(cc_apps.head())
print(cc_apps.describe())
print(cc_apps.info())
```

### Replace placeholders and inspect missing values

```python
cc_apps = cc_apps.replace("?", np.nan)
print(cc_apps.isna().sum())
```

### Split before preprocessing

```python
train_df, test_df = train_test_split(
    cc_apps,
    test_size=0.33,
    random_state=42,
)
```

### Impute train and test safely

```python
train_df = train_df.replace("?", np.nan)
test_df = test_df.replace("?", np.nan)

train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
test_df.fillna(train_df.mean(numeric_only=True), inplace=True)

for col in train_df.columns:
    if train_df[col].dtype == "object":
        mode_val = train_df[col].mode(dropna=True)[0]
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)
```

### One-hot encode and align columns

```python
train_df = pd.get_dummies(train_df, dtype=int)
test_df = pd.get_dummies(test_df, dtype=int)

test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
```

### Separate features and labels

```python
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.ravel()
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.ravel()
```

### MinMax scaling

```python
scaler = MinMaxScaler(feature_range=(0, 1))
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

### Baseline KNN

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred_knn))
```

### Elbow search for best `k`

```python
error_rates = []

for k in range(1, 41):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(np.mean(y_pred_temp != y_test))

best_k = np.arange(1, 41)[np.argmin(error_rates)]
print(best_k)
```

### GridSearchCV for `k`

```python
param_grid = {"n_neighbors": range(1, 41)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

### GridSearchCV with multiple hyperparameters

```python
param_grid = {
    "n_neighbors": np.arange(1, 11),
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2],
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

### Lecture 11 reminders

- The safe pattern is `split -> impute -> encode -> align columns -> scale -> fit`.
- Fit scalers on training data only.
- `reindex(columns=train_df.columns, fill_value=0)` is critical after `pd.get_dummies`.
- Use Logistic Regression and KNN as complementary baselines.

## Lecture 12: K-Means clustering

### Basic K-Means on iris-like samples

```python
from sklearn.datasets import load_iris

iris = load_iris()
model = KMeans(n_clusters=3, n_init="auto")
model.fit(iris.data)

labels = model.predict(iris.data)
print(labels)
```

### Predict clusters for new points

```python
new_samples = np.array([
    [5.7, 4.4, 1.5, 0.4],
    [6.5, 3.0, 5.5, 1.8],
    [5.8, 2.7, 5.1, 1.9],
])

print(model.predict(new_samples))
```

### Visualize clusters

```python
xs = iris.data[:, 0]
ys = iris.data[:, 2]

plt.scatter(xs, ys, c=labels, alpha=0.6)
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.show()
```

### Plot centroids

```python
centroids = model.cluster_centers_
plt.scatter(xs, ys, c=labels, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 2], marker="D", s=60)
plt.show()
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
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(ks))
plt.show()
```

### Seeds example with crosstab

```python
labels = model.fit_predict(seeds)
df = pd.DataFrame({"labels": labels, "varieties": true_varieties})
ct = pd.crosstab(df["labels"], df["varieties"])
print(ct)
```

### Standardization with pipeline

```python
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3, n_init="auto")
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)

labels = pipeline.predict(samples)
centers = pipeline.named_steps["kmeans"].cluster_centers_
```

### Practical workflow for repo datasets

```python
points = pd.read_csv("points.csv")
new_points = pd.read_csv("new_points.csv")
seeds = pd.read_csv("seeds.csv")
wine = pd.read_csv("wine.data", header=None)
fish = pd.read_csv("fish.csv")
```

```python
model = KMeans(n_clusters=3, n_init="auto")
labels = model.fit_predict(points)
new_labels = model.predict(new_points)
```

### Lecture 12 reminders

- Standardize features before clustering when scales differ.
- `model.inertia_` gets smaller as `k` increases, so look for the elbow instead of the minimum.
- `pd.crosstab()` is useful only when you have known labels to compare against.
- Use `predict()` for new points after fitting the model.

## Cross-Lecture Evaluation Patterns

### Accuracy

```python
accuracy_score(y_test, y_pred)
model.score(X_test, y_test)
```

### Confusion matrix

```python
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### Classification report

```python
print(classification_report(y_test, y_pred))
```

### RMSE

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
```

### Crosstab for clustering

```python
pd.crosstab(df["labels"], df["true_class"])
```

## Repo-Specific Final Exam Patterns

The current `final_files/` folder adds two useful review workflows on top of Lectures 7-12:

### Final Q1 pattern

```python
train_1 = pd.read_csv("train_1.csv", index_col=0)
train_2 = pd.read_csv("train_2.csv", index_col=0)
train = pd.concat([train_1, train_2], ignore_index=True)
```

Typical tasks in the repo notebook:

- inspect dtypes and null counts
- fill categorical columns with mode
- fill selected numeric columns with mode or median-like fixes
- label-encode loan-style categorical fields
- inspect correlation with a heatmap
- fit a simple OLS regression such as `LoanAmount ~ ApplicantIncome`

### Final Q2 pattern

```python
telecom_churn = pd.read_csv("telecom_churn.csv", index_col=0)

X = telecom_churn[
    ["total_day_charge", "total_eve_charge", "total_night_charge", "customer_service_calls"]
]
y = telecom_churn["churn"]
```

Typical tasks in the repo notebook:

- compare multiple `test_size` values
- search for the best `k`
- fit the final KNN model with the chosen `k`
- predict churn on `new_data.csv`

## Critical Reminders

1. Use `random_state=` when a question needs reproducibility.
2. Use `stratify=y` for classification splits.
3. Split before preprocessing to avoid data leakage.
4. Fit scalers and imputers on training data only.
5. Align test columns with train columns after one-hot encoding.
6. Use `ols()` for lecture-style regression work.
7. Use accuracy and confusion matrices for classification tasks.
8. Standardize before K-Means if feature scales differ.
9. Use elbow plots to choose `k` in both KNN and K-Means style questions.
10. Keep plots labeled and readable because many notebook questions rely on interpretation.

## Fast Exam Checklist

- Read the dataset and inspect nulls first.
- Decide whether the task is stats, regression, classification, preprocessing, or clustering.
- Pick the matching lecture workflow from this cheatsheet.
- Write the cleanest version of the pipeline first, then print the key metric or plot.
- If the question asks for interpretation, explain the metric or chart in plain language after the code.
