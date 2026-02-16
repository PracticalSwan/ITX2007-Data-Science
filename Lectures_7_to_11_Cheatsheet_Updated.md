# Python Data Science Cheatsheet (Lectures 7-11)
## Exams Provide Step-by-Step Instructions - Use This as Reference

---

## **LECTURE 7: INTRODUCTION TO STATISTICS**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
%matplotlib inline
```

---

### **1. Descriptive Statistics**

#### **Mean & Median**
```python
# For a specific column or subset
mean_val = df['column_name'].mean()
median_val = df['column_name'].median()

# For groups
df.groupby('category')['column_name'].agg(['mean', 'median'])

# Multiple aggregation with .agg()
mean_median_consumption = df.groupby('category')['column'].agg(['mean', 'median'])
print(mean_median_consumption)

# Example from actual notebook:
be_consumption = food_consumption[food_consumption['country'] == 'Belgium']
usa_consumption = food_consumption[food_consumption['country'] == 'USA']
print("Belgium - Mean:", be_consumption['consumption'].mean())
print("USA - Median:", usa_consumption['consumption'].median())
```

#### **Quantiles**
```python
# Quartiles (0, 0.25, 0.5, 0.75, 1)
np.quantile(df['column'], [0, 0.25, 0.5, 0.75, 1])

# Quintiles
np.quantile(df['column'], [0, 0.2, 0.4, 0.6, 0.8, 1])

# Deciles
np.quantile(df['column'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Example from actual notebook:
print(np.quantile(food_consumption['co2_emission'], [0,0.25,0.5,0.75,1]))
print(np.quantile(food_consumption['co2_emission'], [0,0.2,0.4,0.6,0.8,1]))
print(np.quantile(food_consumption['co2_emission'], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
```

#### **Variance & Standard Deviation**
```python
# For a single column
variance = df['column'].var()
std_dev = df['column'].std()

# For groups
df.groupby('category')['column'].agg(['var', 'std'])

# Example from actual notebook:
print(food_consumption.groupby('food_category')['co2_emission'].agg(['var', 'std']))

# Create histogram per category
food_consumption[food_consumption['food_category'] == 'beef']['co2_emission'].hist()
plt.show()
```

#### **Histograms**
```python
# Simple histogram
df['column'].hist()
plt.show()

# Histogram with bins
df['column'].hist(bins=10)
plt.show()

# Grouped histograms
sns.displot(data=df, x='column', col='category', col_wrap=2, bins=9)
plt.show()

# Example from actual notebook:
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']
rice_consumption['co2_emission'].hist()
plt.show()
print(rice_consumption['co2_emission'].agg(['mean', 'median']))
```

---

### **2. Outlier Detection (IQR Method)**

```python
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

# Example from actual notebook:
emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()
q1 = np.quantile(emissions_by_country, 0.25)
q3 = np.quantile(emissions_by_country, 0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = emissions_by_country[(emissions_by_country < lower) | (emissions_by_country > upper)]
print(outliers)
non_outliers = emissions_by_country[(emissions_by_country >= lower) | (emissions_by_country <= upper)]
```

---

### **3. Probability & Distributions**

#### **Discrete Probability Distribution**
```python
# Count occurrences and calculate probabilities
counts = df['category'].value_counts()
probs = counts / df.shape[0]
print(probs)

# Example from actual notebook:
amir_deals = pd.read_csv('amir_deals.csv')
counts = amir_deals['product'].value_counts()
probs = counts / amir_deals.shape[0]
print(probs)
```

#### **Expected Value (for Discrete Distributions)**
```python
# Create probability distribution
size_dist = df['column'].value_counts() / df.shape[0]
size_dist = size_dist.reset_index()
size_dist.columns = ['value', 'probability']

# Calculate expected value
expected_value = np.sum(size_dist['value'] * size_dist['probability'])
print('The expected value is', expected_value)

# Probability of condition
prob_4_or_more = np.sum(size_dist[size_dist['value'] >= 4]['probability'])
print('The probability is', prob_4_or_more)

# Example from actual notebook:
restaurant_groups = pd.read_csv('restaurant_groups.csv')
size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups.shape[0]
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']
expected_value = np.sum(size_dist['group_size'] * size_dist['prob'])
print('The expected value is', expected_value)
groups_4_or_more = size_dist[size_dist['group_size'] >= 4]
prob_4_or_more = np.sum(groups_4_or_more['prob'])
print('The probability is ', prob_4_or_more)
```

#### **Continuous Uniform Distribution**
```python
min_time = 0
max_time = 30

# Probability less than X (min=0, max=30)
prob_less_than_5 = uniform.cdf(5, 0, 30)
print(prob_less_than_5)

# Probability greater than X
prob_greater_than_5 = 1 - uniform.cdf(5, 0, 30)
print(prob_greater_than_5)

# Probability between A and B
prob_between_10_and_20 = uniform.cdf(20, 0, 30) - uniform.cdf(10, 0, 30)
print(prob_between_10_and_20)
```

---

### **4. Sampling**

```python
# Set seed for reproducibility
np.random.seed(24)

# Sample without replacement
sample_without_replacement = df.sample(5)
print(sample_without_replacement)

# Sample with replacement
sample_with_replacement = df.sample(5, replace=True)
print(sample_with_replacement)

# Sampling distribution of mean
np.random.seed(104)
sample_means = []
for i in range(100):
    samp_20 = df['column'].sample(20, replace=True)
    samp_20_mean = np.mean(samp_20)
    sample_means.append(samp_20_mean)

# Plot sampling distribution
sample_means_series = pd.Series(sample_means)
sample_means_series.hist()
plt.show()
```

---

### **5. Correlation**

```python
# Scatterplot
sns.scatterplot(x='col1', y='col2', data=df)
plt.show()

# Scatterplot with trendline
sns.lmplot(x='col1', y='col2', data=df, ci=None)
plt.show()

# Calculate correlation coefficient
correlation = df['col1'].corr(df['col2'])
print(f"Correlation: {correlation}")

# Example from actual notebook:
world_happiness = pd.read_csv('world_happiness.csv')
sns.scatterplot(x='life_exp', y='happiness_score', data=world_happiness)
plt.show()
sns.lmplot(x='life_exp', y='happiness_score', data=world_happiness, ci=None)
plt.show()
cor = world_happiness['life_exp'].corr(world_happiness['happiness_score'])
print(cor)
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

# Example from actual notebook - improving correlation with transformation:
world_happiness['log_gdp_per_cap'] = np.log(world_happiness['gdp_per_cap'])
sns.scatterplot(x='log_gdp_per_cap', y='happiness_score', data=world_happiness)
plt.show()
cor = world_happiness['log_gdp_per_cap'].corr(world_happiness['happiness_score'])
print(cor)
```

---

## **LECTURE 8: REGRESSION**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
```

---

### **1. Visualizing Relationships**

```python
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

# Example from actual notebook:
taiwan_real_estate = pd.read_csv('Taiwan_real_estate2.csv')
sns.scatterplot(x='n_convenience', y='price_twd_msq', data=taiwan_real_estate)
sns.regplot(x='n_convenience', y='price_twd_msq', data=taiwan_real_estate, ci=None, scatter_kws={'alpha':0.5})
```

---

### **2. Simple Linear Regression**

```python
# Create model: dependent ~ independent
model = ols('dependent_var ~ independent_var', data=df)
model_result = model.fit()

# Print model parameters (intercept and slope)
print(model_result.params)

# Get model summary (R-squared, p-values, etc.)
print(model_result.summary())

# Example from actual notebook:
swedish_motor_insurance = pd.read_csv('swedish_motor_insurance.csv')
mdl_payment_vs_claims = ols('total_payment_sek ~ n_claims', data=swedish_motor_insurance)
mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
print(mdl_payment_vs_claims.params)

taiwan_real_estate = pd.read_csv('taiwan_real_estate2.csv')
mdl_price_vs_conv = ols('price_twd_msq ~ n_convenience', data=taiwan_real_estate)
mdl_price_vs_conv = mdl_price_vs_conv.fit()
print(mdl_price_vs_conv.params)
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

# Example from actual notebook:
fish = pd.read_csv('fish.csv')
mdl_mass_vs_species = ols('mass_g ~ species + 0', data=fish).fit()
print(mdl_mass_vs_species.params)

# Grouped histograms by species:
sns.displot(data=fish, x='mass_g', col='species', col_wrap=2, bins=9)
plt.show()
```

---

### **4. Making Predictions**

```python
# Create explanatory data
explanatory_data = pd.DataFrame({'independent_var': np.arange(0, 11)})

# Use model to predict
predictions = model_result.predict(explanatory_data)

# Create prediction dataframe
prediction_data = explanatory_data.assign(
    dependent_var = predictions
)
print(prediction_data)

# Example from actual notebook:
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})
price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)
prediction_data = explanatory_data.assign(price_twd_msq = price_twd_msq)
print(prediction_data)

# Manual calculation using model parameters:
coeffs = mdl_price_vs_conv.params
intercept = coeffs[0]
slope = coeffs[1]
price_twd_msq = intercept + slope * explanatory_data
print(price_twd_msq)
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

# Example from actual notebook:
perch = fish[fish['species'] == 'Perch']
perch['length_cm_cubed'] = perch['length_cm'] ** 3
mdl_perch = ols('mass_g ~ length_cm_cubed', data=perch).fit()
print(mdl_perch.params)

explanatory_data = pd.DataFrame({
    'length_cm_cubed': np.arange(10, 41, 5) ** 3,
    'length_cm': np.arange(10, 41, 5)
})
prediction_data = explanatory_data.assign(mass_g = mdl_perch.predict(explanatory_data))
print(prediction_data)
```

#### **Square Root Transformation**
```python
# Add sqrt variable
df['sqrt_var'] = np.sqrt(df['var'])

# Fit model
model = ols('dependent_var ~ sqrt_var', data=df)
model_result = model.fit()
print(model_result.params)

# Example from actual notebook:
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])
sns.regplot(x="sqrt_dist_to_mrt_m", y="price_twd_msq", data=taiwan_real_estate, ci=None)
plt.show()

mdl_price_vs_dist = ols("price_twd_msq ~ sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()
print(mdl_price_vs_dist.params)

explanatory_data = pd.DataFrame({
    "sqrt_dist_to_mrt_m": np.sqrt(np.arange(0, 81, 10) ** 2),
    "dist_to_mrt_m": np.arange(0, 81, 10) ** 2
})
prediction_data = explanatory_data.assign(
    price_twd_msq = mdl_price_vs_dist.predict(explanatory_data)
)
print(prediction_data)
```

#### **Fourth Root Transformation (for ad clicks/impressions)**
```python
# Add qdrt (fourth root) variable
df['qdrt_n_impressions'] = df['n_impressions'] ** 0.25
df['qdrt_n_clicks'] = df['n_clicks'] ** 0.25

# Fit model
model = ols('qdrt_clicks ~ qdrt_impressions', data=df)
model_result = model.fit()

# Example from actual notebook:
ad_conversion = pd.read_csv('ad_conversion.csv')
ad_conversion['qdrt_n_impressions'] = ad_conversion['n_impressions'] ** 0.25
ad_conversion['qdrt_n_clicks'] = ad_conversion['n_clicks'] ** 0.25

sns.regplot(x='qdrt_n_impressions', y='qdrt_n_clicks', data=ad_conversion, ci=None)
plt.show()

mdl_click_vs_impression = ols('qdrt_n_clicks ~ qdrt_n_impressions', data=ad_conversion).fit()
print(mdl_click_vs_impression.params)

# Back-transform predictions:
prediction_data['n_clicks'] = prediction_data['qdrt_n_clicks'] ** 4
```

---

### **6. Visualizing Predictions**

```python
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

# Example from actual notebook:
fig = plt.figure()
sns.regplot(x='length_cm_cubed', y='mass_g', data=perch, ci=None)
sns.scatterplot(data=prediction_data, x='length_cm_cubed', y='mass_g', color='red', marker='s')
plt.show()
```

---

### **7. Model Evaluation Metrics**

```python
# Get model summary
print(model_result.summary())

# Extract R-squared
r_squared = model_result.rsquared
print(f"R-squared: {r_squared}")

# Extract RSE (Residual Standard Error)
rse = np.sqrt(model_result.mse_resid)
print(f"RSE: {rse}")

# Example from actual notebook:
print(mdl_click_vs_impression.summary())
r_squared = mdl_click_vs_impression.rsquared
rse = np.sqrt(mdl_click_vs_impression.mse_resid)
print(f"\n--- Interpretation ---")
print(f"R-squared: {r_squared}")
print(f"RSE: {rse}")
print("Back-transformed errors vary with click magnitude")
```

---

## **LECTURE 9: K-NEAREST NEIGHBORS (KNN)**

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

---

### **1. Basic KNN Setup**

```python

# Load data
df = pd.read_csv('data.csv')

# Select features and target
X = df[['feature1', 'feature2']].values
y = df['target'].values

# Create and fit KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)

# Example from actual notebook:
churn_df = pd.read_csv("churn_df.csv")
X = churn_df[["account_length", "customer_service_calls"]].values
y = churn_df["churn"].values
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
```

---

### **2. Making Predictions with KNN**

```python
# New data to predict
X_new = np.array([[val1, val2],
                  [val3, val4],
                  [val5, val6]])

# Make predictions
y_pred = knn.predict(X_new)
print("Predictions:", y_pred)

# Example from actual notebook:
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
y_pred = knn.predict(X_new)
print("Predictions:", y_pred)
```

---

### **3. Train/Test Split**

```python
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

# Example from actual notebook:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                   random_state=21,
                                                   stratify=y)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

---

### **4. Finding Optimal k**

```python
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

# Example from actual notebook:
neighbors = np.arange(1, 26)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.figure(figsize=(8, 6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, list(train_accuracies.values()), label='Training Accuracy')
plt.plot(neighbors, list(test_accuracies.values()), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
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

# Example from actual notebook:
X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

---

## **LECTURE 10: ADVANCED ML TECHNIQUES**

```python
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report
import numpy as np
```

---

### **1. Cross-Validation**

```python
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

# Example from actual notebook:
sales_df = pd.read_csv('sales_df.csv', index_col=0)
X = sales_df.drop(["tv","sales"], axis=1).values
y = sales_df["sales"].values

kf = KFold(n_splits=6, shuffle=True, random_state=5)
reg = LinearRegression()
cv_scores = cross_val_score(reg, X, y, cv=kf)

print(f'CV scores: {cv_scores}')
print(f'Mean: {np.mean(cv_scores)}')
print(f'STD: {np.std(cv_scores)}')
print(f'Confidence interval: {np.quantile(cv_scores, [0.025, 0.975])}')
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

# Example from actual notebook:
X_cat = music_df["genre"].values.reshape(-1, 1)
X_num = music_df.drop(["genre", "popularity"], axis=1).values

imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
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

# Example from actual notebook:
le = LabelEncoder()
X_train_cat = le.fit_transform(X_train_cat)
X_test_cat = le.transform(X_test_cat)
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

# Example from actual notebook:
music_dummies = pd.get_dummies(music_df['genre'], drop_first=True)
music_dummies = pd.concat([music_df, music_dummies], axis=1)
music_dummies = music_dummies.drop('genre', axis=1)
```

---

### **4. Combining Preprocessed Data**

```python
# Combine numerical and categorical features
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)

print(X_train.shape, X_test.shape)

# Example from actual notebook:
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)
```

---

### **5. Using Pipelines**

```python
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

# Example from actual notebook:
steps = [('imputation', SimpleImputer()),
         ('Log_reg', LogisticRegression(max_iter=100000))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))
```

---

### **6. Linear Regression for Prediction**

```python
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

# Example from actual notebook:
sales_df = pd.read_csv('sales_df.csv', index_col=0)
X = sales_df["radio"].values
y = sales_df["sales"].values
X = X.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X, y)
predictions = reg.predict(X)

plt.scatter(X, y, color="blue")
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")
plt.show()
```

---

### **7. Model Evaluation Metrics**

```python
# R-squared
r_squared = reg.score(X_test, y_test)
print(f"R^2: {r_squared}")

# RMSE
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Example from actual notebook:
sales_df = pd.read_csv('sales_df.csv', index_col=0)
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

r_squared = reg.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))
```

---

## **LECTURE 11: COMPLETE ML PIPELINE (CREDIT CARD PREDICTOR)**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

---

### **1. Data Loading & Initial Exploration**

```python

# Load data without headers if needed
df = pd.read_csv('data.csv', header=None)

# Explore data
print(df.head())
print(df.describe())
print(df.info())
print(df.tail())

# Example from actual notebook:
cc_apps = pd.read_csv("cc_approvals.data", header=None)
cc_apps.head()

print(cc_apps.describe())
print('\n')
print(cc_apps.info())
print('\n')
cc_apps.tail(17)
```

---

### **2. Missing Value Handling**

```python
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

# Example from actual notebook:
cc_apps = cc_apps.replace('?', np.nan)
cc_apps.loc[[2,7,10,14]] = cc_apps.loc[[2,7,10,14]].fillna(
    cc_apps.mean(numeric_only=True))
print(cc_apps.isna().sum())

for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])
print(cc_apps.isna().sum())
```

---

### **3. Label Encoding for Categorical Data**

```python
# Encode each object column
for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])

# Example from actual notebook:
le = LabelEncoder()
for col in list(cc_apps):
    if cc_apps[col].dtypes=='object':
        cc_apps[col]=le.fit_transform(cc_apps[col])
```

---

### **4. Train/Test Split (Before Preprocessing - INCORRECT APPROACH)**

⚠️ **Note:** This approach fits on test data (data leakage). See proper approach below.

```python
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

# Example from actual notebook:
cc_apps = cc_apps.drop([11,13], axis=1)
cc_apps = cc_apps.values
X, y = cc_apps[:,0:12], cc_apps[:,13]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

---

### **5. Proper Split-Then-Preprocess Approach (CORRECT)**

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

# Example from actual notebook:
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)
cc_apps_train = cc_apps_train.replace('?', np.nan)
cc_apps_test = cc_apps_test.replace('?', np.nan)

cc_apps_train.fillna(cc_apps_train.mean(numeric_only=True), inplace=True)
cc_apps_test.fillna(cc_apps_train.mean(numeric_only=True), inplace=True)

print(cc_apps_train.isnull().sum())
print(cc_apps_test.isnull().sum())
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

# Example from actual notebook:
cc_apps_train = pd.get_dummies(cc_apps_train)  # try with argument dtype = int
cc_apps_test = pd.get_dummies(cc_apps_test)

# Reindex columns of test set aligning with train set
cc_apps_test = cc_apps_test.reindex(columns=cc_apps_train.columns, fill_value=0)
print(cc_apps_train)
```

---

### **7. Feature Scaling (MinMaxScaler)**

```python
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

# Example from actual notebook:
X_train = cc_apps_train.iloc[:, :-1].values
y_train = cc_apps_train.iloc[:, -1].values.ravel()
X_test = cc_apps_test.iloc[:, :-1].values
y_test = cc_apps_test.iloc[:, -1].values.ravel()

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
print(rescaledX_train.shape)
print(rescaledX_test.shape)
```

---

### **8. Logistic Regression**

```python
# Create classifier
logreg = LogisticRegression(max_iter=1000)

# Fit to training data
logreg.fit(rescaledX_train, y_train)

# Make predictions
y_pred = logreg.predict(rescaledX_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Example from actual notebook:
logreg = LogisticRegression(max_iter=1000)
logreg.fit(rescaledX_train, y_train)

y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

### **9. K-Nearest Neighbors**

```python
# With default k=5
knn = KNeighborsClassifier()
knn.fit(rescaledX_train, y_train)

y_pred_knn = knn.predict(rescaledX_test)

accuracy = knn.score(rescaledX_test, y_test)
print(f"Accuracy (k=5): {accuracy}")

print(confusion_matrix(y_test, y_pred_knn))

# Example from actual notebook:
knn = KNeighborsClassifier()
knn.fit(rescaledX_train, y_train)

y_pred_knn = knn.predict(rescaledX_test)

print("Accuracy of KNN classifier (k=5): ", knn.score(rescaledX_test, y_test))
print(confusion_matrix(y_test, y_pred_knn))
```

---

### **10. Finding Best k - Elbow Method**

```python
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
print(confusion_matrix(y_test, y_pred_elbow))

# Example from actual notebook from notebook:
error_rates = []
k_range = range(1, 41)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(rescaledX_train, y_train)
    y_pred_temp = knn_temp.predict(rescaledX_test)
    error_rates.append(np.mean(y_pred_temp != y_test))

best_k_elbow = k_range[np.argmin(error_rates)]

print(f"Best k using Elbow Method: {best_k_elbow}")

knn_elbow = KNeighborsClassifier(n_neighbors=best_k_elbow)
knn_elbow.fit(rescaledX_train, y_train)
y_pred_elbow = knn_elbow.predict(rescaledX_test)
accuracy_elbow = knn_elbow.score(rescaledX_test, y_test)

print(f"Accuracy with k={best_k_elbow} (Elbow Method): {accuracy_elbow:.4f}")

print("\nConfusion Matrix with best k (Elbow Method):")
print(confusion_matrix(y_test, y_pred_elbow))
```

---

### **11. Finding Best k - GridSearchCV**

```python
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

# Example from actual notebook:
knn_cv = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 41)}

knn_gscv = GridSearchCV(knn_cv, param_grid, cv=5, scoring='accuracy')
knn_gscv.fit(rescaledX_train, y_train)

best_k_cv = knn_gscv.best_params_['n_neighbors']
best_score_cv = knn_gscv.best_score_

print(f"Best k using GridSearchCV: {best_k_cv}")
print(f"Best cross-validation accuracy: {best_score_cv:.4f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k_cv)
best_knn.fit(rescaledX_train, y_train)
test_accuracy = best_knn.score(rescaledX_test, y_test)

print(f"Test set accuracy with k={best_k_cv}: {test_accuracy:.4f}")

y_pred_best_k = best_knn.predict(rescaledX_test)
print("\nConfusion Matrix with best k:")
print(confusion_matrix(y_test, y_pred_best_k))
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

# Example from actual notebook:
param_grid = {
    'n_neighbors': np.arange(1, 11),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(rescaledX_train, y_train)

print("Best Hyperparameters:", grid.best_params_)
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
# Convert categorical to binary (0/1)
df['binary_col'] = np.where(df['category'] == 'DesiredValue', 1, 0)

# Example from actual notebook:
music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
```

### **3. Checking Data Correlations**
```python
# Correlation matrix (only numeric)
print(df.corr(numeric_only=True))

# Single correlation
print(df['col1'].corr(df['col2']))

# Example from actual notebook:
print(cc_apps.corr(numeric_only=True))
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
- **Split BEFORE preprocessing to avoid data leakage** (critical for Lecture 11)
- Use `stratify=y` in train_test_split for classification
- Fit scalers on training data only, not on test data
- Use fit_transform on training data, transform only on test data
- For GridSearchCV, ensure you fit on training data and evaluate on test data separately
