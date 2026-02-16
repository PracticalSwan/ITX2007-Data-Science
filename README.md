# ITX2007 Data Science - Course Files

This repository contains all assignments and datasets for the **ITX2007 Data Science** course at Assumption University of Thailand. The course covers fundamental to intermediate data science concepts using Python and Jupyter Notebooks.

**Lecturer:** Dr. Thanachai Thumthawatworn

**Student:** Sithu Win San (6726077)

---

## Repository Structure

### Lecture 2: Data Fundamentals
**Topics:** Introduction to data types, basic data manipulation, and visualization

- **Lecture File:**
  - `6726077_SithuWinSan_L2.ipynb` - Main lecture assignment covering data fundamentals

- **Practice Files:**
  - `midterm_practice_L2.ipynb` - Midterm practice exercises
  - `netflix.ipynb` - Netflix dataset analysis practice

- **Datasets:**
  - `netflix_data.csv` - Netflix movies and shows dataset
  - `L2_Superstore_Sales.csv` - Superstore sales data
  - `brics.csv` - BRICS countries data
  - `cars.csv` - Automobile dataset
  - `color_data2.csv` - Color-related dataset

**Skills Covered:**
- Working with CSV files
- Basic data exploration
- Data type understanding
- Simple visualizations

---

### Lecture 3: Exploratory Data Analysis (EDA) - Part 1
**Topics:** Data profiling, statistical analysis, and advanced visualization

- **Lecture File:**
  - `6726077_SithuWinSan_L3.ipynb` - Main lecture assignment on EDA techniques

- **Practice Files:**
  - `midterm_practice_L3.ipynb` - Comprehensive EDA practice exercises

- **Datasets:**
  - `avocados.csv` - Avocado prices and sales
  - `avocados_2016.csv` - 2016 avocado dataset
  - `actors_movies.csv` - Movie actors data
  - `S&P500.csv` - S&P 500 stock data
  - `WorldBank_GDP.csv`, `WorldBank_POP.csv`, `WorldBank_POP_NEW.csv` - World Bank economic indicators
  - `sales_subset.csv` - Sales transaction subset
  - `temperatures.csv` - Temperature measurements
  - `thailand_2005_2010_temps.csv` - Thailand temperature data
  - `countries_avg_temp_20_30.csv` - Average country temperatures
  - `homelessness.csv` - Homelessness statistics
  - `gdp_growing_2010_2018.csv`, `gdp_growth_2010_2018_pct.csv` - GDP growth data

- **Pickled Data Objects:**
  - `business_owners.p`, `casts.p`, `census.p`, `crews.p` - Movie and demographic data
  - `cta_calendar.p`, `cta_ridership.p` - Chicago Transit Authority data
  - `financials.p` - Financial records
  - `land_use.p`, `licenses.p` - Urban planning and licensing data
  - `movie_to_genres.p`, `movies.p`, `ratings.p` - Movie metadata
  - `sequels.p`, `stations.p`, `taglines.p` - Movie sequels and descriptions
  - `taxi_owners.p`, `taxi_vehicles.p` - Taxi fleet information
  - `ward.p`, `zip_demo.p` - Geographic and demographic data

- **Visualizations:**
  - `gdp_growth_lineplot.png` - GDP growth visualization
  - `avoplotto.pkl` - Pickled plot objects

**Skills Covered:**
- Data profiling and summary statistics
- Distribution analysis
- Correlation analysis
- Advanced plotting techniques
- Handling multiple data formats

---

### Lecture 4: Data Cleaning & Preprocessing
**Topics:** Data quality assessment, missing values, and data transformation

- **Lecture File:**
  - `6726077_SithuWinSan_L4.ipynb` - Main lecture assignment on data cleaning and preprocessing

- **Practice Files:**
  - `midterm_practice_L4.ipynb` - Midterm practice exercises for data cleaning

- **Datasets (Airline Data):**
  - `Airline.csv`, `Airline.txt` - Complete airline dataset
  - `Airlines_unclean.csv` - Unclean airline data
  - `AirlineV1.csv`, `AirlineV2.csv`, `AirlineV4.csv` - Different versions of airline data

- **Datasets (Other):**
  - `divorce.csv` - Divorce statistics
  - `ds_salaries.csv`, `ds_salaries - Copy.csv`, `ds_salaries_clean.csv` - Data science job salaries
  - `Salaries_with_date_of_response.csv`, `Salary_Rupee_USD.csv` - Salary data with conversion
  - `Salary_Job_Category.csv`, `Salary_Job_Category.txt` - Salary by job category
  - `Salary_No_Job_Category.txt` - Salary data without category
  - `clean_books.csv` - Cleaned book data
  - `clean_unemployment.csv` - Cleaned unemployment data

**Skills Covered:**
- Identifying and handling missing values
- Detecting and treating outliers
- Data type conversion
- String manipulation
- Standardization and normalization
- Data quality assessment

---

### Lecture 5: Exploratory Data Analysis (EDA) - Part 2
**Topics:** Advanced EDA techniques and hypothesis testing

- **Lecture File:**
  - `6726077_SithuWinSan_L5.ipynb` - Main lecture assignment on advanced EDA and statistical testing

- **Practice Files:**
  - `midterm_practice_L5.ipynb` - Midterm practice exercises for advanced EDA

- **Datasets:**
  - `Airlines_unclean.csv` - Unclean airline data for advanced analysis
  - `divorce.csv` - Divorce data analysis
  - `Salaries_with_date_of_response.csv` - Time-series salary data
  - `Salary_Rupee_USD.csv` - International salary comparison

**Skills Covered:**
- Advanced statistical analysis
- Hypothesis testing
- Multivariate analysis
- Feature relationships
- Business insight generation

---

### Lecture 6: Data Joining & Merging
**Topics:** Combining multiple data sources and relational data operations

- **Lecture File:**
  - `6726077_SithuWinSan_L6.ipynb` - Main lecture assignment on data merging and joins

- **Practice Files:**
  - `midterm_practice_L6.ipynb` - Midterm practice exercises for data joining and merging

- **Datasets (Multiple Sources for Merging):**
  - **Inventory Data:** `inv_jul.csv`, `inv_aug.csv`, `inv_sep.csv` (with corresponding `.txt` versions)
  - **Music Data:** `tracks_master.csv`, `tracks_ride.csv`, `tracks_st.csv` (with `.txt` versions)
  - **Movie Data:** `pop_movies.csv`, `actors_movies.csv`, `tmdb_movies.csv`, `toy_story.csv`
  - **Movie Metadata:** `tdmb_movie_to_genres.csv`, `tdmb_taglines.csv` (TMDB API data)
  - **Music Metadata:** `music_unclean.csv`
  
- **Economic & Geographic Data:**
  - `GDP.csv`, `GDP.txt` - GDP data
  - `inflation.csv`, `inflation.txt` - Inflation rates
  - `unemployment.csv`, `unemployment.txt` - Unemployment statistics
  - `WorldBank_GDP.csv`, `WorldBank_POP.csv` - World Bank indicators
  - `S&P500.csv` - Stock market data
  - `ds_salaries.csv` - Data science salaries

- **Chicago Data:**
  - `Business_Licenses.csv` - Chicago business licenses
  - `Wards_Census.csv`, `Wards_Offices.csv`, `Wards_Offices_Altered.csv` - Chicago ward data
  - Pickled files: `cta_calendar.p`, `cta_ridership.p`, `business_owners.p`, `licenses.p`, `taxi_owners.p`, `taxi_vehicles.p`, `stations.p`

- **Relational Data:**
  - Pickled objects: `movies.p`, `ratings.p`, `casts.p`, `crews.p`, `movie_to_genres.p`, `sequels.p`, `taglines.p`
  - Pickled objects: `census.p`, `land_use.p`, `financials.p`, `ward.p`, `zip_demo.p`

**Skills Covered:**
- Inner, outer, left, and right joins
- Merging on multiple columns
- Handling duplicate keys
- Concatenating datasets
- Data integrity checks
- Complex multi-table queries

---

### Lecture 7: Introduction to Statistics
**Topics:** Statistical foundations, probability distributions, and hypothesis testing

- **Lecture File:**
  - `6726077_SithuWinSan_L7.ipynb` - Main lecture assignment on statistical analysis

- **Practice Files:**
  - `notebook.ipynb` - Additional statistical practice
  - `WorldOldestBusinesses.ipynb` - Case study on world's oldest businesses

- **Datasets:**
  - `food_consumption.csv` - Global food consumption patterns
  - `world_happiness.csv` - World happiness index data
  - `amir_deals.csv` - Sales deals dataset
  - `restaurant_groups.csv`, `restuaraunt_groups.txt` - Restaurant chain data

**Skills Covered:**
- Descriptive statistics
- Probability distributions (normal, binomial, etc.)
- Statistical inference
- Hypothesis testing (t-tests, chi-square)
- Confidence intervals
- Statistical significance
- A/B testing fundamentals

---

### Lecture 9: Machine Learning - Classification
**Topics:** K-Nearest Neighbors (KNN), classification algorithms, and model evaluation

- **Lecture File:**
  - `6726077_SithuWinSan_L9.ipynb` - Main lecture assignment on classification with KNN

- **Datasets:**
  - `churn_df.csv`, `churn_df.txt` - Customer churn classification data
  - `diabetes.csv`, `diabetes_clean.csv` - Diabetes prediction dataset
  - `music.csv`, `music_df.txt`, `music_unclean.csv`, `music_unclean.txt`, `music_unsort.csv` - Music classification data
  - `advertising_and_sales_clean.csv` - Advertising campaigns with sales outcomes
  - `sales_df.csv` - Sales data for classification tasks
  - `telecom_churn_clean.csv` - Telecommunications customer churn data

**Skills Covered:**
- K-Nearest Neighbors (KNN) classifier
- Training and testing splits
- Model accuracy evaluation
- Hyperparameter tuning (number of neighbors)
- Overfitting and underfitting detection
- Classification performance metrics
- Feature selection for classification

---

### Lecture 10: Advanced Machine Learning
**Topics:** Cross-validation, model tuning, advanced preprocessing, and model pipelines

- **Lecture File:**
  - `6726077_SithuWinSan_L10.ipynb` - Main lecture assignment on advanced ML techniques

- **Datasets:**
  - `churn_df.csv`, `churn_df.txt` - Customer churn for advanced classification
  - `diabetes.csv`, `diabetes_clean.csv` - Diabetes prediction for model evaluation
  - `music.csv`, `music_df.txt`, `music_unclean.csv`, `music_unclean.txt`, `music_unsort.csv` - Music data with categorical features
  - `advertising_and_sales_clean.csv` - Advertising data for regression models
  - `sales_df.csv` - Sales data for supervised learning
  - `telecom_churn_clean.csv` - Telecom churn for model comparison

**Skills Covered:**
- Cross-validation techniques (K-Fold)
- Model performance evaluation with cross-validation
- SimpleImputer for missing value handling
- LabelEncoder for categorical data transformation
- One-hot encoding with pandas
- Combining numerical and categorical features
- Multiple model comparison
- Root Mean Squared Error (RMSE) calculation
- Model selection and hyperparameter optimization

---

### Lecture 8: Regression Analysis
**Topics:** Linear regression, multiple regression, and predictive modeling

- **Lecture File:**
  - `6726077_SithuWinSan_L8.ipynb` - Main lecture assignment on regression techniques

- **Datasets:**
  - `fish.csv` - Fish market data for regression analysis
  - `taiwan_real_estate2.csv` - Taiwan real estate prices
  - `swedish_motor_insurance.csv` - Insurance claims data
  - `sp500_yearly_returns.csv` - S&P 500 historical returns
  - `ad_conversion.csv` - Advertising conversion rates
  - `churn.csv` - Customer churn prediction data

**Skills Covered:**
- Simple linear regression
- Multiple linear regression
- Model evaluation (R², RMSE, MAE)
- Feature selection
- Residual analysis
- Multicollinearity detection
- Prediction and forecasting
- Model interpretation

---

### Midterm Examination Files
**Topics:** Comprehensive assessment of data science fundamentals, EDA, and data preprocessing

- **Midterm Notebooks:**
  - `ITX2007_6726077_SithuWinSan_Midterm_2_2025_Q1.ipynb` - Midterm Question 1 submission
  - `ITX2007_6726077_SithuWinSan_Midterm_2_2025_Q2.ipynb` - Midterm Question 2 submission

- **Datasets:**
  - `employees.csv` - Employee demographic and performance data
  - `foods_data.csv` - Food items with nutritional information
  - `food_mg.csv` - Food magnesium content dataset

**Exam Coverage:**
- Application of all concepts from Lectures 2-6
- Real-world data analysis scenarios
- Data cleaning and preprocessing techniques
- Exploratory data analysis
- Data transformation and feature engineering
- Statistical analysis and insights generation
- Data visualization best practices

**Note:** This folder contains the official midterm examination submissions for Semester 2, 2025. The exam tested comprehensive understanding of pandas, data manipulation, statistical analysis, and problem-solving skills covered throughout the course.

---

### Lecture 11: Machine Learning - Credit Card Prediction
**Topics:** Credit card approval prediction, advanced preprocessing, hyperparameter tuning, and model comparison

- **Lecture File:**
  - `6726077_SithuWinSan_L11.ipynb` - Main lecture assignment on credit card prediction

- **Datasets:**
  - `cc_approvals.data` - Credit card approval application data

**Skills Covered:**
- Handling missing values with mean imputation
- Categorical data encoding using LabelEncoder
- MinMaxScaler for feature normalization
- Logistic Regression for binary classification
- K-Nearest Neighbors (KNN) classifier
- Elbow Method for finding optimal k value
- GridSearchCV for hyperparameter tuning with cross-validation
- Confusion matrix analysis
- Model accuracy evaluation and comparison
- Train-test splitting strategies
- Feature selection and correlation analysis

**Key Models Implemented:**
- Logistic Regression with `max_iter=1000`
- KNN Classifier with k=5 (default)
- KNN with optimal k from Elbow Method
- KNN with optimal k from GridSearchCV (5-fold CV)
- Hyperparameter grid search including: n_neighbors, weights, algorithm, and distance metrics (p=1,2)

---

## Technology Stack

- **Language:** Python 3
- **Notebooks:** Jupyter Notebook
- **Key Libraries:**
  - pandas - Data manipulation and analysis
  - NumPy - Numerical computing
  - Matplotlib & Seaborn - Data visualization
  - Scikit-learn - Machine learning algorithms (classification, regression, cross-validation)
  - pickle - Data serialization

---

## Dataset Overview

| Lecture | Primary Datasets | Record Count (Approximate) | Focus Area |
|---------|------------------|---------------------------|-----------|
| 2 | Netflix, Superstore, Cars, BRICS | Varies | Fundamentals |
| 3 | Avocados, Movies, Economics, Temperature | 1000-100,000+ | EDA Techniques |
| 4 | Airlines, Salaries, Divorce, Books | 1000-10,000 | Data Cleaning |
| 5 | Airlines, Salaries, Divorce | 1000-5,000 | Advanced EDA |
| 6 | Movies, Music, Economics, Chicago, Inventory | Variable | Data Merging |
| Midterm | Employees, Foods, Food Nutrients | Variable | Comprehensive Assessment |
| 7 | Food Consumption, Happiness, Businesses | 100-10,000 | Statistics |
| 8 | Fish, Real Estate, Insurance, Stocks | 500-5,000 | Regression |
| 9 | Churn, Diabetes, Music, Advertising, Sales | Variable | ML Classification |
| 10 | Churn, Diabetes, Music, Advertising, Sales | Variable | Advanced ML |
| 11 | Credit Card Approvals | Variable | Real-world ML Application |

---

## File Naming Convention

- **Main Assignments:** `6726077_SithuWinSan_L[X].ipynb` where X is the lecture number
- **Practice Assignments:** `midterm_practice_L[X].ipynb`
- **Specialty Assignments:** Named by topic (e.g., `netflix.ipynb`)

---

## How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PracticalSwan/ITX2007-Data-Science.git
   cd ITX2007-Data-Science
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Navigate to the desired lecture folder and open the corresponding `.ipynb` file**

---

## Learning Outcomes

By completing all assignments in this repository, students will be able to:

- ✅ Load, explore, and understand different data formats
- ✅ Perform comprehensive exploratory data analysis
- ✅ Identify and handle data quality issues
- ✅ Create meaningful visualizations
- ✅ Merge and combine multiple data sources
- ✅ Generate actionable business insights from data
- ✅ Apply statistical methods to validate findings
- ✅ Conduct hypothesis testing and statistical inference
- ✅ Build and evaluate regression models
- ✅ Implement classification algorithms (K-Nearest Neighbors)
- ✅ Apply cross-validation for robust model evaluation
- ✅ Handle missing values and categorical features
- ✅ Perform hyperparameter tuning and model optimization
- ✅ Make predictions using machine learning techniques
- ✅ Build real-world binary classification models (Logistic Regression, KNN)
- ✅ Implement Elbow Method for optimal hyperparameter selection
- ✅ Use GridSearchCV for comprehensive hyperparameter tuning
- ✅ Compare model performance using accuracy metrics and confusion matrices
- ✅ Feature scaling and normalization techniques
- ✅ Evaluate classifier performance on test data

---

## Notes

- All datasets are provided in CSV, pickle (`.p`), and text (`.txt`) formats for flexibility
- Some notebooks include practice exercises from midterm preparation
- Data sizes vary; some datasets (Netflix, airlines) are larger and may take time to process
- Pickle files contain pre-processed data objects for advanced analyses
3
---

*Last Updated: February 16, 2026*
