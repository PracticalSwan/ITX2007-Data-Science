# ITX2007 Data Science - Course Files

This repository contains all assignments and datasets for the **ITX2007 Data Science** course at Assumption University of Thailand. The course covers fundamental to intermediate data science concepts using Python and Jupyter Notebooks.

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

- **Practice Files:**
  - `midterm_practice_L6.ipynb` - Midterm practice exercises for data joining and merging

- **Lecture File:**
  - `6726077_SithuWinSan_L6.ipynb` - Main lecture assignment on data merging and joins

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

## Technology Stack

- **Language:** Python 3
- **Notebooks:** Jupyter Notebook
- **Key Libraries:**
  - pandas - Data manipulation and analysis
  - NumPy - Numerical computing
  - Matplotlib & Seaborn - Data visualization
  - Scikit-learn - Machine learning basics
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

---

## Notes

- All datasets are provided in CSV, pickle (`.p`), and text (`.txt`) formats for flexibility
- Some notebooks include practice exercises from midterm preparation
- Data sizes vary; some datasets (Netflix, airlines) are larger and may take time to process
- Pickle files contain pre-processed data objects for advanced analyses
3
---

*Last Updated: January 2, 2026*
