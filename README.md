# ITX2007 Data Science Course Files

This repository contains the coursework, practice notebooks, datasets, and exam materials for **ITX2007 Data Science** at Assumption University of Thailand.

**Lecturer:** Dr. Thanachai Thumthawatworn  
**Student:** Sithu Win San (6726077)  
**Documentation sync:** 2026-04-04

## Repository Snapshot

- Covers Lectures 2-12, midterm submissions, final-practice notebooks, and final exam materials.
- Includes **29 notebooks**, **101 CSV files**, **24 TXT files**, and **38 pickle files** across the lecture, midterm, and final-exam folders.
- The newest academic additions in the repo are:
  - `Lecture11_DataSets - Credit Card Predictor/`
  - `Lecture12_DataSets - Unsupervised ML/`
  - `final_files/` with the Semester 2, 2025 final exam notebooks and datasets

## Top-Level Structure

| Path | What it contains |
|------|------------------|
| `Lecture_2/` | Lecture 2 notebook, practice notebook, and introductory CSV datasets |
| `Lecture_3/` | Lecture 3 EDA notebooks, CSV datasets, pickle datasets, and saved visuals |
| `Lecture_4/` | Lecture 4 data-cleaning notebooks and cleaning datasets |
| `Lecture5_DataSets - EDA Part 2/` | Lecture 5 advanced EDA notebook and related CSV files |
| `Lecture06_DataSets - Data Joining/` | Lecture 6 notebooks plus the largest collection of relational CSV and pickle datasets |
| `Lecture7_Data Sets - Intro2Statistics/` | Lecture 7 statistics notebook, extra practice notebooks, and final-practice notebook |
| `Lecture8 - Regression/` | Lecture 8 regression notebook, final-practice notebook, and regression datasets |
| `Lecture9_DataSets - ML Part 1/` | Lecture 9 notebook, final-practice notebook, and KNN/classification datasets |
| `Lecture10_DataSets - ML Part 1/` | Lecture 10 notebook, final-practice notebook, and preprocessing / evaluation datasets |
| `Lecture11_DataSets - Credit Card Predictor/` | Lecture 11 credit-card prediction notebook, final-practice notebook, and `cc_approvals.data` |
| `Lecture12_DataSets - Unsupervised ML/` | Lecture 12 K-Means notebook, final-practice notebook, and clustering datasets |
| `midterm_files/` | Midterm Q1 and Q2 notebooks with their supporting datasets |
| `final_files/` | Final exam Q1 and Q2 notebooks with current exam datasets and prediction inputs |
| `Lectures_7_to_12_Cheatsheet.md` | Full lecture reference for statistics, regression, ML pipelines, and clustering |
| `FINAL_EXAM_Cheatsheet.md` | Compressed exam-first quick reference based on Lectures 7-12 and the current final notebooks |
| `CLAUDE.md` | Repository guidance for agent workflows and notebook-editing expectations |

## Lecture Map

| Lecture | Folder | Main notebook(s) | Focus | Key datasets |
|---------|--------|------------------|-------|--------------|
| 2 | `Lecture_2/` | `6726077_SithuWinSan_L2.ipynb`, `midterm_practice_L2.ipynb`, `netflix.ipynb` | CSV loading, pandas basics, visualization | `netflix_data.csv`, `L2_Superstore_Sales.csv`, `cars.csv`, `brics.csv` |
| 3 | `Lecture_3/` | `6726077_SithuWinSan_L3.ipynb`, `midterm_practice_L3.ipynb` | EDA, profiling, grouping, visualization | `avocados.csv`, `temperatures.csv`, `homelessness.csv`, World Bank CSVs, many `.p` data objects |
| 4 | `Lecture_4/` | `6726077_SithuWinSan_L4.ipynb`, `midterm_practice_L4.ipynb` | Data cleaning, missing values, outliers, type fixes | airline datasets, salary datasets, `divorce.csv` |
| 5 | `Lecture5_DataSets - EDA Part 2/` | `6726077_SithuWinSan_L5.ipynb`, `midterm_practice_L5.ipynb` | Advanced EDA and feature relationships | `Airlines_unclean.csv`, `divorce.csv`, salary CSVs |
| 6 | `Lecture06_DataSets - Data Joining/` | `6726077_SithuWinSan_L6.ipynb`, `midterm_practice_L6.ipynb` | Merge, join, concat, multi-table reasoning | inventory CSVs, TMDB/movie data, Chicago data, economic files |
| 7 | `Lecture7_Data Sets - Intro2Statistics/` | `6726077_SithuWinSan_L7.ipynb`, `final_practice_L7.ipynb`, `WorldOldestBusinesses.ipynb`, `notebook.ipynb` | Statistics, sampling, inference | `food_consumption.csv`, `world_happiness.csv`, `amir_deals.csv`, `restaurant_groups.csv` |
| 8 | `Lecture8 - Regression/` | `6726077_SithuWinSan_L8.ipynb`, `final_practice_L8.ipynb` | Linear regression and model interpretation | `fish.csv`, `taiwan_real_estate2.csv`, `ad_conversion.csv`, `swedish_motor_insurance.csv` |
| 9 | `Lecture9_DataSets - ML Part 1/` | `6726077_SithuWinSan_L9.ipynb`, `final_practice_L9.ipynb` | KNN, train/test split, evaluation | `churn_df.csv`, `diabetes.csv`, `sales_df.csv`, music datasets |
| 10 | `Lecture10_DataSets - ML Part 1/` | `6726077_SithuWinSan_L10.ipynb`, `final_practice_L10.ipynb` | Cross-validation, imputation, encoding, pipelines | `music.csv`, `music_unclean.csv`, `telecom_churn_clean.csv` |
| 11 | `Lecture11_DataSets - Credit Card Predictor/` | `6726077_SithuWinSan_L11.ipynb`, `final_practice_L11.ipynb` | Credit-card approval classification, scaling, GridSearchCV | `cc_approvals.data` |
| 12 | `Lecture12_DataSets - Unsupervised ML/` | `6726077_SithuWinSan_L12.ipynb`, `final_practice_L12.ipynb` | K-Means clustering, elbow method, clustering evaluation | `points.csv`, `new_points.csv`, `seeds.csv`, `wine.data`, `fish.csv` |

## Exam Materials

### Midterm

`midterm_files/` contains:

- `ITX2007_6726077_SithuWinSan_Midterm_2_2025_Q1.ipynb`
- `ITX2007_6726077_SithuWinSan_Midterm_2_2025_Q2.ipynb`
- `employees.csv`
- `foods_data.csv`
- `food_mg.csv`

The midterm material is centered on Lectures 2-6: data loading, cleaning, exploratory analysis, and feature-level reasoning.

### Final

`final_files/` contains the current final exam submissions and supporting inputs:

- `ITX2007_6726077_SithuWinSan_Final_2_2025_Q1.ipynb`
- `ITX2007_6726077_SithuWinSan_Final_2_2025_Q2.ipynb`
- `train_1.csv`
- `train_2.csv`
- `telecom_churn.csv`
- `new_data.csv`
- `X_new.txt`

The final notebooks currently cover two main workflows:

1. **Q1**: combine `train_1.csv` and `train_2.csv`, clean loan-style training data, encode categorical features, inspect correlations, and fit a simple OLS regression.
2. **Q2**: build and compare KNN churn models across different test sizes, search for the best `k`, and predict churn on `new_data.csv`.

## Documentation Guide

These are the main documentation files that should stay in sync with the notebooks:

- `README.md`: repo structure, lecture map, setup, and exam-material overview
- `Lectures_7_to_12_Cheatsheet.md`: detailed study reference for the second half of the course
- `FINAL_EXAM_Cheatsheet.md`: quick exam workflow reference
- `CLAUDE.md`: notebook workflow guidance for repo-aware agents

When a new lecture notebook, practice notebook, or exam folder is added, update the relevant doc above in the same change.

## Technology Stack

- **Language:** Python 3
- **Environment:** Jupyter Notebook
- **Core libraries used across the notebooks:**
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `statsmodels`
  - `scikit-learn`
  - `pickle`

## Setup

```bash
git clone https://github.com/PracticalSwan/ITX2007-Data-Science.git
cd ITX2007-Data-Science
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn jupyter
jupyter notebook
```

Open the notebook you want from its lecture folder and keep the datasets in the same folder as the notebook that uses them.

## Naming Conventions

- Main lecture notebooks: `6726077_SithuWinSan_L[X].ipynb`
- Midterm practice notebooks: `midterm_practice_L[X].ipynb`
- Final practice notebooks: `final_practice_L[X].ipynb`
- Midterm submissions: `ITX2007_6726077_SithuWinSan_Midterm_2_2025_Q[1-2].ipynb`
- Final submissions: `ITX2007_6726077_SithuWinSan_Final_2_2025_Q[1-2].ipynb`

## Working Conventions

- Most notebooks are code-heavy and contain little or no markdown, so the companion cheatsheets are the main narrative reference.
- Datasets are generally co-located with the notebook that uses them.
- The later lectures build directly on earlier ones:
  - Lecture 7 introduces statistics and inference.
  - Lecture 8 moves into regression.
  - Lectures 9-11 focus on supervised ML workflows.
  - Lecture 12 adds unsupervised clustering.
- Lecture 11 and the final exam both include preprocessing pipelines, so pay close attention to leakage-safe workflows:
  - split before preprocessing
  - fit transformers on training data only
  - align train and test feature columns before modeling

## Learning Outcomes

By working through the full repository, the student practices how to:

- load and inspect datasets in several tabular formats
- clean inconsistent or incomplete data
- explore distributions and relationships visually
- merge and reshape multi-table data
- apply statistical reasoning and inference
- fit and interpret regression models
- train and evaluate classification models
- build preprocessing pipelines with scaling, imputation, and encoding
- tune models with cross-validation and grid search
- cluster unlabeled data with K-Means and evaluate the result

## Notes

- This is a coursework repository, so keep the original student submissions intact and update docs around them rather than rewriting notebook intent.
- `Lecture12_DataSets - Unsupervised ML/6726077_SithuWinSan_L12.ipynb` currently has local modifications, and `final_files/` is present as untracked content in the working tree. The docs in this repo now reflect those materials.
