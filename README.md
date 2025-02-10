# Evaluating Column Type Annotation Models and Benchmarks

This repository contains a set of Python scripts developed as part of the work for the paper **"Evaluating Column Type Annotation Models and Benchmarks"**. The code in this repository provides tools to process datasets, analyze overlaps between test and training sets, detect ambiguous column annotations, and even leverage the OpenAI API to predict column types.

## Overview of Scripts

### 1. `overlap_analysis.py`
- **Purpose:**  
  Computes the overlap between the test set and training set of the SATO dataset.
- **Functionality:**  
  - Loads the SATO dataset (e.g., `sato_cv_0.csv` for the test set and `sato_cv_1.csv`–`sato_cv_4.csv` for the training set).
  - Tokenizes column entries and measures the proportion of overlapping tokens between each test instance and matching training instances.
  - Categorizes test indices based on overlap thresholds (e.g., 100%, ≤60%, ≤30%, ≤10%).
  - Saves the overlap results as a JSON file (`overlap_dict.json`).

### 2. `ambiguity_analysis.py`
- **Purpose:**  
  Detects ambiguous column annotations within the datasets.
- **Functionality:**  
  - Analyzes the datasets, to detect columns where the same or similar data entries are associated with different column type labels.
  - Identifies ambiguous columns and outputs results (including table IDs, data, and flags indicating ambiguity) as a compressed CSV file (`same_column_dif_type.zip`).

### 3. `column_type_predictor.py`
- **Purpose:**  
  Uses GPT 3.5 to predict column types from provided column data.
- **Functionality:**  
  - Contains prompt templates for single-column classification as well as few-shot multi-column examples.
  - Defines a clean, modular function that sends a prompt along with column data to the OpenAI ChatCompletion API.
  - Returns the predicted column type based on the model's response.

## Datasets

- **SATO (Viznet) Dataset:**  
  The SATO dataset is divided into several CSV files:
  - `sato_cv_0.csv` is typically used as the test set.
  - `sato_cv_1.csv`–`sato_cv_4.csv` serve as the training set.

- **SOTAB Dataset:**  
  Contains ambiguous column information to further evaluate column type annotation models under conditions of data ambiguity.

## How to Run

1. **Install Dependencies:**  
   Ensure you have Python 3 installed. Install the required packages via pip:
   ```bash
   pip install pandas openai scikit-learn
