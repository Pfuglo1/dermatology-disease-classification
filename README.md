# dermatology-disease-classification

# Deep Learning–Based Dermatology Disease Classification System
 
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?logo=scipy&logoColor=white)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
 
A data analysis and preprocessing pipeline for a **multi-class deep learning classifier** that distinguishes between **six erythemato-squamous skin diseases** using clinical observations and histopathological findings. The project covers data cleaning, exploratory analysis, statistical testing and feature engineering, preparing the data for a TensorFlow/Keras neural network.
 
---
 
## Table of Contents
 
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Author](#author)
---
 
## Problem Statement
 
Erythemato-squamous diseases are a group of skin conditions that are notoriously hard to tell apart. They share clinical features such as redness (erythema) and scaling, often with only subtle differences. Even a biopsy may not settle the diagnosis, because the diseases also share many histopathological features.
 
The goal is to build a model that classifies a patient into one of six diseases from **12 clinical** and **22 histopathological** attributes. This is a **multi-class classification** problem.
 
| Class | Disease | Patients |
|---|---|---|
| 1 | Psoriasis | 112 |
| 2 | Seborrheic Dermatitis | 61 |
| 3 | Lichen Planus | 72 |
| 4 | Pityriasis Rosea | 49 |
| 5 | Chronic Dermatitis | 52 |
| 6 | Pityriasis Rubra Pilaris | 20 |
 
---
 
## Dataset
 
**Source:** [Dermatology — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/33/dermatology)
 
- **Records:** 366 patients
- **Features:** 34 (33 ordinal + age)
- **Target:** `class` (6 diseases)
Most attributes are graded on an **ordinal 0–3 scale** (0 = absent, 3 = the largest amount possible). `family_history` is binary (0/1), and `age` is a continuous value in years.
 
<details>
<summary><b>Feature groups (click to expand)</b></summary>
**Clinical attributes**
erythema, scaling, definite_borders, itching, koebner_phenomenon, polygonal_papules, follicular_papules, oral_mucosal_involvement, knee_and_elbow_involvement, scalp_involvement, family_history, age
 
**Histopathological attributes**
melanin_incontinence, eosinophils_in_the_infiltrate, PNL_infiltrate, fibrosis_of_the_papillary_dermis, exocytosis, acanthosis, hyperkeratosis, parakeratosis, clubbing_of_the_rete_ridges, elongation_of_the_rete_ridges, thinning_of_the_suprapapillary_epidermis, spongiform_pustule, munro_microabcess, focal_hypergranulosis, disappearance_of_the_granular_layer, vacuolisation_and_damage_of_basal_layer, spongiosis, saw_tooth_appearance_of_retes, follicular_horn_plug, perifollicular_parakeratosis, inflammatory_monoluclear_infiltrate, band_like_infiltrate
 
</details>
---
 
## Project Workflow
 
### 1. Data Cleaning
- No null values and no duplicate rows.
- **Hidden missing values:** `age` was stored as text because 8 records contained `'?'`. These were replaced with the **median age (35)** and the column was converted to integer.
### 2. Outlier Treatment & Feature Reduction
- Applied **IQR capping** to all numerical columns.
- After capping, **17 columns became constant** (zero variance) and were dropped, leaving **17 features + target**.
### 3. Exploratory Data Analysis
- Class distribution analysis. The data is moderately imbalanced: Psoriasis makes up 31% and Pityriasis Rubra Pilaris only 5%.
- Histograms, KDE plots, box plots and frequency tables for every feature.
- Correlation heatmap and feature–target correlation chart.
- Pair plot coloured by disease class.
### 4. Statistical Analysis
| Test | Purpose | Result |
|---|---|---|
| **One-way ANOVA** | Do feature means differ across the 6 diseases? | All 17 features significant (p < 0.05) |
| **Chi-square test of independence** | Is each ordinal feature associated with disease class? | All features significant (p < 0.05) |
| **Shapiro–Wilk** | Normality check | No feature is normally distributed |
 
Based on the normality results, a **Yeo-Johnson power transform** was applied to reduce skewness.
 
### 5. Feature Engineering
Four domain-inspired composite features were created:
 
| Feature | Formula | Clinical intuition |
|---|---|---|
| `psoriasis_severity_index` | scaling + parakeratosis + hyperkeratosis + elongation_of_rete_ridges | Classic markers of psoriatic epidermal change |
| `inflammatory_activity_score` | PNL_infiltrate + exocytosis + spongiosis + inflammatory_mononuclear_infiltrate | Overall inflammatory burden |
| `epidermal_damage_ratio` | thinning_of_suprapapillary_epidermis / (hyperkeratosis + 1) | Epidermal thinning relative to thickening |
| `clinical_histopathological_interaction` | itching × spongiosis | Links a clinical symptom with its tissue-level counterpart (typical of dermatitis) |
 
The final modelling dataset has **21 features + target**.
 
---
 
## Key Findings
 
- 🔬 Every retained feature differs significantly across the six diseases, which confirms the dataset carries strong diagnostic signal.
- ⚖️ The class imbalance (112 vs. 20 patients) means the model should be judged on **per-class recall and macro-F1**, not accuracy alone.
- 📉 Ordinal 0–3 features are inherently non-normal. Power transforms help, but tree-based models or one-hot encoding may also be worth exploring.
---
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistics | SciPy (`f_oneway`, `chi2_contingency`, `shapiro`) |
| Preprocessing | scikit-learn (`PowerTransformer`, `LabelEncoder`, `StandardScaler`) |
| Deep learning | TensorFlow / Keras |
| Environment | Google Colab |
 
---
 
## How to Run
 
### Option 1: Google Colab (recommended)
1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/33/dermatology) and save it with column headers as `dermatology_with_columns.csv`.
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. Upload the CSV to `/content/`.
4. Run all cells.
### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/dermatology-disease-classification.git
cd dermatology-disease-classification
pip install -r requirements.txt
jupyter notebook
```
Then update the `path` variable in the first code cell to point to your local copy of the CSV.
 
---
 
## Project Structure
 
```
dermatology-disease-classification/
├── Dermatology_Disease_Classification.ipynb   # Cleaning, EDA, statistics, feature engineering
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```
 
---
 
## Roadmap
 
- [x] Data cleaning (hidden `'?'` values in age)
- [x] Outlier treatment and constant-feature removal
- [x] Exploratory data analysis
- [x] Statistical testing (ANOVA, Chi-square, Shapiro–Wilk)
- [x] Feature transformation & engineering
- [ ] Stratified train/test split and feature scaling
- [ ] Multi-class neural network (Keras, softmax output, `to_categorical` targets)
- [ ] Evaluation with confusion matrix, classification report and macro-F1
- [ ] Baseline comparison (Logistic Regression, Random Forest)
**Technical notes for the modelling stage**
- **Revisit the outlier capping.** On ordinal 0–3 features, IQR capping treats rare-but-real high grades as outliers. That is why 17 features collapsed to constants and others gained an artificial `2.5` level. Several of the dropped features are the **hallmark markers** of specific diseases: `band_like_infiltrate` and `polygonal_papules` for lichen planus, `follicular_horn_plug` for pityriasis rubra pilaris, and `munro_microabcess` for psoriasis. Skipping capping for ordinal features (and capping only `age` if needed) should noticeably improve class separation.
- Fit the power transformer on the training split only, ideally inside a scikit-learn `Pipeline`, to avoid data leakage.
- Use `class_weight` or stratified sampling to handle the minority class (Pityriasis Rubra Pilaris, 20 patients).
---
 
## Author
 
**Patricia Fuglo**
 
⭐ If you found this project useful, consider giving it a star!
 
