# Breast Cancer ML Classifier

A machine learning classifier to assist radiologists in distinguishing malignant from benign breast masses using cell nucleus morphology features.

---

## Business Question

> Can we develop a machine learning model that uses cell morphology features to help radiologists differentiate malignant from benign breast masses, thereby reducing unnecessary biopsies without compromising cancer detection rates?

---

## Repository Structure

```
breast-cancer-classifier/
├── .github
│   └── pull_request_template.md
├── data
│   ├── processed
│   └── raw
├── experiments
├── models
├── notebooks
│   └── 02_preprocessing.ipynb
├── reports
│   └── figures
├── src
│   └── utils
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── repo_workflow.md
└── SETUP.md
```

---

## Dataset

- **Source:** [UCI ML Repository — Wisconsin Breast Cancer Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Samples:** 569 patients
- **Features:** 30 continuous features derived from digitized FNA (fine needle aspirate) images
- **Target:** Binary — Malignant (212) / Benign (357)
- **Feature groups:** Each of 10 nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) captured as mean, standard error, and worst value
- **Data quality:** No missing values

---

## Methodology

1. **Data Cleaning** — Missing value check, duplicate detection, outlier analysis (IQR)
2. **EDA** — Class distribution, feature distributions by diagnosis, box plots, scatter plots, correlation heatmap
3. **Preprocessing** — Target encoding, StandardScaler normalization, stratified 80/20 train-test split
4. **Baseline Model** — Logistic Regression
5. **Advanced Models** — Random Forest, Support Vector Machine (SVM)
6. **Hyperparameter Tuning** — GridSearchCV with 10-fold cross-validation, optimized for recall
7. **Evaluation** — AUC-ROC curves, confusion matrices, feature importance, threshold tuning

---

## Model Results

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | Missed Cancers (FN) | Unnecessary Biopsies (FP) |
|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest (Default) | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest (Tuned) | TBD | TBD | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Key Findings

- TBD — Top predictive features (e.g., worst concave points, worst radius)
- TBD — Recommended clinical threshold and sensitivity/specificity trade-off
- TBD — Best performing model for clinical use case

---

## Clinical Interpretation

- **Sensitivity is the priority metric** — a false negative (missed cancer) is far more costly than a false positive (unnecessary biopsy)
- **Threshold tuning** allows the model to be calibrated to clinical risk tolerance
- **Feature importance** maps directly to measurable cell morphology properties, supporting radiologist trust and interpretability

---

## Limitations & Next Steps

- Small dataset (569 patients) — real deployment requires larger, more diverse cohorts
- Features are computed from FNA images, not raw image pixels — deep learning on raw images may improve performance
- External validation on independent hospital data required before clinical use
- Future work: SHAP values for individual prediction explanations, ensemble stacking

---

## Team

| Name | GitHub | Primary Role |
|---|---|---|
| Marie Perry | [@mvrieperry](https://github.com/mvrieperry) | Evaluation & Integration Lead |
| Sarah Creighton | [@sarahcreighton](https://github.com/sarahcreighton) | EDA & Multicollinearity Lead |
| Rajesh Detroja | [@Rajesh-Detroja](https://github.com/Rajesh-Detroja) | Linear Modeling Lead |
| Sean Rampersad | [@seanlr-github](https://github.com/seanlr-github) | Tree-based Modeling Lead |

---

## Technical Stack

- **Language:** Python >= 3.11
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn *(see also [pyproject.toml](https://github.com/sarahcreighton/breast-cancer-classifier/blob/main/pyproject.toml))*
- **Environment:** [uv](https://github.com/astral-sh/uv)
- **IDE:** VS Code

---

## Setup
**Step 1**: Clone the repository
```bash
# Clone the repo
git clone https://github.com/[username]/breast-cancer-classifier.git
cd breast-cancer-classifier
```

**Step 2**: Follow the instrutions in [SETUP.md](https://github.com/sarahcreighton/breast-cancer-classifier/blob/main/SETUP.md)
- creates the `wdbc-env` virtual environment
- downloads and installs package dependencies (requires  [`uv`](https://github.com/astral-sh/uv)).

---
_Last Updated: 2026-03-01_
