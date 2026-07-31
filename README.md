# CreditWise Loan System

An intelligent loan approval prediction system built for **SecureTrust Bank**, a mid-sized financial company offering personal and home loans across urban and rural India. CreditWise uses machine learning to predict whether a loan application should be **Approved** or **Rejected**, before final human verification.

## Problem Statement

SecureTrust Bank currently relies on a manual loan verification process — loan officers manually check income proofs, employment details, and credit history for hundreds of applications every day. This process is slow, inconsistent, and prone to bias, leading to two costly outcomes:

1. **Good customers get rejected** → loss of business
2. **High-risk customers get approved** → financial losses

CreditWise addresses this by learning approval patterns from historical applicant data to deliver faster, more consistent, and unbiased loan decisions.

## Dataset

The dataset contains **1,000 applicant records** with **20 columns**, each row representing a loan applicant's personal, financial, and credit profile. Roughly 5% of values across all columns are missing and were imputed during preprocessing.

| Column | Description |
|---|---|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business / Car |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Undergraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / MNC / Self / Unemployed |
| **Loan_Approved** (target) | 1 = Approved, 0 = Rejected |

## Project Workflow

1. **Data Cleaning** — Missing numerical values imputed with mean; missing categorical values imputed with mode (`SimpleImputer`)
2. **Exploratory Data Analysis** — Distribution plots for approval status, gender, marital status, education, age, and income; boxplots to inspect outliers and compare feature spread across approval classes
3. **Feature Engineering**
   - Dropped `Applicant_ID` (non-predictive identifier)
   - Label-encoded `Education_Level` and the target `Loan_Approved`
   - One-hot encoded `Employment_Status`, `Loan_Purpose`, `Marital_Status`, `Property_Area`, `Gender`, `Employer_Category`
   - Added squared terms for `DTI_Ratio` and `Credit_Score` to capture non-linear effects
4. **Correlation Analysis** — Identified `Credit_Score` (+0.45) and `DTI_Ratio` (−0.44) as the two strongest predictors of loan approval, well ahead of `Applicant_Income` (+0.12)
5. **Train/Test Split** — 80/20 split with `StandardScaler` applied to all features
6. **Model Training & Evaluation** — Three classifiers trained and compared on precision, recall, F1-score, and accuracy

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **88.0%** | 0.78 | 0.84 | **0.81** |
| Naive Bayes | 86.0% | 0.81 | 0.70 | 0.75 |
| K-Nearest Neighbors | 78.5% | 0.67 | 0.57 | 0.62 |

**Logistic Regression** was the best-performing model, offering the strongest balance of precision and recall for identifying approvable loan applications.

### Key Insight
Credit Score and Debt-to-Income Ratio are far stronger predictors of loan approval than income alone — reinforcing that a borrower's repayment discipline matters more than raw earnings.

## Tech Stack

- **Language:** Python 3
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes)
- **Preprocessing:** SimpleImputer, LabelEncoder, OneHotEncoder, StandardScaler

## Project Structure

```
CreditWise-Loan-System/
├── credit_wise.ipynb              # Main notebook: EDA, preprocessing, modeling
├── loan_approval_data.csv         # Raw dataset (1,000 applicant records)
├── CreditWise_Loan_System.pdf     # Problem statement & dataset documentation
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone <repo-url>
cd CreditWise-Loan-System

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch the notebook
jupyter notebook credit_wise.ipynb
```

## Future Improvements

- Address class imbalance (e.g., SMOTE) to improve recall on rejected-loan cases
- Hyperparameter tuning via GridSearchCV/RandomizedSearchCV
- Test ensemble models (Random Forest, XGBoost) for potential accuracy gains
- Add SHAP/feature-importance explainability for regulatory transparency
- Deploy as an API/web app for real-time loan screening

## Author

Built by Joy as a portfolio project demonstrating an end-to-end ML pipeline for financial risk prediction.
