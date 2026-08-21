
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ============================================================
# FILE / COLUMN SETTINGS
# ============================================================

DATA_PATH = Path(__file__).parent / "loan_approval_data.csv"

TARGET_COL = "Loan_Approved"
ID_COL = "Applicant_ID"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


# ============================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_model():

    df = load_dataset()

    working_df = df.copy()

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    working_df["DTI_Ratio_sq"] = working_df["DTI_Ratio"] ** 2
    working_df["Credit_Score_sq"] = working_df["Credit_Score"] ** 2

    # Remove Applicant ID because it is not useful for prediction
    working_df = working_df.drop(columns=[ID_COL]).copy()

    # Remove rows where target is missing
    working_df = working_df.dropna(subset=[TARGET_COL]).copy()

    # --------------------------------------------------------
    # Encode Target
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    working_df[TARGET_COL] = label_encoder.fit_transform(
        working_df[TARGET_COL].astype(str)
    )

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X = working_df.drop(columns=[TARGET_COL])
    y = working_df[TARGET_COL]

    # --------------------------------------------------------
    # Feature Lists
    # --------------------------------------------------------

    categorical_features = [
        "Employment_Status",
        "Loan_Purpose",
        "Marital_Status",
        "Property_Area",
        "Gender",
        "Employer_Category",
        "Education_Level",
    ]

    numeric_features = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Dependents",
        "DTI_Ratio_sq",
        "Credit_Score_sq",
    ]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="mean"),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop="first",
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    # --------------------------------------------------------
    # Logistic Regression Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # Model Evaluation
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, y_pred),
            3,
        ),
        "precision": round(
            precision_score(y_test, y_pred),
            3,
        ),
        "recall": round(
            recall_score(y_test, y_pred),
            3,
        ),
        "f1": round(
            f1_score(y_test, y_pred),
            3,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ),
    }

    return pipeline, metrics, working_df


# ============================================================
# BUILD STREAMLIT INPUT FORM
# ============================================================

def build_input_frame():

    df = load_dataset()

    # --------------------------------------------------------
    # Get categorical options directly from dataset
    # --------------------------------------------------------

    categorical_options = {
        "Employment_Status": sorted(
            df["Employment_Status"]
            .dropna()
            .unique()
            .tolist()
        ),
        "Loan_Purpose": sorted(
            df["Loan_Purpose"]
            .dropna()
            .unique()
            .tolist()
        ),
        "Marital_Status": sorted(
            df["Marital_Status"]
            .dropna()
            .unique()
            .tolist()
        ),
        "Property_Area": sorted(
            df["Property_Area"]
            .dropna()
            .unique()
            .tolist()
        ),
        "Gender": sorted(
            df["Gender"]
            .dropna()
            .unique()
            .tolist()
        ),
        "Employer_Category": sorted(
            df["Employer_Category"]
            .dropna()
            .unique()
            .tolist()
        ),
        "Education_Level": sorted(
            df["Education_Level"]
            .dropna()
            .unique()
            .tolist()
        ),
    }

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    st.sidebar.header("Applicant Details")

    values = {}

    # --------------------------------------------------------
    # Financial Information
    # IMPORTANT:
    # These ranges are kept close to the dataset distribution.
    # --------------------------------------------------------

    values["Applicant_Income"] = st.sidebar.number_input(
        "Applicant income",
        min_value=0.0,
        max_value=20000.0,
        value=10000.0,
        step=500.0,
    )

    values["Coapplicant_Income"] = st.sidebar.number_input(
        "Co-applicant income",
        min_value=0.0,
        max_value=10000.0,
        value=5000.0,
        step=500.0,
    )

    # --------------------------------------------------------
    # Employment
    # --------------------------------------------------------

    values["Employment_Status"] = st.sidebar.selectbox(
        "Employment status",
        categorical_options["Employment_Status"],
    )

    # --------------------------------------------------------
    # Personal Information
    # --------------------------------------------------------

    values["Age"] = st.sidebar.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=32,
        step=1,
    )

    values["Marital_Status"] = st.sidebar.selectbox(
        "Marital status",
        categorical_options["Marital_Status"],
    )

    values["Dependents"] = st.sidebar.number_input(
        "Dependents",
        min_value=0,
        max_value=6,
        value=1,
        step=1,
    )

    # --------------------------------------------------------
    # Credit Information
    # --------------------------------------------------------

    values["Credit_Score"] = st.sidebar.number_input(
        "Credit score",
        min_value=550,
        max_value=799,
        value=700,
        step=1,
    )

    values["Existing_Loans"] = st.sidebar.number_input(
        "Existing loans",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

    # --------------------------------------------------------
    # DTI
    #
    # IMPORTANT:
    # Dataset uses values around 0.10 - 0.60.
    # Do NOT use 18, 20, 30 etc.
    # --------------------------------------------------------

    values["DTI_Ratio"] = st.sidebar.number_input(
        "DTI ratio",
        min_value=0.10,
        max_value=0.60,
        value=0.35,
        step=0.01,
    )

    # --------------------------------------------------------
    # Savings / Collateral / Loan
    # --------------------------------------------------------

    values["Savings"] = st.sidebar.number_input(
        "Savings",
        min_value=0.0,
        max_value=20000.0,
        value=10000.0,
        step=500.0,
    )

    values["Collateral_Value"] = st.sidebar.number_input(
        "Collateral value",
        min_value=0.0,
        max_value=50000.0,
        value=25000.0,
        step=1000.0,
    )

    values["Loan_Amount"] = st.sidebar.number_input(
        "Loan amount",
        min_value=1000.0,
        max_value=40000.0,
        value=20000.0,
        step=1000.0,
    )

    values["Loan_Term"] = st.sidebar.number_input(
        "Loan term (months)",
        min_value=6,
        max_value=360,
        value=24,
        step=1,
    )

    # --------------------------------------------------------
    # Categorical Loan Information
    # --------------------------------------------------------

    values["Loan_Purpose"] = st.sidebar.selectbox(
        "Loan purpose",
        categorical_options["Loan_Purpose"],
    )

    values["Property_Area"] = st.sidebar.selectbox(
        "Property area",
        categorical_options["Property_Area"],
    )

    values["Education_Level"] = st.sidebar.selectbox(
        "Education level",
        categorical_options["Education_Level"],
    )

    values["Gender"] = st.sidebar.selectbox(
        "Gender",
        categorical_options["Gender"],
    )

    values["Employer_Category"] = st.sidebar.selectbox(
        "Employer category",
        categorical_options["Employer_Category"],
    )

    # --------------------------------------------------------
    # Feature Engineering
    # Must match training features exactly
    # --------------------------------------------------------

    values["DTI_Ratio_sq"] = (
        values["DTI_Ratio"] ** 2
    )

    values["Credit_Score_sq"] = (
        values["Credit_Score"] ** 2
    )

    # Convert dictionary to DataFrame
    return pd.DataFrame([values])


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():

    # --------------------------------------------------------
    # Page Configuration
    # --------------------------------------------------------

    st.set_page_config(
        page_title="CreditWise Loan Predictor",
        page_icon="💳",
        layout="wide",
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.title("💳 CreditWise Loan Approval Predictor")

    st.write(
        "This Streamlit app uses a trained Logistic Regression "
        "model to estimate whether a loan application should be "
        "approved or rejected."
    )

    # --------------------------------------------------------
    # Train Model
    # --------------------------------------------------------

    pipeline, metrics, _ = train_model()

    # --------------------------------------------------------
    # Input Form
    # --------------------------------------------------------

    st.sidebar.markdown("### Prediction Input")

    input_frame = build_input_frame()

    # --------------------------------------------------------
    # Prediction Button
    # --------------------------------------------------------

    if st.sidebar.button(
        "Predict Loan Outcome",
        use_container_width=True,
    ):

        # Prediction
        prediction = pipeline.predict(
            input_frame
        )[0]

        # Probability
        probabilities = pipeline.predict_proba(
            input_frame
        )[0]

        # Probability of class 1
        approval_probability = probabilities[1]

        # Label
        label = (
            "Approved"
            if prediction == 1
            else "Rejected"
        )

        # ----------------------------------------------------
        # Prediction Result
        # ----------------------------------------------------

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Outcome",
                label,
            )

        with col2:
            st.metric(
                "Approval Probability",
                f"{approval_probability * 100:.1f}%",
            )

        # ----------------------------------------------------
        # Status Message
        # ----------------------------------------------------

        if label == "Approved":

            st.success(
                "The model predicts that this applicant "
                "is likely to receive loan approval."
            )

        else:

            st.warning(
                "The model predicts that this applicant "
                "is likely to be rejected."
            )

        # ----------------------------------------------------
        # Show Probability Explanation
        # ----------------------------------------------------

        st.markdown("### Prediction Confidence")

        probability_percentage = (
            approval_probability * 100
        )

        if probability_percentage >= 70:

            st.info(
                f"High approval likelihood: "
                f"{probability_percentage:.1f}%"
            )

        elif probability_percentage >= 50:

            st.info(
                f"Moderate approval likelihood: "
                f"{probability_percentage:.1f}%"
            )

        else:

            st.info(
                f"Low approval likelihood: "
                f"{probability_percentage:.1f}%"
            )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown("---")

    st.markdown("### 📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}",
    )

    col2.metric(
        "Precision",
        f"{metrics['precision']:.3f}",
    )

    col3.metric(
        "Recall",
        f"{metrics['recall']:.3f}",
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1']:.3f}",
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown("### Confusion Matrix")

    cm = metrics["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Rejected",
            "Actual Approved",
        ],
        columns=[
            "Predicted Rejected",
            "Predicted Approved",
        ],
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.markdown("### Dataset Preview")

    st.dataframe(
        load_dataset().head(10),
        use_container_width=True,
    )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()

