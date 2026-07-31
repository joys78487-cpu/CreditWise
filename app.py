import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "loan_approval_data.csv"
TARGET_COL = "Loan_Approved"
ID_COL = "Applicant_ID"


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_dataset()
    working_df = df.copy()

    working_df["DTI_Ratio_sq"] = working_df["DTI_Ratio"] ** 2
    working_df["Credit_Score_sq"] = working_df["Credit_Score"] ** 2
    working_df = working_df.drop(columns=[ID_COL]).copy()
    working_df = working_df.dropna(subset=[TARGET_COL]).copy()

    label_encoder = LabelEncoder()
    working_df[TARGET_COL] = label_encoder.fit_transform(working_df[TARGET_COL].astype(str))

    X = working_df.drop(columns=[TARGET_COL])
    y = working_df[TARGET_COL]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return pipeline, metrics, working_df


def build_input_frame():
    df = load_dataset()
    categorical_options = {
        "Employment_Status": sorted(df["Employment_Status"].dropna().unique().tolist()),
        "Loan_Purpose": sorted(df["Loan_Purpose"].dropna().unique().tolist()),
        "Marital_Status": sorted(df["Marital_Status"].dropna().unique().tolist()),
        "Property_Area": sorted(df["Property_Area"].dropna().unique().tolist()),
        "Gender": sorted(df["Gender"].dropna().unique().tolist()),
        "Employer_Category": sorted(df["Employer_Category"].dropna().unique().tolist()),
        "Education_Level": sorted(df["Education_Level"].dropna().unique().tolist()),
    }

    st.sidebar.header("Applicant Details")
    values = {}
    values["Applicant_Income"] = st.sidebar.number_input("Applicant income", min_value=0.0, value=50000.0, step=1000.0)
    values["Coapplicant_Income"] = st.sidebar.number_input("Co-applicant income", min_value=0.0, value=15000.0, step=1000.0)
    values["Employment_Status"] = st.sidebar.selectbox("Employment status", categorical_options["Employment_Status"])
    values["Age"] = st.sidebar.number_input("Age", min_value=18, max_value=80, value=32, step=1)
    values["Marital_Status"] = st.sidebar.selectbox("Marital status", categorical_options["Marital_Status"])
    values["Dependents"] = st.sidebar.number_input("Dependents", min_value=0, max_value=6, value=1, step=1)
    values["Credit_Score"] = st.sidebar.number_input("Credit score", min_value=300, max_value=850, value=700, step=1)
    values["Existing_Loans"] = st.sidebar.number_input("Existing loans", min_value=0, max_value=10, value=1, step=1)
    values["DTI_Ratio"] = st.sidebar.number_input("DTI ratio", min_value=0.0, max_value=100.0, value=18.0, step=0.1)
    values["Savings"] = st.sidebar.number_input("Savings", min_value=0.0, value=100000.0, step=1000.0)
    values["Collateral_Value"] = st.sidebar.number_input("Collateral value", min_value=0.0, value=500000.0, step=1000.0)
    values["Loan_Amount"] = st.sidebar.number_input("Loan amount", min_value=1000.0, value=200000.0, step=1000.0)
    values["Loan_Term"] = st.sidebar.number_input("Loan term (months)", min_value=6, max_value=360, value=24, step=1)
    values["Loan_Purpose"] = st.sidebar.selectbox("Loan purpose", categorical_options["Loan_Purpose"])
    values["Property_Area"] = st.sidebar.selectbox("Property area", categorical_options["Property_Area"])
    values["Education_Level"] = st.sidebar.selectbox("Education level", categorical_options["Education_Level"])
    values["Gender"] = st.sidebar.selectbox("Gender", categorical_options["Gender"])
    values["Employer_Category"] = st.sidebar.selectbox("Employer category", categorical_options["Employer_Category"])

    values["DTI_Ratio_sq"] = values["DTI_Ratio"] ** 2
    values["Credit_Score_sq"] = values["Credit_Score"] ** 2

    return pd.DataFrame([values])


def main():
    st.set_page_config(page_title="CreditWise Loan Predictor", page_icon="💳", layout="wide")
    st.title("CreditWise Loan Approval Predictor")
    st.write("This Streamlit app uses a trained logistic regression model to estimate whether a loan application should be approved or rejected.")

    pipeline, metrics, _ = train_model()

    st.sidebar.markdown("### Prediction Input")
    input_frame = build_input_frame()

    if st.sidebar.button("Predict Loan Outcome", use_container_width=True):
        prediction = pipeline.predict(input_frame)[0]
        probability = pipeline.predict_proba(input_frame)[0][1]
        label = "Approved" if prediction == 1 else "Rejected"

        st.subheader("Prediction Result")
        st.metric("Outcome", label)
        st.metric("Approval Probability", f"{probability * 100:.1f}%")

        if label == "Approved":
            st.success("The model predicts this applicant is likely to receive loan approval.")
        else:
            st.warning("The model predicts this applicant is likely to be rejected.")

    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.markdown("### Confusion Matrix")
    cm = metrics["confusion_matrix"]
    st.write(cm)

    st.markdown("### Dataset Preview")
    st.dataframe(load_dataset().head(10))


if __name__ == "__main__":
    main()
