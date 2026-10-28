
import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="wide"
)


@st.cache_resource
def load_model():
    df = pd.read_csv("job_salary_prediction_dataset.csv")

    target = "salary"

    numeric_features = [
        "experience_years",
        "skills_count",
        "certifications"
    ]

    categorical_features = [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work"
    ]

    feature_columns = numeric_features + categorical_features

    df = df.dropna(subset=[target])

    X = df[feature_columns]
    y = df[target]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)

    return model, df, feature_columns


model, df, feature_columns = load_model()

st.title("💼 Job Salary Prediction")
st.write(
    "Estimate a salary based on job role, experience, "
    "education, skills, and other job-related details."
)

st.divider()

with st.form("salary_prediction_form"):
    st.subheader("Enter Job Details")

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox(
            "Job Title",
            sorted(df["job_title"].dropna().unique())
        )

        experience_years = st.slider(
            "Years of Experience",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )

        education_level = st.selectbox(
            "Education Level",
            sorted(df["education_level"].dropna().unique())
        )

        skills_count = st.slider(
            "Number of Skills",
            min_value=int(df["skills_count"].min()),
            max_value=int(df["skills_count"].max()),
            value=int(df["skills_count"].median())
        )

        certifications = st.slider(
            "Number of Certifications",
            min_value=int(df["certifications"].min()),
            max_value=int(df["certifications"].max()),
            value=int(df["certifications"].median())
        )

    with col2:
        industry = st.selectbox(
            "Industry",
            sorted(df["industry"].dropna().unique())
        )

        company_size = st.selectbox(
            "Company Size",
            sorted(df["company_size"].dropna().unique())
        )

        location = st.selectbox(
            "Location",
            sorted(df["location"].dropna().unique())
        )

        remote_work = st.selectbox(
            "Remote Work",
            sorted(df["remote_work"].dropna().unique())
        )

    submitted = st.form_submit_button(
        "Predict Salary",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "job_title": job_title,
        "experience_years": experience_years,
        "education_level": education_level,
        "skills_count": skills_count,
        "industry": industry,
        "company_size": company_size,
        "location": location,
        "remote_work": remote_work,
        "certifications": certifications
    }], columns=feature_columns)

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Predicted Salary")
    st.metric(
        label="Estimated salary in dataset units",
        value=f"{prediction:,.2f}"
    )

    st.caption(
        "This is an estimate from a machine learning model, "
        "not a guaranteed salary offer."
    )

with st.expander("About this project"):
    st.write(
        "This app uses a Linear Regression model with "
        "one-hot encoding for categorical features. "
        "The model is trained from the repository's CSV dataset."
    )
