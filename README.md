# 💼 Job Salary Prediction

An interactive web application built with **Streamlit** and **Scikit-Learn** to estimate job salaries based on professional experience, education, skills, industry, company size, and work mode.

---

## 🚀 Live Demo

Access the live web application on Streamlit Community Cloud:  
👉 **[salary-prediction-job.streamlit.app](https://salary-prediction-job.streamlit.app/)**

---

## 📌 Project Overview

Predicting candidate salary expectations accurately is critical for talent acquisition and compensation planning. This project utilizes an end-to-end machine learning regression pipeline trained on 250,000+ job records to predict salary outcomes based on user inputs.

### Key Features
- **Interactive UI**: Clean, dual-column Streamlit interface with responsive sliders and dropdown menus.
- **Machine Learning Pipeline**: Complete preprocessing pipeline using `ColumnTransformer` with `SimpleImputer` and `OneHotEncoder` coupled with `LinearRegression`.
- **Dynamic Input Validation**: Dynamically loads unique categorical choices and ranges directly from the dataset.
- **Instant Inference**: Real-time salary estimation rendered as a formatted metric.

---

## 📊 Features & Dataset Schema

The model predicts `salary` using the following features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `job_title` | Categorical | Title of the position (e.g. AI Engineer, Data Analyst, Frontend Developer) |
| `experience_years` | Numerical | Years of professional work experience (0 – 20) |
| `education_level` | Categorical | Highest completed education level (Bachelor, Master, PhD, etc.) |
| `skills_count` | Numerical | Total number of relevant technical and soft skills |
| `certifications` | Numerical | Number of professional certifications held |
| `industry` | Categorical | Industry sector (e.g. Healthcare, Telecom, Media, Retail) |
| `company_size` | Categorical | Scale of the organization (Small, Medium, Large) |
| `location` | Categorical | Country or geographic region of the role |
| `remote_work` | Categorical | Remote policy (No, Hybrid, Yes) |

---

## 🛠️ Project Structure

```text
Salary-prediction/
│
├── app.py                             # Streamlit web application & inference pipeline
├── job_salary_prediction_dataset.csv  # Dataset used for training and feature values
├── job_salary_prediction.ipynb        # Exploratory Data Analysis (EDA) and modeling notebook
├── requirements.txt                   # Project Python dependencies
└── README.md                          # Project documentation
```

---

## ⚙️ Installation & Local Setup

### Prerequisites
- Python 3.9+ (Python 3.10 to 3.12 recommended)
- `pip` or virtual environment manager (`venv`)

### 1. Clone the Repository
```bash
git clone https://github.com/vaibhavhivrale/Salary-prediction.git
cd Salary-prediction
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

---

## 🧠 Model & Pipeline Architecture

The application constructs an end-to-end `scikit-learn` pipeline:

1. **Numerical Pipeline**:
   - `SimpleImputer(strategy="median")`
2. **Categorical Pipeline**:
   - `SimpleImputer(strategy="most_frequent")`
   - `OneHotEncoder(handle_unknown="ignore")`
3. **Model**:
   - `LinearRegression()`
4. **Caching**:
   - Decorated with `@st.cache_resource` for fast in-memory query processing after initial model fit.

---

## 📦 Dependencies

- `streamlit`
- `pandas`
- `scikit-learn`

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
