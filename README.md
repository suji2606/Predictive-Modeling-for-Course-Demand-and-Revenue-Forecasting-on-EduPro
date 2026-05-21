# EduPro Predictive Analytics Dashboard

A full machine learning pipeline + interactive Streamlit dashboard for predicting **course enrollment demand** and **revenue forecasting** on the EduPro online learning platform.

---

## 📁 Project Structure

```
edupro/
├── app.py                        # Streamlit dashboard (main entry point)
├── model_train.py                # ML model training & evaluation
├── data_prep.py                  # Data merging & feature engineering
├── EduPro_Online_Platform.xlsx   # Dataset (3 sheets: Courses, Teachers, Transactions)
├── requirements.txt              # Python dependencies
├── models/                       # Auto-generated after first run
│   ├── EnrollmentCount_model.pkl
│   ├── TotalRevenue_model.pkl
│   └── metadata.pkl
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / Download the project folder

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Pre-train models

Models are auto-trained on first app launch, but you can run training manually:

```bash
python model_train.py
```

This will print model evaluation metrics for all 5 algorithms.

---

## 🚀 Run the Dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPIs, enrollment by category, revenue distribution |
| **Demand Predictor** | Predict enrollments for a new course |
| **Revenue Forecast** | Forecast revenue at course & category level |
| **Feature Insights** | Feature importance + business-level takeaways |
| **Model Performance** | MAE, RMSE, R² leaderboard across all models |

---

## 🤖 Models Used

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Regularized Linear |
| Lasso Regression | Sparse Linear |
| Random Forest | Ensemble (Trees) |
| Gradient Boosting | Ensemble (Boosting) |

---

## 📐 Feature Engineering

- **Price bands**: Free / Low / Medium / High  
- **Duration buckets**: Short / Medium / Long / Extended  
- **Rating tiers**: Low / Medium / High / Top  
- **Experience buckets**: Junior / Mid / Senior / Expert  
- **Expertise-category match**: Binary indicator  
- **Historical aggregates**: Past enrollments, avg revenue, revenue per enrollment  

---

## 📈 Key Results (on test set)

**Enrollment Count Prediction:**  
- Best Model: Linear Regression | R² = 0.238 | MAE = 9.4 enrollments

**Revenue Prediction:**  
- Best Model: Ridge Regression | R² = 0.990 | MAE = $1,522

> Revenue prediction is highly accurate. Enrollment count variance is harder to predict purely from course metadata — time-series signals could improve R² further.

---

## 📦 Dataset Summary

| Sheet | Rows | Description |
|---|---|---|
| Courses | 60 | Course metadata (category, level, price, rating) |
| Teachers | 60 | Instructor profiles (expertise, experience, rating) |
| Transactions | 10,000 | Enrollment records with amounts |
| Users | 3,000 | Learner profiles |
