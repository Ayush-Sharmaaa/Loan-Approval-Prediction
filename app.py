import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit App Title
st.title("🏦 Loan Approval Prediction System")
st.markdown("Predict whether a loan will be approved based on applicant details.")

# Load Dataset
df = pd.read_csv("loan_data.csv")

# Drop Unnecessary Column and Missing Data
df.drop("Loan_ID", axis=1, inplace=True)
df.dropna(inplace=True)  # drop rows with any missing values

# Encode Categorical Variables
label_cols = ['Gender', 'Married', 'Education', 'Loan_Status']
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

# EDA
st.subheader("📊 Data Exploration")
fig1, ax1 = plt.subplots()
ax1.hist(df['ApplicantIncome'], bins=30, color='skyblue', edgecolor='black')
ax1.set_title("Applicant Income Distribution")
ax1.set_xlabel("Income")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.hist(df['LoanAmount'], bins=30, color='orange', edgecolor='black')
ax2.set_title("Loan Amount Distribution")
ax2.set_xlabel("Loan Amount")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

approved = df[df['Loan_Status'] == 1]['Credit_History']
not_approved = df[df['Loan_Status'] == 0]['Credit_History']
fig3, ax3 = plt.subplots()
ax3.hist([approved, not_approved], label=['Approved', 'Not Approved'], bins=3, edgecolor='black')
ax3.legend()
ax3.set_title("Loan Status by Credit History")
ax3.set_xlabel("Credit History (1 = Good, 0 = Bad)")
ax3.set_ylabel("Number of Applicants")
st.pyplot(fig3)

# Train-Test Split
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Numerical Columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

st.subheader("📈 Model Performance")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"**{name}** - Accuracy: {acc:.4f}")
    st.text(classification_report(y_test, preds))

# Prediction Section
st.subheader("🔮 Make a Prediction")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])

# Convert inputs to match model format
new_app = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],
    'Married': [1 if married == "Yes" else 0],
    'Education': [0 if education == "Graduate" else 1],
    'ApplicantIncome': [income],
    'LoanAmount': [loan_amount],
    'Credit_History': [credit_history]
})

new_app_scaled = scaler.transform(new_app)

selected_model = st.selectbox("Choose a Model", list(models.keys()))

if st.button("Predict Loan Approval"):
    prediction = models[selected_model].predict(new_app_scaled)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Not Approved"
    st.subheader(f"Prediction: {result}")
