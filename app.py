import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ---------- Page Config ----------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------- App Title ----------
st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("""
    <style>
        .big-font { font-size: 18px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar (Navigation) ----------
st.sidebar.title("📌 Navigation")
option = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Explorer", "🔮 Predict", "📈 Model Performance"]
)

# ---------- Load Model & Scaler ----------
@st.cache_resource
def load_artifacts():
    model = load_model('loan_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------- Load Dataset (for exploration) ----------
@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset.csv')
    # Clean column names (trim spaces)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------- HOME PAGE ----------
if option == "🏠 Home":
    st.header("📋 Welcome to the Loan Approval System")
    st.write("""
        This application uses an Artificial Neural Network (ANN) to predict whether a loan application will be **Approved** or **Rejected**.
        The model is trained on historical loan data with an accuracy of **92.74%**.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Records", df.shape[0])
    col2.metric("✅ Approved", df[df['loan_status'].str.strip() == 'Approved'].shape[0])
    col3.metric("❌ Rejected", df[df['loan_status'].str.strip() == 'Rejected'].shape[0])
    
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(10))
    
    st.subheader("📈 Quick Statistics")
    st.write(df.describe())

# ---------- DATA EXPLORER PAGE ----------
elif option == "📊 Data Explorer":
    st.header("🔍 Explore Loan Data")
    
    # Filters
    st.sidebar.subheader("Filter Options")
    education = st.sidebar.selectbox("Education", ["All"] + df['education'].unique().tolist())
    self_employed = st.sidebar.selectbox("Self Employed", ["All"] + df['self_employed'].unique().tolist())
    
    filtered_df = df.copy()
    if education != "All":
        filtered_df = filtered_df[filtered_df['education'] == education]
    if self_employed != "All":
        filtered_df = filtered_df[filtered_df['self_employed'] == self_employed]
    
    st.subheader(f"📋 Data (Showing {len(filtered_df)} rows)")
    st.dataframe(filtered_df)
    
    # Visualizations
    st.subheader("📊 Visualizations")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Education vs Loan Status
    edu_status = pd.crosstab(df['education'], df['loan_status'])
    edu_status.plot(kind='bar', ax=axes[0], stacked=True, color=['#ff9999','#66b3ff'])
    axes[0].set_title("Education vs Loan Status")
    axes[0].set_xlabel("Education")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Loan Status")
    
    # Self Employed vs Loan Status
    se_status = pd.crosstab(df['self_employed'], df['loan_status'])
    se_status.plot(kind='bar', ax=axes[1], stacked=True, color=['#ff9999','#66b3ff'])
    axes[1].set_title("Self Employed vs Loan Status")
    axes[1].set_xlabel("Self Employed")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Loan Status")
    
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("🧮 Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# ---------- PREDICTION PAGE ----------
elif option == "🔮 Predict":
    st.header("🔮 Predict Loan Approval")
    st.write("Enter customer details below and click **Predict**")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            income_annum = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=20000000, step=100000)
        with col2:
            loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=15)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
            residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=5000000, step=100000)
            commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=2000000, step=100000)
            luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=3000000, step=100000)
            bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=4000000, step=100000)
        
        submitted = st.form_submit_button("🔮 Predict")
    
    if submitted:
        # 1. Create input DataFrame
        input_data = pd.DataFrame({
            'no_of_dependents': [no_of_dependents],
            'education': [education],
            'self_employed': [self_employed],
            'income_annum': [income_annum],
            'loan_amount': [loan_amount],
            'loan_term': [loan_term],
            'cibil_score': [cibil_score],
            'residential_assets_value': [residential_assets_value],
            'commercial_assets_value': [commercial_assets_value],
            'luxury_assets_value': [luxury_assets_value],
            'bank_asset_value': [bank_asset_value]
        })
        
        # 2. One-hot encode (education, self_employed)
        input_data = pd.get_dummies(input_data, columns=['education', 'self_employed'], drop_first=True)
        
        # 3. Ensure all columns match training (fill missing with 0)
        # Training columns (after encoding): we know them from earlier
        expected_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                         'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                         'luxury_assets_value', 'bank_asset_value', 
                         'education_Not Graduate', 'self_employed_Yes']
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_cols]
        
        # 4. Scale numerical columns
        num_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                    'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                    'luxury_assets_value', 'bank_asset_value']
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # 5. Predict
        prob = model.predict(input_data)[0][0]
        prediction = "Approved ✅" if prob > 0.5 else "Rejected ❌"
        confidence = prob if prob > 0.5 else 1 - prob
        
        # 6. Display result
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Progress bar
        confidence = float(prob if prob > 0.5 else 1 - prob)
        st.progress(confidence)
        
        # Gauge like indicator
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(['Probability'], [prob], color='#4CAF50' if prob>0.5 else '#f44336')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Approval Probability')
        ax.axvline(0.5, color='black', linestyle='--')
        st.pyplot(fig)

# ---------- MODEL PERFORMANCE PAGE ----------
elif option == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    st.write("""
        - **Accuracy:** 92.74%
        - **Loss:** 0.1658
        - **Precision:** 0.96 (for Approved class)
        - **Recall:** 0.92 (for Approved class)
        - **F1-Score:** 0.94 (for Approved class)
    """)
    
    # Placeholder for confusion matrix (you can load from saved file if any)
    st.subheader("📊 Confusion Matrix")
    cm = np.array([[296, 22], [41, 495]])  # approximate values based on your results
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader("📋 Classification Report")
    report = """
    | Class | Precision | Recall | F1-Score |
    |-------|-----------|--------|----------|
    | Rejected (0) | 0.88 | 0.93 | 0.91 |
    | Approved (1) | 0.96 | 0.92 | 0.94 |
    | **Accuracy** | | **0.93** | |
    """
    st.markdown(report)