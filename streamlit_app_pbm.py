
# PBM Contract Analysis Streamlit MVP App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

st.set_page_config(page_title="PBM Contract Analysis MVP", layout="wide")
st.title("💊 PBM Contract Analysis via AI")
st.markdown("Predictive modeling and cost optimization for payer strategy")

st.sidebar.header("Upload Your PBM Contract Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    df_encoded = pd.get_dummies(df, columns=['Drug_Tier', 'Channel', 'PA_Required', 'Step_Therapy'])
    X = df_encoded.drop(columns=['Contract_ID', 'Projected_Cost_PMPM'])
    y = df_encoded['Projected_Cost_PMPM']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 🔍 Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    importance = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    st.markdown("### 📊 Feature Importance")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax)
    ax.set_title("Feature Importance - Random Forest")
    st.pyplot(fig)

    st.sidebar.header("🧠 Cost Simulation Inputs")
    selected_rebate = st.sidebar.slider("Rebate %", min_value=10, max_value=40, value=25)
    selected_copay = st.sidebar.slider("Average Copay", min_value=0, max_value=100, value=20)

    if 'Rebate_%' in df.columns and 'Copay' in df.columns:
        sim_df = df.copy()
        sim_df['Rebate_%'] = selected_rebate
        sim_df['Copay'] = selected_copay

        sim_df_encoded = pd.get_dummies(sim_df, columns=['Drug_Tier', 'Channel', 'PA_Required', 'Step_Therapy'])
        sim_X = sim_df_encoded[X.columns]
        sim_prediction = model.predict(sim_X)

        st.markdown("### 💡 Simulated Projected PMPM Cost")
        st.write(f"Average Simulated Cost PMPM: ${np.mean(sim_prediction):.2f}")
        st.line_chart(sim_prediction)

        if st.button("📥 Download Payer Strategy PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="PBM Contract Analysis Strategy Report", ln=True, align="C")
            pdf.ln(10)
            pdf.multi_cell(0, 10, f"Mean Squared Error: {mse:.2f}")
            pdf.multi_cell(0, 10, f"R² Score: {r2:.2f}")
            pdf.multi_cell(0, 10, f"Simulated Rebate: {selected_rebate}%")
            pdf.multi_cell(0, 10, f"Simulated Copay: ${selected_copay}")
            pdf.multi_cell(0, 10, f"Average Simulated PMPM Cost: ${np.mean(sim_prediction):.2f}")
            pdf.output("PBM_Strategy_Report.pdf")
            with open("PBM_Strategy_Report.pdf", "rb") as file:
                st.download_button("Download Report", file, file_name="PBM_Strategy_Report.pdf")
else:
    st.info("👈 Upload a PBM dataset to get started.")
