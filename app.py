import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
kmeans_model = joblib.load("kmeans_model.pkl")
final_model = joblib.load("final_model.sav")

# Load clustered data for visualization
df = pd.read_csv("Clustered_Customer_Data.csv")

st.title("Customer Segmentation App")
st.markdown('<style>body{background-color: #f4f6f7;}</style>', unsafe_allow_html=True)

# Input form
with st.form("customer_form"):
    balance = st.number_input("Balance", step=0.001, format="%.6f")
    balance_frequency = st.number_input("Balance Frequency", step=0.001, format="%.6f")
    purchases = st.number_input("Purchases", step=0.01, format="%.2f")
    oneoff_purchases = st.number_input("OneOff Purchases", step=0.01, format="%.2f")
    installments_purchases = st.number_input("Installments Purchases", step=0.01, format="%.2f")
    cash_advance = st.number_input("Cash Advance", step=0.01, format="%.6f")
    purchases_frequency = st.number_input("Purchases Frequency", step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input("OneOff Purchases Frequency", step=0.01, format="%.6f")
    purchases_installment_frequency = st.number_input("Purchases Installments Frequency", step=0.01, format="%.6f")
    cash_advance_frequency = st.number_input("Cash Advance Frequency", step=0.01, format="%.6f")
    cash_advance_trx = st.number_input("Cash Advance Trx", step=1)
    purchases_trx = st.number_input("Purchases TRX", step=1)
    credit_limit = st.number_input("Credit Limit", step=0.1, format="%.1f")
    payments = st.number_input("Payments", step=0.01, format="%.6f")
    minimum_payments = st.number_input("Minimum Payments", step=0.01, format="%.6f")
    prc_full_payment = st.number_input("PRC Full Payment", step=0.01, format="%.6f")
    tenure = st.number_input("Tenure", step=1)

    # Collect inputs in same order as training
    features = np.array([[balance, balance_frequency, purchases, oneoff_purchases,
                          installments_purchases, cash_advance, purchases_frequency,
                          oneoff_purchases_frequency, purchases_installment_frequency,
                          cash_advance_frequency, cash_advance_trx, purchases_trx,
                          credit_limit, payments, minimum_payments, prc_full_payment,
                          tenure]])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Predictions
    cluster_label = kmeans_model.predict(features)[0]
    final_label = final_model.predict(features)[0]

    st.success(f"Cluster (KMeans): {cluster_label}")
    st.success(f"Predicted Segment (Decision Tree): {final_label}")

    # Show histograms of the predicted cluster
    cluster_df = df[df['Cluster'] == cluster_label]
    st.subheader(f"Cluster {cluster_label} Feature Distributions")
    plt.rcParams["figure.figsize"] = (15, 3)

    for col in cluster_df.drop(['Cluster'], axis=1).columns:
        fig, ax = plt.subplots()
        sns.histplot(cluster_df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {col} (Cluster {cluster_label})")
        st.pyplot(fig)


