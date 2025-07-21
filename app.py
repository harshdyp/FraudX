import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from suspicious_by_model import detect_suspicious_transactions, explain_suspicion
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model, scaler, and feature importances
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        model = joblib.load(os.path.join(current_dir, 'fraud_detection_model.joblib'))
        scaler = joblib.load(os.path.join(current_dir, 'fraud_detection_scaler.joblib'))
        feature_importances = joblib.load(os.path.join(current_dir, 'feature_importances.joblib'))
        return model, scaler, feature_importances
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, scaler, feature_importances = load_artifacts()

# --- UI Enhancements ---
def set_theme():
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
    if theme == "Dark":
        st.markdown("""
        <style>
        body, .stApp { background-color: #18191A !important; color: #E4E6EB !important; }
        .big-font { color: #FF4B4B; }
        .kpi-card { background: #23272F; color: #E4E6EB; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .big-font { color: #FF4B4B; }
        .kpi-card { background: #f8f9fa; color: #31333F; }
        </style>
        """, unsafe_allow_html=True)

set_theme()

st.set_page_config(page_title="FraudX - Advanced Fraud Detection", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("FraudX Navigation")
st.sidebar.info("""
- Upload your transaction CSV
- Detect suspicious transactions
- Download results
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**About FraudX**\n\nA modern, explainable, and interactive fraud detection system.")

# --- Tabs for Multi-Page Navigation ---
tabs = st.tabs(["Upload & Analyze", "Feature Importance", "About"])

with tabs[0]:
    st.title("🚨 FraudX: Advanced Fraud Detection System")
    st.markdown("""
    <style>
    .big-font { font-size:22px !important; }
    .medium-font { font-size:16px !important; }
    .kpi-card { border-radius: 8px; padding: 16px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload a CSV file with transactions", type="csv")
    threshold = st.sidebar.slider("Suspicion Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Adjust the threshold for flagging transactions as suspicious.")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("### Data Preview:")
            st.dataframe(data.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if st.button("🔍 Detect Suspicious Transactions"):
            with st.spinner("Detecting suspicious transactions..."):
                suspicious_df = detect_suspicious_transactions(data, model, scaler, threshold=threshold)
                suspicious_df['Suspicion_Reasons'] = suspicious_df.apply(lambda row: explain_suspicion(row, feature_importances, data), axis=1)

            # --- KPI Dashboard ---
            total_transactions = len(suspicious_df)
            suspicious_transactions = suspicious_df['Is_Suspicious'].sum()
            suspicious_percentage = (suspicious_transactions / total_transactions) * 100
            st.markdown(f"<div class='kpi-card'><b>Total Transactions:</b> {total_transactions} &nbsp;&nbsp; <b>Suspicious:</b> {suspicious_transactions} ({suspicious_percentage:.2f}%)</div>", unsafe_allow_html=True)

            # --- Advanced Visualizations ---
            st.subheader("📊 Data Visualizations")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_pie = px.pie(
                    values=[suspicious_transactions, total_transactions - suspicious_transactions],
                    names=['Suspicious', 'Not Suspicious'],
                    title='Suspicious vs Non-Suspicious Transactions',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                reason_counts = suspicious_df[suspicious_df['Is_Suspicious']]['Suspicion_Reasons'].str.split(', ', expand=True).stack().value_counts()
                fig_bar = px.bar(
                    x=reason_counts.index,
                    y=reason_counts.values,
                    title='Top Suspicion Reasons',
                    labels={'x': 'Reason', 'y': 'Count'},
                    color=reason_counts.values,
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col3:
                # Histogram of Suspicion Scores
                fig_hist = px.histogram(suspicious_df, x='Suspicion_Score', nbins=30, title='Suspicion Score Distribution', color='Is_Suspicious', color_discrete_map={True: '#FF4B4B', False: '#36CFC9'})
                st.plotly_chart(fig_hist, use_container_width=True)

            # --- Geographical visualization ---
            st.subheader("🌍 Average Suspicion Score by Country")
            country_scores = suspicious_df.groupby('Sender_Country')['Suspicion_Score'].mean().reset_index()
            fig_geo = px.choropleth(
                country_scores,
                locations='Sender_Country',
                locationmode='country names',
                color='Suspicion_Score',
                hover_name='Sender_Country',
                projection='natural earth',
                title='Average Suspicion Score by Country',
                color_continuous_scale=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig_geo, use_container_width=True)

            # --- Transaction Details ---
            st.subheader("📝 Transaction Details")
            suspicious_only = suspicious_df[suspicious_df['Is_Suspicious']]
            if not suspicious_only.empty:
                selected_transaction = st.selectbox("Select a suspicious transaction to view details:", 
                                                    suspicious_only['Transaction_ID'])
                if selected_transaction:
                    transaction_details = suspicious_only[suspicious_only['Transaction_ID'] == selected_transaction].iloc[0]
                    st.write(transaction_details)

                    # Show historical patterns for the sender
                    sender_history = data[data['Sender_ID'] == transaction_details['Sender_ID']]
                    st.write(f"Sender's Transaction History (Last 5 transactions):")
                    st.dataframe(sender_history.sort_values('Date', ascending=False).head(), use_container_width=True)

                    # Compare with past transactions
                    st.subheader("Comparison with Past Transactions")
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_amount = sender_history['Transaction_Amount'].mean()
                        st.markdown(f"<p class='medium-font'>Average Transaction Amount: ${avg_amount:.2f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='big-font'>Current Transaction Amount: ${transaction_details['Transaction_Amount']:.2f}</p>", unsafe_allow_html=True)
                        amount_diff = (transaction_details['Transaction_Amount'] - avg_amount) / avg_amount * 100 if avg_amount else 0
                        st.markdown(f"<p class='medium-font'>Difference: {amount_diff:.2f}%</p>", unsafe_allow_html=True)
                    with col2:
                        avg_velocity = sender_history['Transaction_Velocity'].mean()
                        st.markdown(f"<p class='medium-font'>Average Transaction Velocity: {avg_velocity:.2f}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='big-font'>Current Transaction Velocity: {transaction_details['Transaction_Velocity']:.2f}</p>", unsafe_allow_html=True)
                        velocity_diff = (transaction_details['Transaction_Velocity'] - avg_velocity) / avg_velocity * 100 if avg_velocity else 0
                        st.markdown(f"<p class='medium-font'>Difference: {velocity_diff:.2f}%</p>", unsafe_allow_html=True)

            # --- Download Results ---
            suspicious_transactions = suspicious_df[suspicious_df['Is_Suspicious']]
            suspicious_transactions.to_csv('suspicious_transactions.csv', index=False)
            st.success(f"Saved {len(suspicious_transactions)} suspicious transactions to suspicious_transactions.csv")
            st.download_button(
                label="Download Suspicious Transactions CSV",
                data=suspicious_transactions.to_csv(index=False),
                file_name="suspicious_transactions.csv",
                mime="text/csv"
            )

with tabs[1]:
    st.header("🔎 Feature Importance Analysis")
    if feature_importances:
        fi_df = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Feature Importances", color="Importance", color_continuous_scale=px.colors.sequential.Blues)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.dataframe(fi_df, use_container_width=True)
    else:
        st.warning("Feature importances not available.")

with tabs[2]:
    st.header("About FraudX")
    st.markdown("""
    **FraudX** is a modern, explainable, and interactive fraud detection system built with Python, Streamlit, and machine learning.\
    - Upload your transaction data and get instant fraud analysis.\
    - Visualize suspicious patterns and feature importances.\
    - Download results and tune the model threshold.\
    - Built for transparency, usability, and real-world deployment.
    """)