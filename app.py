import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from suspicious_by_model import detect_suspicious_transactions, explain_suspicion
import os\

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model, scaler, and feature importances
model = joblib.load(os.path.join(current_dir, 'fraud_detection_model.joblib'))
scaler = joblib.load(os.path.join(current_dir, 'fraud_detection_scaler.joblib'))
feature_importances = joblib.load(os.path.join(current_dir, 'feature_importances.joblib'))
def main():
    st.title("Advanced Fraud Detection System")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #FF4B4B;
    }
    .medium-font {
        font-size:16px !important;
        color: #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        if st.button("Detect Suspicious Transactions"):
            suspicious_df = detect_suspicious_transactions(data, model, scaler)
            suspicious_df['Suspicion_Reasons'] = suspicious_df.apply(lambda row: explain_suspicion(row, feature_importances, data), axis=1)

            # Overall Statistics
            total_transactions = len(suspicious_df)
            suspicious_transactions = suspicious_df['Is_Suspicious'].sum()
            suspicious_percentage = (suspicious_transactions / total_transactions) * 100

            st.subheader("Overall Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<p class='big-font'>Total Transactions: {total_transactions}</p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<p class='big-font'>Suspicious Transactions: {suspicious_transactions} ({suspicious_percentage:.2f}%)</p>", unsafe_allow_html=True)

            # Visualizations
            st.subheader("Data Visualizations")

            # Pie chart of suspicious transactions
            fig_pie = px.pie(
                values=[suspicious_transactions, total_transactions - suspicious_transactions],
                names=['Suspicious', 'Not Suspicious'],
                title='Suspicious vs Non-Suspicious Transactions',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie)

            # Bar chart of suspicion reasons
            reason_counts = suspicious_df[suspicious_df['Is_Suspicious']]['Suspicion_Reasons'].str.split(', ', expand=True).stack().value_counts()
            fig_bar = px.bar(
                x=reason_counts.index,
                y=reason_counts.values,
                title='Top Suspicion Reasons',
                labels={'x': 'Reason', 'y': 'Count'},
                color=reason_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_bar)

            # Geographical visualization
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
            st.plotly_chart(fig_geo)

            # Transaction Details
            st.subheader("Transaction Details")
            selected_transaction = st.selectbox("Select a transaction to view details:", 
                                                suspicious_df[suspicious_df['Is_Suspicious']]['Transaction_ID'])
            
            if selected_transaction:
                transaction_details = suspicious_df[suspicious_df['Transaction_ID'] == selected_transaction].iloc[0]
                st.write(transaction_details)

                # Show historical patterns for the sender
                sender_history = data[data['Sender_ID'] == transaction_details['Sender_ID']]
                st.write(f"Sender's Transaction History (Last 5 transactions):")
                st.write(sender_history.sort_values('Date', ascending=False).head())

                # Compare with past transactions
                st.subheader("Comparison with Past Transactions")
                col1, col2 = st.columns(2)
                with col1:
                    avg_amount = sender_history['Transaction_Amount'].mean()
                    st.markdown(f"<p class='medium-font'>Average Transaction Amount: ${avg_amount:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='big-font'>Current Transaction Amount: ${transaction_details['Transaction_Amount']:.2f}</p>", unsafe_allow_html=True)
                    amount_diff = (transaction_details['Transaction_Amount'] - avg_amount) / avg_amount * 100
                    st.markdown(f"<p class='medium-font'>Difference: {amount_diff:.2f}%</p>", unsafe_allow_html=True)
                
                with col2:
                    avg_velocity = sender_history['Transaction_Velocity'].mean()
                    st.markdown(f"<p class='medium-font'>Average Transaction Velocity: {avg_velocity:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='big-font'>Current Transaction Velocity: {transaction_details['Transaction_Velocity']:.2f}</p>", unsafe_allow_html=True)
                    velocity_diff = (transaction_details['Transaction_Velocity'] - avg_velocity) / avg_velocity * 100
                    st.markdown(f"<p class='medium-font'>Difference: {velocity_diff:.2f}%</p>", unsafe_allow_html=True)

            # Save results
            suspicious_transactions = suspicious_df[suspicious_df['Is_Suspicious']]
            suspicious_transactions.to_csv('suspicious_transactions.csv', index=False)
            st.success(f"Saved {len(suspicious_transactions)} suspicious transactions to suspicious_transactions.csv")

if __name__ == "__main__":
    main()