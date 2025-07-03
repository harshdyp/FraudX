import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and feature importances
model = joblib.load('fraud_detection_model.joblib')
scaler = joblib.load('fraud_detection_scaler.joblib')
feature_importances = joblib.load('feature_importances.joblib')

# Load the new transactions
new_transactions = pd.read_csv('new_transactions.csv')

def preprocess_data(df):
    # Convert categorical variables to numerical
    categorical_columns = ['Sender_Country', 'Receiver_Country', 'Payment_Method', 'Transaction_Currency']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    # Ensure all columns used during training are present
    for col in scaler.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only the columns used during training
    df_encoded = df_encoded[scaler.feature_names_in_]
    
    return df_encoded

def detect_suspicious_transactions(df, model, scaler, threshold=0.5):
    X = preprocess_data(df)
    X_scaled = scaler.transform(X)
    
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    df['Suspicion_Score'] = y_pred_proba
    df['Is_Suspicious'] = y_pred_proba > threshold
    
    return df

def explain_suspicion(row, feature_importances, threshold=0.05):
    reasons = []
    for feature, importance in feature_importances.items():
        if importance > threshold:
            if feature in ['Transaction_Amount', 'Transaction_Velocity']:
                if row[feature] > np.mean(new_transactions[feature]) + np.std(new_transactions[feature]):
                    reasons.append(f"High {feature.replace('_', ' ')}")
            elif feature in ['Unusual_Time', 'Multiple_Currency_Conversions'] and row[feature]:
                reasons.append(feature.replace('_', ' '))
            elif feature in ['Sender_Country', 'Receiver_Country'] and row['Sender_Country'] != row['Receiver_Country']:
                reasons.append("Cross-border transaction")
            elif feature == 'Is_Known_Fraudster' and row[feature]:
                reasons.append("Known fraudster involved")
    return ', '.join(reasons) if reasons else "Unknown"

# Detect suspicious transactions
suspicious_df = detect_suspicious_transactions(new_transactions, model, scaler)

# Add suspicion reasons
suspicious_df['Suspicion_Reasons'] = suspicious_df.apply(lambda row: explain_suspicion(row, feature_importances), axis=1)

# Generate summary report
suspicious_summary = suspicious_df[suspicious_df['Is_Suspicious']].groupby('Suspicion_Reasons').size().reset_index(name='Count')
suspicious_summary = suspicious_summary.sort_values('Count', ascending=False)

# Print statistics
total_transactions = len(suspicious_df)
suspicious_transactions = suspicious_df['Is_Suspicious'].sum()

print(f"Total Transactions: {total_transactions}")
print(f"Suspicious Transactions: {suspicious_transactions} ({suspicious_transactions/total_transactions:.2%})")
print("\nTop 10 Suspicion Reasons:")
print(suspicious_summary.head(10))

# Save only the suspicious transactions
suspicious_transactions = suspicious_df[suspicious_df['Is_Suspicious']]
suspicious_transactions.to_csv('new_suspicious_transactions.csv', index=False)
suspicious_summary.to_csv('new_suspicious_transactions_summary.csv', index=False)

print("Analysis complete. Results have been saved to CSV files.")
print(f"Saved {len(suspicious_transactions)} suspicious transactions to new_suspicious_transactions.csv")

# Update the model with new data
X_new = preprocess_data(new_transactions.drop(['Transaction_ID', 'Date', 'Time', 'Sender_ID', 'Receiver_ID', 'Is_Fraudulent'], axis=1))
y_new = new_transactions['Is_Fraudulent']

X_new_scaled = scaler.transform(X_new)
model.fit(X_new_scaled, y_new)

# Save the updated model
joblib.dump(model, 'fraud_detection_model.joblib')
print("Model has been updated with new data and saved.")