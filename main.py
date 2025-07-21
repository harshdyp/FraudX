import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Initialize Faker for random data generation
fake = Faker()

# Constants
NUM_TRANSACTIONS = 10000
FRAUD_PERCENTAGE = 0.05
NUM_KNOWN_FRAUDSTERS = 100
NUM_SANCTIONED_ENTITIES = 50

# Generate known fraudsters and sanctioned entities
known_fraudsters = [str(uuid.uuid4()) for _ in range(NUM_KNOWN_FRAUDSTERS)]
sanctioned_entities = [str(uuid.uuid4()) for _ in range(NUM_SANCTIONED_ENTITIES)]

# Save known fraudsters and sanctioned entities to CSV
pd.DataFrame(known_fraudsters, columns=['Fraudster_ID']).to_csv('known_fraudsters.csv', index=False)
pd.DataFrame(sanctioned_entities, columns=['Sanctioned_Entity_ID']).to_csv('sanctioned_entities.csv', index=False)

# Define possible countries and payment methods
countries = ["USA", "India", "UK", "Canada", "Germany", "Australia", "France", "Netherlands", "Brazil", "Japan"]
payment_methods = ["bank transfer", "UPI", "PayPal", "Skrill", "Payoneer", "Cryptocurrency"]
currencies = {
    "USA": "USD", "India": "INR", "UK": "GBP", "Canada": "CAD", "Germany": "EUR",
    "Australia": "AUD", "France": "EUR", "Netherlands": "EUR", "Brazil": "BRL", "Japan": "JPY"
}

def generate_transaction_data():
    data = []
    for _ in range(NUM_TRANSACTIONS):
        sender_country = random.choice(countries)
        receiver_country = random.choice(countries)
        
        transaction_date = fake.date_time_between(start_date='-1y', end_date='now')
        transaction_amount = round(random.uniform(1, 100000), 2)
        
        is_fraudulent = random.random() < FRAUD_PERCENTAGE
        is_known_fraudster = random.random() < 0.02
        is_sanctioned_entity = random.random() < 0.01

        # New features
        device_types = ["Mobile", "Desktop", "Tablet"]
        browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
        transaction_channels = ["Online", "POS", "ATM"]
        merchant_categories = ["Grocery", "Electronics", "Clothing", "Travel", "Dining", "Fuel", "Healthcare", "Entertainment"]
        cities = ["New York", "London", "Mumbai", "Berlin", "Toronto", "Sydney", "Paris", "Amsterdam", "Sao Paulo", "Tokyo"]
        merchant_names = [fake.company() for _ in range(100)]
        device_type = random.choice(device_types)
        browser = random.choice(browsers)
        ip_address_change = random.random() < (0.3 if is_fraudulent else 0.05)
        account_age_days = random.randint(1, 3650) if not is_fraudulent else random.randint(1, 180)
        num_linked_cards = random.randint(1, 5) if not is_fraudulent else random.randint(1, 8)
        vpn_usage = random.random() < (0.2 if is_fraudulent else 0.03)
        transaction_channel = random.choice(transaction_channels)
        is_new_device = random.random() < (0.25 if is_fraudulent else 0.05)
        is_new_location = random.random() < (0.2 if is_fraudulent else 0.04)
        merchant_name = random.choice(merchant_names)
        merchant_category = random.choice(merchant_categories)
        transaction_city = random.choice(cities)
        latitude = round(random.uniform(-90, 90), 6)
        longitude = round(random.uniform(-180, 180), 6)

        if is_fraudulent or is_known_fraudster or is_sanctioned_entity:
            transaction_amount = round(random.uniform(1, 200000), 2)
            transaction_velocity = max(1, int(random.gauss(8, 3)))
            unusual_time = random.random() < 0.6
            multiple_currency_conversions = random.random() < 0.3
            repeated_failed_attempts = max(0, int(random.gauss(3, 2)))
        else:
            transaction_velocity = max(1, int(random.gauss(3, 1)))
            unusual_time = random.random() < 0.2
            multiple_currency_conversions = random.random() < 0.1
            repeated_failed_attempts = max(0, int(random.gauss(1, 1)))
        
        transaction = {
            "Transaction_ID": str(uuid.uuid4()),
            "Date": transaction_date.strftime('%Y-%m-%d'),
            "Time": transaction_date.strftime('%H:%M:%S'),
            "Sender_ID": random.choice(known_fraudsters) if is_known_fraudster else str(uuid.uuid4()),
            "Receiver_ID": random.choice(sanctioned_entities) if is_sanctioned_entity else str(uuid.uuid4()),
            "Sender_Country": sender_country,
            "Receiver_Country": receiver_country,
            "Payment_Method": random.choice(payment_methods),
            "Transaction_Amount": transaction_amount,
            "Transaction_Currency": currencies[sender_country],
            "Transaction_Velocity": transaction_velocity,
            "Unusual_Time": unusual_time,
            "Multiple_Currency_Conversions": multiple_currency_conversions,
            "Repeated_Failed_Attempts": repeated_failed_attempts,
            "Device_Type": device_type,
            "Browser": browser,
            "IP_Address_Change": ip_address_change,
            "Account_Age_Days": account_age_days,
            "Num_Linked_Cards": num_linked_cards,
            "VPN_Usage": vpn_usage,
            "Transaction_Channel": transaction_channel,
            "Is_New_Device": is_new_device,
            "Is_New_Location": is_new_location,
            "Merchant_Name": merchant_name,
            "Merchant_Category": merchant_category,
            "Transaction_City": transaction_city,
            "Latitude": latitude,
            "Longitude": longitude,
            "Is_Fraudulent": is_fraudulent,
            "Is_Known_Fraudster": is_known_fraudster,
            "Is_Sanctioned_Entity": is_sanctioned_entity
        }
        data.append(transaction)
    
    return pd.DataFrame(data)

# Generate the main transaction dataset
df = generate_transaction_data()
df.to_csv('transactions.csv', index=False)
print("Transaction dataset has been generated and saved.")

# Prepare data for model training
X = df.drop(['Transaction_ID', 'Date', 'Time', 'Sender_ID', 'Receiver_ID', 'Is_Fraudulent'], axis=1)
y = df['Is_Fraudulent']

# Convert categorical variables to numerical
X = pd.get_dummies(X, columns=[
    'Sender_Country', 'Receiver_Country', 'Payment_Method', 'Transaction_Currency',
    'Device_Type', 'Browser', 'Transaction_Channel', 'Merchant_Name', 'Merchant_Category', 'Transaction_City'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Print the classification report
print("Model Performance:")
print(classification_report(y_test, y_pred))

# Save the model, scaler, and feature importances
joblib.dump(rf_classifier, 'fraud_detection_model.joblib')
joblib.dump(scaler, 'fraud_detection_scaler.joblib')
feature_importances = dict(zip(X.columns, rf_classifier.feature_importances_))
joblib.dump(feature_importances, 'feature_importances.joblib')
print("Model, scaler, and feature importances have been saved.")