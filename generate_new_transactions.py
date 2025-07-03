import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker for random data generation
fake = Faker()

# Constants
NUM_NEW_TRANSACTIONS = 5000
FRAUD_PERCENTAGE = 0.045  # 4.5% suspicious transactions

# Load known fraudsters
known_fraudsters = pd.read_csv('known_fraudsters.csv')['Fraudster_ID'].tolist()

# Define possible countries and payment methods
countries = ["USA", "India", "UK", "Canada", "Germany"]
payment_methods = ["bank transfer", "credit card", "debit card", "PayPal", "Cryptocurrency"]
currencies = {"USA": "USD", "India": "INR", "UK": "GBP", "Canada": "CAD", "Germany": "EUR"}

def generate_new_transaction_data():
    data = []
    for _ in range(NUM_NEW_TRANSACTIONS):
        sender_country = random.choice(countries)
        receiver_country = random.choice(countries)
        
        transaction_date = fake.date_time_between(start_date='-1m', end_date='now')
        transaction_amount = round(random.uniform(1, 100000), 2)
        
        is_fraudulent = random.random() < FRAUD_PERCENTAGE
        is_known_fraudster = is_fraudulent and random.random() < 0.3
        
        if is_fraudulent:
            transaction_amount = round(random.uniform(50000, 200000), 2)
            transaction_velocity = random.randint(5, 15)
            unusual_time = random.random() < 0.7
            multiple_currency_conversions = random.random() < 0.4
            repeated_failed_attempts = random.randint(3, 10)
        else:
            transaction_velocity = random.randint(1, 4)
            unusual_time = random.random() < 0.1
            multiple_currency_conversions = random.random() < 0.05
            repeated_failed_attempts = random.randint(0, 2)
        
        transaction = {
            "Transaction_ID": str(uuid.uuid4()),
            "Date": transaction_date.strftime('%Y-%m-%d'),
            "Time": transaction_date.strftime('%H:%M:%S'),
            "Sender_ID": random.choice(known_fraudsters) if is_known_fraudster else str(uuid.uuid4()),
            "Receiver_ID": str(uuid.uuid4()),
            "Sender_Country": sender_country,
            "Receiver_Country": receiver_country,
            "Payment_Method": random.choice(payment_methods),
            "Transaction_Amount": transaction_amount,
            "Transaction_Currency": currencies[sender_country],
            "Transaction_Velocity": transaction_velocity,
            "Unusual_Time": unusual_time,
            "Multiple_Currency_Conversions": multiple_currency_conversions,
            "Repeated_Failed_Attempts": repeated_failed_attempts,
            "Is_Fraudulent": is_fraudulent,
            "Is_Known_Fraudster": is_known_fraudster
        }
        data.append(transaction)
    
    return pd.DataFrame(data)

# Generate new transaction dataset
new_df = generate_new_transaction_data()
new_df.to_csv('new_transactions.csv', index=False)
print("New transaction dataset has been generated and saved.")