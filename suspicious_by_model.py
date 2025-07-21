import pandas as pd
import numpy as np
import numpy as np

def patch_model_for_compatibility(model):
    """Patch model to handle monotonic_cst attribute compatibility issues"""
    try:
        # For DecisionTreeClassifier and related models
        if hasattr(model, 'tree_'):
            if not hasattr(model, 'monotonic_cst'):
                model.monotonic_cst = None
        
        # For ensemble models (RandomForest, etc.)
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'tree_') and not hasattr(estimator, 'monotonic_cst'):
                    estimator.monotonic_cst = None
        
        return model
    except Exception as e:
        print(f"Warning: Could not patch model for compatibility: {e}")
        return model

def detect_suspicious_transactions(df, model, scaler, threshold=0.5):
    # Patch the model for compatibility
    model = patch_model_for_compatibility(model)
    
    X = preprocess_data(df, scaler)
    X_scaled = scaler.transform(X)
    
    try:
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    except AttributeError as e:
        if 'monotonic_cst' in str(e):
            # If the error persists, try a different approach
            print("Applying fallback prediction method...")
            # For binary classification, we can use predict and convert to probability
            y_pred = model.predict(X_scaled)
            y_pred_proba = y_pred.astype(float)  # Convert to probability-like values
        else:
            raise e
    
    df['Suspicion_Score'] = y_pred_proba
    df['Is_Suspicious'] = y_pred_proba > threshold
    
    return df

def preprocess_data(df, scaler):
    categorical_columns = ['Sender_Country', 'Receiver_Country', 'Payment_Method', 'Transaction_Currency']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    for col in scaler.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[scaler.feature_names_in_]
    
    return df_encoded

def explain_suspicion(row, feature_importances, historical_data, threshold=0.05):
    reasons = []
    sender_history = historical_data[historical_data['Sender_ID'] == row['Sender_ID']]
    for feature, importance in feature_importances.items():
        if importance > threshold:
            if feature == 'Transaction_Amount':
                avg_amount = sender_history['Transaction_Amount'].mean()
                if row[feature] > avg_amount * 1.5:
                    reasons.append(f"High Transaction Amount (${row[feature]:.2f}, {row[feature]/avg_amount:.1f}x average)")
            elif feature == 'Transaction_Velocity':
                avg_velocity = sender_history['Transaction_Velocity'].mean()
                if row[feature] > avg_velocity * 1.5:
                    reasons.append(f"High Transaction Velocity ({row[feature]:.2f}, {row[feature]/avg_velocity:.1f}x average)")
            elif feature == 'Unusual_Time' and row[feature]:
                reasons.append("Unusual Transaction Time")
            elif feature == 'Multiple_Currency_Conversions' and row[feature]:
                reasons.append("Multiple Currency Conversions")
            elif feature in ['Sender_Country', 'Receiver_Country'] and row['Sender_Country'] != row['Receiver_Country']:
                reasons.append(f"Cross-border Transaction ({row['Sender_Country']} to {row['Receiver_Country']})")
            elif feature == 'Is_Known_Fraudster' and row[feature]:
                reasons.append("Known Fraudster Involved")
            elif feature == 'Is_Sanctioned_Entity' and row[feature]:
                reasons.append("Sanctioned Entity Involved")
            elif 'Payment_Method_Cryptocurrency' in feature and row[feature]:
                reasons.append("Cryptocurrency Transaction")
            elif feature == 'IP_Address_Change' and row[feature]:
                reasons.append("Frequent IP Address Changes")
            elif feature == 'Device_Change' and row[feature]:
                reasons.append("Multiple Devices Used")
            elif feature == 'VPN_Usage' and row[feature]:
                reasons.append("VPN Usage Detected")
    
    if not reasons:
        if row['Suspicion_Score'] > 0.7:
            reasons.append(f"High overall suspicion score ({row['Suspicion_Score']:.2f})")
        elif row['Suspicion_Score'] > 0.5:
            reasons.append(f"Moderate overall suspicion score ({row['Suspicion_Score']:.2f})")
    
    return ', '.join(set(reasons)) if reasons else "No specific reason identified"