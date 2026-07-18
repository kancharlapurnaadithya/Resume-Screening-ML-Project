# src/train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("⏳ Loading datasets from 'data/' folder...")
    # Data Loading (Step 2)
    try:
        df_train = pd.read_csv('data/train.csv')
        df_val = pd.read_csv('data/validation.csv')
        df_test = pd.read_csv('data/test.csv')
    except FileNotFoundError:
        print("❌ Error: 'data/' folder lo files levu. Please check file paths!")
        return

    print("⚙️ Data Preprocessing & Encoding started...")
    # Features and Targets separation
    X_train = df_train.drop(columns=['candidate_id', 'selected'])
    y_train = df_train['selected']

    X_val = df_val.drop(columns=['candidate_id', 'selected'])
    y_val = df_val['selected']

    X_test = df_test.drop(columns=['candidate_id', 'selected'])
    y_test = df_test['selected']

    # Categorical Columns Encoding (Step 3 & 4 Fix)
    categorical_cols = ['gender', 'degree', 'specialization', 'location', 'university_tier']
    le = LabelEncoder()

    for col in categorical_cols:
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    print("✅ Encoding completed successfully.")

    # Model Training (Step 5)
    print("\n🚀 Model training start avuthundi...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("🎉 Model Training Successfully Completed!")

    # Validation Evaluation (Step 6)
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"📊 Validation Dataset Accuracy: {val_accuracy * 100:.2f}%")

    # Final Test Evaluation (Step 7)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"🥇 Final Test Dataset Accuracy: {test_accuracy * 100:.2f}%\n")

    print("📋 Final Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("-" * 50)

    # Feature Importance Analysis (Bonus Step)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("🔥 Top 5 Features driving Candidate Selection:")
    for f in range(5):
        print(f"{f + 1}. {X_train.columns[indices[f]]} ({importances[indices[f]]*100:.2f}%)")

if __name__ == "__main__":
    main()
