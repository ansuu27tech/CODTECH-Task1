import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Set plotting style
sns.set_theme(style="whitegrid")

def main():
    print("--- Customer Churn Analysis and Prediction ---")
    data_path = 'data/customer_churn.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("\n1. Data Loading...")
    df = pd.read_csv(data_path)
    print(df.head())
    
    print("\n2. Data Cleaning & Missing Value Handling...")
    # Convert TotalCharges to numeric, coercing errors to NaN
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop CustomerID as it's not useful for prediction
    df = df.drop(columns=['CustomerID'])
    
    print("\n3. Exploratory Data Analysis (EDA)...")
    # 3.1 Churn Distribution Chart
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution Analysis')
    plt.savefig('churn_distribution.png')
    plt.close()
    
    # 3.2 Monthly Charges Analysis
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, palette='Set1')
    plt.title('Monthly Charges Distribution by Churn')
    plt.savefig('monthly_charges_analysis.png')
    plt.close()
    
    # 3.3 Tenure Analysis
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='Tenure', data=df, palette='Set3')
    plt.title('Tenure Analysis by Churn')
    plt.savefig('tenure_analysis.png')
    plt.close()

    print("\n4. Feature Engineering & Label Encoding...")
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Label encode all categorical features including target 'Churn'
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    print("\n5. Correlation Analysis...")
    # 5.1 Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    print("\n6. Train-Test Split...")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    print("\n7. Model Training (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("\n8. Model Evaluation...")
    y_pred = model.predict(X_test)
    
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    # Use zero_division parameter for small mock datasets where precision/recall might be undefined
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\n9. Feature Importance Graph...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Saved all visualizations to current directory.")
    
    print("\n10. Prediction System Test...")
    # Test prediction on a sample
    sample = X_test.iloc[[0]]
    prediction = model.predict(sample)
    pred_label = le_dict['Churn'].inverse_transform(prediction)
    print(f"Sample Features:\n{sample.to_string(index=False)}")
    print(f"Predicted Churn: {pred_label[0]}")
    print("\n--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
