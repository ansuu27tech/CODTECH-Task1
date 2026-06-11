import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download NLTK data if not already present
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """
    Cleans the text: lowercases, removes punctuation, tokenizes, and removes stopwords.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization and Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def predict_spam(text, vectorizer, model):
    """
    Predicts whether a given text is spam or ham.
    """
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

def main():
    print("--- Email Spam Classifier Using Machine Learning ---")
    data_path = 'data/spam.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    print("1. Loading dataset...")
    df = pd.read_csv(data_path)
    print(df.head())
    
    print("\n2. Data Cleaning and Preprocessing...")
    df = df.dropna()
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    print("\n3. Feature Extraction (TF-IDF)...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    
    print("\n4. Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    print("\n5. Model Training (Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("\n6. Model Evaluation...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n7. Visualization (Confusion Matrix)...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to 'confusion_matrix.png'.")
    
    print("\n8. Testing Prediction Function...")
    sample_text = "Congratulations! You've won a $1000 gift card. Click here."
    prediction = predict_spam(sample_text, vectorizer, model)
    print(f"Sample Text: '{sample_text}'")
    print(f"Prediction: {prediction.upper()}")

if __name__ == "__main__":
    main()
