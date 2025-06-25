#!/usr/bin/env python3
"""
Improved Resume Screening AI - Python Script Version
This script provides a comprehensive analysis and machine learning pipeline for resume category prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Text processing
import re
import pickle
import time
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully!")

def clean_resume_enhanced(text):
    """Enhanced text cleaning function"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print(" Loading dataset...")
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {df['Category'].nunique()}")
    
    # Clean resume texts
    print(" Cleaning resume texts...")
    df['Resume_Cleaned'] = df['Resume'].apply(clean_resume_enhanced)
    
    # Label encoding
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Category'])
    
    # Balance classes if needed
    category_counts = df['Category'].value_counts()
    min_count = category_counts.min()
    max_count = category_counts.max()
    imbalance_ratio = max_count / min_count
    
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:
        print("Balancing dataset...")
        balanced_df = df.groupby('Category', group_keys=False).apply(
            lambda x: x.sample(max_count, replace=True)
        ).reset_index(drop=True)
        df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df, le

def create_features(df):
    """Create TF-IDF features"""
    print("Creating TF-IDF features...")
    
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_tfidf = tfidf.fit_transform(df['Resume_Cleaned'])
    y = df['Category_Encoded']
    
    print(f"Feature matrix shape: {X_tfidf.shape}")
    print(f"Number of features: {X_tfidf.shape[1]}")
    
    return X_tfidf, y, tfidf

def train_and_evaluate_models(X_tfidf, y):
    """Train and evaluate multiple models"""
    print("Training and evaluating models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'SVC': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Multinomial NB': MultinomialNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': time.time() - start_time
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results, X_train, X_test, y_train, y_test

def save_models(results, tfidf, le):
    """Save the best model and preprocessing components"""
    print("Saving models...")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Save components
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    with open('clf.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save metadata
    model_metadata = {
        'model_name': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'training_time': results[best_model_name]['training_time'],
        'feature_count': tfidf.get_feature_names_out().shape[0],
        'categories': list(le.classes_),
        'created_at': datetime.now().isoformat()
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(model_metadata, f)
    
    print("All models saved successfully!")
    return best_model_name, best_model

def test_prediction(best_model, tfidf, le):
    """Test the prediction with sample resumes"""
    print(" Testing prediction function...")
    
    test_resumes = [
        # Data Science resume
        """
        Data Scientist with 5+ years of experience in machine learning, 
        deep learning, and statistical analysis. Proficient in Python, 
        R, SQL, and various ML frameworks including TensorFlow and PyTorch. 
        Experienced in developing predictive models, A/B testing, and 
        data visualization. Strong background in mathematics and statistics.
        """,
        
        # Software Development resume
        """
        Senior Software Engineer with expertise in full-stack development 
        using Java, Spring Boot, React, and Node.js. Experience with 
        microservices architecture, cloud platforms (AWS, Azure), and 
        DevOps practices. Led development teams and delivered scalable 
        applications for enterprise clients.
        """,
        
        # Marketing resume
        """
        Marketing Manager with 8 years of experience in digital marketing, 
        brand management, and customer acquisition. Expertise in Google 
        Analytics, Facebook Ads, SEO, and content marketing strategies. 
        Successfully increased brand awareness and conversion rates for 
        multiple companies.
        """
    ]
    
    for i, resume in enumerate(test_resumes, 1):
        # Clean text
        cleaned_text = clean_resume_enhanced(resume)
        
        # Vectorize
        text_vectorized = tfidf.transform([cleaned_text])
        
        # Predict
        prediction = best_model.predict(text_vectorized)[0]
        category_name = le.inverse_transform([prediction])[0]
        
        # Get confidence
        confidence = 0.0
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(text_vectorized)[0]
            confidence = max(proba) * 100
        
        print(f"\\n📄 Test Resume {i}:")
        print(f"Category: {category_name}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Text length: {len(cleaned_text)} characters")

def main():
    """Main execution function"""
    print("Starting Improved Resume Screening AI Analysis")
    print("=" * 60)
    
    # Load and preprocess data
    df, le = load_and_preprocess_data()
    
    # Create features
    X_tfidf, y, tfidf = create_features(df)
    
    # Train and evaluate models
    results, X_train, X_test, y_train, y_test = train_and_evaluate_models(X_tfidf, y)
    
    # Save models
    best_model_name, best_model = save_models(results, tfidf, le)
    
    # Test prediction
    test_prediction(best_model, tfidf, le)
    
    # Final summary
    print("\\n Analysis Complete!")
    print("=" * 40)
    print(f"Dataset size: {len(df)} resumes")
    print(f"Categories: {len(df['Category'].unique())}")
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("\\n Saved files:")
    print("- tfidf.pkl: TF-IDF vectorizer")
    print("- clf.pkl: Best model")
    print("- encoder.pkl: Label encoder")
    print("- model_metadata.pkl: Model information")
    print("\\n Ready for deployment!")

if __name__ == "__main__":
    main() 
