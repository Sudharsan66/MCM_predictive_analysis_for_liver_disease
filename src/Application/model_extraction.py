import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVC': LinearSVC(random_state=42, max_iter=10000, dual=False),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Neural Network': MLPClassifier(random_state=42, max_iter=500),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

def evaluate_models(X_train, X_test, y_train, y_test):
    best_model = None
    best_accuracy = 0
    accuracy, precision, recall, f1 = None, None, None, None
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            accuracy = acc
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1, best_model

def main():
    file_path = 'src\\Dataset\\Liver Patient Dataset (LPD)_train.csv'
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data.columns = data.columns.str.strip()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])
    data['Gender of the patient'] = data['Gender of the patient'].map({'Male': 1, 'Female': 0})

    nafld_columns = ['Age of the patient', 'Total Bilirubin', 'ALB Albumin', 'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase']
    lft_columns = ['Total Bilirubin', 'Direct Bilirubin', 'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin', 'A/G Ratio Albumin and Globulin Ratio']
    albi_columns = ['Total Bilirubin', 'ALB Albumin']

    results = {}
    
    for feature_set, columns in {'NAFLD': nafld_columns, 'LFT': lft_columns, 'ALBI': albi_columns}.items():
        print(f'Evaluating models for {feature_set}')
        X = data[columns]
        y = data['Result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        results[feature_set] = evaluate_models(X_train, X_test, y_train, y_test)
    
    for feature_set, result in results.items():
        accuracy, precision, recall, f1, best_model = result
        if best_model:
            model_name = f'best_model_{feature_set.lower()}.pkl'
            with open(model_name, 'wb') as f:
                pickle.dump(best_model, f)
            print(f'Saved best model for {feature_set} as {model_name}')

    for feature_set, result in results.items():
        accuracy, precision, recall, f1, best_model = result
        print(f'{feature_set} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

if __name__ == "__main__":
    main()