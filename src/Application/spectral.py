import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import matplotlib.pyplot as plt

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVC': LinearSVC(random_state=42, max_iter=10000, dual='auto'),
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
        try:
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
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    return accuracy, precision, recall, f1, best_model

def optimal_spectral_clusters(data, max_clusters=10):
    distances = pairwise_distances(data, metric='euclidean')
    laplacian = np.diag(distances.sum(axis=1)) - distances
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), np.diff(eigenvalues[:max_clusters + 1]), marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Eigengap')
    plt.title('Eigengap Heuristic')
    plt.show()
    
    eigengap = np.diff(eigenvalues[:max_clusters + 1])
    optimal_k = np.argmax(eigengap) + 1
    return optimal_k

def spectral_clustering(data, feature_columns):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[feature_columns])
    
    optimal_k = optimal_spectral_clusters(data_scaled)
    # print("Optimal Cluster Count: ", optimal_k)
    try:
        spectral = SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors', random_state=42)
        clusters = spectral.fit_predict(data_scaled)
    except ValueError as e:
        print(f"Spectral Clustering failed: {e}")
        clusters = np.zeros(data_scaled.shape[0])
    
    data['Cluster'] = clusters
    return data

def main():
    file_path = 'src\\Dataset\\Liver Patient Dataset (LPD)_train.csv'
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data.columns = data.columns.str.strip()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = ['Gender of the patient']
    
    imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])
    
    encoder = OneHotEncoder(drop='first')
    gender_encoded = encoder.fit_transform(data[['Gender of the patient']]).toarray()
    data['Gender of the patient'] = gender_encoded
    
    nafld_columns = ['Age of the patient', 'Total Bilirubin', 'ALB Albumin', 'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase']
    lft_columns = ['Total Bilirubin', 'Direct Bilirubin', 'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase', 'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin', 'A/G Ratio Albumin and Globulin Ratio']
    albi_columns = ['Total Bilirubin', 'ALB Albumin']

    results = {}
    
    for feature_set, columns in {'NAFLD': nafld_columns, 'LFT': lft_columns, 'ALBI': albi_columns}.items():
        print(f'Performing Spectral Clustering for {feature_set}')
        spectral_data = spectral_clustering(data.copy(), columns)
        # print("Final Spectral data",spectral_data)
        X = spectral_data[columns + ['Cluster']]
        y = spectral_data['Result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
        results[f'Spectral_{feature_set}'] = evaluate_models(X_train, X_test, y_train, y_test)

    best_metrics = compare_metrics_and_export(results)

    for dataset, model_info in best_metrics.items():
        if model_info['model'] is not None:
            with open(model_info['file_name'], 'wb') as f:
                pickle.dump(model_info['model'], f)

    print(best_metrics)

def compare_metrics_and_export(results):
    best_metrics = {}
    for dataset in ['NAFLD', 'LFT', 'ALBI']:
        best_metrics[dataset] = {
            'model': None,
            'accuracy': 0,
            'file_name': ''
        }
        
        if f'Spectral_{dataset}' in results and results[f'Spectral_{dataset}'][0] > best_metrics[dataset]['accuracy']:
            best_metrics[dataset] = {
                'model': results[f'Spectral_{dataset}'][4],
                'accuracy': results[f'Spectral_{dataset}'][0],
                'precision': results[f'Spectral_{dataset}'][1],
                'recall': results[f'Spectral_{dataset}'][2],
                'f1': results[f'Spectral_{dataset}'][3],
                'file_name': f'best_model_{dataset.lower()}_spectral.pkl'
            }
    
    return best_metrics

if __name__ == "__main__":
    main()
