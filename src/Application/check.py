import joblib

model_path = 'src\\Application\\backend\\best_model_lft_kmeans.pkl'
model = joblib.load(model_path)

# Check the number of features the model expects
print(f'The model expects {model.n_features_in_} features.')