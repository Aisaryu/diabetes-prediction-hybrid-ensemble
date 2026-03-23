import numpy as np
import pandas as pd
import os
import kagglehub
import joblib # Added for saving the model
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, recall_score, roc_auc_score, mean_squared_error,
                             accuracy_score, precision_score, average_precision_score,
                             cohen_kappa_score, brier_score_loss, confusion_matrix)
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# --- Research Parameters ---
RANDOM_STATE = 42
SAMPLE_SIZE = None 

# 1. Download/Load Dataset
print("Downloading/Loading dataset...")
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
file_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")
df = pd.read_csv(file_path)

# 2. Target and Features
X_all_raw = df.drop('Diabetes_binary', axis=1)
y_all = df['Diabetes_binary']

# --- Subsampling ---
if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
    print(f"Subsampling to {SAMPLE_SIZE} for computational efficiency...")
    X_all_raw, _, y_all, _ = train_test_split(
        X_all_raw, y_all, train_size=SAMPLE_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )

# --- Updated Hybrid SMOTE Class ---
class HybridGaussianAdaptiveSMOTE:
    def __init__(self, k_neighbors=5, m_neighbors=10, sigma_scale=0.2, random_state=42):
        self.k = k_neighbors
        self.m = m_neighbors
        self.sigma = sigma_scale
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, X, y):
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        unique, counts = np.unique(y_arr, return_counts=True)
        min_class = unique[np.argmin(counts)]
        X_min, X_maj = X_arr[y_arr == min_class], X_arr[y_arr != min_class]
        n_to_syn = len(X_maj) - len(X_min)
        if n_to_syn <= 0: return X, y

        nn_global = NearestNeighbors(n_neighbors=self.m + 1, n_jobs=-1).fit(X_arr)
        _, indices_global = nn_global.kneighbors(X_min)
        inner_indices, danger_indices = [], []

        for i, neighbors in enumerate(indices_global):
            neighbor_classes = y_arr[neighbors[1:]]
            if np.sum(neighbor_classes[:self.k] == min_class) > (self.k / 2):
                inner_indices.append(i)
            else:
                danger_indices.append(i)

        X_inner, X_danger = X_min[inner_indices], X_min[danger_indices]
        syn_samples = []

        def get_target_neighbor(query, population):
            if len(population) == 0: return query
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(population)
            _, ind = nn.kneighbors([query])
            return population[ind[0][0]]

        for _ in range(n_to_syn):
            src = X_danger[self.rng.randint(len(X_danger))] if len(X_danger) > 0 else X_inner[self.rng.randint(len(X_inner))]
            target = get_target_neighbor(src, X_inner) if len(X_inner) > 0 else get_target_neighbor(src, X_danger)
            vec = target - src
            dist = np.linalg.norm(vec)
            mean = src + (vec * 0.5)
            scale = (dist * self.sigma) + 1e-6
            cov = np.eye(src.shape[0]) * (scale**2)
            noise = self.rng.multivariate_normal(mean=np.zeros(src.shape[0]), cov=cov)
            syn_samples.append(mean + noise)

        return np.vstack([X_arr, np.array(syn_samples)]), np.hstack([y_arr, np.full(len(syn_samples), min_class)])


# --- Final Model Deployment Saving ---
def save_final_model():
    print("--- Training final model on ALL data for deployment... ---")
    
    # 1. Prepare full data
    imp = KNNImputer(n_neighbors=5)
    X_imp = imp.fit_transform(X_all_raw)
    
    scaler = StandardScaler()
    X_scl = scaler.fit_transform(X_imp)
    
    # 2. Augment full training data
    print("Applying HybridGaussianAdaptiveSMOTE (this may take a moment)...")
    hgas = HybridGaussianAdaptiveSMOTE(random_state=RANDOM_STATE)
    X_res, y_res = hgas.fit_resample(X_scl, y_all)
    
    # 3. Train final Hybrid Ensemble
    print("Training the Voting Classifier Ensemble...")
    rf_opt = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_opt = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE)
    hybrid_ens = VotingClassifier(estimators=[('rf', rf_opt), ('xgb', xgb_opt)], voting='soft')
    
    hybrid_ens.fit(X_res, y_res)
    
    # 4. Save artifacts
    print("Saving model artifacts to .pkl files...")
    joblib.dump(hybrid_ens, 'hybrid_ensemble_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imp, 'imputer.pkl')
    joblib.dump(list(X_all_raw.columns), 'feature_names.pkl') 
    print("Model saved successfully! You are ready for Phase 2.")

# Execute the saving function
if __name__ == "__main__":
    save_final_model()
