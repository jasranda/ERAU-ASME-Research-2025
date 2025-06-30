#!/usr/bin/env python
# coding: utf-8

# # IMECE 2025

# In[1]:


#pip install optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from itertools import product

# ----------------------
# Step 1: Prepare Data
# ----------------------

# Dataset 1: Temperature
data_temp = pd.DataFrame({
    'Model': ['Model 6', 'Model 9', 'Model 3', 'Model 5', 'Model 1', 'Model 7', 'Model 2', 'Model 8', 'Model 4'],
    'Nozzle_Temp': [252.5, 252.5, 238.3, 224.2, 252.5, 252.5, 266.7, 252.5, 280.8],
    'Bed_Temp': [96.7, 61.7, 85, 85, 85, 73.3, 85, 108.3, 85],
    'Ranking_Temp': [1, 2, 3, 4, 5, 6, 7, 8, 9]
})

# Dataset 2: Speed & Thickness
data_speed = pd.DataFrame({
    'Model': ['Model 3', 'Model 8', 'Model 4', 'Model 6', 'Model 9', 'Model 2', 'Model 7', 'Model 5', 'Model 1'],
    'Speed': [120, 173.3, 120, 146.6, 66.6, 120, 93.3, 120, 120],
    'Thickness': [150, 175, 225, 175, 175, 200, 175, 125, 175],
    'Ranking_Speed': [1, 2, 3, 4, 5, 6, 7, 8, 9]
})

# Merge datasets and average rankings
merged = pd.merge(data_temp, data_speed, on='Model')
merged['Ranking'] = merged[['Ranking_Temp', 'Ranking_Speed']].mean(axis=1)

X = merged[['Nozzle_Temp', 'Bed_Temp', 'Speed', 'Thickness']]
y = merged['Ranking']

# ----------------------
# Step 2: Bayesian Optimization with Optuna
# ----------------------

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0)
    }

    model = GradientBoostingRegressor(**params, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10000)

print("\nBest hyperparameters from Bayesian Optimization:")
print(study.best_params)

# ----------------------
# Step 3: Train Final Model with Best Parameters
# ----------------------

best_params = study.best_params
model = GradientBoostingRegressor(**best_params, random_state=0)
model.fit(X, y)

# ----------------------
# Step 4: Predict Best Parameter Combination
# ----------------------

def predict_best(model, bounds_dict, resolution=30):
    ranges = [np.linspace(v[0], v[1], resolution) for v in bounds_dict.values()]
    grid_points = np.array(list(product(*ranges)))
    preds = model.predict(grid_points)
    best_idx = np.argmin(preds)
    best_point = grid_points[best_idx]
    return dict(zip(bounds_dict.keys(), best_point))

bounds = {
    'Nozzle_Temp': (224.2, 280.8),
    'Bed_Temp': (61.7, 108.3),
    'Speed': (66.6, 173.3),
    'Thickness': (125, 225)
}

best_combination = predict_best(model, bounds)

print("\n=== 🔧 Optimal Printing Settings for Highest Reliability ===")
for k, v in best_combination.items():
    print(f"{k}: {v:.2f}")


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# Set font to Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'

# Get feature importances and clean feature names
importances = model.feature_importances_
feature_names = [name.replace('_', ' ') for name in X.columns]

# Sort features by importance
sorted_idx = np.argsort(importances)
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

# Choose colors for bars
colors = plt.cm.viridis(np.linspace(0.4, 1.0, len(sorted_names)))

# Plot
plt.figure(figsize=(8, 5))
plt.barh(sorted_names, sorted_importances, color=colors)
plt.xlabel('Importance', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
#plt.title('Feature Importance in Reliability Prediction', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('IMECE_importance_features_1.svg')
plt.show()

# Print raw values (optional)
print(dict(zip(feature_names, importances)))

