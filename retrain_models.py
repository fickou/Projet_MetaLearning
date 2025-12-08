#!/usr/bin/env python3
"""
Script pour réentraîner tous les modèles avec la version actuelle de scikit-learn
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

print("=" * 60)
print("RÉENTRAÎNEMENT DES MODÈLES")
print("=" * 60)

# Créer le dossier models
if not os.path.exists('models'):
    os.makedirs('models')

# Chargement des données
print("\n[1/5] Chargement des données...")
df = pd.read_csv('donnees/datatraining.txt')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
print(f"✓ Dataset chargé : {df.shape}")

# Préparation
X = df.drop('Occupancy', axis=1)
y = df['Occupancy']

# Split
print("\n[2/5] Split des données...")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Scaling
print("\n[3/5] Normalisation...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler sauvegardé")

# Modèles de base
print("\n[4/5] Entraînement des 4 modèles de base...")
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

for name, model in models.items():
    print(f"  • {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_val_scaled, y_val)
    filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Acc={acc:.4f}")

# Meta-features
print("\n[5/5] Génération du meta-dataset...")


def get_meta_features(models, X_scaled):
    meta_features = []
    for i in range(len(X_scaled)):
        row_features = []
        sample = X_scaled[i].reshape(1, -1)
        for name, model in models.items():
            probas = model.predict_proba(sample)[0]
            conf_max = np.max(probas)
            probas_sorted = np.sort(probas)
            margin = probas_sorted[-1] - probas_sorted[-2]
            row_features.extend([conf_max, margin])
        meta_features.append(row_features)
    return np.array(meta_features)


meta_X = get_meta_features(models, X_val_scaled)

# Meta labels
meta_y = []
y_val_array = y_val.values

for i in range(len(X_val_scaled)):
    sample = X_val_scaled[i].reshape(1, -1)
    true_label = y_val_array[i]
    best_model_idx = -1
    best_conf = -1
    
    for idx, (name, model) in enumerate(models.items()):
        pred = model.predict(sample)[0]
        probas = model.predict_proba(sample)[0]
        conf = np.max(probas)
        
        if pred == true_label and conf > best_conf:
            best_conf = conf
            best_model_idx = idx
    
    if best_model_idx == -1:
        for idx, (name, model) in enumerate(models.items()):
            probas = model.predict_proba(sample)[0]
            conf = np.max(probas)
            if conf > best_conf:
                best_conf = conf
                best_model_idx = idx
    
    meta_y.append(best_model_idx)

meta_y = np.array(meta_y)

# Meta-modèle
print(f"  • Meta-features: {meta_X.shape}")
meta_model = KNeighborsClassifier(n_neighbors=5)
meta_model.fit(meta_X, meta_y)

with open('models/meta_model_knn.pkl', 'wb') as f:
    pickle.dump(meta_model, f)
print("✓ Meta-modèle KNN entraîné et sauvegardé")

print("\n" + "=" * 60)
print("✓ TOUS LES MODÈLES ONT ÉTÉ RÉENTRAÎNÉS AVEC SUCCÈS")
print("=" * 60)
