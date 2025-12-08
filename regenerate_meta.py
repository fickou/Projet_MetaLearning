#!/usr/bin/env python3
"""
Script pour régénérer le meta-dataset avec une meilleure stratégie de sélection
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("RÉGÉNÉRATION DU META-DATASET")
print("=" * 60)

# Chargement des données
print("\n[1/4] Chargement des données...")
df = pd.read_csv('donnees/datatraining.txt')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

X = df.drop('Occupancy', axis=1)
y = df['Occupancy']

# Split (même random_state pour cohérence)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Chargement des modèles
print("\n[2/4] Chargement des modèles...")
models = {}
model_files = {
    "Decision Tree": "models/decision_tree_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "SVM": "models/svm_model.pkl",
    "Naive Bayes": "models/naive_bayes_model.pkl"
}

for name, filepath in model_files.items():
    with open(filepath, 'rb') as f:
        models[name] = pickle.load(f)
    print(f"  ✓ {name}")

# Meta-features
print("\n[3/4] Génération des méta-features...")


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

# NOUVELLE STRATÉGIE DE SÉLECTION
print("\n[4/4] Génération des labels avec stratégie améliorée...")
meta_y = []
y_val_array = y_val.values

# Compteurs pour vérifier la distribution
selection_counts = {name: 0 for name in models.keys()}

for i in range(len(X_val_scaled)):
    sample = X_val_scaled[i].reshape(1, -1)
    true_label = y_val_array[i]
    
    # Collecter tous les modèles corrects avec leurs scores
    correct_models = []
    
    for idx, (name, model) in enumerate(models.items()):
        pred = model.predict(sample)[0]
        probas = model.predict_proba(sample)[0]
        conf = np.max(probas)
        margin = probas[pred] - probas[1-pred] if len(probas) == 2 else 0
        
        if pred == true_label:
            # Score composite : confiance + margin + petit bonus aléatoire pour départager
            score = conf + 0.5 * margin + np.random.uniform(0, 0.01)
            correct_models.append((idx, name, score))
    
    # Si au moins un modèle correct, prendre celui avec le meilleur score
    if correct_models:
        best_idx, best_name, _ = max(correct_models, key=lambda x: x[2])
        meta_y.append(best_idx)
        selection_counts[best_name] += 1
    else:
        # Fallback : prendre le plus confiant (même si faux)
        best_idx = -1
        best_conf = -1
        for idx, (name, model) in enumerate(models.items()):
            probas = model.predict_proba(sample)[0]
            conf = np.max(probas)
            if conf > best_conf:
                best_conf = conf
                best_idx = idx
                best_name = name
        meta_y.append(best_idx)
        selection_counts[best_name] += 1

meta_y = np.array(meta_y)

print("\n📊 Distribution des modèles sélectionnés :")
for name, count in selection_counts.items():
    pct = 100 * count / len(meta_y)
    print(f"  • {name:20s}: {count:4d} ({pct:5.1f}%)")

# Export CSV
print("\n[5/5] Export du CSV...")
columns = []
for name in models.keys():
    columns.extend([f"Conf_{name}", f"Margin_{name}"])

df_meta = pd.DataFrame(meta_X, columns=columns)
df_meta['Target_Model_Index'] = meta_y
df_meta['Target_Model_Name'] = df_meta['Target_Model_Index'].apply(lambda i: list(models.keys())[i])

output_csv = 'donnees/meta_features.csv'
df_meta.to_csv(output_csv, index=False)
print(f"✓ Fichier sauvegardé : {output_csv}")

# Réentraîner le meta-modèle
print("\n[6/6] Réentraînement du meta-modèle KNN...")
from sklearn.neighbors import KNeighborsClassifier
meta_model = KNeighborsClassifier(n_neighbors=5)
meta_model.fit(meta_X, meta_y)

with open('models/meta_model_knn.pkl', 'wb') as f:
    pickle.dump(meta_model, f)
print("✓ Meta-modèle KNN mis à jour")

print("\n" + "=" * 60)
print("✓ META-DATASET RÉGÉNÉRÉ AVEC SUCCÈS")
print("=" * 60)
