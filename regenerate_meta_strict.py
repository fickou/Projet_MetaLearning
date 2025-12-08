#!/usr/bin/env python3
"""
Script conforme au document du TP - Section 7
Stratégie : Le classifieur correct devient la cible
Fallback : Le plus confiant si échec total
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("RÉGÉNÉRATION META-DATASET (CONFORME AU TP)")
print("=" * 60)

# Chargement
print("\n[1/4] Chargement des données...")
df = pd.read_csv('donnees/datatraining.txt')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

X = df.drop('Occupancy', axis=1)
y = df['Occupancy']

# Split
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

# STRATÉGIE CONFORME AU DOCUMENT
print("\n[4/4] Génération des labels (stratégie du TP)...")
meta_y = []
y_val_array = y_val.values

selection_counts = {name: 0 for name in models.keys()}

for i in range(len(X_val_scaled)):
    sample = X_val_scaled[i].reshape(1, -1)
    true_label = y_val_array[i]
    
    selected_idx = None
    selected_name = None
    
    # Chercher le premier classifieur correct (dans l'ordre du dict)
    for idx, (name, model) in enumerate(models.items()):
        pred = model.predict(sample)[0]
        if pred == true_label:
            selected_idx = idx
            selected_name = name
            break  # Prendre le premier correct
    
    # Fallback : si aucun correct, prendre le plus confiant
    if selected_idx is None:
        best_conf = -1
        for idx, (name, model) in enumerate(models.items()):
            probas = model.predict_proba(sample)[0]
            conf = np.max(probas)
            if conf > best_conf:
                best_conf = conf
                selected_idx = idx
                selected_name = name
    
    meta_y.append(selected_idx)
    selection_counts[selected_name] += 1

meta_y = np.array(meta_y)

print("\n📊 Distribution (stratégie stricte du TP) :")
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
print("✓ CONFORME AU DOCUMENT DU TP")
print("=" * 60)
