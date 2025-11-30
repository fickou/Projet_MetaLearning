# app/utils/pipeline.py
# Fonctions utilitaires pour le pipeline : extraction des meta-features,
# conversion safe des scores en probabilités, et pipeline complet.
# L'idée : fournir une fonction full_prediction_pipeline(...) réutilisable
# par Flask.

import os
import pickle
import numpy as np
from sklearn.base import ClassifierMixin
from typing import Tuple, Dict

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax stable pour vecteurs/arrays 1D ou 2D."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=1, keepdims=True)


def probas_from_model(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """
    Retourne un array (n_samples, n_classes) de probabilités pour un modèle donné.
    Si le modèle possède predict_proba(), on l'utilise.
    Sinon si il possède decision_function(), on convertit via softmax.
    """
    # Cas privilégié
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # Sinon essayer decision_function
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Certains modèles retournent shape (n_samples,) pour binaire -> convertir en deux classes
        scores = np.atleast_2d(scores)
        # si 1D convert to 2D (n_samples, 1) -> transformer en deux colonnes [ -s, s ]
        if scores.shape[0] == 1 and X.shape[0] == 1:
            # rare cas pour single sample -> transpose
            scores = scores.T
        if scores.ndim == 1 or scores.shape[1] == 1:
            # binaire : transformer en logits pour deux classes
            s = scores.ravel()
            s2 = np.vstack([-s, s]).T
            return softmax(s2)
        else:
            return softmax(scores)
    # fallback : prédiction one-hot sur predict()
    preds = model.predict(X)
    # déterminer nombre de classes (supposons 2)
    n = len(np.unique(preds))
    probs = np.zeros((X.shape[0], n))
    for i, p in enumerate(preds):
        probs[i, int(p)] = 1.0
    return probs


def extract_meta_features_for_model(model: ClassifierMixin, X: np.ndarray) -> Tuple[float, float]:
    """
    Pour une instance X (shape (1, n_features)), retourne (confiance_max, margin).
    - confiance_max = max(probabilities)
    - margin = |p(class0) - p(class1)|
    """
    proba = probas_from_model(model, X)[0]  # shape (n_classes,)
    # Si classes > 2, on prend les deux plus grandes pour margin
    sorted_idx = np.argsort(proba)[::-1]
    pmax = proba[sorted_idx[0]]
    if proba.size >= 2:
        margin = abs(proba[sorted_idx[0]] - proba[sorted_idx[1]])
    else:
        margin = pmax  # cas improbable
    return float(pmax), float(margin)


def build_meta_vector(models: Dict[int, ClassifierMixin], X_scaled: np.ndarray) -> np.ndarray:
    """
    Construit le vecteur meta (1, 8) à partir des modèles passés.
    models : dictionnaire index -> modèle (doit être dans l'ordre attendu)
    """
    meta = []
    # on s'attend à 4 modèles indexés 0..3
    for i in range(4):
        m = models[i]
        conf, margin = extract_meta_features_for_model(m, X_scaled)
        meta.extend([conf, margin])
    return np.array(meta).reshape(1, -1)


def full_prediction_pipeline(
    X: np.ndarray,
    scaler,
    models: Dict[int, ClassifierMixin],
    meta_knn
) -> Tuple[str, int, float]:
    """
    Pipeline complet :
    - X : (n_samples, 6) (on supporte n_samples mais Flask enverra 1)
    - scaler : StandardScaler ou similaire
    - models : dict {0: model_dt, 1: model_rf, 2: model_svm, 3: model_nb}
    - meta_knn : modèle KNN entraîné sur meta-features

    Retour :
    - label_final : "Occupé" / "Inoccupé"
    - selected_model_idx : index du modèle choisi (int)
    - confidence : probabilité max renvoyée par le modèle choisi (float)
    """
    # Normalisation (on suppose scaler déjà fit)
    X_scaled = scaler.transform(X)

    # Pour chaque sample, on peut faire la même procédure ; ici on gère 1 sample
    meta = build_meta_vector(models, X_scaled)

    # KNN prédit l'indice du meilleur modèle
    selected_idx = int(meta_knn.predict(meta)[0])

    # Exécuter le modèle choisi sur X_scaled
    chosen_model = models[selected_idx]
    y_pred = chosen_model.predict(X_scaled)[0]

    # Récupérer la probabilité de confiance du modèle choisi
    proba = probas_from_model(chosen_model, X_scaled)[0]
    conf = float(np.max(proba))

    # Map label numeric -> string (on suppose 0 = inoccupé, 1 = occupé)
    label_map = {0: "Inoccupé", 1: "Occupé"}
    label_final = label_map.get(int(y_pred), str(int(y_pred)))

    return label_final, selected_idx, conf


def load_models_from_folder(model_folder: str):
    """
    Charge scaler, modèles et meta_knn depuis un dossier donné.
    Attendu :
      scaler.pkl
      model_dt.pkl
      model_rf.pkl
      model_svm.pkl
      model_nb.pkl
      meta_knn.pkl

    Retour : scaler, models_dict, meta_knn
    """
    def load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    scaler_path = os.path.join(model_folder, "scaler.pkl")
    dt_path = os.path.join(model_folder, "model_dt.pkl")
    rf_path = os.path.join(model_folder, "model_rf.pkl")
    svm_path = os.path.join(model_folder, "model_svm.pkl")
    nb_path = os.path.join(model_folder, "model_nb.pkl")
    meta_knn_path = os.path.join(model_folder, "meta_knn.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler missing: {scaler_path}")

    scaler = load_pkl(scaler_path)
    model_dt = load_pkl(dt_path)
    model_rf = load_pkl(rf_path)
    model_svm = load_pkl(svm_path)
    model_nb = load_pkl(nb_path)
    meta_knn = load_pkl(meta_knn_path)

    models = {
        0: model_dt,
        1: model_rf,
        2: model_svm,
        3: model_nb
    }

    return scaler, models, meta_knn
