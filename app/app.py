# app/app.py
# Application Flask - Interface réelle avec pipeline Meta-Learning
# Usage: python app.py

from flask import Flask, render_template, request, flash
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "dev_secret_key_for_ui_test"

# -------------------------------------------------------------------
# 1) CHARGEMENT DES MODÈLES
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

models = {}
scaler = None
meta_model = None

def load_models():
    global models, scaler, meta_model
    try:
        # ---- Load scaler ----
        with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
            print("Scaler chargé.")
            scaler = pickle.load(f)

        # ---- Load Base Models ----
        model_files = {
            "Decision Tree": "decision_tree_model.pkl",
            "Random Forest": "random_forest_model.pkl",
            "SVM": "svm_model.pkl",
            "Naive Bayes": "naive_bayes_model.pkl"
        }

        for name, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                    print(f"Modèle chargé : {name}")
            else:
                print(f"⚠️ Modèle manquant : {name} ({path})")

        # ---- Load Meta-model ----
        with open(os.path.join(MODELS_DIR, 'meta_model_knn.pkl'), 'rb') as f:
            meta_model = pickle.load(f)
            print("Meta modèle KNN chargé.")

        print("\n=== Tous les modèles ont été chargés avec succès ===\n")

    except Exception as e:
        print(f"\n❌ ERREUR CHARGEMENT MODÈLES : {e}\n")

load_models()

# -------------------------------------------------------------------
# 2) PIPELINE DE PRÉDICTION RÉEL
# -------------------------------------------------------------------

def real_prediction(values_dict):
    """
    Pipeline complet :
    - Scaling
    - Prédictions des 4 modèles
    - Construction des meta-features (confidence + margin)
    - Meta-model → sélection du meilleur modèle
    - Retour prédiction + confiance + détails de tous les modèles
    """
    if not scaler or not meta_model or not models:
        return "Erreur", "Modèles non chargés", 0.0, {}, []

    try:
        # ---- 1) Features dans l'ordre utilisé à l'entraînement ----
        feature_vector = [
            float(values_dict['temp']),
            float(values_dict['hum']),
            float(values_dict['light']),
            float(values_dict['co2']),
            float(values_dict['hr'])
        ]

        # ---- 2) Scaling ----
        vector_scaled = scaler.transform([feature_vector])

        # ---- 3) Meta-features (conf_max + margin pour chaque modèle) ----
        meta_features_row = []
        base_predictions = {}
        all_models_details = {}
        meta_features_list = []

        # Ordre : DT → RF → SVM → NB
        for name, model in models.items():
            # Prédiction et probabilités
            probas = model.predict_proba(vector_scaled)[0]
            prediction = int(model.predict(vector_scaled)[0])
            
            # Meta-feature 1 : Confiance maximale
            conf_max = float(np.max(probas))
            
            # Meta-feature 2 : Margin (différence entre top-1 et top-2)
            probas_sorted = np.sort(probas)
            margin = float(probas_sorted[-1] - probas_sorted[-2])
            
            # Ajout au vecteur de meta-features
            meta_features_row.extend([conf_max, margin])
            
            # Stockage pour affichage
            meta_features_list.append({
                'model': name,
                'confidence': conf_max,
                'margin': margin
            })
            
            # Stockage des prédictions de base
            base_predictions[name] = {
                "pred": prediction,
                "conf": conf_max,
                "margin": margin,
                "proba_occupied": float(probas[1]) if len(probas) > 1 else float(probas[0]),
                "proba_unoccupied": float(probas[0]) if len(probas) > 1 else 1 - float(probas[0])
            }
            
            # Détails complets pour chaque modèle
            all_models_details[name] = {
                'prediction': 'Occupé' if prediction == 1 else 'Inoccupé',
                'confidence': conf_max,
                'confidence_percent': f"{int(conf_max * 100)}%",
                'margin': margin,
                'margin_percent': f"{int(margin * 100)}%"
            }

        # Reshape pour le meta-modèle
        meta_features_row = np.array(meta_features_row).reshape(1, -1)

        # ---- 4) Meta-model choisit le meilleur modèle ----
        idx = int(meta_model.predict(meta_features_row)[0])
        
        model_names = list(models.keys())
        
        if 0 <= idx < len(model_names):
            selected_model = model_names[idx]
        else:
            selected_model = model_names[0]

        # ---- 5) Résultat final ----
        final_pred = base_predictions[selected_model]["pred"]
        final_conf = base_predictions[selected_model]["conf"]
        
        label = "Occupé" if final_pred == 1 else "Inoccupé"

        return label, selected_model, final_conf, all_models_details, meta_features_list

    except Exception as e:
        print(f"❌ Erreur dans le pipeline : {e}")
        import traceback
        traceback.print_exc()
        return "Erreur", "Pipeline Error", 0.0, {}, []

# -------------------------------------------------------------------
# 3) ROUTES FLASK
# -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_name = None
    confidence_percent = None
    confidence_float = None
    all_models_details = {}
    meta_features_list = []

    form_values = {"temp":"", "hum":"", "light":"", "co2":"", "hr":"", "time":""}

    if request.method == "POST":
        # --- Reprendre les valeurs du formulaire ---
        for key in form_values.keys():
            form_values[key] = request.form.get(key, "").strip()

        # --- Vérification simple ---
        required_keys = ["temp", "hum", "light", "co2", "hr"]

        if any(form_values[k] == "" for k in required_keys):
            flash("Merci de remplir tous les champs.", "warning")
            return render_template("index.html", 
                                 prediction=None, 
                                 form=form_values,
                                 all_models_details={},
                                 meta_features_list=[])

        try:
            # Vérifier la validité numérique
            _ = [float(form_values[k]) for k in required_keys]

        except:
            flash("Veuillez entrer des valeurs numériques valides.", "danger")
            return render_template("index.html", 
                                 prediction=None, 
                                 form=form_values,
                                 all_models_details={},
                                 meta_features_list=[])

        # ---- Exécuter la vraie prédiction ----
        prediction, model_name, confidence_float, all_models_details, meta_features_list = real_prediction(form_values)

        confidence_percent = f"{int(round(confidence_float * 100))}%"

        return render_template(
            "index.html",
            prediction=prediction,
            model_name=model_name,
            confidence_percent=confidence_percent,
            confidence_float=confidence_float,
            form=form_values,
            all_models_details=all_models_details,
            meta_features_list=meta_features_list
        )

    return render_template("index.html", 
                         prediction=None, 
                         form=form_values,
                         all_models_details={},
                         meta_features_list=[])

# -------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)