# app/app.py
# Application Flask - Interface Réelle
# Usage: python app.py

from flask import Flask, render_template, request, flash
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "dev_secret_key_for_ui_test"

# --- Chargement des Modèles ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

models = {}
scaler = None
meta_model = None

def load_models():
    global models, scaler, meta_model
    try:
        # Scaler
        with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Base Models
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
            else:
                print(f"Warning: Model {name} not found at {path}")

        # Meta Model
        with open(os.path.join(MODELS_DIR, 'meta_model_knn.pkl'), 'rb') as f:
            meta_model = pickle.load(f)
            
        print("Tous les modèles ont été chargés avec succès.")
        
    except Exception as e:
        print(f"Erreur lors du chargement des modèles : {e}")

# Charger au démarrage
load_models()

# --- Logique de Prédiction (Pipeline) ---
def real_prediction(values_dict):
    """
    Pipeline complet : Scaling -> Base Models -> Meta Features -> KNN -> Final Prediction
    """
    if not scaler or not meta_model or not models:
        return "Erreur", "Modèles non chargés", 0.0

    try:
        # 1. Préparation des features (Ordre important !)
        # Temperature, Humidity, Light, CO2, HumidityRatio
        # Mapping form keys to feature order
        # Form keys: temp, hum, light, co2, hr
        feature_vector = [
            float(values_dict['temp']),
            float(values_dict['hum']),
            float(values_dict['light']),
            float(values_dict['co2']),
            float(values_dict['hr'])
        ]
        
        # Scaling
        vector_scaled = scaler.transform([feature_vector])
        
        # 2. Prédictions de base et Méta-features
        meta_features_row = []
        base_predictions = {}
        
        # L'ordre d'itération doit être le même que lors de l'entraînement !
        # Dans le notebook, on a itéré sur models.items(). 
        # Ici on doit s'assurer de l'ordre. Le dict python 3.7+ préserve l'ordre d'insertion.
        # Pour être sûr, on peut trier par nom ou fixer une liste.
        # Supposons que l'ordre d'insertion dans 'models' ici (DT, RF, SVM, NB) est le même que dans le notebook.
        # Dans le notebook : DT, RF, SVM, NB. Ici aussi. C'est bon.
        
        for name, model in models.items():
            probas = model.predict_proba(vector_scaled)[0]
            conf_max = np.max(probas)
            probas_sorted = np.sort(probas)
            margin = probas_sorted[-1] - probas_sorted[-2]
            
            meta_features_row.extend([conf_max, margin])
            base_predictions[name] = {
                'pred': model.predict(vector_scaled)[0],
                'conf': conf_max
            }
            
        # 3. Choix du modèle via Meta-Modèle
        meta_features_row = np.array(meta_features_row).reshape(1, -1)
        selected_model_idx = meta_model.predict(meta_features_row)[0]
        
        # Récupérer le nom du modèle via l'index
        # Attention : selected_model_idx est un entier qui correspond à l'index dans la liste des clés de 'models'
        model_names = list(models.keys())
        if 0 <= selected_model_idx < len(model_names):
            selected_model_name = model_names[selected_model_idx]
        else:
            # Fallback
            selected_model_name = model_names[0]

        # 4. Résultat final
        final_pred_val = base_predictions[selected_model_name]['pred']
        final_conf = base_predictions[selected_model_name]['conf']
        
        label = "Occupé" if final_pred_val == 1 else "Inoccupé"
        
        return label, selected_model_name, final_conf

    except Exception as e:
        print(f"Erreur de prédiction : {e}")
        return "Erreur", str(e), 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_name = None
    confidence = None
    # Form keys match HTML name attributes
    form_values = {"temp":"", "hum":"", "light":"", "co2":"", "hr":"", "time":""}

    if request.method == "POST":
        for key in form_values.keys():
            form_values[key] = request.form.get(key, "").strip()

        if any(v == "" for v in form_values.values() if key != "time"): # Time might be optional or unused
            flash("Merci de remplir tous les champs.", "warning")
            return render_template("index.html", prediction=None, form=form_values)

        try:
            # Simple numeric check
            _ = float(form_values['temp'])
            _ = float(form_values['hum'])
            _ = float(form_values['light'])
            _ = float(form_values['co2'])
            _ = float(form_values['hr'])
        except ValueError:
            flash("Valeurs numériques invalides.", "danger")
            return render_template("index.html", prediction=None, form=form_values)

        # Real Prediction
        prediction, model_name, confidence = real_prediction(form_values)

        display_conf = f"{int(round(confidence * 100))}%"
        
        return render_template(
            "index.html",
            prediction=prediction,
            model_name=model_name,
            confidence_percent=display_conf,
            confidence_float=confidence,
            form=form_values
        )

    return render_template("index.html", prediction=None, form=form_values)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
