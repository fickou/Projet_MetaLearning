# app/app.py
# Application Flask - Interface only (fake predictions)
# Usage: python app.py

from flask import Flask, render_template, request, flash, redirect, url_for
import random

app = Flask(__name__)
app.secret_key = "dev_secret_key_for_ui_test"  # change in prod

# Simple helper to produce a fake prediction based on a simple heuristic
def fake_prediction(values_dict):
    """
    values_dict: dict of feature name -> float (strings already converted)
    Returns: (prediction_label, model_name, confidence_float)
    """
    try:
        co2 = float(values_dict.get("co2", 0))
        light = float(values_dict.get("light", 0))
        temp = float(values_dict.get("temp", 0))
    except ValueError:
        # fallback
        return "Informatique", "Modèle factice", 0.0

    # Heuristic: if CO2 high or light moderate+ and temp moderate => occupied
    score = 0.0
    score += max(0, (co2 - 400) / 1000)  # CO2 contributes
    score += max(0, (light - 200) / 1000)  # light contributes lightly
    score += 0.05 if 18 <= temp <= 28 else 0.0

    # clip and convert to percentage
    conf = min(0.98, max(0.1, score))
    # random small noise to avoid same result each time
    conf = round(conf + random.uniform(-0.05, 0.05), 3)
    conf = max(0.05, min(0.99, conf))

    label = "Occupé" if conf >= 0.35 else "Inoccupé"
    # choose a model name randomly among plausible ones (for UI demo)
    model = random.choice(["Random Forest", "SVM", "Decision Tree", "Naive Bayes"])

    return label, model, conf

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_name = None
    confidence = None
    form_values = {"temp":"", "hum":"", "light":"", "co2":"", "hr":"", "time":""}

    if request.method == "POST":
        # retrieve & simple validation
        for key in form_values.keys():
            form_values[key] = request.form.get(key, "").strip()

        # check required fields
        if any(v == "" for v in form_values.values()):
            flash("Merci de remplir tous les champs avant de lancer la prédiction.", "warning")
            return render_template("index.html", prediction=None, form=form_values)

        # try convert numeric (basic front-end validation too)
        try:
            # convert to floats to ensure validity
            _ = [float(v) for v in form_values.values()]
        except ValueError:
            flash("Merci de saisir des valeurs numériques valides.", "danger")
            return render_template("index.html", prediction=None, form=form_values)

        # generate fake prediction
        prediction, model_name, confidence = fake_prediction(form_values)

        # format confidence for display (percentage)
        display_conf = f"{int(round(confidence * 100))}%"  # for progress bar width usage
        # pass confidence as both numeric and string
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
    # debug True for development
    app.run(debug=True, host="0.0.0.0", port=5000)
