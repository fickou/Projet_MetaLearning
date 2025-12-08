# app.py - Version corrigée pour Windows (sans caractères Unicode)
# Application Flask complète avec Meta-Learning

import pickle
import numpy as np
import os
import logging
import traceback
import secrets
import warnings
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from flask import Flask, render_template, request, flash, jsonify
from flask_caching import Cache
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Supprimer les warnings sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

class Config:
    """Configuration centralisée de l'application"""
    
    # Chemins
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR.parent / 'models'
    
    # Sécurité
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    
    # Serveur
    HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
    PORT = int(os.environ.get('SERVER_PORT', 5000))
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Cache
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 300))
    
    # Rate limiting
    RATE_LIMIT = os.environ.get('RATE_LIMIT', '100/hour')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'app.log'
    
    # Plages de validation des capteurs
    SENSOR_RANGES = {
        'temp': {'min': -30, 'max': 60, 'unit': '°C'},
        'hum': {'min': 0, 'max': 100, 'unit': '%'},
        'light': {'min': 0, 'max': 2000, 'unit': 'Lux'},
        'co2': {'min': 0, 'max': 5000, 'unit': 'ppm'},
        'hr': {'min': 0, 'max': 0.02, 'unit': 'ratio'}
    }
    
    # Fichiers de modèles requis
    REQUIRED_MODELS = {
        'scaler': 'scaler.pkl',
        'meta_model': 'meta_model_knn.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'svm': 'svm_model.pkl',
        'naive_bayes': 'naive_bayes_model.pkl'
    }

# -------------------------------------------------------------------
# LOGGING (VERSION WINDOWS-COMPATIBLE)
# -------------------------------------------------------------------

def setup_logging():
    """Configuration du système de logging compatible Windows"""
    log_dir = Path(Config.LOG_FILE).parent
    log_dir.mkdir(exist_ok=True)
    
    # Configurer le logger
    logger = logging.getLogger('occupancy_detection')
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Formatter sans caractères Unicode pour Windows
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler fichier avec encodage UTF-8
    file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Handler console avec encodage forcé
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Nettoyer les caractères problématiques pour Windows
                msg = msg.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    console_handler = SafeStreamHandler()
    console_handler.setFormatter(formatter)
    
    # Ajouter les handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Supprimer le handler par défaut
    logger.propagate = False
    
    return logger

logger = setup_logging()

# -------------------------------------------------------------------
# EXCEPTIONS PERSONNALISÉES
# -------------------------------------------------------------------

class ValidationError(Exception):
    """Exception pour les erreurs de validation"""
    pass

class ModelLoadingError(Exception):
    """Exception pour les erreurs de chargement des modèles"""
    pass

# -------------------------------------------------------------------
# GESTIONNAIRE DE MODÈLES
# -------------------------------------------------------------------

class ModelManager:
    """Gestionnaire centralisé des modèles avec lazy loading"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.meta_model = None
        self._initialized = True
        
        try:
            self._check_required_files()
            self.load_models()
        except Exception as e:
            logger.critical("Erreur d'initialisation du ModelManager: %s", e)
            raise ModelLoadingError(f"Impossible de charger les modèles: {e}")
    
    def _check_required_files(self):
        """Vérifier que tous les fichiers requis existent"""
        missing_files = []
        for name, filename in Config.REQUIRED_MODELS.items():
            filepath = Config.MODELS_DIR / filename
            if not filepath.exists():
                missing_files.append(f"{name} ({filename})")
        
        if missing_files:
            error_msg = f"Fichiers manquants: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    def load_models(self):
        """Charger tous les modèles depuis le disque"""
        try:
            logger.info("Debut du chargement des modeles...")
            
            # Charger le scaler
            scaler_path = Config.MODELS_DIR / Config.REQUIRED_MODELS['scaler']
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("[OK] Scaler charge")
            
            # Charger les modèles de base
            model_mapping = {
                'Decision Tree': Config.REQUIRED_MODELS['decision_tree'],
                'Random Forest': Config.REQUIRED_MODELS['random_forest'],
                'SVM': Config.REQUIRED_MODELS['svm'],
                'Naive Bayes': Config.REQUIRED_MODELS['naive_bayes']
            }
            
            for name, filename in model_mapping.items():
                model_path = Config.MODELS_DIR / filename
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                logger.info("[OK] Modele charge: %s", name)
            
            # Charger le méta-modèle
            meta_path = Config.MODELS_DIR / Config.REQUIRED_MODELS['meta_model']
            with open(meta_path, 'rb') as f:
                self.meta_model = pickle.load(f)
            logger.info("[OK] Meta-modele KNN charge")
            
            logger.info("=== Tous les modeles charges avec succes ===")
            
        except Exception as e:
            logger.error("ERREUR chargement modeles: %s", e)
            logger.error(traceback.format_exc())
            raise
    
    def is_ready(self):
        """Vérifier si tous les modèles sont chargés"""
        return (self.scaler is not None and 
                self.meta_model is not None and 
                len(self.models) == 4)

# Initialiser le gestionnaire de modèles
try:
    model_manager = ModelManager()
    logger.info("[OK] Gestionnaire de modeles initialise")
except Exception as e:
    logger.critical("Impossible d'initialiser les modeles: %s", e)
    model_manager = None

# -------------------------------------------------------------------
# VALIDATION DES DONNÉES
# -------------------------------------------------------------------

def validate_sensor_data(data_dict: Dict[str, str]) -> Dict[str, float]:
    """
    Valider les données des capteurs
    """
    required_keys = ['temp', 'hum', 'light', 'co2', 'hr']
    
    # Vérifier les champs requis
    missing_keys = [k for k in required_keys if k not in data_dict or not data_dict[k]]
    if missing_keys:
        raise ValidationError(f"Champs manquants: {', '.join(missing_keys)}")
    
    validated_data = {}
    errors = []
    
    # Valider et convertir chaque valeur
    for key in required_keys:
        try:
            value = float(data_dict[key])
            
            # Vérifier les plages
            if key in Config.SENSOR_RANGES:
                ranges = Config.SENSOR_RANGES[key]
                if not (ranges['min'] <= value <= ranges['max']):
                    errors.append(
                        f"{key}: {value}{ranges['unit']} hors plage "
                        f"[{ranges['min']}, {ranges['max']}]{ranges['unit']}"
                    )
            
            validated_data[key] = value
            
        except ValueError:
            errors.append(f"{key}: valeur non numerique '{data_dict[key]}'")
        except Exception as e:
            errors.append(f"{key}: erreur de validation - {str(e)}")
    
    if errors:
        raise ValidationError(" | ".join(errors))
    
    # Ajouter le timestamp optionnel
    if 'time' in data_dict and data_dict['time']:
        try:
            validated_data['time'] = int(data_dict['time'])
        except ValueError:
            logger.warning("Timestamp invalide ignore: %s", data_dict['time'])
    
    return validated_data

# -------------------------------------------------------------------
# PIPELINE DE PRÉDICTION
# -------------------------------------------------------------------

@lru_cache(maxsize=128)
def cached_prediction(temp: float, hum: float, light: float, co2: float, hr: float) -> Tuple:
    """
    Pipeline de prédiction avec cache LRU
    """
    data_dict = {
        'temp': temp,
        'hum': hum,
        'light': light,
        'co2': co2,
        'hr': hr
    }
    
    return _prediction_pipeline(data_dict)

def _prediction_pipeline(data_dict: Dict[str, float]) -> Tuple:
    """
    Pipeline de prédiction principal
    """
    if not model_manager or not model_manager.is_ready():
        raise RuntimeError("Modeles non disponibles")
    
    try:
        # ---- 1) Préparation des features ----
        feature_vector = np.array([
            data_dict['temp'],
            data_dict['hum'],
            data_dict['light'],
            data_dict['co2'],
            data_dict['hr']
        ]).reshape(1, -1)
        
        # ---- 2) Scaling ----
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vector_scaled = model_manager.scaler.transform(feature_vector)
        
        # ---- 3) Prédictions des modèles de base ----
        meta_features_row = []
        base_predictions = {}
        all_models_details = {}
        
        for name, model in model_manager.models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probas = model.predict_proba(vector_scaled)[0]
                prediction = int(model.predict(vector_scaled)[0])
            
            # Calcul des métriques
            conf_max = float(np.max(probas))
            probas_sorted = np.sort(probas)
            margin = float(probas_sorted[-1] - probas_sorted[-2])
            
            # Meta-features
            meta_features_row.extend([conf_max, margin])
            
            # Stockage des résultats
            base_predictions[name] = {
                "prediction": prediction,
                "confidence": conf_max,
                "margin": margin,
                "proba_occupied": float(probas[1]) if len(probas) > 1 else float(probas[0]),
                "proba_unoccupied": float(probas[0]) if len(probas) > 1 else 1 - float(probas[0])
            }
            
            # Détails pour l'affichage
            all_models_details[name] = {
                'prediction': 'Occupe' if prediction == 1 else 'Inoccupe',
                'confidence': conf_max,
                'confidence_percent': f"{int(conf_max * 100)}%",
                'margin': margin,
                'margin_percent': f"{int(margin * 100)}%"
            }
        
        # ---- 4) Sélection du modèle par méta-modèle ----
        meta_features_row = np.array(meta_features_row).reshape(1, -1)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idx = int(model_manager.meta_model.predict(meta_features_row)[0])
            
            model_names = list(model_manager.models.keys())
            
            if 0 <= idx < len(model_names):
                selected_model = model_names[idx]
            else:
                logger.warning("Index de modele invalide %s, fallback sur confiance max", idx)
                selected_model = max(
                    base_predictions.items(),
                    key=lambda x: x[1]['confidence']
                )[0]
                
        except Exception as e:
            logger.error("Erreur meta-modele: %s", e)
            selected_model = list(model_manager.models.keys())[0]
        
        # ---- 5) Résultat final ----
        final_result = base_predictions[selected_model]
        label = "Occupe" if final_result['prediction'] == 1 else "Inoccupe"
        
        # Créer la liste des méta-features pour l'affichage
        meta_features_list = [
            {
                'model': name,
                'confidence': details['confidence'],
                'margin': details['margin']
            }
            for name, details in base_predictions.items()
        ]
        
        return (
            label,
            selected_model,
            final_result['confidence'],
            all_models_details,
            meta_features_list
        )
        
    except Exception as e:
        logger.error("Erreur dans le pipeline: %s", e)
        raise

# -------------------------------------------------------------------
# INITIALISATION FLASK
# -------------------------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)

# Sécurité CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Cache
cache = Cache(app)

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.RATE_LIMIT]
)

# -------------------------------------------------------------------
# MIDDLEWARE
# -------------------------------------------------------------------

@app.before_request
def before_request():
    """Middleware exécuté avant chaque requête"""
    request.start_time = datetime.now()

@app.after_request
def after_request(response):
    """Middleware exécuté après chaque requête"""
    if hasattr(request, 'start_time'):
        processing_time = (datetime.now() - request.start_time).total_seconds() * 1000
        logger.debug(
            "%s %s - %s - %.2fms",
            request.method, request.path, response.status_code, processing_time
        )
    return response

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@limiter.limit("30/minute")
def index():
    """Route principale - Interface web"""
    prediction = None
    model_name = None
    confidence_percent = None
    confidence_float = None
    all_models_details = {}
    meta_features_list = []
    
    form_values = {"temp": "", "hum": "", "light": "", "co2": "", "hr": "", "time": ""}
    
    if request.method == "POST":
        for key in form_values.keys():
            form_values[key] = request.form.get(key, "").strip()
        
        try:
            validated_data = validate_sensor_data(form_values)
            start_time = datetime.now()
            
            if all(form_values[k] for k in ['temp', 'hum', 'light', 'co2', 'hr']):
                prediction_result = cached_prediction(
                    float(form_values['temp']),
                    float(form_values['hum']),
                    float(form_values['light']),
                    float(form_values['co2']),
                    float(form_values['hr'])
                )
            else:
                prediction_result = _prediction_pipeline(validated_data)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            prediction, model_name, confidence_float, all_models_details, meta_features_list = prediction_result
            confidence_percent = f"{int(round(confidence_float * 100))}%"
            
            logger.info(
                "Prediction: %s | Modele: %s | Confiance: %s | Temps: %.2fms",
                prediction, model_name, confidence_percent, processing_time
            )
            
        except ValidationError as e:
            flash(f"Erreur de validation : {str(e)}", "danger")
            logger.warning("Validation error: %s", e)
            
        except Exception as e:
            flash(f"Erreur lors de la prediction : {str(e)}", "danger")
            logger.error("Prediction error: %s", e)
    
    return render_template(
        "index.html",
        prediction=prediction,
        model_name=model_name,
        confidence_percent=confidence_percent,
        confidence_float=confidence_float,
        form=form_values,
        all_models_details=all_models_details,
        meta_features_list=meta_features_list,
        sensor_ranges=Config.SENSOR_RANGES
    )

@app.route("/api/predict", methods=["POST"])
@limiter.limit("60/minute")
@cache.cached(timeout=300, query_string=True)
def api_predict():
    """API REST pour les prédictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Donnees JSON requises",
                "success": False
            }), 400
        
        validated_data = validate_sensor_data(data)
        start_time = datetime.now()
        prediction_result = _prediction_pipeline(validated_data)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        label, model_name, confidence_float, all_models_details, meta_features_list = prediction_result
        
        response = {
            "success": True,
            "prediction": label,
            "model": model_name,
            "confidence": confidence_float,
            "confidence_percent": f"{int(confidence_float * 100)}%",
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
            "all_models": all_models_details,
            "meta_features": meta_features_list
        }
        
        logger.info("API Prediction - %s - %s - %.3f", label, model_name, confidence_float)
        
        return jsonify(response), 200
        
    except ValidationError as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 400
        
    except Exception as e:
        logger.error("API Error: %s", e)
        return jsonify({
            "error": "Erreur interne du serveur",
            "success": False
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint de vérification de santé"""
    health_status = {
        "status": "healthy" if model_manager and model_manager.is_ready() else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model_manager is not None,
        "models_count": len(model_manager.models) if model_manager else 0,
        "version": "1.0.0"
    }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return jsonify(health_status), status_code

@app.route("/api/sensor-ranges", methods=["GET"])
def get_sensor_ranges():
    """Retourne les plages de valeurs acceptées pour les capteurs"""
    return jsonify({
        "sensor_ranges": Config.SENSOR_RANGES,
        "timestamp": datetime.now().isoformat()
    }), 200

# -------------------------------------------------------------------
# GESTION DES ERREURS
# -------------------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                          error_code=404, 
                          error_message="Page non trouvee"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal Server Error: %s", error)
    return render_template('error.html', 
                          error_code=500,
                          error_message="Erreur interne du serveur"), 500

@app.errorhandler(429)
def ratelimit_error(error):
    return jsonify({
        "error": "Trop de requetes. Veuillez ralentir.",
        "success": False
    }), 429

# -------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# -------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Créer le répertoire des logs
        log_dir = Path(Config.LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)
        
        if model_manager and model_manager.is_ready():
            print("=" * 60)
            print("DETECTION D'OCCUPATION - META LEARNING")
            print(f"Server: http://{Config.HOST}:{Config.PORT}")
            print(f"Debug mode: {Config.DEBUG}")
            print(f"Cache timeout: {Config.CACHE_DEFAULT_TIMEOUT}s")
            print(f"Rate limit: {Config.RATE_LIMIT}")
            print(f"Log level: {Config.LOG_LEVEL}")
            print("=" * 60)
            
            app.run(
                host=Config.HOST,
                port=Config.PORT,
                debug=Config.DEBUG,
                threaded=True,
                use_reloader=False
            )
        else:
            print("=" * 60)
            print("ERREUR: Les modeles n'ont pas pu etre charges.")
            print("Verifiez que tous les fichiers .pkl sont presents dans 'models/'")
            print("Fichiers requis:")
            for name, filename in Config.REQUIRED_MODELS.items():
                filepath = Config.MODELS_DIR / filename
                status = "[OK]" if filepath.exists() else "[MISSING]"
                print(f"  {status} {filename}")
            print("=" * 60)
            
    except Exception as e:
        print(f"ERREUR FATALE: {e}")