import os
import pickle
import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


# ============================================================
# BRCA FEDERATED EDGE HEALTHCARE API
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# CONFIGURATION
# ============================================================

EDGE_DEVICE = "EDGE-HOSPITAL-A"
MODEL_VERSION = "FED-GLOBAL-V1"
FEDERATED_ROUND = 10
TOTAL_ROUNDS = 10
HOSPITAL_COUNT = 4


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension"
]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
scaler = None

model_error = None
scaler_error = None


# ============================================================
# SERIALIZATION LOADER
# ============================================================

def load_serialized_file(path):
    """
    Load a pickle/joblib serialized file.

    Tries joblib first and pickle second.
    """

    errors = []

    # --------------------------------------------------------
    # METHOD 1: JOBLIB
    # --------------------------------------------------------

    try:
        return joblib.load(path)

    except Exception as e:
        errors.append(
            f"joblib.load failed: {type(e).__name__}: {e}"
        )

    # --------------------------------------------------------
    # METHOD 2: PICKLE
    # --------------------------------------------------------

    try:
        with open(path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        errors.append(
            f"pickle.load failed: {type(e).__name__}: {e}"
        )

    # --------------------------------------------------------
    # BOTH FAILED
    # --------------------------------------------------------

    raise RuntimeError(
        " | ".join(errors)
    )


# ============================================================
# LOAD GLOBAL MODEL
# ============================================================

print()
print("=" * 70)
print("             BRCA FEDERATED EDGE API")
print("=" * 70)

print()
print("Project directory:")
print(BASE_DIR)

print()
print("Global model:")
print(MODEL_PATH)

print()
print("Scaler:")
print(SCALER_PATH)


print()
print("-" * 70)
print("LOADING GLOBAL MODEL")
print("-" * 70)

if not os.path.exists(MODEL_PATH):

    model_error = f"Model file not found: {MODEL_PATH}"

    print()
    print("!!! GLOBAL MODEL LOAD FAILED !!!")
    print(model_error)

else:

    try:

        loaded_model = load_serialized_file(MODEL_PATH)

        # ----------------------------------------------------
        # MODEL PACKAGE
        # ----------------------------------------------------

        if isinstance(loaded_model, dict):

            print()
            print("Model package detected.")
            print("Keys:", list(loaded_model.keys()))

            if "model" in loaded_model:

                model = loaded_model["model"]

            else:

                raise ValueError(
                    "Model package does not contain 'model' key."
                )

            if "num_features" in loaded_model:

                print(
                    "Number of features:",
                    loaded_model["num_features"]
                )

            if "classes" in loaded_model:

                print(
                    "Classes:",
                    loaded_model["classes"]
                )

            if "algorithm" in loaded_model:

                print(
                    "Algorithm:",
                    loaded_model["algorithm"]
                )

            if "rounds" in loaded_model:

                print(
                    "Federated rounds:",
                    loaded_model["rounds"]
                )

        else:

            model = loaded_model

        # ----------------------------------------------------
        # MODEL VALIDATION
        # ----------------------------------------------------

        if not hasattr(model, "predict"):

            raise ValueError(
                "Loaded object does not have a predict() method."
            )

        print()
        print("GLOBAL MODEL LOADED SUCCESSFULLY")
        print("Model type:", type(model).__name__)

        if hasattr(model, "coef_"):

            print(
                "Coefficient shape:",
                model.coef_.shape
            )

        if hasattr(model, "intercept_"):

            print(
                "Intercept:",
                model.intercept_
            )

        if hasattr(model, "classes_"):

            print(
                "Model classes:",
                model.classes_
            )

    except Exception as e:

        model_error = e

        print()
        print("!!! GLOBAL MODEL LOAD FAILED !!!")
        print("Error type:", type(e).__name__)
        print("Error:", e)


# ============================================================
# LOAD SCALER
# ============================================================

print()
print("-" * 70)
print("LOADING SCALER")
print("-" * 70)

if not os.path.exists(SCALER_PATH):

    scaler_error = f"Scaler file not found: {SCALER_PATH}"

    print()
    print("!!! SCALER LOAD FAILED !!!")
    print(scaler_error)

else:

    try:

        loaded_scaler = load_serialized_file(SCALER_PATH)

        # ----------------------------------------------------
        # SCALER PACKAGE
        # ----------------------------------------------------

        if isinstance(loaded_scaler, dict):

            print()
            print("Scaler package detected.")
            print("Keys:", list(loaded_scaler.keys()))

            if "scaler" in loaded_scaler:

                scaler = loaded_scaler["scaler"]

            else:

                raise ValueError(
                    "Scaler package does not contain 'scaler' key."
                )

            # ------------------------------------------------
            # GET FEATURE NAMES FROM PACKAGE
            # ------------------------------------------------

            if "feature_names" in loaded_scaler:

                package_features = loaded_scaler["feature_names"]

                if package_features:

                    FEATURE_NAMES = package_features

                    print(
                        "Feature names loaded from scaler package."
                    )

            if "num_features" in loaded_scaler:

                print(
                    "Number of features:",
                    loaded_scaler["num_features"]
                )

        else:

            scaler = loaded_scaler

        # ----------------------------------------------------
        # SCALER VALIDATION
        # ----------------------------------------------------

        if not hasattr(scaler, "transform"):

            raise ValueError(
                "Loaded scaler does not have transform() method."
            )

        print()
        print("SCALER LOADED SUCCESSFULLY")
        print("Scaler type:", type(scaler).__name__)

    except Exception as e:

        scaler_error = e

        print()
        print("!!! SCALER LOAD FAILED !!!")
        print("Error type:", type(e).__name__)
        print("Error:", e)


# ============================================================
# STARTUP STATUS
# ============================================================

print()
print("=" * 70)
print("        BRCA FEDERATED EDGE HEALTHCARE API")
print("=" * 70)

print()
print("Configuration")
print("-" * 70)

print("Edge Device       :", EDGE_DEVICE)
print("Model Version     :", MODEL_VERSION)
print(
    "Federated Round   :",
    f"{FEDERATED_ROUND}/{TOTAL_ROUNDS}"
)
print("Hospitals         :", HOSPITAL_COUNT)
print("Features          :", len(FEATURE_NAMES))

print()
print("Model Status")
print("-" * 70)

print(
    "Global Model      :",
    "LOADED" if model is not None else "NOT LOADED"
)

print(
    "Scaler            :",
    "LOADED" if scaler is not None else "NOT LOADED"
)

if model_error:

    print()
    print("Model error:", model_error)

if scaler_error:

    print()
    print("Scaler error:", scaler_error)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "project": "BRCA Federated Edge Healthcare",
        "status": "running",
        "edge_device": EDGE_DEVICE,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "federated_round": FEDERATED_ROUND,
        "total_rounds": TOTAL_ROUNDS,
        "hospitals": HOSPITAL_COUNT
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "healthy"
        if model is not None and scaler is not None
        else "degraded",

        "edge_device": EDGE_DEVICE,

        "model_loaded": model is not None,

        "scaler_loaded": scaler is not None,

        "model_version": MODEL_VERSION,

        "federated_round": FEDERATED_ROUND,

        "total_rounds": TOTAL_ROUNDS,

        "hospitals": HOSPITAL_COUNT,

        "features": len(FEATURE_NAMES)

    })


# ============================================================
# STATUS
# ============================================================

@app.route("/api/status", methods=["GET"])
def status():

    return jsonify({

        "edge_device": EDGE_DEVICE,

        "model": {
            "loaded": model is not None,
            "version": MODEL_VERSION,
            "algorithm": (
                type(model).__name__
                if model is not None
                else None
            )
        },

        "scaler": {
            "loaded": scaler is not None,
            "type": (
                type(scaler).__name__
                if scaler is not None
                else None
            )
        },

        "federated_learning": {
            "current_round": FEDERATED_ROUND,
            "total_rounds": TOTAL_ROUNDS,
            "hospitals": HOSPITAL_COUNT,
            "aggregation": "FedAvg"
        }

    })


# ============================================================
# FEATURES
# ============================================================

@app.route("/api/features", methods=["GET"])
def features():

    return jsonify({

        "count": len(FEATURE_NAMES),

        "features": FEATURE_NAMES

    })


# ============================================================
# HOSPITALS
# ============================================================

@app.route("/api/hospitals", methods=["GET"])
def hospitals():

    hospital_data = [

        {
            "id": "HOSPITAL-A",
            "name": "Hospital A",
            "edge_device": "EDGE-A",
            "status": "ONLINE",
            "patients": 143,
            "benign": 90,
            "malignant": 53
        },

        {
            "id": "HOSPITAL-B",
            "name": "Hospital B",
            "edge_device": "EDGE-B",
            "status": "ONLINE",
            "patients": 142,
            "benign": 89,
            "malignant": 53
        },

        {
            "id": "HOSPITAL-C",
            "name": "Hospital C",
            "edge_device": "EDGE-C",
            "status": "ONLINE",
            "patients": 142,
            "benign": 89,
            "malignant": 53
        },

        {
            "id": "HOSPITAL-D",
            "name": "Hospital D",
            "edge_device": "EDGE-D",
            "status": "ONLINE",
            "patients": 142,
            "benign": 89,
            "malignant": 53
        }

    ]

    return jsonify({

        "count": len(hospital_data),

        "hospitals": hospital_data

    })


# ============================================================
# FEDERATED STATUS
# ============================================================

@app.route("/api/federated/status", methods=["GET"])
def federated_status():

    return jsonify({

        "status": "COMPLETED",

        "algorithm": "FedAvg",

        "current_round": FEDERATED_ROUND,

        "total_rounds": TOTAL_ROUNDS,

        "clients": HOSPITAL_COUNT,

        "connected_clients": HOSPITAL_COUNT,

        "fit_fraction": 1.0,

        "evaluation_fraction": 1.0,

        "aggregation": "Federated Averaging",

        "global_accuracy": 0.9914

    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None:

        return jsonify({

            "success": False,

            "error": "Global model is not loaded.",

            "details": str(model_error)
            if model_error
            else None

        }), 500

    # --------------------------------------------------------
    # CHECK SCALER
    # --------------------------------------------------------

    if scaler is None:

        return jsonify({

            "success": False,

            "error": "Scaler is not loaded.",

            "details": str(scaler_error)
            if scaler_error
            else None

        }), 500

    # --------------------------------------------------------
    # READ REQUEST
    # --------------------------------------------------------

    try:

        data = request.get_json(silent=True)

        if data is None:

            return jsonify({

                "success": False,

                "error": "Request body must contain JSON data."

            }), 400

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURE_NAMES
            if feature not in data
        ]

        if missing_features:

            return jsonify({

                "success": False,

                "error": "Missing required features.",

                "missing_features": missing_features

            }), 400

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        values = []

        for feature in FEATURE_NAMES:

            value = data[feature]

            try:

                value = float(value)

            except (TypeError, ValueError):

                return jsonify({

                    "success": False,

                    "error": f"Invalid value for {feature}.",

                    "value": value

                }), 400

            values.append(value)

        X_df = pd.DataFrame(
            [values],
            columns=FEATURE_NAMES
        )

        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        X_scaled = scaler.transform(X_df)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(X_scaled)

        prediction_value = int(prediction[0])

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        benign_probability = None
        malignant_probability = None
        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_scaled)[0]

            classes = list(model.classes_)

            probability_map = {
                int(cls): float(prob)
                for cls, prob in zip(
                    classes,
                    probabilities
                )
            }

            benign_probability = probability_map.get(
                0,
                0.0
            )

            malignant_probability = probability_map.get(
                1,
                0.0
            )

            confidence = max(
                benign_probability,
                malignant_probability
            )

        # ----------------------------------------------------
        # DIAGNOSIS
        # ----------------------------------------------------

        if prediction_value == 0:

            diagnosis = "BENIGN"

        else:

            diagnosis = "MALIGNANT"

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "diagnosis": diagnosis,

            "prediction": prediction_value,

            "confidence": round(
                confidence * 100,
                2
            ) if confidence is not None else None,

            "benign_probability": round(
                benign_probability * 100,
                2
            ) if benign_probability is not None else None,

            "malignant_probability": round(
                malignant_probability * 100,
                2
            ) if malignant_probability is not None else None,

            "edge_device": EDGE_DEVICE,

            "algorithm": type(model).__name__,

            "model_version": MODEL_VERSION,

            "federated_round": FEDERATED_ROUND

        })

    # --------------------------------------------------------
    # PREDICTION ERROR
    # --------------------------------------------------------

    except Exception as e:

        return jsonify({

            "success": False,

            "error": "Prediction failed.",

            "details": str(e),

            "error_type": type(e).__name__

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error": "API endpoint not found."

    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "success": False,

        "error": "Internal server error."

    }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("API Endpoints")
    print("-" * 70)

    print(
        "Home              : "
        "http://127.0.0.1:5000/"
    )

    print(
        "Health            : "
        "http://127.0.0.1:5000/api/health"
    )

    print(
        "Status            : "
        "http://127.0.0.1:5000/api/status"
    )

    print(
        "Features          : "
        "http://127.0.0.1:5000/api/features"
    )

    print(
        "Hospitals         : "
        "http://127.0.0.1:5000/api/hospitals"
    )

    print(
        "Federated Status  : "
        "http://127.0.0.1:5000/api/federated/status"
    )

    print(
        "Prediction        : "
        "http://127.0.0.1:5000/api/predict"
    )

    print("=" * 70)
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )