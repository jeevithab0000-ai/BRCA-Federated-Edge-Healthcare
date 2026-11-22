import os
import joblib
import numpy as np
import flwr as fl

from sklearn.linear_model import SGDClassifier


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

SERVER_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


PROJECT_DIR = os.path.dirname(
    SERVER_DIR
)


MODEL_DIR = os.path.join(

    PROJECT_DIR,

    "models"

)


os.makedirs(

    MODEL_DIR,

    exist_ok=True

)


GLOBAL_MODEL_FILE = os.path.join(

    MODEL_DIR,

    "global_model.pkl"

)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_FEATURES = 30

NUM_ROUNDS = 10


# ============================================================
# CUSTOM FEDAVG STRATEGY
# ============================================================

class SaveModelStrategy(
    fl.server.strategy.FedAvg
):


    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.final_parameters = None


    # ========================================================
    # AGGREGATE AND STORE GLOBAL PARAMETERS
    # ========================================================

    def aggregate_fit(

        self,

        server_round,

        results,

        failures

    ):


        aggregated_parameters, aggregated_metrics = (

            super().aggregate_fit(

                server_round,

                results,

                failures

            )

        )


        # ----------------------------------------------------
        # STORE FINAL PARAMETERS
        # ----------------------------------------------------

        if aggregated_parameters is not None:

            self.final_parameters = (

                aggregated_parameters

            )


            print("\n")

            print(
                f"GLOBAL PARAMETERS AGGREGATED - ROUND {server_round}"
            )


        return (

            aggregated_parameters,

            aggregated_metrics

        )


# ============================================================
# WEIGHTED METRICS
# ============================================================

def weighted_average(
    metrics
):


    if not metrics:

        return {}


    total_examples = 0

    total_accuracy = 0.0


    for num_examples, metric in metrics:


        if "accuracy" in metric:


            total_examples += (

                num_examples

            )


            total_accuracy += (

                num_examples

                *

                metric["accuracy"]

            )


    if total_examples == 0:

        return {}


    return {

        "accuracy":

        total_accuracy / total_examples

    }


# ============================================================
# CREATE STRATEGY
# ============================================================

strategy = SaveModelStrategy(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=4,

    min_evaluate_clients=4,

    min_available_clients=4,

    fit_metrics_aggregation_fn=weighted_average,

    evaluate_metrics_aggregation_fn=weighted_average

)


# ============================================================
# SERVER DISPLAY
# ============================================================

print("\n")

print("=" * 70)

print(
    "        BRCA FEDERATED LEARNING SERVER"
)

print("=" * 70)


print(
    "\nFederated Learning Configuration"
)

print(
    "-" * 70
)


print(
    "Dataset              : WDBC Breast Cancer Dataset"
)

print(
    "Hospitals             : 4"
)

print(
    "Hospital A            : Edge Device A"
)

print(
    "Hospital B            : Edge Device B"
)

print(
    "Hospital C            : Edge Device C"
)

print(
    "Hospital D            : Edge Device D"
)

print(
    "Federated Algorithm   : FedAvg"
)

print(
    f"Federated Rounds      : {NUM_ROUNDS}"
)

print(
    "Minimum Fit Clients   : 4"
)

print(
    "Minimum Eval Clients  : 4"
)

print(
    "Fit Fraction          : 100%"
)

print(
    "Evaluation Fraction   : 100%"
)

print(
    "Server Address        : 127.0.0.1:8080"
)

print(
    "-" * 70
)


# ============================================================
# START SERVER
# ============================================================

print(
    "\nStarting Federated Server..."
)

print(
    "Waiting for Hospital A, B, C and D..."
)


history = fl.server.start_server(

    server_address="127.0.0.1:8080",

    config=fl.server.ServerConfig(

        num_rounds=NUM_ROUNDS

    ),

    strategy=strategy

)


# ============================================================
# FEDERATED TRAINING FINISHED
# ============================================================

print("\n")

print("=" * 70)

print(
    "FEDERATED TRAINING COMPLETED"
)

print("=" * 70)


# ============================================================
# CHECK FINAL PARAMETERS
# ============================================================

if strategy.final_parameters is None:


    print(
        "\nERROR: No final federated parameters received!"
    )


else:


    print(
        "\nConverting final federated parameters..."
    )


    # --------------------------------------------------------
    # CONVERT FLOWER PARAMETERS
    # --------------------------------------------------------

    parameters_ndarrays = (

        fl.common.parameters_to_ndarrays(

            strategy.final_parameters

        )

    )


    print(
        "\nNumber of parameter arrays:"
    )

    print(
        len(
            parameters_ndarrays
        )
    )


    # --------------------------------------------------------
    # GET COEFFICIENTS
    # --------------------------------------------------------

    coefficients = np.array(

        parameters_ndarrays[0],

        dtype=np.float64

    )


    intercept = np.array(

        parameters_ndarrays[1],

        dtype=np.float64

    )


    print(
        "\nCoefficient Shape:"
    )

    print(
        coefficients.shape
    )


    print(
        "\nIntercept Shape:"
    )

    print(
        intercept.shape
    )


    # ========================================================
    # CREATE GLOBAL SGD MODEL
    # ========================================================

    print(
        "\nCreating final global SGDClassifier..."
    )


    global_model = SGDClassifier(

        loss="log_loss",

        learning_rate="optimal",

        alpha=0.0001,

        random_state=42

    )


    # --------------------------------------------------------
    # SET LEARNED PARAMETERS
    # --------------------------------------------------------

    global_model.coef_ = coefficients


    global_model.intercept_ = intercept


    global_model.classes_ = np.array(

        [0, 1]

    )


    global_model.n_features_in_ = (

        NUM_FEATURES

    )


    # ========================================================
    # MODEL PACKAGE
    # ========================================================

    model_package = {

        "model": global_model,

        "num_features": NUM_FEATURES,

        "classes": {

            0: "BENIGN",

            1: "MALIGNANT"

        },

        "algorithm": "FedAvg with SGDClassifier",

        "rounds": NUM_ROUNDS

    }


    # ========================================================
    # SAVE MODEL
    # ========================================================

    print(
        "\nSaving global model..."
    )


    joblib.dump(

        model_package,

        GLOBAL_MODEL_FILE

    )


    print("\n")

    print("=" * 70)

    print(
        "GLOBAL MODEL SAVED SUCCESSFULLY"
    )

    print("=" * 70)


    print(
        "\nModel Location:"
    )

    print(
        GLOBAL_MODEL_FILE
    )


    print(
        "\nModel Classes:"
    )

    print(
        global_model.classes_
    )


    print(
        "\nNumber of Features:"
    )

    print(
        global_model.n_features_in_
    )


# ============================================================
# SERVER COMPLETED
# ============================================================

print("\n")

print("=" * 70)

print(
    "SERVER PROCESS COMPLETED"
)

print("=" * 70)