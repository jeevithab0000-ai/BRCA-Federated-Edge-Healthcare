import os
import sys
import joblib
import numpy as np
import pandas as pd
import flwr as fl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# START HOSPITAL CLIENT
# ============================================================

def start_hospital_client(
    hospital_name,
    dataset_path,
    scaler_path,
    server_address="127.0.0.1:8080"
):

    print("\n")
    print("=" * 70)
    print(f"{hospital_name.upper()} / FEDERATED EDGE DEVICE")
    print("=" * 70)

    print("\nDataset:")
    print(dataset_path)

    print("\nScaler:")
    print(scaler_path)


    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not os.path.exists(dataset_path):

        print("\nERROR: Dataset file not found!")

        print(dataset_path)

        sys.exit()


    # ========================================================
    # CHECK SCALER
    # ========================================================

    if not os.path.exists(scaler_path):

        print("\nERROR: Scaler file not found!")

        print(scaler_path)

        sys.exit()


    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading hospital dataset...")

    df = pd.read_csv(dataset_path)

    print("\nDataset Shape:")
    print(df.shape)


    # ========================================================
    # CHECK TARGET COLUMN
    # ========================================================

    if "diagnosis" not in df.columns:

        print("\nERROR: diagnosis column not found!")

        print("\nAvailable columns:")

        print(df.columns.tolist())

        sys.exit()


    # ========================================================
    # SHOW TARGET VALUES
    # ========================================================

    print("\nTarget values before conversion:")

    print(
        df["diagnosis"].value_counts()
    )


    # ========================================================
    # CONVERT TARGET VALUES
    # ========================================================

    def convert_target(value):

        value = str(value).strip().upper()

        if value in ["0", "BENIGN", "B"]:

            return 0

        elif value in ["1", "MALIGNANT", "M"]:

            return 1

        else:

            return np.nan


    df["diagnosis"] = df["diagnosis"].apply(
        convert_target
    )


    # ========================================================
    # REMOVE INVALID TARGET ROWS
    # ========================================================

    df = df.dropna(
        subset=["diagnosis"]
    )


    df["diagnosis"] = df["diagnosis"].astype(
        int
    )


    print("\nTarget values after conversion:")

    print(
        df["diagnosis"].value_counts()
    )


    print("\nAvailable classes:")

    print(
        np.unique(
            df["diagnosis"]
        )
    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_columns = [

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


    # ========================================================
    # CHECK FEATURES
    # ========================================================

    missing_columns = [

        column

        for column in feature_columns

        if column not in df.columns

    ]


    if len(missing_columns) > 0:

        print("\nERROR: Missing feature columns!")

        print(missing_columns)

        print("\nAvailable columns:")

        print(
            df.columns.tolist()
        )

        sys.exit()


    # ========================================================
    # CREATE FEATURES AND TARGET
    # ========================================================

    X = df[
        feature_columns
    ].copy()


    y = df[
        "diagnosis"
    ].copy()


    # ========================================================
    # CONVERT FEATURES TO NUMERIC
    # ========================================================

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )


    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    if X.isnull().sum().sum() > 0:

        print(
            "\nMissing feature values found."
        )

        print(
            "Replacing missing values with column median..."
        )

        X = X.fillna(
            X.median()
        )


    print("\nNumber of Features:")

    print(
        X.shape[1]
    )


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    print("\nTraining distribution:")

    print(
        y_train.value_counts()
    )


    print("\nTesting distribution:")

    print(
        y_test.value_counts()
    )


    # ========================================================
    # LOAD SHARED SCALER
    # ========================================================

    print("\nLoading shared scaler...")


    scaler_data = joblib.load(
        scaler_path
    )


    # ========================================================
    # IMPORTANT FIX
    # SCALER FILE MAY CONTAIN A DICTIONARY
    # ========================================================

    if isinstance(
        scaler_data,
        dict
    ):

        print(
            "\nScaler package detected."
        )

        print(
            "Scaler package keys:"
        )

        print(
            scaler_data.keys()
        )


        # Try common scaler keys

        if "scaler" in scaler_data:

            scaler = scaler_data[
                "scaler"
            ]

        elif "standard_scaler" in scaler_data:

            scaler = scaler_data[
                "standard_scaler"
            ]

        elif "model" in scaler_data:

            scaler = scaler_data[
                "model"
            ]

        else:

            print(
                "\nERROR: Could not find StandardScaler inside dictionary!"
            )

            print(
                "\nDictionary contents:"
            )

            print(
                scaler_data.keys()
            )

            sys.exit()


    else:

        scaler = scaler_data


    # ========================================================
    # CHECK SCALER
    # ========================================================

    print("\nScaler type:")

    print(
        type(scaler)
    )


    if not hasattr(
        scaler,
        "transform"
    ):

        print(
            "\nERROR: Loaded scaler does not support transform()!"
        )

        print(
            "Actual object:"
        )

        print(
            scaler
        )

        sys.exit()


    # ========================================================
    # SCALE TRAIN AND TEST DATA
    # ========================================================

    print(
        "\nApplying shared StandardScaler..."
    )


    X_train_scaled = scaler.transform(
        X_train
    )


    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    X_train_scaled = np.asarray(
        X_train_scaled,
        dtype=np.float64
    )


    X_test_scaled = np.asarray(
        X_test_scaled,
        dtype=np.float64
    )


    y_train = np.asarray(
        y_train,
        dtype=np.int64
    )


    y_test = np.asarray(
        y_test,
        dtype=np.int64
    )


    # ========================================================
    # INITIAL MODEL
    # ========================================================

    model = SGDClassifier(

        loss="log_loss",

        penalty="l2",

        alpha=0.0001,

        max_iter=1,

        tol=None,

        learning_rate="optimal",

        random_state=42,

        warm_start=True

    )


    # ========================================================
    # FEDERATED CLIENT
    # ========================================================

    class HospitalClient(
        fl.client.NumPyClient
    ):


        # ====================================================
        # INITIAL PARAMETERS
        # ====================================================

        def get_parameters(
            self,
            config
        ):


            # Initialize model if not fitted

            if not hasattr(
                model,
                "coef_"
            ):


                model.partial_fit(

                    X_train_scaled,

                    y_train,

                    classes=np.array(
                        [0, 1]
                    )

                )


            return [

                model.coef_.astype(
                    np.float64
                ),

                model.intercept_.astype(
                    np.float64
                )

            ]


        # ====================================================
        # SET GLOBAL PARAMETERS
        # ====================================================

        def set_parameters(
            self,
            parameters
        ):


            model.coef_ = np.asarray(

                parameters[0],

                dtype=np.float64

            )


            model.intercept_ = np.asarray(

                parameters[1],

                dtype=np.float64

            )


            model.classes_ = np.array(

                [0, 1]

            )


            model.n_features_in_ = (

                X_train_scaled.shape[1]

            )


        # ====================================================
        # FEDERATED TRAINING
        # ====================================================

        def fit(
            self,
            parameters,
            config
        ):


            print("\n")
            print("=" * 60)

            print(
                f"{hospital_name.upper()} - LOCAL FEDERATED TRAINING"
            )

            print("=" * 60)


            # -----------------------------------------------
            # RECEIVE GLOBAL MODEL
            # -----------------------------------------------

            self.set_parameters(
                parameters
            )


            print(
                "\nGlobal model parameters received."
            )


            # -----------------------------------------------
            # LOCAL TRAINING
            # -----------------------------------------------

            model.partial_fit(

                X_train_scaled,

                y_train,

                classes=np.array(
                    [0, 1]
                )

            )


            # -----------------------------------------------
            # TRAINING PREDICTION
            # -----------------------------------------------

            predictions = model.predict(

                X_train_scaled

            )


            accuracy = accuracy_score(

                y_train,

                predictions

            )


            print(

                f"\n{hospital_name} Training Accuracy: "

                f"{accuracy * 100:.2f}%"

            )


            print(

                "\nPredicted classes:"

            )

            print(

                np.unique(
                    predictions
                )

            )


            # -----------------------------------------------
            # RETURN PARAMETERS
            # -----------------------------------------------

            return (

                [

                    model.coef_.astype(
                        np.float64
                    ),

                    model.intercept_.astype(
                        np.float64
                    )

                ],

                len(
                    X_train_scaled
                ),

                {

                    "accuracy": float(
                        accuracy
                    )

                }

            )


        # ====================================================
        # FEDERATED EVALUATION
        # ====================================================

        def evaluate(
            self,
            parameters,
            config
        ):


            print("\n")
            print("=" * 60)

            print(
                f"{hospital_name.upper()} - FEDERATED EVALUATION"
            )

            print("=" * 60)


            # -----------------------------------------------
            # LOAD GLOBAL PARAMETERS
            # -----------------------------------------------

            self.set_parameters(
                parameters
            )


            # -----------------------------------------------
            # PREDICTION
            # -----------------------------------------------

            predictions = model.predict(

                X_test_scaled

            )


            # -----------------------------------------------
            # ACCURACY
            # -----------------------------------------------

            accuracy = accuracy_score(

                y_test,

                predictions

            )


            print(

                f"\n{hospital_name} Evaluation Accuracy: "

                f"{accuracy * 100:.2f}%"

            )


            print(

                "\nPredicted classes:"

            )

            print(

                np.unique(
                    predictions
                )

            )


            # -----------------------------------------------
            # CLASSIFICATION REPORT
            # -----------------------------------------------

            print(
                "\nClassification Report:"
            )


            print(

                classification_report(

                    y_test,

                    predictions,

                    labels=[0, 1],

                    target_names=[

                        "Benign",

                        "Malignant"

                    ],

                    zero_division=0

                )

            )


            # -----------------------------------------------
            # LOSS
            # -----------------------------------------------

            loss = 1.0 - accuracy


            return (

                float(loss),

                len(
                    X_test_scaled
                ),

                {

                    "accuracy": float(
                        accuracy
                    )

                }

            )


    # ========================================================
    # START FLOWER CLIENT
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"CONNECTING {hospital_name.upper()} TO FEDERATED SERVER"
    )

    print("=" * 70)


    print(
        f"\nServer Address: {server_address}"
    )


    fl.client.start_numpy_client(

        server_address=server_address,

        client=HospitalClient()

    )