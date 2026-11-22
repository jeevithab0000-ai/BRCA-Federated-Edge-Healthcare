import os
import sys


EDGE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HOSPITAL_DIR = os.path.dirname(
    EDGE_DIR
)

HOSPITALS_DIR = os.path.dirname(
    HOSPITAL_DIR
)

PROJECT_DIR = os.path.dirname(
    HOSPITALS_DIR
)


sys.path.append(
    PROJECT_DIR
)


from federated_client import start_hospital_client


DATASET_PATH = os.path.join(
    HOSPITAL_DIR,
    "data.csv"
)


SCALER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "scaler.pkl"
)


start_hospital_client(

    hospital_name="Hospital B",

    dataset_path=DATASET_PATH,

    scaler_path=SCALER_PATH,

    server_address="127.0.0.1:8080"

)