import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler


print("=" * 70)
print("CREATING SHARED WDBC SCALER")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# models folder -> project folder
PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

HOSPITALS_DIR = os.path.join(
    PROJECT_DIR,
    "hospitals"
)

SCALER_FILE = os.path.join(
    CURRENT_DIR,
    "scaler.pkl"
)


# ============================================================
# HOSPITAL FILES
# ============================================================

hospital_files = [

    os.path.join(
        HOSPITALS_DIR,
        "hospital_A",
        "data.csv"
    ),

    os.path.join(
        HOSPITALS_DIR,
        "hospital_B",
        "data.csv"
    ),

    os.path.join(
        HOSPITALS_DIR,
        "hospital_C",
        "data.csv"
    ),

    os.path.join(
        HOSPITALS_DIR,
        "hospital_D",
        "data.csv"
    )

]


# ============================================================
# LOAD ALL HOSPITAL DATA
# ============================================================

datasets = []

for file in hospital_files:

    print("\nChecking:")
    print(file)

    if not os.path.exists(file):

        print("ERROR: File not found!")

        continue

    df = pd.read_csv(file)

    print("Loaded:", df.shape)

    datasets.append(df)


if len(datasets) == 0:

    print("\nERROR: No hospital datasets found.")

    raise SystemExit


# ============================================================
# COMBINE DATA
# ============================================================

full_data = pd.concat(
    datasets,
    ignore_index=True
)

print("\nCombined dataset shape:")
print(full_data.shape)


# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

drop_columns = [

    "diagnosis",
    "target",
    "id",
    "ID",
    "Unnamed: 0"

]


for column in drop_columns:

    if column in full_data.columns:

        full_data = full_data.drop(
            columns=[column]
        )


# ============================================================
# KEEP NUMERIC FEATURES
# ============================================================

X = full_data.select_dtypes(
    include=["number"]
).copy()


print("\nNumber of features:")
print(X.shape[1])


print("\nFeature names:")
print(X.columns.tolist())


# ============================================================
# CLEAN DATA
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# CREATE SCALER
# ============================================================

scaler = StandardScaler()

scaler.fit(X)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_FILE
)


print("\n" + "=" * 70)
print("SHARED SCALER CREATED SUCCESSFULLY")
print("=" * 70)

print("\nScaler saved at:")
print(SCALER_FILE)

print("\nNumber of features:")
print(scaler.n_features_in_)