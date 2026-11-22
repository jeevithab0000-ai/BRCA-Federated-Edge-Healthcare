import os
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "brca_processed.csv"
)

HOSPITALS_DIR = os.path.join(
    BASE_DIR,
    "hospitals"
)


# ============================================================
# SETTINGS
# ============================================================

HOSPITALS = [
    "hospital_A",
    "hospital_B",
    "hospital_C",
    "hospital_D"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("WDBC DATASET - 4 HOSPITAL SPLITTER")
print("=" * 60)

print("\nProject directory:")
print(BASE_DIR)

print("\nInput dataset:")
print(INPUT_FILE)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("\nERROR: Processed dataset not found!")

    print("\nExpected:")
    print(INPUT_FILE)

    print("\nFirst run:")
    print("preprocessing/preprocess.py")

    raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")

print("Total records:", len(df))
print("Total columns:", len(df.columns))


# ============================================================
# CHECK TARGET
# ============================================================

if "diagnosis" not in df.columns:

    print("\nERROR: 'diagnosis' target column not found!")

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise SystemExit


print("\nTarget distribution BEFORE splitting:")

print(df["diagnosis"].value_counts())


# ============================================================
# SHUFFLE DATA
# ============================================================

# Randomly shuffle the records so that each hospital
# receives a representative mixture of patients.

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# CREATE HOSPITAL FOLDERS
# ============================================================

os.makedirs(
    HOSPITALS_DIR,
    exist_ok=True
)


# ============================================================
# SPLIT DATA INTO 4 HOSPITALS
# ============================================================

hospital_data = []

total_records = len(df)

base_size = total_records // len(HOSPITALS)

remainder = total_records % len(HOSPITALS)

start = 0


for i, hospital in enumerate(HOSPITALS):

    # Give the first hospitals one extra record
    # when the total cannot be divided equally.

    size = base_size

    if i < remainder:
        size += 1

    end = start + size

    hospital_df = df.iloc[
        start:end
    ].copy()

    hospital_data.append(
        hospital_df
    )

    start = end


# ============================================================
# SAVE EACH HOSPITAL DATASET
# ============================================================

print("\n")
print("=" * 60)
print("CREATING HOSPITAL DATASETS")
print("=" * 60)


for hospital, hospital_df in zip(
    HOSPITALS,
    hospital_data
):

    hospital_dir = os.path.join(
        HOSPITALS_DIR,
        hospital
    )

    edge_dir = os.path.join(
        hospital_dir,
        "edge"
    )

    # Create directories

    os.makedirs(
        hospital_dir,
        exist_ok=True
    )

    os.makedirs(
        edge_dir,
        exist_ok=True
    )

    # Data file

    output_file = os.path.join(
        hospital_dir,
        "data.csv"
    )

    # Save

    hospital_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + hospital)

    print(
        "Records:",
        len(hospital_df)
    )

    print(
        "Saved:",
        output_file
    )

    print("Target distribution:")

    print(
        hospital_df["diagnosis"].value_counts()
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 60)
print("HOSPITAL SPLITTING COMPLETED")
print("=" * 60)

print("\nTotal original records:")
print(total_records)

print("\nRecords distributed:")

total_split = 0

for hospital, hospital_df in zip(
    HOSPITALS,
    hospital_data
):

    count = len(hospital_df)

    total_split += count

    print(
        f"{hospital}: {count} records"
    )


print("\nTotal distributed:")
print(total_split)


# ============================================================
# VERIFY
# ============================================================

if total_split == total_records:

    print("\nSUCCESS!")
    print(
        "All records were distributed correctly."
    )

else:

    print("\nERROR!")
    print(
        "Record count mismatch."
    )


# ============================================================
# FINAL STRUCTURE
# ============================================================

print("\n")
print("=" * 60)
print("EXPECTED PROJECT STRUCTURE")
print("=" * 60)

print("""
hospitals/
│
├── hospital_A/
│   ├── data.csv
│   └── edge/
│
├── hospital_B/
│   ├── data.csv
│   └── edge/
│
├── hospital_C/
│   ├── data.csv
│   └── edge/
│
└── hospital_D/
    ├── data.csv
    └── edge/
""")


print("=" * 60)
print("NEXT STEP: CREATE EDGE A, B, C AND D")
print("=" * 60)