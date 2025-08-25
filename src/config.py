import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

# File names
RAW_FILE = os.path.join(DATA_RAW, "kpi_dataset.csv")
PROCESSED_FILE = os.path.join(DATA_PROCESSED, "kpi_cleaned.csv")

TRAIN_FILE = os.path.join(DATA_RAW, "FS-data-80475.csv")
TEST_FILE = os.path.join(DATA_RAW, "FS-data-80475-2025-all-months.csv")
SAMPLE_FILE = os.path.join(DATA_RAW, "sample_data.csv")
