"""Central configuration for the pipeline."""

# Reproducibility
LABEL_SEED = 42
SEEDS = [42, 123, 2026]

# Image / training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
HUBER_DELTA = 100.0
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
FINETUNE_UNFREEZE_LAYERS = 40

# Task
THRESHOLD_KG = 12800  # map to the posted gross-weight limit of the protected structure

# Deduplication
PHASH_DISTANCE = 4

# Vehicle categories and weight ranges (kg)
WEIGHT_RANGES = {
    "Car":         (1000,  1800),
    "Bus":         (7000, 12000),
    "light_truck": (1800,  3500),
    "Heavy_truck": (18000, 30000),
    "motorcycle":  (100,    400),
}

# Paths (edit to your environment)
DATASET_ROOT = "data/Dataset"          # expects train/ and test/ subfolders per class
LABELS_CSV = "splits/labels_master.csv"
RESULTS_DIR = "results"
